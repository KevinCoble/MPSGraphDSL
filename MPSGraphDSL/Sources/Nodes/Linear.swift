//
//  Linear.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 3/12/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///   Node for a 'Linear'' Neural Network layer, a linear layer that projects the last dimension of the input.  Similar to PyTorch.Linear
///
public class Linear : UnaryNode {
    let outputFeatures: Int
    let addBias: Bool
    let activationFunction: ActivationFunction
    
    var weightInitialization: WeightInitialization = .HeUniform
    var biasInitialValue: Double = 0.0
    
    var lossNode: String? = nil
    var weightLearningOptions: LearningOptions = LearningOptions(clipping: nil, optimizer: .stochasticGradientDescent)
    var biasLearningOptions: LearningOptions = LearningOptions(clipping: nil, optimizer: .stochasticGradientDescent)

    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    var totalParameterCount: Int = 0

    /// Add an affine linear transform to the graph.  Input size comes from the last dimension of the input tensor
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - outputFeatures: The size of the output of the layer.  Output shape will be input shape with last dimension replaced with this value
    ///   - addBias: (Optional) If true an additive bias term will be created.  Default is true
    ///   - activationFunction: (Optional) The activation function for the layer.  Default is .none
    ///   - name: The name for this node and its associated tensor.  The variable nodes will be named &ltname&gt_weights' and &ltname&gt_biases'
    public init(input: String? = nil, outputFeatures: Int, addBias: Bool = true, activationFunction: ActivationFunction = .none, name: String) {
        self.outputFeatures = outputFeatures
        self.addBias = addBias
        self.activationFunction = activationFunction
        switch (activationFunction) {
            case .none, .tanh, .sigmoid:
                weightInitialization = .XavierGlorotNormal
        case .relu, .leakyRelu, .leakyReluFromTensor, .gelu, .gelu_tanh_approx, .elu:
                weightInitialization = .HeNormal
        }
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        totalParameterCount = 0

        //  Get the input tensor
        var inputTensor = try graph.getUnaryTensor(name: inputName)
        var inputShape: TensorShape
        if let shape = inputTensor.shape {
            inputShape = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        let inputFeatures = inputShape.dimensions.last!
        
        //  Determine if we have an activation function
        let haveActivationFunction: Bool
        switch (activationFunction) {
        case .none:
            haveActivationFunction = false
        default:
            haveActivationFunction = true
        }

        //  Add the weights variable
        let weightShape = TensorShape([inputFeatures, outputFeatures])
        let weightType = DataType(from: inputTensor.dataType)
        let weights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: inputFeatures, numOutput: outputFeatures)
        let weightData = weights.getData()
        let weightName = graph.getFullName(name)! + "_weights"
        let weightTensor = graph.mpsgraph.variable(with: weightData, shape: weightShape.getMPSShape(), dataType: weights.type.getMPSDataType(), name: weightName)
        suffixes.append("_weights")
        addedTensors.append(weightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        let node = try Variable.createWeightInitializationVariable(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: inputFeatures, numOutput: outputFeatures, name: weightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: weightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  If this is a learning layer - add the weights to the list to get assignment operations for
        if let lossNode = lossNode {
            let learningVariable = LearningVariable(variable: node, tensor: weightTensor, loss: lossNode, learningOptions: weightLearningOptions)
            graph.learningVariables.append(learningVariable)
            totalParameterCount += weightShape.totalSize
        }
        
        //  If using a bias term, add the bias variable
        var biasTensor: MPSGraphTensor? = nil
        if (addBias) {
            //  Get the bias shape
            let biasShape = TensorShape([outputFeatures])
            
            let biases = CreateTensor.constantValues(type: weightType, shape: biasShape, initialValue: biasInitialValue)
            let biasData = biases.getData()
            let biasName = graph.getFullName(name)! + "_biases"
            biasTensor = graph.mpsgraph.variable(with: biasData, shape: biasShape.getMPSShape(), dataType: biases.type.getMPSDataType(), name: biasName)
            suffixes.append("_biases")
            addedTensors.append(biasTensor)
            
            //  If we are adding load or reset assignments, put this variable on the list for load assignments
            let node = Variable(dataType: weightType, shape: biasShape, initialValue: biasInitialValue, name: biasName)
            if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
                let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: biasTensor!, sourceTensor: nil)
                graph.loadResetAssignList.append(loadResetAssignInfo)
            }

            //  If this is a learning layer - add the biases to the list to get assignment operations for
            if let lossNode = lossNode {
                let learningVariable = LearningVariable(variable: node, tensor: biasTensor!, loss: lossNode, learningOptions: biasLearningOptions)
                graph.learningVariables.append(learningVariable)
                totalParameterCount += biasShape.totalSize
            }
       }
        
        //  If the input tensor is 1D, add a single row dimension so matrix multiply doesn't complain
        if (inputShape.numDimensions == 1) {
            let inputReshape = graph.mpsgraph.reshape(inputTensor, shape: [NSNumber(value: 1), NSNumber(value: inputShape.dimensions[0])], name: graph.getFullName(name)! + "_inputReshape")
            addedTensors.append(inputReshape)
            suffixes.append("_inputReshape")
            inputTensor = inputReshape
        }
        
        //  Add the matrix multiply
        let haveDownStreamOperation = (addBias || haveActivationFunction || inputShape.numDimensions == 1)
        let matrixTensorName : String
        if (haveDownStreamOperation) {
            matrixTensorName = graph.getFullName(name)! + "_matrixMult"
            suffixes.append("_matrixMult")
        }
        else {
            matrixTensorName = graph.getFullName(name)!
            targetIndices.append(suffixes.count)
            suffixes.append("")
        }
        let matrixMultTensor = graph.mpsgraph.matrixMultiplication(primary: inputTensor, secondary: weightTensor, name: matrixTensorName)
        addedTensors.append(matrixMultTensor)
        
        //  If the input was one dimensional, reshape the output to 1 dimensional
        var finalTensor = matrixMultTensor
        if (inputShape.numDimensions == 1) {
            let reshapeTensorName : String
            if (addBias || haveActivationFunction) {
                reshapeTensorName = graph.getFullName(name)! + "_outputReshape"
                suffixes.append("_outputReshape")
            }
            else {
                reshapeTensorName = graph.getFullName(name)!
                targetIndices.append(suffixes.count)
                suffixes.append("")
            }
            let outputReshape = graph.mpsgraph.reshape(matrixMultTensor, shape: [NSNumber(value: outputFeatures)], name: reshapeTensorName)
            addedTensors.append(outputReshape)
            finalTensor = outputReshape
        }

        //  If using a bias term, add the addition operation
        if (addBias) {
            if let biasTensor = biasTensor {
                var biasTensorName = graph.getFullName(name)!
                if (haveActivationFunction) {
                    biasTensorName += "_biasAdded"
                    suffixes.append("_biasAdded")
                }
                else {
                    targetIndices.append(suffixes.count)
                    suffixes.append("")
                }
                let additionTensor = graph.mpsgraph.addition(matrixMultTensor, biasTensor, name: biasTensorName)
                addedTensors.append(additionTensor)
                finalTensor = additionTensor
            }
            else {
                throw GenericMPSGraphDSLErrors.InternalError
            }
        }
        
        //  Add any activation function
        if let activationTensor = try activationFunction.addActivation(graph: graph, inputTensor: finalTensor, name: name) {
            addedTensors.append(activationTensor)
            targetIndices.append(suffixes.count)
            suffixes.append("")
        }

        return addedTensors
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }

    ///  Modifier to set the initialization info for the random initialization of the weights  Default depends on activation function
    public func weightInitialization(initializerInfo: WeightInitialization) -> Linear {
        weightInitialization = initializerInfo
        return self
    }
    
    ///  Modifier to set the initialization value for the initialization of the biases.  Default is 0
    public func biasInitialValue(initialValue: Double) -> Linear {
        self.biasInitialValue = initialValue
        return self
    }
    
    /// Modifier to configure the layer's variables to learn
    /// - Parameters:
    ///   - mode: lossNode: the name of the loss calculation in the Graph
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String) -> Linear {
        self.lossNode = lossNode
        return self
    }
    
    /// Modifier to set the optimizer used for learning the weight variable
    /// - Parameter optimizer: the optimizer method to use for learning the weights.  Defaults to stochastic gradient descent
    /// - Returns: The modified layer
    public func weightOptimizer(_ optimizer: LearningOptimizer) -> Linear {
        weightLearningOptions = LearningOptions(clipping: weightLearningOptions.clipping, optimizer: optimizer)
        return self
    }
    
    /// Modifier to set all the learning options for the weight variable
    /// - Parameter options: The LearningOptions structure with all the learning options
    /// - Returns: The modified layer
    public func weightLearningOptions(_ options: LearningOptions) -> Linear {
        weightLearningOptions = options
        return self
    }
    
    /// Modifier to set the optimizer used for learning the bias variable
    /// - Parameter optimizer: the optimizer method to use for learning the biases.  Defaults to stochastic gradient descent
    /// - Returns: The modified layer
    public func biasOptimizer(_ optimizer: LearningOptimizer) -> Linear {
        biasLearningOptions = LearningOptions(clipping: biasLearningOptions.clipping, optimizer: optimizer)
        return self
    }
    
    /// Modifier to set all the learning options for the bias variable
    /// - Parameter options: The LearningOptions structure with all the learning options
    /// - Returns: The modified layer
    public func biasLearningOptions(_ options: LearningOptions) -> Linear {
        biasLearningOptions = options
        return self
    }

    override func getNumberOfParameters() throws -> Int {
        return totalParameterCount
    }
}
