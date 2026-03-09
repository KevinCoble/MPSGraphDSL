//
//  FullyConnectedLayer.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 12/8/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

/// Method for initializing weights or biases in a FullyConnectedLayer
public enum WeightInitialization {
    ///  Uniformly distributed between the minimum and the maximum
    case uniform(min: Double, max: Double)
    ///  A normal distribution with the given mean and standard deviation
    case normal(mean: Double, standardDeviation: Double)
    ///  A uniform distribution using the Xavier/Glorot parameters.  Default for sigmoid and tanh activations
    case XavierGlorotUniform
    ///  A uniform distribution using the He parameters.  Default for ReLU and associated activations
    case HeUniform
    ///  A normal distribution using the Xavier/Glorot parameters.
    case XavierGlorotNormal
    ///  A uniform distribution using the He parameters
    case HeNormal
}

///   Node for a fully-connected Neural Network layer, with specified output shape and activation function
///
public class FullyConnectedLayer : UnaryNode {
    let outputShape: TensorShape
    let ignoreBroadcastShape: Bool
    let activationFunction: ActivationFunction
    var useBias: Bool = true
    var weightInitialization: WeightInitialization
    var biasInitialValue: Double = 0.0
    
    var lossNode: String? = nil
    var learningOptimizer: LearningOptimizer = .stochasticGradientDescent
    var gradientClipping: (min: Double, max: Double)? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    var totalParameterCount: Int = 0

    ///  Constructor for a fully connected node
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - outputShape: The shape of the output tensor.  This dictates the number of fully-connected nodes in the layer
    ///   - ignoreBroadcastShape: (Optional) If true all non-batch input dimensions are combined into one dimension for the matrix multiplication, making it truly 'fully connected'.  Default is false
    ///   - activationFunction: The activation function for the layer
    ///   - name: The name for this node and its associated tensor.  The variable nodes will be named &ltname&gt_weights' and &ltname&gt_biases'
    public init(input: String? = nil, outputShape: TensorShape, ignoreBroadcastShape: Bool = false, activationFunction: ActivationFunction, name: String) {
        self.outputShape = outputShape
        self.ignoreBroadcastShape = ignoreBroadcastShape
        self.activationFunction = activationFunction
        switch (activationFunction) {
            case .none, .tanh, .sigmoid:
                weightInitialization = .XavierGlorotNormal
        case .relu, .leakyRelu, .leakyReluFromTensor, .gelu, .elu:
                weightInitialization = .HeNormal
        }
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
//        Input.    Output.    ignore   Matrix     bias     numBroadcast    input reshape    output reshape
//
//        2           3        NA        2x3        3            0            no              no
//        4x2        4x3       no        2x3        3            1            no              no
//        4x2        4x3       yes       8x12       4x3          0            yes (8)         yes (4x3)
//        4x5x2      4x3       no        10x3       3            1            yes (4x10)      no
//        4x5x2      4x3       yes       20x12      4x3          0            yes (20)        yes (4x3)
//        4x3        4         no        12x4       4            0            no              no
        
        suffixes = []
        targetIndices = []
        totalParameterCount = 0

        //  See if we have an activation function
        let haveActivationFunction: Bool
        switch (activationFunction) {
        case .none:
            haveActivationFunction = false
        default:
            haveActivationFunction = true
        }

        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        var inputShape: TensorShape
        if let shape = inputTensor.shape {
            inputShape = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        //  If this is a batch graph, get the input shape minus the batch dimension
        var batchedInput = false
        if (graph.batchGraph) {
            if (!inputShape.firstDimensionIsBatchSize(graph.batchSize)) { throw GenericMPSGraphDSLErrors.InvalidShape }
            batchedInput = true
            inputShape = inputShape.shapeWithRemovedBatchDimension()
        }
        
        //  Find how many (non-batch) dimensions we are broadcasting across
        var numBroadcastDimensions = inputShape.numMatchingDimensions(outputShape)
        if (numBroadcastDimensions == inputShape.numDimensions) { numBroadcastDimensions -= 1 }     //  Must have at least one non-broadcast dimension
        if (numBroadcastDimensions == outputShape.numDimensions) { numBroadcastDimensions -= 1 }     //  Must have at least one non-broadcast dimension
        if (ignoreBroadcastShape) { numBroadcastDimensions = 0 }
        
        //  Get the needed input shape
        var neededInputShapeDimensions: [Int] = []
        if (numBroadcastDimensions > 0) {
            for dim in 0..<numBroadcastDimensions {
                neededInputShapeDimensions.append(inputShape.dimensions[dim])
            }
        }
        var totalInputNonBroadcastSize = 1
        for dim in numBroadcastDimensions..<inputShape.numDimensions {
            totalInputNonBroadcastSize *= inputShape.dimensions[dim]
        }
        neededInputShapeDimensions.append(totalInputNonBroadcastSize)
        if (neededInputShapeDimensions.count == 1) { neededInputShapeDimensions.insert(1, at: 0) }      //  matmul has issues with 1D input
        var neededInputShape = TensorShape(neededInputShapeDimensions)

        //  See if we have to reshape the input tensor
        let inputReshapeTensor : MPSGraphTensor
        if inputShape != neededInputShape {
            //  Add a reshape node
            if (batchedInput) { neededInputShape = neededInputShape.shapeWithAddedBatchDimension(graph.batchSize) }
            let reshapeName = graph.getFullName(name)! + "_inputReshape"
            inputReshapeTensor = graph.mpsgraph.reshape(inputTensor, shape: neededInputShape.getMPSShape(), name: reshapeName)
            addedTensors.append(inputReshapeTensor)
            suffixes.append("_inputReshape")
        }
        else {
            inputReshapeTensor = inputTensor
        }
        
        //  Get the output non-broadcast size
        var totalOutputNonBroadcastSize = 1
        for dim in numBroadcastDimensions..<outputShape.numDimensions {
            totalOutputNonBroadcastSize *= outputShape.dimensions[dim]
        }
        
        //  Add the weights variable
        let weightShape = TensorShape([totalInputNonBroadcastSize, totalOutputNonBroadcastSize])
        let weightType = DataType(from: inputTensor.dataType)
        let weights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: totalInputNonBroadcastSize, numOutput: totalOutputNonBroadcastSize)
        let weightData = weights.getData()
        let weightName = graph.getFullName(name)! + "_weights"
        let weightTensor = graph.mpsgraph.variable(with: weightData, shape: weightShape.getMPSShape(), dataType: weights.type.getMPSDataType(), name: weightName)
        suffixes.append("_weights")
        addedTensors.append(weightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        let node = try Variable.createWeightInitializationVariable(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: totalInputNonBroadcastSize, numOutput: totalOutputNonBroadcastSize, name: weightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: weightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  If this is a learning layer - add the weights to the list to get assignment operations for
        if let lossNode = lossNode {
            let learningVariable = LearningVariable(variable: node, tensor: weightTensor, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
            graph.learningVariables.append(learningVariable)
            totalParameterCount += weightShape.totalSize
        }
        
        //  If using a bias term, add the bias variable
        var biasTensor: MPSGraphTensor? = nil
        if (useBias) {
            //  Get the bias shape - output shape without the broadcast dimensions
            let outputDimensions = outputShape.dimensions
            let biasDimensions = Array(outputDimensions[numBroadcastDimensions..<outputDimensions.count])
            let biasShape = TensorShape(biasDimensions)
            
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
                let learningVariable = LearningVariable(variable: node, tensor: biasTensor!, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
                graph.learningVariables.append(learningVariable)
                totalParameterCount += biasShape.totalSize
            }
       }
        
        //  Add the matrix multiply
        let haveDownStreamOperation = (useBias || haveActivationFunction)
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
        let matrixMultTensor = graph.mpsgraph.matrixMultiplication(primary: inputReshapeTensor, secondary: weightTensor, name: matrixTensorName)
        addedTensors.append(matrixMultTensor)
        
        //  Get the output shape - with batch dimension if needed
        var desiredOutputShape = outputShape
        if (batchedInput) {
            desiredOutputShape = outputShape.shapeWithAddedBatchDimension(graph.batchSize)
        }
        
        //  If the shape isn't the output shape (with possible batch prefix), reshape to it
        let matrixMultResult: MPSGraphTensor
        if let matrixMultShape = matrixMultTensor.shape {
            if (!desiredOutputShape.matchesMPSShape(matrixMultShape)) {
                //  Add a reshape node
                let reshapeName = graph.getFullName(name)! + "_outputReshape"
                let outputReshapeTensor = graph.mpsgraph.reshape(matrixMultTensor, shape: desiredOutputShape.getMPSShape(), name: reshapeName)
                addedTensors.append(outputReshapeTensor)
                suffixes.append("_outputReshape")
                matrixMultResult = outputReshapeTensor
            }
            else {
                matrixMultResult = matrixMultTensor
            }
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        //  If using a bias term, add the addition operation
        var finalTensor = matrixMultResult
        if (useBias) {
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
                let additionTensor = graph.mpsgraph.addition(matrixMultResult, biasTensor, name: biasTensorName)
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

    ///  Modifier to turn off addition of a bias term after the matrix multiplication
    public func noBiasTerm() -> FullyConnectedLayer {
        useBias = false
        return self
    }
    
    ///  Modifier to set the initialization info for the random initialization of the weights
    public func weightInitialization(initializerInfo: WeightInitialization) -> FullyConnectedLayer {
        weightInitialization = initializerInfo
        return self
    }
    
    ///  Modifier to set the initialization value for the initialization of the biases
    public func biasInitialValue(initialValue: Double) -> FullyConnectedLayer {
        self.biasInitialValue = initialValue
        return self
    }
    
    /// Modifier to configure the layer's variables to learn
    /// - Parameters:
    ///   - mode: lossNode: the name of the loss calculation in the Graph
    ///   - using: (Optional) the optimizer method to use for learning.  Defaults to stochastic gradient descent
    ///   - gradientClipping: (Optional) defaults to nil.  A tuple with the minimum and maximum gradient values allowed in the back-propogation for this node.  The gradient is clipped to this range before being used by the optimizer
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String, using: LearningOptimizer = .stochasticGradientDescent, gradientClipping: (min: Double, max: Double)? = nil) -> FullyConnectedLayer {
        self.lossNode = lossNode
        self.learningOptimizer = using
        self.gradientClipping = gradientClipping
        return self
    }
    
    override func getNumberOfParameters() throws -> Int {
        return totalParameterCount
    }
}
