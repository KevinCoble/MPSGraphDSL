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
    let activationFunction: ActivationFunction
    var useBias: Bool = true
    var weightInitialization: WeightInitialization
    var biasInitialValue: Double = 0.0
    var lossNode: String? = nil
    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    ///  Constructor for a fully connected node
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - outputShape: The shape of the output tensor.  This dictates the number of fully-connected nodes in the layer
    ///   - activationFunction: The activation function for the layer
    ///   - name: The name for this node and its associated tensor.  The variable nodes will be named &ltname&gt_weights' and &ltname&gt_biases'
    public init(input: String? = nil, outputShape: TensorShape, activationFunction: ActivationFunction, name: String) {
        self.outputShape = outputShape
        self.activationFunction = activationFunction
        switch (activationFunction) {
            case .none, .tanh, .sigmoid:
                weightInitialization = .XavierGlorotNormal
            case .relu, .leakyRelu, .leakyReluFromTensor:
                weightInitialization = .HeNormal
        }
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        
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
        var neededInputShape: TensorShape? = nil
        if (graph.batchGraph && inputShape.firstDimensionIsBatchSize(graph.batchSize)) {
            batchedInput = true
            inputShape = inputShape.shapeWithRemovedBatchDimension()
            neededInputShape = TensorShape([graph.batchSize, inputShape.totalSize]) // reshape the input tensor to a [batchSize, x]
        }
        else {
            //  Not a batch input anymore, just reshape to [1, x], where x is the input length
            if (inputShape.numDimensions != 2 || inputShape.dimensions[0] != 1) {
                neededInputShape = TensorShape([1, inputShape.totalSize])
            }
        }

        //  See if we have to reshape the input tensor
        let inputReshapeTensor : MPSGraphTensor
        if let neededInputShape = neededInputShape {
            //  Add a reshape node
            let reshapeName = graph.getFullName(name)! + "_inputReshape"
            inputReshapeTensor = graph.mpsgraph.reshape(inputTensor, shape: neededInputShape.getMPSShape(), name: reshapeName)
            addedTensors.append(inputReshapeTensor)
            suffixes.append("_inputReshape")
        }
        else {
            inputReshapeTensor = inputTensor
        }
        
        //  Add the weights variable
        var weightDimensions : [Int] = [inputShape.totalSize]
        weightDimensions += outputShape.dimensions
        let weightShape = TensorShape(weightDimensions)
        let weightType = DataType(from: inputTensor.dataType)
        let numInputs = inputShape.totalSize
        let numOutputs = outputShape.totalSize
        let weights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: numInputs, numOutput: numOutputs)
        let weightData = weights.getData()
        let weightName = graph.getFullName(name)! + "_weights"
        let weightTensor = graph.mpsgraph.variable(with: weightData, shape: weightShape.getMPSShape(), dataType: weights.type.getMPSDataType(), name: weightName)
        suffixes.append("_weights")
        addedTensors.append(weightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        let node = try Variable.createWeightInitializationVariable(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: numInputs, numOutput: numOutputs, name: weightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: weightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  If this is a learning layer - add the weights to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: node, tensor: weightTensor, loss: lossNode))
        }
        
        //  If using a bias term, add the bias variable
        var biasTensor: MPSGraphTensor? = nil
        if (useBias) {
            let biases = CreateTensor.constantValues(type: weightType, shape: outputShape, initialValue: biasInitialValue)
            let biasData = biases.getData()
            let biasName = graph.getFullName(name)! + "_biases"
            biasTensor = graph.mpsgraph.variable(with: biasData, shape: outputShape.getMPSShape(), dataType: biases.type.getMPSDataType(), name: biasName)
            suffixes.append("_biases")
            addedTensors.append(biasTensor)
            
            //  If we are adding load or reset assignments, put this variable on the list for load assignments
            let node = Variable(dataType: weightType, shape: outputShape, initialValue: biasInitialValue, name: biasName)
            if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
                let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: biasTensor!, sourceTensor: nil)
                graph.loadResetAssignList.append(loadResetAssignInfo)
            }

            //  If this is a learning layer - add to the biases to the list to get assignment operations for
            if let lossNode = lossNode {
                graph.learningVariables.append((variable: node, tensor: biasTensor!, loss: lossNode))
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
    /// - Parameter lossNode: the name of the loss calculation in the Graph
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String) -> FullyConnectedLayer {
        self.lossNode = lossNode
        return self
    }
}
