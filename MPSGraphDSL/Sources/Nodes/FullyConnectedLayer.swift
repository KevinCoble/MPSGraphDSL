//
//  FullyConnectedLayer.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 12/8/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Enumeration for selecting the activation function of a Neural Network layer
public enum ActivationFunction {
    case none
    case relu
    case tanh
    case sigmoid
    case leakyRelu(Double)
    case leakyReluFromTensor(String)
}

///   Node for a fully-connected Neural Network layer, with specified output shape and activation function
///
public class FullyConnectedLayer : UnaryNode {
    let outputShape: TensorShape
    let activationFunction: ActivationFunction
    var useBias: Bool = true
    var weightInitialMinimum : Double = -0.5
    var weightInitialMaximum : Double = 0.5
    var biasInitialMinimum : Double = -0.5
    var biasInitialMaximum : Double = 0.5
    var lossNode: String? = nil
    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    ///  Constructor for a fully connected node
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - outputShape: The shape of the output tensor.  This dictates the number of fully-connected nodes in the layer
    ///   - name: The name for this node and its associated tensor.  The variable nodes will be named &ltname&gt_weights' and &ltname&gt_biases'
    public init(input: String? = nil, outputShape: TensorShape, activationFunction: ActivationFunction, name: String) {
        self.outputShape = outputShape
        self.activationFunction = activationFunction
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

        //  See if we have to reshape the input tensor to a [1, x], where x is the input length
        let inputReshapeTensor : MPSGraphTensor
        if (inputShape.numDimensions != 2 || inputShape.dimensions[0] != 1) {
            //  Add a reshape node
            let reshapeName = graph.getFullName(name)! + "_inputReshape"
            let newShape = TensorShape([1, inputShape.totalSize])
            inputReshapeTensor = graph.mpsgraph.reshape(inputTensor, shape: newShape.getMPSShape(), name: reshapeName)
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
        let weightRange = try ParameterRange(minimum: weightInitialMinimum, maximum: weightInitialMaximum)
        let weights = TensorFloat32(shape: weightShape, randomValueRange: weightRange)
        let weightData = weights.getData()
        let weightName = graph.getFullName(name)! + "_weights"
        let weightTensor = graph.mpsgraph.variable(with: weightData, shape: weightShape.getMPSShape(), dataType: weights.type.getMPSDataType(), name: weightName)
        suffixes.append("_weights")
        addedTensors.append(weightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        let node = Variable(dataType: .float32, shape: weightShape, randomValueRange: weightRange, name: weightName)
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
            let biasRange = try ParameterRange(minimum: biasInitialMinimum, maximum: biasInitialMaximum)
            let biases = TensorFloat32(shape: outputShape, randomValueRange: biasRange)
            let biasData = biases.getData()
            let biasName = graph.getFullName(name)! + "_biases"
            biasTensor = graph.mpsgraph.variable(with: biasData, shape: outputShape.getMPSShape(), dataType: biases.type.getMPSDataType(), name: biasName)
            suffixes.append("_biases")
            addedTensors.append(biasTensor)
            
            //  If we are adding load or reset assignments, put this variable on the list for load assignments
            let node = Variable(dataType: .float32, shape: outputShape, randomValueRange: biasRange, name: biasName)
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
        
        //  If the shape isn't the output shape, reshape to it
        let matrixMultResult: MPSGraphTensor
        if let matrixMultShape = matrixMultTensor.shape {
            if (!outputShape.matchesMPSShape(matrixMultShape)) {
                //  Add a reshape node
                let reshapeName = graph.getFullName(name)! + "_outputReshape"
                let outputReshapeTensor = graph.mpsgraph.reshape(matrixMultTensor, shape: outputShape.getMPSShape(), name: reshapeName)
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
                let additionTensor = graph.mpsgraph.addition(matrixMultResult, biasTensor, name: graph.getFullName(name))
                addedTensors.append(additionTensor)
                finalTensor = additionTensor
            }
            else {
                throw GenericMPSGraphDSLErrors.InternalError
            }
        }
        
        //  Add any activation function
        if let activationTensor = try addActivation(activation: activationFunction, graph: graph, inputTensor: finalTensor) {
            addedTensors.append(activationTensor)
            targetIndices.append(suffixes.count)
            suffixes.append("")
        }

        return addedTensors
    }
    
    func addActivation(activation: ActivationFunction, graph: Graph, inputTensor: MPSGraphTensor) throws -> MPSGraphTensor? {
        switch (activation) {
            case .none:
                return nil
            case .relu:
                let reLUResult = graph.mpsgraph.reLU(with: inputTensor, name: graph.getFullName(name))
                return reLUResult
            case .tanh:
                let tanhResult = graph.mpsgraph.tanh(with: inputTensor, name: graph.getFullName(name))
                return tanhResult
            case .sigmoid:
                let sigmoidResult = graph.mpsgraph.sigmoid(with: inputTensor, name: graph.getFullName(name))
                return sigmoidResult
            case .leakyRelu(let alpha):
                let leakyReLUResult = graph.mpsgraph.leakyReLU(with: inputTensor, alpha: alpha, name: graph.getFullName(name))
                return leakyReLUResult
            case .leakyReluFromTensor(let alphaName):
                if let alphaNode = graph.findNamedNode(alphaName) {
                    let leakyReLUResult = graph.mpsgraph.leakyReLU(with: inputTensor, alphaTensor: alphaNode.mpstensor, name: graph.getFullName(name))
                    return leakyReLUResult
                }
                else {
                    throw MPSGraphDSLErrors.NamedTensorNotFound(alphaName)
                }
        }

    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }

    ///  Modifier to turm of addition of a bias term after the matrix multiplication
    public func noBiasTerm() -> FullyConnectedLayer {
        useBias = false
        return self
    }
    
    ///  Modifier to set the range for the random initialization of the weights
    public func weightInitialRange(min: Double, max: Double) -> FullyConnectedLayer {
        weightInitialMinimum = min
        weightInitialMaximum = max
        return self
    }
    
    ///  Modifier to set the range for the random initialization of the biases
    public func biasInitialRange(min: Double, max: Double) -> FullyConnectedLayer {
        biasInitialMinimum = min
        biasInitialMaximum = max
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
