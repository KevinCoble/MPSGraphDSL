//
//  RNNLayer.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 1/1/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///  Class that wraps the standard MPSGraph singleGateRNN node
public class SingleGateRNN : UnaryNode {
    let recurrentWeight: String?
    let inputWeight: String?
    let bias: String?
    let initState: String?
    let mask: String?
    let descriptor: MPSGraphSingleGateRNNDescriptor

    var suffixes: [String] = []

    /// Constructor for a single-gate RNN operation.  Direct wrap of MPSGraph functions - see Apple documentation for input shape limits, etc.
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used
    ///   - recurrentWeight: (Optional) The name of the tensor that will provide the recurrent weights (usually a Variable).  If nil the previous node's output will be used
    ///   - inputWeight: (Optional) The name of the tensor that will provide the input weights (usually a Variable).  If nil no input weights are used  Defaults to nil
    ///   - bias: (Optional) The name of the tensor that will provide the bias values (usually a Variable).  If nil no bias values are used  Defaults to nil
    ///   - initState: (Optional) The name of the tensor that will provide the initial state values.  If nil a zero tensor is used.  Defaults to nil
    ///   - mask: (Optional) The name of the tensor containing the mask m, if missing the operation assumes ones.
    ///   - descriptor: The LSTM options structure
    ///   - name: (Optional) The name for this node and its associated tensors.  One or two tensors will be produced
    public init(input: String? = nil, recurrentWeight: String? = nil, inputWeight: String? = nil, bias: String? = nil, initState: String? = nil, mask: String? = nil, descriptor: MPSGraphSingleGateRNNDescriptor, name: String? = nil) {
        self.recurrentWeight = recurrentWeight
        self.inputWeight = inputWeight
        self.bias = bias
        self.initState = initState
        self.mask = mask
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }
    
    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Get the recurrent weight tensor
        let recurrentMPSTensor = try graph.getOptionalTensor(recurrentWeight)
        
        //  Get the input weight tensor
        let inputMPSTensor: MPSGraphTensor?
        if let inputWeight = inputWeight {
            if let addedNode = graph.findNamedNode(inputWeight) {
                inputMPSTensor = addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(inputWeight)
            }
        }
        else {
            //  nil input weight tensor node name - set to nil to not use
            inputMPSTensor = nil
        }
        
        //  Get the bias tensor
        let biasMPSTensor: MPSGraphTensor?
        if let bias = bias {
            if let addedNode = graph.findNamedNode(bias) {
                biasMPSTensor = addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(bias)
            }
        }
        else {
            //  nil bias tensor node name - set to nil to not use
            biasMPSTensor = nil
        }
        
        //  Get the init state tensor
        let initStateMPSTensor: MPSGraphTensor?
        if let initState = initState {
            if let addedNode = graph.findNamedNode(initState) {
                initStateMPSTensor = addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(initState)
            }
        }
        else {
            //  nil init state tensor node name - set to nil to not use
            initStateMPSTensor = nil
        }
        
        //  Get the mask tensor
        let maskMPSTensor: MPSGraphTensor?
        if let mask = mask {
            if let addedNode = graph.findNamedNode(mask) {
                maskMPSTensor = addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(mask)
            }
        }
        else {
            //  nil mask tensor node name - set to nil to not use
            maskMPSTensor = nil
        }

        //  Get the output suffixes
        suffixes = []
        suffixes.append("_state")
        if (descriptor.training) { suffixes.append("_trainingState") }
        //  Add the node
        if (maskMPSTensor == nil) {
            if (inputMPSTensor == nil && biasMPSTensor == nil) {
                let rnnResults = graph.mpsgraph.singleGateRNN(inputTensor, recurrentWeight: recurrentMPSTensor, initState: initStateMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
                
                //  Return the result
                return rnnResults
            }
            else {
                let rnnResults = graph.mpsgraph.singleGateRNN(inputTensor, recurrentWeight: recurrentMPSTensor, inputWeight: inputMPSTensor, bias:biasMPSTensor, initState: initStateMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
                
                //  Return the result
                return rnnResults
            }
        }
        else {
            let rnnResults = graph.mpsgraph.singleGateRNN(inputTensor, recurrentWeight: recurrentMPSTensor, inputWeight: inputMPSTensor, bias:biasMPSTensor, initState: initStateMPSTensor, mask: maskMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
            
            //  Return the result
            return rnnResults
        }
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

///  Layer for a standard single-gate RNN.  All weights are created as Variables
public class RNNLayer: UnaryNode {
    let stateSize: Int
    
    var activation: MPSGraphRNNActivation = .tanh
    var recurrentWeightInitialMinimum : Double = -0.5
    var recurrentWeightInitialMaximum : Double = 0.5
    var inputWeightInitialMinimum : Double = -0.5
    var inputWeightInitialMaximum : Double = 0.5
    var biasInitialMinimum : Double = -0.5
    var biasInitialMaximum : Double = 0.5
    var createLastState: Bool = true
    var targetLoop: Bool = false
    var targetLast: Bool = true
    var lossNode: String? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []

    /// Constructor for an RNN layer.
    /// State size is passed in.  Number of features and number of inputs are determined from the input tensor
    /// Default setup is used.  To change, use supplied modifiers
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used  Must be of shape \[T, N, I\]
    ///   - stateSize: The number of state datum (neurons) in the cell and state tensors
    ///   - name: (Optional) The name for this node and its associated tensors.  One or two tensors will be produced
    public init(input: String? = nil, stateSize: Int, name: String? = nil) {
        self.stateSize = stateSize
        super.init(input: input, name: name)
    }
    
    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Get the shape of the input tensor and determine number of features and number of inputs
        //lstm        let numFeatures: Int
        let numInputs: Int
        if let inputShape = inputTensor.shape {
            if (inputShape.count != 3) { throw MPSGraphLSTMGRUErrors.InputTensorNot3D }
            //lstm            numFeatures = Int(truncating: inputShape[1])
            numInputs = Int(truncating: inputShape[2])
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        //  Add the recurrent weights variable
        let rweightsShape = TensorShape([stateSize, stateSize])       //  [H, H]
        let rweightsRange = try ParameterRange(minimum: recurrentWeightInitialMinimum, maximum: recurrentWeightInitialMaximum)
        let rweights = TensorFloat32(shape: rweightsShape, randomValueRange: rweightsRange)
        let rweightData = rweights.getData()
        let rweightName = graph.getFullName(name)! + "_recurrentWeights"
        let rweightTensor = graph.mpsgraph.variable(with: rweightData, shape: rweightsShape.getMPSShape(), dataType: rweights.type.getMPSDataType(), name: rweightName)
        suffixes.append("_recurrentWeights")
        addedTensors.append(rweightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        var node = Variable(dataType: .float32, shape: rweightsShape, randomValueRange: rweightsRange, name: rweightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: rweightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }
        
        //  If this is a learning layer - add the recurrent weights to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: node, tensor: rweightTensor, loss: lossNode))
        }
        
        //  Add the input weights variable
        let iweightsShape = TensorShape([stateSize, numInputs])       //  [H, I]
        let iweightsRange = try ParameterRange(minimum: recurrentWeightInitialMinimum, maximum: recurrentWeightInitialMaximum)
        let iweights = TensorFloat32(shape: iweightsShape, randomValueRange: iweightsRange)
        let iweightData = iweights.getData()
        let iweightName = graph.getFullName(name)! + "_inputWeights"
        let iweightTensor = graph.mpsgraph.variable(with: iweightData, shape: iweightsShape.getMPSShape(), dataType: iweights.type.getMPSDataType(), name: iweightName)
        suffixes.append("_inputWeights")
        addedTensors.append(iweightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        node = Variable(dataType: .float32, shape: iweightsShape, randomValueRange: iweightsRange, name: iweightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: iweightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }
        
        //  If this is a learning layer - add the input weights to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: node, tensor: iweightTensor, loss: lossNode))
        }
        
        //  Add the bias variable
        let biasShape = TensorShape([stateSize])       //  [H]
        let biasRange = try ParameterRange(minimum: recurrentWeightInitialMinimum, maximum: recurrentWeightInitialMaximum)
        let bias = TensorFloat32(shape: biasShape, randomValueRange: biasRange)
        let biasData = bias.getData()
        let biasName = graph.getFullName(name)! + "_bias"
        let biasTensor = graph.mpsgraph.variable(with: biasData, shape: biasShape.getMPSShape(), dataType: bias.type.getMPSDataType(), name: biasName)
        suffixes.append("_bias")
        addedTensors.append(biasTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        node = Variable(dataType: .float32, shape: biasShape, randomValueRange: biasRange, name: biasName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: biasTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }
        
        //  If this is a learning layer - add the bias to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: node, tensor: biasTensor, loss: lossNode))
        }
        
        //  Create the descriptor
        let descriptor = MPSGraphSingleGateRNNDescriptor()
        descriptor.activation = activation
        descriptor.bidirectional = false
        descriptor.reverse = false
        descriptor.training = (lossNode != nil)
        
        //  Get the suffixes for the output tensor
        if (targetLoop) { targetIndices.append(suffixes.count) }
        suffixes.append("_state")
        if (descriptor.training) {
            targetIndices.append(suffixes.count)
            suffixes.append("_z")
        }
        
        let rnn = graph.mpsgraph.singleGateRNN(inputTensor, recurrentWeight: rweightTensor, inputWeight: iweightTensor, bias: biasTensor, initState: nil, descriptor: descriptor, name: graph.getFullName(name))
        addedTensors += rnn
        
        //  If requested to create a 'last state' output, do so
        if (createLastState) {
            let stateSliceName = graph.getFullName(name)! + "_lastStateSlice"
            let stateSlice = graph.mpsgraph.sliceTensor(rnn.first!, dimension: 0, start: -1, length: 1, name: stateSliceName)
            addedTensors.append(stateSlice)
            suffixes.append("_lastStateSlice")
            
            var stateShape = stateSlice.shape!
            stateShape.removeFirst()
            
            let stateReshapeName = graph.getFullName(name)! + "_lastState"
            let stateReshape = graph.mpsgraph.reshape(stateSlice, shape: stateShape, name: stateReshapeName)
            addedTensors.append(stateReshape)
            if (targetLast) { targetIndices.append(suffixes.count)}
            suffixes.append("_lastState")
        }
        
        if (!targetModes.isEmpty && targetIndices.isEmpty) {
            throw MPSGraphLSTMGRUErrors.NoConfiguredTargetTensorsForTargettedNode
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
    
    ///  Modifier to set the activation function for the RNN layer.  Default is tanh
    public func activationFunction(_ activation: MPSGraphRNNActivation) -> RNNLayer {
        self.activation = activation
        return self
    }
    
    ///  Modifier to set the range for the random initializer of the recurrent weights.  Default is -0.5 to 0.5
    public func recurrentWeightInitialRange(min: Double, max: Double) -> RNNLayer {
        recurrentWeightInitialMinimum = min
        recurrentWeightInitialMaximum = max
        return self
    }
    
    ///  Modifier to set the range for the random initializer of the input weights.  Default is -0.5 to 0.5
    public func inputWeightInitialRange(min: Double, max: Double) -> RNNLayer {
        inputWeightInitialMinimum = min
        inputWeightInitialMaximum = max
        return self
    }
    
    ///  Modifier to set the range for the random initializer of the bias.  Default is -0.5 to 0.5
    public func biasInitialRange(min: Double, max: Double) -> RNNLayer {
        biasInitialMinimum = min
        biasInitialMaximum = max
        return self
    }
    
    ///  Modifier to set the node output options
    /// - Parameters:
    ///   - createLastState: (Optional) if true the last state tensor of the time loop is separated out and made it's own output tensor with suffix "_lastState".  Defaults to true
    ///   - targetLoop: (Optional)  If true the time loop output (state) is marked as target tensors if the node is marked as a target.  Defaults to false
    ///   - targetLast: (Optional)  If true the last-time output (state ) is marked as target tensors if the node is marked as a target.  Defaults to false
    /// - Returns: The LSTMLayer node
    public func setOutput(createLastState: Bool = true, targetLoop: Bool = false, targetLast: Bool = true) -> RNNLayer {
        self.createLastState = createLastState
        self.targetLoop = targetLoop
        self.targetLast = targetLast
        return self
    }
    
    ///  Modifier to set the LSTM layer to learn with respect to a loss calculation.  The z tensor will be output if this is used
    public func learnWithRespectTo(_ lossNode: String) -> RNNLayer {
        self.lossNode = lossNode
        return self
    }
}
