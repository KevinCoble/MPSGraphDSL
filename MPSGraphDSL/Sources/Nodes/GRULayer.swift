//
//  GRULayer.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 1/4/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///  Class that wraps the standard MPSGraph GRU node
public class GRU: UnaryNode {
    let recurrentWeight: String?
    let inputWeight: String?
    let bias: String?
    let initState: String?
    let mask: String?
    let secondaryBias: String?
    let descriptor: MPSGraphGRUDescriptor

    var suffixes: [String] = []

    /// Constructor for an GRU operation.  Direct wrap of MPSGraph functions - see Apple documentation for input shape limits, etc.
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used
    ///   - recurrentWeight: (Optional) The name of the tensor that will provide the recurrent weights (usually a Variable).  If nil the previous node's output will be used
    ///   - inputWeight: (Optional) The name of the tensor that will provide the input weights (usually a Variable).  If nil a diagonal unit matrix is used.  Defaults to nil
    ///   - bias: (Optional) The name of the tensor that will provide the bias values (usually a Variable).  If nil no bias values are used  Defaults to nil
    ///   - initState: (Optional) The name of the tensor that will provide the initial state values.  If nil a zero tensor is used.  Defaults to nil
    ///   - mask: (Optional) The name of the tensor containing the mask m, if missing the operation assumes ones.
    ///   - secondaryBias: (Optional) The name of the tensor containing the secondary bias tensor., if missing the operation assumes zeroes.  Only used with reset_after = YES
    ///   - descriptor: The GRU options structure
    ///   - name: (Optional) The name for this node and its associated tensors.  One or two tensors will be produced
    public init(input: String? = nil, recurrentWeight: String? = nil, inputWeight: String? = nil, bias: String? = nil, initState: String? = nil, mask: String? = nil, secondaryBias: String? = nil, descriptor: MPSGraphGRUDescriptor, name: String? = nil) {
        self.recurrentWeight = recurrentWeight
        self.inputWeight = inputWeight
        self.bias = bias
        self.initState = initState
        self.mask = mask
        self.secondaryBias = secondaryBias
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
        
        //  Get the secondary bias tensor
        let secondaryBiasMPSTensor: MPSGraphTensor?
        if let secondaryBias = secondaryBias {
            if let addedNode = graph.findNamedNode(secondaryBias) {
                secondaryBiasMPSTensor = addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(secondaryBias)
            }
        }
        else {
            //  nil peephole tensor node name - set to nil to not use
            secondaryBiasMPSTensor = nil
        }

        //  Get the output suffixes
        suffixes = []
        suffixes.append("_state")
        if (descriptor.training) { suffixes.append("_trainingState") }
        
        //  Add the node
        if (maskMPSTensor == nil && secondaryBiasMPSTensor == nil) {
            if (initStateMPSTensor == nil) {
                let gruResults = graph.mpsgraph.GRU(inputTensor, recurrentWeight: recurrentMPSTensor, inputWeight: inputMPSTensor, bias: biasMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
                
                //  Return the result
                return gruResults
            }
            else {
                let gruResults = graph.mpsgraph.GRU(inputTensor, recurrentWeight: recurrentMPSTensor, inputWeight: inputMPSTensor, bias: biasMPSTensor, initState: initStateMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
                
                //  Return the result
                return gruResults
            }
        }
        else {
            let gruResults = graph.mpsgraph.GRU(inputTensor, recurrentWeight: recurrentMPSTensor, inputWeight: inputMPSTensor, bias: biasMPSTensor, initState: initStateMPSTensor, mask: maskMPSTensor, secondaryBias: secondaryBiasMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
            
            //  Return the result
            return gruResults
        }
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}


///  Layer for a standard GRU.  All weights are created as Variables
public class GRULayer: UnaryNode {
    let stateSize: Int
    
    var outputGateActivation: MPSGraphRNNActivation = .tanh
    var resetGateActivation: MPSGraphRNNActivation = .sigmoid
    var updateGateActivation: MPSGraphRNNActivation = .sigmoid
    var recurrentWeightInitialization: WeightInitialization
    var recurrentWeightOrthogonalization: Bool = true
    var inputWeightInitialization: WeightInitialization
    var biasInitialValue: Double = 0.0
    var bidirectional: Bool = false
    var createLastState: Bool = true
    var targetLoop: Bool = false
    var targetLast: Bool = true

    var lossNode: String? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []

    /// Constructor for an GRU layer.
    /// State size is passed in.  Number of features and number of inputs are determined from the input tensor
    /// Default setup is used.  To change, use supplied modifiers
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used  Must be of shape \[T, N, I\]
    ///   - stateSize: The number of state datum (neurons) in the cell and state tensors
    ///   - name: (Optional) The name for this node and its associated tensors.  One to three tensors will be produced
    public init(input: String? = nil, stateSize: Int, name: String? = nil) {
        self.stateSize = stateSize
        switch (outputGateActivation) {
            case .relu:
            recurrentWeightInitialization = .HeNormal
            inputWeightInitialization = .HeNormal
        default:
            recurrentWeightInitialization = .XavierGlorotNormal
            inputWeightInitialization = .XavierGlorotNormal
        }
        super.init(input: input, name: name)
    }
    
    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        let weightType = DataType(from: inputTensor.dataType)

        //  Get the shape of the input tensor and determine number of features and number of inputs
        let numInputs: Int
        if let inputShape = inputTensor.shape {
            if (inputShape.count != 3) { throw MPSGraphNeuralNetErrors.InputTensorNot3D }
            //lstm            numFeatures = Int(truncating: inputShape[1])
            numInputs = Int(truncating: inputShape[2])
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        //  Add the recurrent weights variable
        let rweightsShape : TensorShape
        if (bidirectional) {
            rweightsShape = TensorShape([2, stateSize, stateSize])       //  [2, 3H, H] for bidirectional
        }
        else {
            rweightsShape = TensorShape([stateSize, stateSize])       //  [3H, H]
        }
        var rweights: Tensor
        if (recurrentWeightOrthogonalization) {
            let squareWeightsShape = TensorShape([stateSize, stateSize])
            rweights = try CreateTensor.createOrthogonalWeightInitializationTensor(type: weightType, shape: squareWeightsShape, initializationInfo: recurrentWeightInitialization, numGates: bidirectional ? 6 : 3)
            if (bidirectional) {
                rweights = try CreateTensor.createReshapedTensor(from: rweights, newShape: rweightsShape)
            }
        }
        else {
            rweights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: rweightsShape, initializationInfo: recurrentWeightInitialization, numInputs: stateSize, numOutput: stateSize)
        }
        let rweightData = rweights.getData()
        let rweightName = graph.getFullName(name)! + "_recurrentWeights"
        let rweightTensor = graph.mpsgraph.variable(with: rweightData, shape: rweightsShape.getMPSShape(), dataType: rweights.type.getMPSDataType(), name: rweightName)
        suffixes.append("_recurrentWeights")
        addedTensors.append(rweightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        var node = try Variable.createWeightInitializationVariable(type: weightType, shape: rweightsShape, initializationInfo: recurrentWeightInitialization, numInputs: stateSize, numOutput: stateSize, orthogonal: recurrentWeightOrthogonalization, name: rweightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: rweightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }
        
        //  If this is a learning layer - add the recurrent weights to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: node, tensor: rweightTensor, loss: lossNode))
        }
        
        //  Add the input weights variable
        let iweightsShape = TensorShape([(bidirectional ? 6 : 3) * stateSize, numInputs])       //  [3H, I] or [6H, I] for bidirectional
        let iweights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: iweightsShape, initializationInfo: inputWeightInitialization, numInputs: numInputs, numOutput: stateSize)
        let iweightData = iweights.getData()
        let iweightName = graph.getFullName(name)! + "_inputWeights"
        let iweightTensor = graph.mpsgraph.variable(with: iweightData, shape: iweightsShape.getMPSShape(), dataType: iweights.type.getMPSDataType(), name: iweightName)
        suffixes.append("_inputWeights")
        addedTensors.append(iweightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        node = try Variable.createWeightInitializationVariable(type: weightType, shape: iweightsShape, initializationInfo: inputWeightInitialization, numInputs: numInputs, numOutput: stateSize, name: iweightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: iweightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }
        
        //  If this is a learning layer - add the input weights to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: node, tensor: iweightTensor, loss: lossNode))
        }
        
        //  Add the bias variable
        let biasShape = TensorShape([(bidirectional ? 6 : 3) * stateSize])       //  [3H] or [6H] for bidirectional
        let biases = CreateTensor.constantValues(type: weightType, shape: biasShape, initialValue: biasInitialValue)
        let biasData = biases.getData()
        let biasName = graph.getFullName(name)! + "_bias"
        let biasTensor = graph.mpsgraph.variable(with: biasData, shape: biasShape.getMPSShape(), dataType: weightType.getMPSDataType(), name: biasName)
        suffixes.append("_bias")
        addedTensors.append(biasTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        node = Variable(dataType: weightType, shape: biasShape, initialValue: biasInitialValue, name: biasName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: biasTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }
        
        //  If this is a learning layer - add the bias to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: node, tensor: biasTensor, loss: lossNode))
        }
        
        //  Create the descriptor
        let descriptor = MPSGraphGRUDescriptor()
        descriptor.bidirectional = bidirectional
        descriptor.flipZ = false
        descriptor.outputGateActivation = outputGateActivation
        descriptor.resetAfter = false
        descriptor.resetGateActivation = resetGateActivation
        descriptor.resetGateFirst = false
        descriptor.reverse = false
        descriptor.training = (lossNode != nil)
        descriptor.updateGateActivation = updateGateActivation
        
        //  Get the suffixes for the output tensor
        if (targetLoop) { targetIndices.append(suffixes.count) }
        suffixes.append("_state")
        if (descriptor.training) {
            targetIndices.append(suffixes.count)
            suffixes.append("_z")
        }
        
//        let desc2 = MPSGraphLSTMDescriptor()
//        desc2.training = (lossNode != nil)
//        var gru = graph.mpsgraph.LSTM(inputTensor, recurrentWeight: rweightTensor, inputWeight: iweightTensor, bias: biasTensor, initState: nil, initCell: nil, descriptor: desc2, name: graph.getFullName(name))
//        gru.removeLast()
        let gru = graph.mpsgraph.GRU(inputTensor, recurrentWeight: rweightTensor, inputWeight: iweightTensor, bias: biasTensor, initState: nil, descriptor: descriptor, name: graph.getFullName(name))
        addedTensors += gru
        
        //  If requested to create a 'last state' output, do so
        if (createLastState) {
            let stateSliceName = graph.getFullName(name)! + "_lastStateSlice"
            let stateSlice = graph.mpsgraph.sliceTensor(gru.first!, dimension: 0, start: -1, length: 1, name: stateSliceName)
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
            throw MPSGraphNeuralNetErrors.NoConfiguredTargetTensorsForTargettedNode
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
    
    ///  Modifier to set the output gate activation function for the GRU layer.  Default is tanh
    public func outputGateActivation(_ activation: MPSGraphRNNActivation) -> GRULayer {
        self.outputGateActivation = activation
        return self
    }
    
    ///  Modifier to set the reset gate activation function for the GRU layer.  Default is sigmoid
    public func resetGateActivation(_ activation: MPSGraphRNNActivation) -> GRULayer {
        self.resetGateActivation = activation
        return self
    }
    
    ///  Modifier to set the update gate activation function for the GRU layer.  Default is sigmoid
    public func updateGateActivation(_ activation: MPSGraphRNNActivation) -> GRULayer {
        self.updateGateActivation = activation
        return self
    }

    ///  Modifier to set the initializer parameters for the recurrent weights
    public func recurrentWeightInitialization(initializerInfo: WeightInitialization, orthogonalize: Bool = true) -> GRULayer {
        recurrentWeightInitialization = initializerInfo
        self.recurrentWeightOrthogonalization = orthogonalize
        return self
    }

    ///  Modifier to set the initializer parameters for the input weights
    public func inputWeightInitialization(initializerInfo: WeightInitialization) -> GRULayer {
        inputWeightInitialization = initializerInfo
        return self
    }
    
    ///  Modifier to set the initialization value for the initialization of the biases
    public func biasInitialValue(initialValue: Double) -> GRULayer {
        self.biasInitialValue = initialValue
        return self
    }
    
    ///  Modifier to make the recurrent layer bidirectional
    public func makeBidirectional() -> GRULayer {
        self.bidirectional = true
        return self
    }

    ///  Modifier to set the node output options
    /// - Parameters:
    ///   - createLastState: (Optional) if true the last state tensor of the time loop is separated out and made it's own output tensor with suffix "_lastState".  Defaults to true
    ///   - targetLoop: (Optional)  If true the time loop output (state) is marked as target tensors if the node is marked as a target.  Defaults to false
    ///   - targetLast: (Optional)  If true the last-time output (state ) is marked as target tensors if the node is marked as a target.  Defaults to false
    /// - Returns: The LSTMLayer node
    public func setOutput(createLastState: Bool = true, targetLoop: Bool = false, targetLast: Bool = true) -> GRULayer {
        self.createLastState = createLastState
        self.targetLoop = targetLoop
        self.targetLast = targetLast
        return self
    }
    
    ///  Modifier to set the LSTM layer to learn with respect to a loss calculation.  The z tensor will be output if this is used
    public func learnWithRespectTo(_ lossNode: String) -> GRULayer {
        self.lossNode = lossNode
        return self
    }
}
