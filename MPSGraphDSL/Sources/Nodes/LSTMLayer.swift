//
//  LSTMLayer.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 12/27/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///  Class that wraps the standard MPSGraph LSTM Node
public class LSTM: UnaryNode {
    let recurrentWeight: String?
    let inputWeight: String?
    let bias: String?
    let initState: String?
    let initCell: String?
    let mask: String?
    let peepHole: String?
    let descriptor: MPSGraphLSTMDescriptor

    var suffixes: [String] = []

    /// Constructor for an LSTM operation.  Direct wrap of MPSGraph functions - see Apple documentation for input shape limits, etc.
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used
    ///   - recurrentWeight: (Optional) The name of the tensor that will provide the recurrent weights (usually a Variable).  If nil the previous node's output will be used
    ///   - inputWeight: (Optional) The name of the tensor that will provide the input weights (usually a Variable).  If nil no input weights are used  Defaults to nil
    ///   - bias: (Optional) The name of the tensor that will provide the bias values (usually a Variable).  If nil no bias values are used  Defaults to nil
    ///   - initState: (Optional) The name of the tensor that will provide the initial state values.  If nil a zero tensor is used.  Defaults to nil
    ///   - initCell: (Optional) The name of the tensor that will provide the initial cell values.  If nil a zero tensor is used  Defaults to nil
    ///   - mask: (Optional) The name of the tensor containing the mask m, if missing the operation assumes ones.
    ///   - peepHole: (Optional) The name of the tensor containing the peephole vector v, if missing the operation assumes zeroes
    ///   - descriptor: The LSTM options structure
    ///   - name: (Optional) The name for this node and its associated tensors.  One to three tensors will be produced
    public init(input: String? = nil, recurrentWeight: String? = nil, inputWeight: String? = nil, bias: String? = nil, initState: String? = nil, initCell: String? = nil, mask: String? = nil, peepHole: String? = nil, descriptor: MPSGraphLSTMDescriptor, name: String? = nil) {
        self.recurrentWeight = recurrentWeight
        self.inputWeight = inputWeight
        self.bias = bias
        self.initState = initState
        self.initCell = initCell
        self.mask = mask
        self.peepHole = peepHole
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
        
        //  Get the init cell tensor
        let initCellMPSTensor: MPSGraphTensor?
        if let initCell = initCell {
            if let addedNode = graph.findNamedNode(initCell) {
                initCellMPSTensor = addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(initCell)
            }
        }
        else {
            //  nil init cell tensor node name - set to nil to not use
            initCellMPSTensor = nil
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
        
        //  Get the peephole tensor
        let peepholeMPSTensor: MPSGraphTensor?
        if let peepHole = peepHole {
            if let addedNode = graph.findNamedNode(peepHole) {
                peepholeMPSTensor = addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(peepHole)
            }
        }
        else {
            //  nil peephole tensor node name - set to nil to not use
            peepholeMPSTensor = nil
        }

        //  Get the output suffixes
        suffixes = []
        suffixes.append("_state")
        if (descriptor.produceCell) { suffixes.append("_cell") }
        if (descriptor.training) { suffixes.append("_trainingState") }

        //  Add the node
        if (maskMPSTensor == nil && peepholeMPSTensor == nil) {
            if (inputMPSTensor == nil && biasMPSTensor == nil) {
                let lstmResults = graph.mpsgraph.LSTM(inputTensor, recurrentWeight: recurrentMPSTensor, initState: initStateMPSTensor, initCell: initCellMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
                
                //  Return the result
                return lstmResults
            }
            else {
                let lstmResults = graph.mpsgraph.LSTM(inputTensor, recurrentWeight: recurrentMPSTensor, inputWeight: inputMPSTensor, bias:biasMPSTensor, initState: initStateMPSTensor, initCell: initCellMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
                
                //  Return the result
                return lstmResults
            }
        }
        else {
            let lstmResults = graph.mpsgraph.LSTM(inputTensor, recurrentWeight: recurrentMPSTensor, inputWeight: inputMPSTensor, bias:biasMPSTensor, initState: initStateMPSTensor, initCell: initCellMPSTensor, mask: maskMPSTensor, peephole: peepholeMPSTensor, descriptor: descriptor, name: graph.getFullName(name))
            
            //  Return the result
            return lstmResults
        }
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

///  Layer for a standard LSTM.  All weights are created as Variables
public class LSTMLayer: UnaryNode {
    let stateSize: Int
    
    //  Configuration
    var activation: MPSGraphRNNActivation = .tanh
    var cellGateActivation: MPSGraphRNNActivation = .tanh
    var forgetGateActivation: MPSGraphRNNActivation = .sigmoid
    var inputGateActivation: MPSGraphRNNActivation = .sigmoid
    var outputGateActivation: MPSGraphRNNActivation = .sigmoid
    var recurrentWeightInitialization: WeightInitialization
    var recurrentWeightOrthogonalization: Bool = true
    var inputWeightInitialization: WeightInitialization
    var biasInitialValue: Double = 0.0
    var bidirectional: Bool = false
    var produceCellOutput: Bool = false
    var createLastState: Bool = true
    var createLastCell: Bool = false
    var targetLoops: Bool = false
    var targetLasts: Bool = true
    
    var lossNode: String? = nil
    var learningOptimizer: LearningOptimizer = .stochasticGradientDescent
    var gradientClipping: (min: Double, max: Double)? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []

    /// Constructor for an LSTM layer.
    /// State size is passed in.  Number of features and number of inputs are determined from the input tensor
    /// Default setup is used.  To change, use supplied modifiers
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used  Must be of shape \[T, N, I\]
    ///   - stateSize: The number of state datum (neurons) in the cell and state tensors
    ///   - name: (Optional) The name for this node and its associated tensors.  One to three tensors will be produced
    public init(input: String? = nil, stateSize: Int, name: String? = nil) {
        self.stateSize = stateSize
        switch (activation) {
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
            rweightsShape = TensorShape([2, 4*stateSize, stateSize])       //  [2, 4H, H] for bidirectional
        }
        else {
            rweightsShape = TensorShape([4*stateSize, stateSize])       //  [4H, H]
        }
        var rweights: Tensor
        if (recurrentWeightOrthogonalization) {
            let squareWeightsShape = TensorShape([stateSize, stateSize])
            rweights = try CreateTensor.createOrthogonalWeightInitializationTensor(type: weightType, shape: squareWeightsShape, initializationInfo: recurrentWeightInitialization, numGates: bidirectional ? 8 : 4)
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
            let learningVariable = LearningVariable(variable: node, tensor: rweightTensor, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
            graph.learningVariables.append(learningVariable)
        }
        
        //  Add the input weights variable
        let iweightsShape = TensorShape([(bidirectional ? 8 : 4) * stateSize, numInputs])       //  [4H, I], [8H, I] for bidirectional
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
            let learningVariable = LearningVariable(variable: node, tensor: iweightTensor, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
            graph.learningVariables.append(learningVariable)
        }
        
        //  Add the bias variable
        let biasShape = TensorShape([(bidirectional ? 8 : 4) * stateSize])       //  [4H] or [8H] for bidirectional
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
            let learningVariable = LearningVariable(variable: node, tensor: biasTensor, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
            graph.learningVariables.append(learningVariable)
        }
        
        //  Create the descriptor
        let descriptor = MPSGraphLSTMDescriptor()
        descriptor.activation = activation
        descriptor.bidirectional = bidirectional
        descriptor.cellGateActivation = cellGateActivation
        descriptor.forgetGateActivation = forgetGateActivation
        descriptor.forgetGateLast = false
        descriptor.inputGateActivation = inputGateActivation
        descriptor.outputGateActivation = outputGateActivation
        descriptor.produceCell = produceCellOutput
        descriptor.reverse = false
        descriptor.training = (lossNode != nil)
        
        //  Get the suffixes for the output tensors
        if (targetLoops) { targetIndices.append(suffixes.count) }
        suffixes.append("_state")
        if (descriptor.produceCell || descriptor.training) {
            if (targetLoops) { targetIndices.append(suffixes.count) }
            suffixes.append("_cell")
        }
        if (descriptor.training) {
            targetIndices.append(suffixes.count)
            suffixes.append("_z")
        }
        
        let lstm = graph.mpsgraph.LSTM(inputTensor, recurrentWeight: rweightTensor, inputWeight: iweightTensor, bias: biasTensor, initState: nil, initCell: nil, descriptor: descriptor, name: graph.getFullName(name))
        addedTensors += lstm
        
        //  If requested to create a 'last state' output, do so
        if (createLastState) {
            let stateSliceName = graph.getFullName(name)! + "_lastStateSlice"
            let stateSlice = graph.mpsgraph.sliceTensor(lstm.first!, dimension: 0, start: -1, length: 1, name: stateSliceName)
            addedTensors.append(stateSlice)
            suffixes.append("_lastStateSlice")
            
            var stateShape = stateSlice.shape!
            stateShape.removeFirst()
            
            let stateReshapeName = graph.getFullName(name)! + "_lastState"
            let stateReshape = graph.mpsgraph.reshape(stateSlice, shape: stateShape, name: stateReshapeName)
            addedTensors.append(stateReshape)
            if (targetLasts) { targetIndices.append(suffixes.count)}
            suffixes.append("_lastState")
        }
        
        //  If requested to create a 'last cell' output, do so
        if (createLastCell) {
            let cellSliceName = graph.getFullName(name)! + "_lastCellSlice"
            let cellSlice = graph.mpsgraph.sliceTensor(lstm.first!, dimension: 0, start: -1, length: 1, name: cellSliceName)
            addedTensors.append(cellSlice)
            suffixes.append("_lastCellSlice")
            
            var cellShape = cellSlice.shape!
            cellShape.removeFirst()
            
            let cellReshapeName = graph.getFullName(name)! + "_lastCell"
            let cellReshape = graph.mpsgraph.reshape(cellSlice, shape: cellShape, name: cellReshapeName)
            addedTensors.append(cellReshape)
            if (targetLasts) { targetIndices.append(suffixes.count)}
            suffixes.append("_lastCell")
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
    
    ///  Modifier to set the activation function for the LSTM layer.  Default is tanh
    public func activationFunction(_ activation: MPSGraphRNNActivation) -> LSTMLayer {
        self.activation = activation
        return self
    }
    
    ///  Modifier to set the cell gate activation function for the LSTM layer.  Default is tanh
    public func cellGateActivationFunction(_ activation: MPSGraphRNNActivation) -> LSTMLayer {
        self.cellGateActivation = activation
        return self
    }
    
    ///  Modifier to set the forget gate activation function for the LSTM layer.  Default is sigmoid
    public func forgetGateActivationFunction(_ activation: MPSGraphRNNActivation) -> LSTMLayer {
        self.forgetGateActivation = activation
        return self
    }
    
    ///  Modifier to set the input gate activation function for the LSTM layer.  Default is sigmoid
    public func inputGateActivationFunction(_ activation: MPSGraphRNNActivation) -> LSTMLayer {
        self.inputGateActivation = activation
        return self
    }
    
    ///  Modifier to set the output gate activation function for the LSTM layer.  Default is sigmoid
    public func outputGateActivationFunction(_ activation: MPSGraphRNNActivation) -> LSTMLayer {
        self.outputGateActivation = activation
        return self
    }
    
    ///  Modifier to set all of the activation function for the LSTM layer to the same function.
    public func allActivationFunctions(_ activation: MPSGraphRNNActivation) -> LSTMLayer {
        self.activation = activation
        self.cellGateActivation = activation
        self.forgetGateActivation = activation
        self.inputGateActivation = activation
        self.outputGateActivation = activation
        return self
    }
    
    ///  Modifier to set the initializer parameters for the recurrent weights
    public func recurrentWeightInitialization(initializerInfo: WeightInitialization, orthogonalize: Bool = true) -> LSTMLayer {
        recurrentWeightInitialization = initializerInfo
        self.recurrentWeightOrthogonalization = orthogonalize
        return self
    }

    ///  Modifier to set the initializer parameters for the input weights
    public func inputWeightInitialization(initializerInfo: WeightInitialization) -> LSTMLayer {
        inputWeightInitialization = initializerInfo
        return self
    }
    
    ///  Modifier to set the initialization value for the initialization of the biases
    public func biasInitialValue(initialValue: Double) -> LSTMLayer {
        self.biasInitialValue = initialValue
        return self
    }
    
    ///  Modifier to make the recurrent layer bidirectional
    public func makeBidirectional() -> LSTMLayer {
        self.bidirectional = true
        return self
    }

    ///  Modifier to set the node output options
    /// - Parameters:
    ///   - produceCellOutput: (Optional) if true the cell state time loop output tensor with suffix "_cell" is produced..  Defaults to false
    ///   - createLastState: (Optional) if true the last state tensor of the time loop is separated out and made it's own output tensor with suffix "_lastState".  Defaults to true
    ///   - createLastCell: (Optional) if true the last cell tensor of the time loop is separated out and made it's own output tensor with suffix "_lastCell".  Defaults to false.  If true produceCellOutput will set true
    ///   - targetLoops: (Optional)  If true the time loop outputs (state and optionally cell) are marked as target tensors if the node is marked as a target.  Defaults to false
    ///   - targetLasts: (Optional)  If true the last-time outputs (state and optionally cell) are marked as target tensors if the node is marked as a target.  Defaults to false
    /// - Returns: The LSTMLayer node
    public func setOutput(produceCellOutput: Bool = false, createLastState: Bool = true, createLastCell: Bool = false, targetLoops: Bool = false, targetLasts: Bool = true) -> LSTMLayer {
        self.produceCellOutput = produceCellOutput
        self.createLastState = createLastState
        self.createLastCell = createLastCell
        if (createLastCell) { self.produceCellOutput = true }
        self.targetLoops = targetLoops
        self.targetLasts = targetLasts
        return self
    }
    
    /// Modifier to set the LSTM layer to learn with respect to a loss calculation.  The z tensor will be output if this is used
    /// - Parameters:
    ///   - mode: lossNode: the name of the loss calculation in the Graph
    ///   - using: (Optional) the optimizer method to use for learning.  Defaults to stochastic gradient descent
    ///   - gradientClipping: (Optional) defaults to nil.  A tuple with the minimum and maximum gradient values allowed in the back-propogation for this node.  The gradient is clipped to this range before being used by the optimizer
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String, using: LearningOptimizer = .stochasticGradientDescent, gradientClipping: (min: Double, max: Double)? = nil) -> LSTMLayer {
        self.lossNode = lossNode
        self.learningOptimizer = using
        self.gradientClipping = gradientClipping
        return self
    }
}
