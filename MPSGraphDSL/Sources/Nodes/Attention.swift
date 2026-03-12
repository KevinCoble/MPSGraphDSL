//
//  Attention.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 3/2/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph



//  emb     = token embedding size
//  bs     = block size - number of tokens in input
//  hs    - head size - size of encoding in query, key, value
//  nh    - number of heads
//  B    = Batch size
//
//
// Tensor                       non-batch shape    batched shape    multi-head shape     batched multi-head shape
//
// Input (token embeddings)         [bs, emb]       [B, bs, emb]        [bs, emb]           [B, bs, emb]
//
// query, key, value Weights        [emb, hs]       [emb, hs]           [nh, emb, hs]       [nh, emb, hs]
//
// matmul in Q, K, V linear         [bs, hs]        [B, bs, hs]         [nh, bs, hs]        [B, nh, bs, hs]
//
// Biases                           [bs, hs]        [bs, hs]            [nh, bs, hs]        [nh, bs, hs]
//
// Bias addition                    [bs, hs]        [B, bs, hs]         [nh, bs, hs]        [B, nh, bs, hs]
//
// transposed K                     [hs, bs]        [B, hs, bs]         [nh, hs, bs]        [B, nh, hs, bs]
//
// QxK                              [bs, bs]        [B, bs, bs]         [nh, bs, bs]        [B, nh, bs, bs]
//
// scaled QxK                       [bs, bs]        [B, bs, bs]         [nh, bs, bs]        [B, nh, bs, bs]
//
// mask                             [bs, bs]        [bs, bs]            [bs, bs]            [bs, bs]
//
// masked QxK                       [bs, bs]        [B, bs, bs]         [nh, bs, bs]        [B, nh, bs, bs]
//
// Softmax                          [bs, bs]        [B, bs, bs]         [nh, bs, bs]        [B, nh, bs, bs]
//
// Softmax x V                      [bs, hs]        [B, bs, hs]         [nh, bs, hs]        [B, nh, bs, hs]
//
// (If number of heads = 1, reshape to remove nh dimension at end)
//
// Transposed Attention                                                 [bs, nh, hs]        [B, bs, nh, hs]

///   Node for a self-attention layer
///         Token embedding size pulled from last dimension of input
///         Data type pulled from input tensor
///
public class SelfAttention : UnaryNode {
    let headSize: Int
    let numHeads: Int
    let masked: Bool
    
    var queryWeightInitialization: WeightInitialization = .XavierGlorotNormal
    var keyWeightInitialization: WeightInitialization = .XavierGlorotNormal
    var valueWeightInitialization: WeightInitialization = .XavierGlorotNormal
    
    var queryHasBias: Bool = false
    var keyHasBias: Bool = false
    var valueHasBias: Bool = false
    var queryBiasInitialValue: Double = 0.0
    var keyBiasInitialValue: Double = 0.0
    var valueBiasInitialValue: Double = 0.0

    var lossNode: String? = nil
    var learningOptimizer: LearningOptimizer = .stochasticGradientDescent
    var gradientClipping: (min: Double, max: Double)? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    var totalParameterCount: Int = 0

    /// Constructor for an optionally masked self attention layer
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used.  Must be of shape blockSize x embeddingSize
    ///   - headSize: The size of the query and key, vectors produced from each token embedding
    ///   - numHeads: (Optional) The number of independent attention heads.  Default is 1.  The head dimension is removed at the end if head count is 1
    ///   - masked: (Optional)  If true the upper-right triangle of the query x keys matrix is masked off, removing the ability of future tokens to affect previous tokens
    ///   - name: The name for this node and its associated tensor.
    public init(input: String? = nil, headSize: Int, numHeads: Int = 1, masked: Bool = false, name: String) {
        self.headSize = headSize
        self.numHeads = numHeads
        self.masked = masked
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
            //  If a batch graph, insert a "head" dimension - sized to 1 so it will be broadcast across the heads
            if (graph.batchGraph) {
                if (shape.count != 3) { throw MPSGraphNeuralNetErrors.AttentionInputShapeNot2D }
                let newShapeSizes: [Int] = [graph.batchSize, 1, Int(truncating: shape[shape.count-2]), Int(truncating: shape.last!)]
                inputShape = TensorShape(newShapeSizes)
                
                inputTensor = graph.mpsgraph.reshape(inputTensor, shape: inputShape.getMPSShape(), name: name! + "_input_reshape")
                addedTensors.append(inputTensor)
                suffixes.append("_input_reshape")
            }
            else {
                if (shape.count != 2) { throw MPSGraphNeuralNetErrors.AttentionInputShapeNot2D }
                inputShape = TensorShape(fromMPS: shape)
            }
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        //  Get the data type, block size and the token size from the last dimension of the input shape
        let tokenSize = inputShape.dimensions[inputShape.numDimensions-1]
        let blockSize = inputShape.dimensions[inputShape.numDimensions-2]
        let dataType = inputTensor.dataType
        
        //  Add the key generation layer
        let addedKeyItems = try SelfAttention.addFullyConnectedBlock(graph: graph, inputTensor: inputTensor, tokenSize: tokenSize, blockSize: blockSize, outputSize: headSize, numHeads: numHeads, dataType: dataType, weightInitialization: keyWeightInitialization, hasBias: keyHasBias, biasInitialValue: keyBiasInitialValue, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer, name: name!, suffix: "_key")
        addedTensors.append(contentsOf: addedKeyItems.addedTensors)
        suffixes.append(contentsOf: addedKeyItems.addedSuffixes)
        let keyTensor = addedKeyItems.addedTensors.last!
        totalParameterCount += addedKeyItems.addedParameters

        //  Add the query generation layer
        let addedQueryItems = try SelfAttention.addFullyConnectedBlock(graph: graph, inputTensor: inputTensor, tokenSize: tokenSize, blockSize: blockSize, outputSize: headSize, numHeads: numHeads, dataType: dataType, weightInitialization: queryWeightInitialization, hasBias: queryHasBias, biasInitialValue: queryBiasInitialValue, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer, name: name!, suffix: "_query")
        addedTensors.append(contentsOf: addedQueryItems.addedTensors)
        suffixes.append(contentsOf: addedQueryItems.addedSuffixes)
        let queryTensor = addedQueryItems.addedTensors.last!
        totalParameterCount += addedQueryItems.addedParameters

        //  If masked, create a mask tensor
        var mask: MPSGraphTensor? = nil
        if (masked) {
            //  Create a mask tensor
            var maskValues : [Double] = []
            for row in 0..<blockSize {
                maskValues += Array(repeating: 0.0, count: row+1)
                let numInfinities = blockSize - (row + 1)
                if (numInfinities > 0) { maskValues += Array(repeating: -Double.infinity, count: row+1) }
            }
            let maskTensor = try CreateTensor.arrayOfValues(type: DataType(from: dataType), shape: TensorShape([blockSize, blockSize]), initialValues: maskValues)
            mask = graph.mpsgraph.constant(maskTensor.getData(), shape: [NSNumber(value: blockSize), NSNumber(value: blockSize)], dataType: dataType)
            addedTensors.append(mask!)
            suffixes.append("_mask")
        }
        
        //  Add the value generation layer
        let addedValueItems = try SelfAttention.addFullyConnectedBlock(graph: graph, inputTensor: inputTensor, tokenSize: tokenSize, blockSize: blockSize, outputSize: headSize, numHeads: numHeads, dataType: dataType, weightInitialization: valueWeightInitialization, hasBias: valueHasBias, biasInitialValue: valueBiasInitialValue, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer, name: name!, suffix: "_value")
        addedTensors.append(contentsOf: addedValueItems.addedTensors)
        suffixes.append(contentsOf: addedValueItems.addedSuffixes)
        let valueTensor = addedValueItems.addedTensors.last!
        totalParameterCount += addedValueItems.addedParameters

        //  Get the scaling constant
        let scaledConstant = 1.0 / sqrt(Double(headSize))

        
        //****   Apples scaledDotProductAttention seem to fail in gradient determination - so we will go with our own   ****
        
//        //  Add the scaledDotProductAttention layer
//        let attention = graph.mpsgraph.scaledDotProductAttention(
//                query: queryTensor,
//                key: keyTensor,
//                value: valueTensor,
//                mask: mask,
//                scale: scaledConstant,
//                name: attentionName
//        )
//        addedTensors.append(attention)
        
        //  Transpose the Key tensor
        let lastKeyDimension = keyTensor.shape!.count - 1
        let transposedKey = graph.mpsgraph.transposeTensor(keyTensor, dimension: lastKeyDimension, withDimension: lastKeyDimension-1, name: name! + "_transposedKey")
        addedTensors.append(transposedKey)
        suffixes.append("_transposedKey")
        
        //  Multiply Q by K^T
        let QxKT = graph.mpsgraph.matrixMultiplication(primary: queryTensor, secondary: transposedKey, name: name! + "_QxKT")
        addedTensors.append(QxKT)
        suffixes.append("_QxKT")
        
        //  Scale by 1 over square root of head size
        let scaleConstant = graph.mpsgraph.constant(scaledConstant, dataType: dataType)
        addedTensors.append(scaleConstant)
        suffixes.append("_scalingConstant")
        var scaledQxKT = graph.mpsgraph.multiplication(QxKT, scaleConstant, name: name! + "_scaledQxKT")
        addedTensors.append(scaledQxKT)
        suffixes.append("_scaledQxKT")
        
        // If masked, add that
        if let mask = mask {
            let maskedScaledQxKT = graph.mpsgraph.addition(scaledQxKT, mask, name: name! + "_maskedScaledQxKT")
            addedTensors.append(maskedScaledQxKT)
            suffixes.append("_maskedScaledQxKT")
            scaledQxKT = maskedScaledQxKT
        }
        
        //  Softmax in the last dimension
        let softMax = graph.mpsgraph.softMax(with: scaledQxKT, axis: scaledQxKT.shape!.count - 1, name: name! + "_softmax")
        addedTensors.append(softMax)
        suffixes.append("_softmax")

        //  Multiply by the value
        let attention = graph.mpsgraph.matrixMultiplication(primary: softMax, secondary: valueTensor, name: name! + "_scaledAttention")
        addedTensors.append(attention)
        suffixes.append("_scaledAttention")

        //  If head count is 1, remove the head dimension
        if (numHeads == 1) {
            if (attention.shape == nil) { throw GenericMPSGraphDSLErrors.UnknownShape }
            var reshapeShape = attention.shape!
            reshapeShape.remove(at: reshapeShape.count - 3)
            
            let reshapedAttention = graph.mpsgraph.reshape( attention, shape: reshapeShape, name: name!)
            addedTensors.append(reshapedAttention)
            suffixes.append("")
        }
        
        //  If head count greater than 1, transpose block and head dimension so the head and headSize dimensions can be concatenated later
        else {
            let transposedAttention = graph.mpsgraph.transposeTensor(attention, dimension: -2, withDimension: -3, name: name!)
            addedTensors.append(transposedAttention)
            suffixes.append("")
        }

        targetIndices.append(addedTensors.count - 1)     //  Last tensor is the normal target
        return addedTensors
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }
    
    /// Modifier to set the weight initialization for the Query projection.  Default is Xavier/Glorot normal distribution
    /// - Parameter initializer: the initializer to be used for the weight matrix
    /// - Returns: The modified layer
    public func setQueryWeightInitializer(_ initializer: WeightInitialization) -> SelfAttention {
        self.queryWeightInitialization = initializer
        return self
    }
    
    /// Modifier to set the weight initialization for the Key projection.  Default is Xavier/Glorot normal distribution
    /// - Parameter initializer: the initializer to be used for the weight matrix
    /// - Returns: The modified layer
    public func setKeyWeightInitializer(_ initializer: WeightInitialization) -> SelfAttention {
        self.keyWeightInitialization = initializer
        return self
    }
    
    /// Modifier to set the weight initialization for the Value projection.  Default is Xavier/Glorot normal distribution
    /// - Parameter initializer: the initializer to be used for the weight matrix
    /// - Returns: The modified layer
    public func setValueWeightInitializer(_ initializer: WeightInitialization) -> SelfAttention {
        self.valueWeightInitialization = initializer
        return self
    }

    /// Modifier to add a bias term to the Query projection.  Default is false
    /// - Returns: The modified layer
    public func addQueryBias() -> SelfAttention {
        self.queryHasBias = true
        return self
    }
    
    /// Modifier to add a bias term to the Key projection.  Default is false
    /// - Returns: The modified layer
    public func addKeyBias() -> SelfAttention {
        self.keyHasBias = true
        return self
    }
    
    /// Modifier to add a bias term to the Value projection.  Default is false
    /// - Returns: The modified layer
    public func addValueBias() -> SelfAttention {
        self.valueHasBias = true
        return self
    }
    
    /// Modifier to set all the bias term use flags for the layer.  Defaults are false
    /// - Parameters:
    ///   - query: use flag for the Query projection
    ///   - key: use flag for the Key projection
    ///   - value: use flag for the Value projection
    /// - Returns: The modified layer
    public func setBiasUse(query: Bool, key: Bool, value: Bool) -> SelfAttention {
        self.queryHasBias = query
        self.keyHasBias = key
        self.valueHasBias = value
        return self
    }

    /// Modifier to change the initialization of the bias term to the Query projection.  Default is 0
    /// - Returns: The modified layer
    public func queryBiasInitialValue(initialValue: Double) -> SelfAttention {
        self.queryBiasInitialValue = initialValue
        return self
    }

    /// Modifier to change the initialization of the bias term to the Key projection.  Default is 0
    /// - Returns: The modified layer
    public func keyBiasInitialValue(initialValue: Double) -> SelfAttention {
        self.keyBiasInitialValue = initialValue
        return self
    }

    /// Modifier to change the initialization of the bias term to the Value projection.  Default is 0
    /// - Returns: The modified layer
    public func queryValueInitialValue(initialValue: Double) -> SelfAttention {
        self.valueBiasInitialValue = initialValue
        return self
    }

    /// Modifier to configure the layer's variables to learn
    /// - Parameters:
    ///   - lossNode: the name of the loss calculation in the Graph
    ///   - using: (Optional) the optimizer method to use for learning.  Defaults to stochastic gradient descent
    ///   - gradientClipping: (Optional) defaults to nil.  A tuple with the minimum and maximum gradient values allowed in the back-propogation for this node.  The gradient is clipped to this range before being used by the optimizer
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String, using: LearningOptimizer = .stochasticGradientDescent, gradientClipping: (min: Double, max: Double)? = nil) -> SelfAttention {
        self.lossNode = lossNode
        self.learningOptimizer = using
        self.gradientClipping = gradientClipping
        return self
    }

    internal static func addFullyConnectedBlock(graph: Graph, inputTensor: MPSGraphTensor, tokenSize: Int, blockSize: Int, outputSize: Int, numHeads: Int, dataType: MPSDataType, weightInitialization: WeightInitialization, hasBias: Bool, biasInitialValue: Double, loss: String?, clipping: (min: Double, max: Double)?, optimizer: LearningOptimizer, name: String, suffix: String) throws -> (addedTensors: [MPSGraphTensor], addedSuffixes: [String], addedParameters: Int) {
        var addedTensors: [MPSGraphTensor] = []
        var addedSuffixes: [String] = []
        var addedParameters = 0

        //  Add the weights variable
        let weightShape = TensorShape([numHeads, tokenSize, outputSize])
        let weightType = DataType(from: dataType)
        let weights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: tokenSize, numOutput: outputSize)
        let weightData = weights.getData()
        let weightName = name + suffix + "_weights"
        let weightTensor = graph.mpsgraph.variable(with: weightData, shape: weightShape.getMPSShape(), dataType: weights.type.getMPSDataType(), name: weightName)
        addedSuffixes.append(suffix + "_weights")
        addedTensors.append(weightTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        let node = try Variable.createWeightInitializationVariable(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: tokenSize, numOutput: outputSize, name: weightName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: weightTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  If this is a learning layer - add the weights to the list to get assignment operations for
        if let lossNode = loss {
            let learningVariable = LearningVariable(variable: node, tensor: weightTensor, loss: lossNode, clipping: clipping, optimizer: optimizer)
            graph.learningVariables.append(learningVariable)
            addedParameters += weightShape.totalSize
        }
        
        //  Add the matrix multiply
        let matrixTensorName = name + suffix + "_matrixMult"
        let matrixMultTensor = graph.mpsgraph.matrixMultiplication(primary: inputTensor, secondary: weightTensor, name: matrixTensorName)
        addedTensors.append(matrixMultTensor)
        addedSuffixes.append(suffix + "_matrixMult")
        
        //  If we need a bias term, add that now
        if hasBias {
            //  Get the bias shape
            let biasShape = TensorShape([numHeads, blockSize, outputSize])
            
            let biases = CreateTensor.constantValues(type: weightType, shape: biasShape, initialValue: biasInitialValue)
            let biasData = biases.getData()
            let biasName = name + suffix + "_biases"
            let biasTensor = graph.mpsgraph.variable(with: biasData, shape: biasShape.getMPSShape(), dataType: weightType.getMPSDataType(), name: biasName)
            addedSuffixes.append(suffix + "_biases")
            addedTensors.append(biasTensor)

            //  If we are adding load or reset assignments, put this variable on the list for load assignments
            let node = Variable(dataType: weightType, shape: biasShape, initialValue: biasInitialValue, name: biasName)
            if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
                let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: biasTensor, sourceTensor: nil)
                graph.loadResetAssignList.append(loadResetAssignInfo)
            }

            //  If this is a learning layer - add the biases to the list to get assignment operations for
            if let loss = loss {
                let learningVariable = LearningVariable(variable: node, tensor: biasTensor, loss: loss, clipping: clipping, optimizer: optimizer)
                graph.learningVariables.append(learningVariable)
                addedParameters += biasShape.totalSize
            }

            let finalTensorName = name + suffix
            let finalTensor = graph.mpsgraph.addition(matrixMultTensor, biasTensor, name: finalTensorName)
            addedTensors.append(finalTensor)
            addedSuffixes.append(suffix)
        }

        return (addedTensors: addedTensors, addedSuffixes: addedSuffixes, addedParameters: addedParameters)
    }
    
    override func getNumberOfParameters() throws -> Int {
        return totalParameterCount
    }
}

///   Node for a cross-attention layer
///         Token embedding size pulled from last dimension of input
///         Data type pulled from input tensor
///
public class CrossAttention : BinaryNode {
    let headSize: Int
    let numHeads: Int
    let masked: Bool
    
    var queryWeightInitialization: WeightInitialization = .XavierGlorotNormal
    var keyWeightInitialization: WeightInitialization = .XavierGlorotNormal
    var valueWeightInitialization: WeightInitialization = .XavierGlorotNormal
    
    var queryHasBias: Bool = false
    var keyHasBias: Bool = false
    var valueHasBias: Bool = false
    var queryBiasInitialValue: Double = 0.0
    var keyBiasInitialValue: Double = 0.0
    var valueBiasInitialValue: Double = 0.0

    var lossNode: String? = nil
    var learningOptimizer: LearningOptimizer = .stochasticGradientDescent
    var gradientClipping: (min: Double, max: Double)? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    var totalParameterCount: Int = 0

    /// Constructor for an optionally masked self attention layer
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand for the queries.  If nil the previous node's output will be used.  Must be of shape blockSize x embeddingSize
    ///   - input: (Optional) The name of the tensor that will provide the input operand for the keys and values.  If nil the previous node's output will be used.  Must be of shape blockSize x embeddingSize
    ///   - headSize: The size of the query and key, vectors produced from each token embedding
    ///   - numHeads: (Optional) The number of independent attention heads.  Default is 1.  The head dimension is removed at the end if head count is 1
    ///   - masked: (Optional)  If true the upper-right triangle of the query x keys matrix is masked off, removing the ability of future tokens to affect previous tokens
    ///   - name: The name for this node and its associated tensor.
    public init(input: String? = nil, crossInput: String? = nil, headSize: Int, numHeads: Int = 1, masked: Bool = false, name: String) {
        self.headSize = headSize
        self.numHeads = numHeads
        self.masked = masked
        super.init(firstInput: input, secondInput: crossInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        totalParameterCount = 0

        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        var inputShape: TensorShape
        if let shape = inputTensors.firstInputTensor.shape {
            inputShape = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        if let shape = inputTensors.secondInputTensor.shape {
            let otherShape = TensorShape(fromMPS: shape)
            if (inputShape != otherShape) { throw MPSGraphNeuralNetErrors.CrossAttentionInputShapesMismatch }
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        //  If a batch graph, insert a "head" dimension to the inputs - sized to 1 so it will be broadcast across the heads
        var firstInputTensor = inputTensors.firstInputTensor
        var secondInputTensor = inputTensors.secondInputTensor
        if (graph.batchGraph) {
            let newShapeSizes: [Int] = [graph.batchSize, 1, inputShape.dimensions[inputShape.dimensions.count-2], inputShape.dimensions.last!]
            inputShape = TensorShape(newShapeSizes)
            
            firstInputTensor = graph.mpsgraph.reshape(firstInputTensor, shape: inputShape.getMPSShape(), name: name! + "_input1_reshape")
            addedTensors.append(firstInputTensor)
            suffixes.append("_input1_reshape")
            
            secondInputTensor = graph.mpsgraph.reshape(secondInputTensor, shape: inputShape.getMPSShape(), name: name! + "_input2_reshape")
            addedTensors.append(firstInputTensor)
            suffixes.append("_input2_reshape")
        }

        //  Get the data type, block size and the token size from the last dimension of the input shape
        let tokenSize = inputShape.dimensions[inputShape.numDimensions-1]
        let blockSize = inputShape.dimensions[inputShape.numDimensions-2]
        let dataType = inputTensors.firstInputTensor.dataType
        
        //  Add the key generation layer
        let addedKeyItems = try SelfAttention.addFullyConnectedBlock(graph: graph, inputTensor: secondInputTensor, tokenSize: tokenSize, blockSize: blockSize, outputSize: headSize, numHeads: numHeads, dataType: dataType, weightInitialization: keyWeightInitialization, hasBias: keyHasBias, biasInitialValue: keyBiasInitialValue, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer, name: name!, suffix: "_key")
        addedTensors.append(contentsOf: addedKeyItems.addedTensors)
        suffixes.append(contentsOf: addedKeyItems.addedSuffixes)
        let keyTensor = addedKeyItems.addedTensors.last!
        totalParameterCount += addedKeyItems.addedParameters

        //  Add the query generation layer
        let addedQueryItems = try SelfAttention.addFullyConnectedBlock(graph: graph, inputTensor: firstInputTensor, tokenSize: tokenSize, blockSize: blockSize, outputSize: headSize, numHeads: numHeads, dataType: dataType, weightInitialization: queryWeightInitialization, hasBias: queryHasBias, biasInitialValue: queryBiasInitialValue, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer, name: name!, suffix: "_query")
        addedTensors.append(contentsOf: addedQueryItems.addedTensors)
        suffixes.append(contentsOf: addedQueryItems.addedSuffixes)
        let queryTensor = addedQueryItems.addedTensors.last!
        totalParameterCount += addedQueryItems.addedParameters

        //  If masked, create a mask tensor
        var mask: MPSGraphTensor? = nil
        if (masked) {
            //  Create a mask tensor
            var maskValues : [Double] = []
            for row in 0..<blockSize {
                maskValues += Array(repeating: 0.0, count: row+1)
                let numInfinities = blockSize - (row + 1)
                if (numInfinities > 0) { maskValues += Array(repeating: -Double.infinity, count: row+1) }
            }
            let maskTensor = try CreateTensor.arrayOfValues(type: DataType(from: dataType), shape: TensorShape([blockSize, blockSize]), initialValues: maskValues)
            mask = graph.mpsgraph.constant(maskTensor.getData(), shape: [NSNumber(value: blockSize), NSNumber(value: blockSize)], dataType: dataType)
            addedTensors.append(mask!)
            suffixes.append("_mask")
        }
        
        //  Add the value generation layer
        let addedValueItems = try SelfAttention.addFullyConnectedBlock(graph: graph, inputTensor: secondInputTensor, tokenSize: tokenSize, blockSize: blockSize, outputSize: headSize, numHeads: numHeads, dataType: dataType, weightInitialization: valueWeightInitialization, hasBias: valueHasBias, biasInitialValue: valueBiasInitialValue, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer, name: name!, suffix: "_value")
        addedTensors.append(contentsOf: addedValueItems.addedTensors)
        suffixes.append(contentsOf: addedValueItems.addedSuffixes)
        let valueTensor = addedValueItems.addedTensors.last!
        totalParameterCount += addedValueItems.addedParameters

        //  Get the scaling constant
        let scaledConstant = 1.0 / sqrt(Double(headSize))


        //****   Apples scaledDotProductAttention seem to fail in gradient determination - so we will go with our own   ****
        
//        //  Add the scaledDotProductAttention layer
//        let attention = graph.mpsgraph.scaledDotProductAttention(
//                query: queryTensor,
//                key: keyTensor,
//                value: valueTensor,
//                mask: mask,
//                scale: scaledConstant,
//                name: attentionName
//        )
//        addedTensors.append(attention)
        
        //  Transpose the Key tensor
        let lastKeyDimension = keyTensor.shape!.count - 1
        let transposedKey = graph.mpsgraph.transposeTensor(keyTensor, dimension: lastKeyDimension, withDimension: lastKeyDimension-1, name: name! + "_transposedKey")
        addedTensors.append(transposedKey)
        suffixes.append("_transposedKey")
        
        //  Multiply Q by K^T
        let QxKT = graph.mpsgraph.matrixMultiplication(primary: queryTensor, secondary: transposedKey, name: name! + "_QxKT")
        addedTensors.append(QxKT)
        suffixes.append("_QxKT")
        
        //  Scale by 1 over square root of head size
        let scaleConstant = graph.mpsgraph.constant(scaledConstant, dataType: dataType)
        
        addedTensors.append(scaleConstant)
        suffixes.append("_scalingConstant")
        var scaledQxKT = graph.mpsgraph.multiplication(QxKT, scaleConstant, name: name! + "_scaledQxKT")
        addedTensors.append(scaledQxKT)
        suffixes.append("_scaledQxKT")
        
        // If masked, add that
        if let mask = mask {
            let maskedScaledQxKT = graph.mpsgraph.addition(scaledQxKT, mask, name: name! + "_maskedScaledQxKT")
            addedTensors.append(maskedScaledQxKT)
            suffixes.append("_maskedScaledQxKT")
            scaledQxKT = maskedScaledQxKT
        }
        
        //  Softmax in the last dimension
        let softMax = graph.mpsgraph.softMax(with: scaledQxKT, axis: scaledQxKT.shape!.count - 1, name: name! + "_softmax")
        addedTensors.append(softMax)
        suffixes.append("_softmax")

        //  Multiply by the value
        let attention = graph.mpsgraph.matrixMultiplication(primary: softMax, secondary: valueTensor, name: name! + "_scaledAttention")
        addedTensors.append(attention)
        suffixes.append("_scaledAttention")

        //  If head count is 1, remove the head dimension
        if (numHeads == 1) {
            if (attention.shape == nil) { throw GenericMPSGraphDSLErrors.UnknownShape }
            var reshapeShape = attention.shape!
            reshapeShape.remove(at: reshapeShape.count - 3)
            
            let reshapedAttention = graph.mpsgraph.reshape( attention, shape: reshapeShape, name: name!)
            addedTensors.append(reshapedAttention)
            suffixes.append("")
        }
        
        //  If head count greater than 1, transpose block and head dimension so the head and headSize dimensions can be concatenated later
        else {
            let transposedAttention = graph.mpsgraph.transposeTensor(attention, dimension: -2, withDimension: -3, name: name!)
            addedTensors.append(transposedAttention)
            suffixes.append("")
        }

        targetIndices.append(addedTensors.count - 1)     //  Last tensor is the normal target
        return addedTensors
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }
    
    /// Modifier to set the weight initialization for the Query projection.  Default is Xavier/Glorot normal distribution
    /// - Parameter initializer: the initializer to be used for the weight matrix
    /// - Returns: The modified layer
    public func setQueryWeightInitializer(_ initializer: WeightInitialization) -> CrossAttention {
        self.queryWeightInitialization = initializer
        return self
    }
    
    /// Modifier to set the weight initialization for the Key projection.  Default is Xavier/Glorot normal distribution
    /// - Parameter initializer: the initializer to be used for the weight matrix
    /// - Returns: The modified layer
    public func setKeyWeightInitializer(_ initializer: WeightInitialization) -> CrossAttention {
        self.keyWeightInitialization = initializer
        return self
    }
    
    /// Modifier to set the weight initialization for the Value projection.  Default is Xavier/Glorot normal distribution
    /// - Parameter initializer: the initializer to be used for the weight matrix
    /// - Returns: The modified layer
    public func setValueWeightInitializer(_ initializer: WeightInitialization) -> CrossAttention {
        self.valueWeightInitialization = initializer
        return self
    }

    /// Modifier to add a bias term to the Query projection.  Default is false
    /// - Returns: The modified layer
    public func addQueryBias() -> CrossAttention {
        self.queryHasBias = true
        return self
    }
    
    /// Modifier to add a bias term to the Key projection.  Default is false
    /// - Returns: The modified layer
    public func addKeyBias() -> CrossAttention {
        self.keyHasBias = true
        return self
    }
    
    /// Modifier to add a bias term to the Value projection.  Default is false
    /// - Returns: The modified layer
    public func addValueBias() -> CrossAttention {
        self.valueHasBias = true
        return self
    }
    
    /// Modifier to set all the bias term use flags for the layer.  Defaults are false
    /// - Parameters:
    ///   - query: use flag for the Query projection
    ///   - key: use flag for the Key projection
    ///   - value: use flag for the Value projection
    /// - Returns: The modified layer
    public func setBiasUse(query: Bool, key: Bool, value: Bool) -> CrossAttention {
        self.queryHasBias = query
        self.keyHasBias = key
        self.valueHasBias = value
        return self
    }

    /// Modifier to change the initialization of the bias term to the Query projection.  Default is 0
    /// - Returns: The modified layer
    public func queryBiasInitialValue(initialValue: Double) -> CrossAttention {
        self.queryBiasInitialValue = initialValue
        return self
    }

    /// Modifier to change the initialization of the bias term to the Key projection.  Default is 0
    /// - Returns: The modified layer
    public func keyBiasInitialValue(initialValue: Double) -> CrossAttention {
        self.keyBiasInitialValue = initialValue
        return self
    }

    /// Modifier to change the initialization of the bias term to the Value projection.  Default is 0
    /// - Returns: The modified layer
    public func queryValueInitialValue(initialValue: Double) -> CrossAttention {
        self.valueBiasInitialValue = initialValue
        return self
    }

    /// Modifier to configure the layer's variables to learn
    /// - Parameters:
    ///   - lossNode: the name of the loss calculation in the Graph
    ///   - using: (Optional) the optimizer method to use for learning.  Defaults to stochastic gradient descent
    ///   - gradientClipping: (Optional) defaults to nil.  A tuple with the minimum and maximum gradient values allowed in the back-propogation for this node.  The gradient is clipped to this range before being used by the optimizer
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String, using: LearningOptimizer = .stochasticGradientDescent, gradientClipping: (min: Double, max: Double)? = nil) -> CrossAttention {
        self.lossNode = lossNode
        self.learningOptimizer = using
        self.gradientClipping = gradientClipping
        return self
    }
    
    override func getNumberOfParameters() throws -> Int {
        return totalParameterCount
    }
}
