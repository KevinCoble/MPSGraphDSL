//
//  ConvolutionNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/21/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node to perform a 2-dimensional convolution operation
public class Convolution2D : BinaryNode {
    let descriptor: MPSGraphConvolution2DOpDescriptor
    
    /// Constructor for an 2D convolution operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - weights: (Optional) The name of the tensor that will provide the weights of the convolution window.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the convolution operation, defining parameters for the operation
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, weights: String? = nil, descriptor: MPSGraphConvolution2DOpDescriptor, name: String? = nil) {
        self.descriptor = descriptor
        super.init(firstInput: input, secondInput: weights, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let convolutionResult = graph.mpsgraph.convolution2D(inputTensors.firstInputTensor, weights: inputTensors.secondInputTensor, descriptor: descriptor, name: graph.getFullName(name))

        //  Return the created MPSGraphTensor
        return [convolutionResult]
    }
}

///   Node to perform a 3-dimensional convolution operation
public class Convolution3D : BinaryNode {
    let descriptor: MPSGraphConvolution3DOpDescriptor
    
    /// Constructor for an 3D convolution operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - weights: (Optional) The name of the tensor that will provide the weights of the convolution window.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the convolution operation, defining parameters for the operation
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, weights: String? = nil, descriptor: MPSGraphConvolution3DOpDescriptor, name: String? = nil) {
        self.descriptor = descriptor
        super.init(firstInput: input, secondInput: weights, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let convolutionResult = graph.mpsgraph.convolution3D(inputTensors.firstInputTensor, weights: inputTensors.secondInputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [convolutionResult]
    }
}



///  Layer for a standard convolution layer.  All parameters will be inferred from input and kernal sizes if not set directly
public class ConvolutionLayer: UnaryNode {
    let kernelSize: [Int]
    let activationFunction: ActivationFunction
    let strides: [Int]
    let numFilters: Int
    
    var assumeExtraDimensionIsBatch: Bool = false
    var filterDimensionLast: Bool = false
    var channelAsDepth: Bool = false

    var useBias: Bool = true
    var weightInitialization: WeightInitialization
    var biasInitialValue: Double = 0.0

    var dilationRateH: Int = 1
    var dilationRateW: Int = 1
    var dilationRateD: Int = 1

    var bottomPadding = 0
    var topPadding = 0
    var leftPadding = 0
    var rightPadding = 0
    var backPadding = 0
    var frontPadding = 0
    var paddingStyle: MPSGraphPaddingStyle = .TF_SAME

    var lossNode: String? = nil
    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    var testWeights = false

    /// Constructor for a 2D convolution layer.
    /// Default setup is used.  To change, use supplied modifiers
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used  Must be of rank 2 or higher
    ///   - kernelHeight: The size of the convolution kernel in the height (row) dimension
    ///   - kernelWidth: The size of the convolution kernel in the width (column) dimension
    ///   - numFilters: (Optional) The number of output layers (filters).  Default is 1
    ///   - activationFunction: (Optional)  The activation function for the layer.  Default is None
    ///   - heightStride: (Optional) The step size for the convolution kernel in the height (row) dimension.  If omitted the stride will be set to 1
    ///   - widthStride: (Optional) The step size for the convolution kernel in the width (column) dimension.  If omitted the stride will be set to 1
    ///   - name: The name for this node and its associated tensors
    public init(input: String? = nil, kernelHeight: Int, kernelWidth: Int, numFilters: Int = 1, activationFunction: ActivationFunction = .none, heightStride: Int = 1, widthStride: Int = 1, name: String) {
        self.kernelSize = [kernelHeight, kernelWidth]
        self.activationFunction = activationFunction
        self.strides = [heightStride, widthStride]
        self.numFilters = numFilters
        switch (activationFunction) {
            case .none, .tanh, .sigmoid:
                weightInitialization = .XavierGlorotNormal
            case .relu, .leakyRelu, .leakyReluFromTensor:
                weightInitialization = .HeNormal
        }
        super.init(input: input, name: name)
    }

    /// Constructor for a 3D convolution layer.
    /// Default setup is used.  To change, use supplied modifiers
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used  Must be of rank 2 or higher
    ///   - kernelHeight: The size of the convolution kernel in the height (row) dimension
    ///   - kernelWidth: The size of the convolution kernel in the width (column) dimension
    ///   - kernelDepth: The size of the convolution kernel in the depth (channel) dimension
    ///   - numFilters: (Optional) The number of output layers (filters).  Default is 1
    ///   - activationFunction: (Optional)  The activation function for the layer.  Default is None
    ///   - heightStride: (Optional) The step size for the convolution kernel in the height (row) dimension.  If omitted the stride will be set to 1
    ///   - widthStride: (Optional) The step size for the convolution kernel in the width (column) dimension.  If omitted the stride will be set to 1
    ///   - depthStride: (Optional) The step size for the convolution kernel in the depth (channel) dimension.  If omitted the stride will be set to 1
    ///   - name: The name for this node and its associated tensors
    public init(input: String? = nil, kernelHeight: Int, kernelWidth: Int, kernelDepth : Int, numFilters: Int = 1, activationFunction: ActivationFunction = .none, heightStride: Int = 1, widthStride: Int = 1, depthStride: Int = 1, name: String) {
        self.kernelSize = [kernelDepth, kernelHeight, kernelWidth]
        self.activationFunction = activationFunction
        self.strides = [depthStride, heightStride, widthStride]
        self.numFilters = numFilters
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
        if (inputShape.dimensions.count < kernelSize.count) { throw MPSGraphNeuralNetErrors.KernelRankGreaterThanInput }
        let weightType = DataType(from: inputTensor.dataType)

        //  If using a bias term, add the bias variable
        var biasTensor: MPSGraphTensor? = nil
        let biasShape = TensorShape([numFilters])
        if (useBias) {
            if (testWeights) {
                var biasConstants: [Float32] = []
                for i in 0..<numFilters {
                    biasConstants.append(0.3 + (Float32(i) * 0.1))
                }
                let constantTensor = try TensorFloat32(shape: biasShape, initialValues: biasConstants)
                let data = constantTensor.getData()
                biasTensor = graph.mpsgraph.constant(data, shape: biasShape.getMPSShape(), dataType: .float32)
            }
            else {
                let biases = CreateTensor.constantValues(type: weightType, shape: biasShape, initialValue: biasInitialValue)
                let biasData = biases.getData()
                let biasName = graph.getFullName(name)! + "_biases"
                biasTensor = graph.mpsgraph.variable(with: biasData, shape: biasShape.getMPSShape(), dataType: biases.type.getMPSDataType(), name: biasName)
                
                //  If we are adding load or reset assignments, put this variable on the list for load assignments
                let node = Variable(dataType: weightType, shape: biasShape, initialValue: biasInitialValue, name: biasName)
                if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
                    let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: biasTensor!, sourceTensor: nil)
                    graph.loadResetAssignList.append(loadResetAssignInfo)
                }
                
                //  If this is a learning layer - add to the biases to the list to get assignment operations for
                if let lossNode = lossNode {
                    graph.learningVariables.append((variable: node, tensor: biasTensor!, loss: lossNode))
                }
            }
            suffixes.append("_biases")
            addedTensors.append(biasTensor)
        }

        //  Process 2D convolutions
        var shapedOutputTensor: MPSGraphTensor
        if (kernelSize.count == 2) {
            //  Shape the input to 4D
            let shapedInputTensor: MPSGraphTensor
            let inputReshapeName = graph.getFullName(name)! + "_inputReshape"
            var numInputChannels = 1
            var haveBatch = false
            if (inputShape.numDimensions == 2) {
                var fourDShape = inputShape.dimensions
                //  Insert batch dimension
                fourDShape.insert(1, at: 0)
                //  Append channel dimension
                fourDShape.append(1)
                shapedInputTensor = graph.mpsgraph.reshape(inputTensor, shape: fourDShape.map { NSNumber(value: $0)}, name: inputReshapeName)
                suffixes.append("_inputReshape")
                addedTensors.append(shapedInputTensor)
            }
            else if (inputShape.numDimensions == 3) {
                var fourDShape = inputShape.dimensions
                if (graph.batchGraph || assumeExtraDimensionIsBatch) {
                    haveBatch = true
                    //  Third dimension is batch (at beginning), so append a channel dimension
                    fourDShape.append(1)
                }
                else {
                    //  Third dimension is channel, so insert a batch dimension and set the number of channels as inputs
                    fourDShape.insert(1, at: 0)
                    numInputChannels = inputShape.dimensions[2]     //  Number of channels is number of input features
                }
                shapedInputTensor = graph.mpsgraph.reshape(inputTensor, shape: fourDShape.map { NSNumber(value: $0)}, name: inputReshapeName)
                suffixes.append("_inputReshape")
                addedTensors.append(shapedInputTensor)
            }
            else {
                //  Assume first dimension is batch
                haveBatch = true
                //  Assume last dimension is channels, and set that as the number of input channels
                numInputChannels = inputShape.dimensions[3]     //  Number of channels is number of input features
                shapedInputTensor = inputTensor
            }

            //  Add the weights variable
            var weightDimensions = kernelSize
            weightDimensions.append(numInputChannels)      //  Number of input features
            weightDimensions.append(numFilters)
            let weightShape = TensorShape(weightDimensions)
            
            let weightTensor: MPSGraphTensor
            if (testWeights) {
                var weightConstants: [Float32] = []
                for _ in 0..<kernelSize[0] * kernelSize[1] {
                    for i in 0..<numFilters {
                        for _ in 0..<numInputChannels {
                            weightConstants.append(Float32(i+1))
                        }
                    }
                }
                let constantTensor = try TensorFloat32(shape: weightShape, initialValues: weightConstants)
                let data = constantTensor.getData()
                weightTensor = graph.mpsgraph.constant(data, shape: weightShape.getMPSShape(), dataType: .float32)
            }
            else {
                let numInputs = kernelSize[0] * kernelSize[1] * numInputChannels
                let numOutputs = 1  //  One output per convolution
                let weights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: numInputs, numOutput: numOutputs)
                let weightData = weights.getData()
                let weightName = graph.getFullName(name)! + "_weights"
                weightTensor = graph.mpsgraph.variable(with: weightData, shape: weightShape.getMPSShape(), dataType: weights.type.getMPSDataType(), name: weightName)
                
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
            }
            suffixes.append("_weights")
            addedTensors.append(weightTensor)

            //  Create the descriptor
            let descriptor = MPSGraphConvolution2DOpDescriptor(
                strideInX: strides[1],
                strideInY: strides[0],
                dilationRateInX: dilationRateW,
                dilationRateInY: dilationRateH,
                groups: 1,
                paddingLeft: leftPadding,
                paddingRight: rightPadding,
                paddingTop: topPadding,
                paddingBottom: bottomPadding,
                paddingStyle: paddingStyle,
                dataLayout: .NHWC,
                weightsLayout: .HWIO
            )!
            
            //  Add the convolution operation
            var haveFollowingTensors = (useBias || (numFilters == 1 || !filterDimensionLast) || !haveBatch || haveActivationFunction)
            let convolutionName: String
            if (haveFollowingTensors) {
                convolutionName = graph.getFullName(name)! + "_convolution"
                suffixes.append("_convolution")
            }
            else {
                convolutionName = graph.getFullName(name)!
                suffixes.append("")
            }
            var convolutionResult = graph.mpsgraph.convolution2D(shapedInputTensor, weights: weightTensor, descriptor: descriptor, name: convolutionName)
            addedTensors.append(convolutionResult)

            //  If a bias term, add it
            if (useBias) {
                if let biasTensor = biasTensor {
                    haveFollowingTensors = ((numFilters == 1 || !filterDimensionLast) || !haveBatch || haveActivationFunction)
                    let biasTensorName: String
                    if (haveFollowingTensors) {
                        biasTensorName = graph.getFullName(name)! + "_biasAddition"
                        suffixes.append("_biasAddition")
                    }
                    else {
                        biasTensorName = graph.getFullName(name)!
                        suffixes.append("")
                    }
                    let additionTensor = graph.mpsgraph.addition(convolutionResult, biasTensor, name: biasTensorName)
                    addedTensors.append(additionTensor)
                    convolutionResult = additionTensor
                }
                else {
                    throw GenericMPSGraphDSLErrors.InternalError
                }
            }
            
            //  Get the resulting output shape
            var outputShape: TensorShape
            if let shape = convolutionResult.shape {
                if (haveBatch) {
                    outputShape = TensorShape(fromMPS: [shape[0], shape[1], shape[2]])
                }
                else {
                    outputShape = TensorShape(fromMPS: [shape[1], shape[2]])
                }
            }
            else {
                throw GenericMPSGraphDSLErrors.UnknownShape
            }

            //  Reshape output
            if (numFilters == 1) {
                haveFollowingTensors = haveActivationFunction
                let outputReshapeName: String
                if (haveFollowingTensors) {
                    outputReshapeName = graph.getFullName(name)! + "_outputReshape"
                    suffixes.append("_outputReshape")
                }
                else {
                    outputReshapeName = graph.getFullName(name)!
                    suffixes.append("")
                }
                shapedOutputTensor = graph.mpsgraph.reshape(convolutionResult, shape: outputShape.getMPSShape(), name: outputReshapeName)
                addedTensors.append(shapedOutputTensor)
            }
            else {
                var outputDimensions = outputShape.dimensions
                outputDimensions.append(numFilters)
                let resultDimensions = convolutionResult.shape!.map(\.int64Value)
                if (outputDimensions[0] != resultDimensions[0] || outputDimensions[1] != resultDimensions[1] || outputDimensions[2] != resultDimensions[2] || outputDimensions[3] != resultDimensions[3]) {
                    haveFollowingTensors = (!filterDimensionLast || haveActivationFunction)
                    let outputReshapeName: String
                    if (haveFollowingTensors) {
                        outputReshapeName = graph.getFullName(name)! + "_outputReshape"
                        suffixes.append("_outputReshape")
                    }
                    else {
                        outputReshapeName = graph.getFullName(name)!
                        suffixes.append("")
                    }
                    shapedOutputTensor = graph.mpsgraph.reshape(convolutionResult, shape: outputDimensions.map {NSNumber(value: $0)}, name: outputReshapeName)
                    addedTensors.append(shapedOutputTensor)
                }
                else {
                    //  Didn't need to reshape
                    shapedOutputTensor = convolutionResult
                }

                //  If needed, move the filter dimension
                if (!filterDimensionLast) {
                    haveFollowingTensors = haveActivationFunction
                    let outputPermutationName: String
                    if (haveFollowingTensors) {
                        outputPermutationName = graph.getFullName(name)! + "_outputPermutation"
                        suffixes.append("_outputPermutation")
                    }
                    else {
                        outputPermutationName = graph.getFullName(name)!
                        suffixes.append("")
                    }
                    let permutation: MPSGraphTensor
                    if (haveBatch) {
                        permutation = graph.mpsgraph.transpose(shapedOutputTensor, permutation: [0, 3, 1, 2], name: outputPermutationName)
                    }
                    else {
                        permutation = graph.mpsgraph.transpose(shapedOutputTensor, permutation: [2, 0, 1], name: outputPermutationName)
                    }
                    addedTensors.append(permutation)
                    shapedOutputTensor = permutation
                }
            }
        }
        
        //  Process 3D convolutions
        else {
            //  Shape the input to 5D
            var shapedInputTensor = inputTensor
            let inputReshapeName = graph.getFullName(name)! + "_inputReshape"
            var numInputChannels = 1
            var haveBatch = false
            if (inputShape.numDimensions == 3) {
                //  DHW or HWC
                var fiveDShape = inputShape.dimensions

                //  If channel as depth, permutate the input tensor
                var reshapeInputTensor = inputTensor
                if (channelAsDepth) {
                    let inputPermutationName = graph.getFullName(name)! + "_inputPermutation"
                    reshapeInputTensor = graph.mpsgraph.transpose(inputTensor, permutation: [2, 0, 1], name: inputPermutationName)
                    suffixes.append("_inputPermutation")
                    addedTensors.append(shapedInputTensor)
                    fiveDShape = [fiveDShape[2], fiveDShape[0], fiveDShape[1]]
                }
                //  Now DHW
                
                //  Insert batch dimension
                fiveDShape.insert(1, at: 0)
                
                //  Append channel dimension
                fiveDShape.append(1)
                shapedInputTensor = graph.mpsgraph.reshape(reshapeInputTensor, shape: fiveDShape.map { NSNumber(value: $0)}, name: inputReshapeName)
                suffixes.append("_inputReshape")
                addedTensors.append(shapedInputTensor)
            }
            else if (inputShape.numDimensions == 4) {
                var fiveDShape = inputShape.dimensions
                if (graph.batchGraph || assumeExtraDimensionIsBatch) {
                    //  NDHW or NHWC
                    haveBatch = true
                    
                    if (channelAsDepth) {
                        //  NHWC - change to NDHW
                        let inputPermutationName = graph.getFullName(name)! + "_inputPermutation"
                        shapedInputTensor = graph.mpsgraph.transpose(inputTensor, permutation: [0, 3, 1, 2], name: inputPermutationName)
                        suffixes.append("_inputPermutation")
                        addedTensors.append(shapedInputTensor)
                        fiveDShape = [fiveDShape[0], fiveDShape[3], fiveDShape[1], fiveDShape[2]]
                    }

                    //  first dimension is batch (at beginning), so append a channel dimension
                    fiveDShape.append(1)
                }
                else {
                    //  DHWC
                    //  Last dimension is channel, so insert a batch dimension and set the number of channels as inputs
                    fiveDShape.insert(1, at: 0)
                    numInputChannels = inputShape.dimensions[3]     //  Number of channels is number of input features
                }
                shapedInputTensor = graph.mpsgraph.reshape(shapedInputTensor, shape: fiveDShape.map { NSNumber(value: $0)}, name: inputReshapeName)
                suffixes.append("_inputReshape")
                addedTensors.append(shapedInputTensor)
            }
            else {
                //  NDHWC - Assume first dimension is batch
                haveBatch = true
                //  Assume last dimension is channels, and set that as the number of input channels
                numInputChannels = inputShape.dimensions[4]     //  Number of channels is number of input features
            }
            
            //  Add the weights variable
            var weightDimensions = kernelSize
            weightDimensions.append(numInputChannels)      //  Number of input features
            weightDimensions.append(numFilters)
            let weightShape = TensorShape(weightDimensions)
            
            let weightTensor: MPSGraphTensor
            if (testWeights) {
                var weightConstants: [Float32] = []
                for _ in 0..<kernelSize[0] * kernelSize[1] * kernelSize[2] {
                    for i in 0..<numFilters {
                        for _ in 0..<numInputChannels {
                            weightConstants.append(Float32(i+1))
                        }
                    }
                }
                let constantTensor = try TensorFloat32(shape: weightShape, initialValues: weightConstants)
                let data = constantTensor.getData()
                weightTensor = graph.mpsgraph.constant(data, shape: weightShape.getMPSShape(), dataType: .float32)
            }
            else {
                let numInputs = kernelSize[0] * kernelSize[1] * kernelSize[2] * numInputChannels
                let numOutputs = 1  //  One output per convolution
                let weights = try CreateTensor.createWeightInitializationTensor(type: weightType, shape: weightShape, initializationInfo: weightInitialization, numInputs: numInputs, numOutput: numOutputs)
                let weightData = weights.getData()
                let weightName = graph.getFullName(name)! + "_weights"
                weightTensor = graph.mpsgraph.variable(with: weightData, shape: weightShape.getMPSShape(), dataType: weights.type.getMPSDataType(), name: weightName)
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
            }
            suffixes.append("_weights")
            addedTensors.append(weightTensor)

            let descriptor = MPSGraphConvolution3DOpDescriptor(
                strideInX: strides[1],
                strideInY: strides[2],
                strideInZ: strides[0],
                dilationRateInX: dilationRateW,
                dilationRateInY: dilationRateH,
                dilationRateInZ: dilationRateD,
                groups: 1,
                paddingLeft: leftPadding,
                paddingRight: rightPadding,
                paddingTop: topPadding,
                paddingBottom: bottomPadding,
                paddingFront: frontPadding,
                paddingBack: backPadding,
                paddingStyle: paddingStyle,
                dataLayout: .NDHWC,
                weightsLayout: .DHWIO
            )!
            
            //  Add the convolution operation
            var haveFollowingTensors = (useBias || (numFilters == 1 || !filterDimensionLast) || channelAsDepth || !haveBatch || haveActivationFunction)
            let convolutionName: String
            if (haveFollowingTensors) {
                convolutionName = graph.getFullName(name)! + "_convolution"
                suffixes.append("_convolution")
            }
            else {
                convolutionName = graph.getFullName(name)!
                suffixes.append("")
            }
            var convolutionResult = graph.mpsgraph.convolution3D(shapedInputTensor, weights: weightTensor, descriptor: descriptor, name: convolutionName)
            addedTensors.append(convolutionResult)

            //  If a bias term, add it
            if (useBias) {
                if let biasTensor = biasTensor {
                    haveFollowingTensors = ((numFilters == 1 || !filterDimensionLast) || channelAsDepth || !haveBatch || haveActivationFunction)
                    let biasTensorName: String
                    if (haveFollowingTensors) {
                        biasTensorName = graph.getFullName(name)! + "_biasAddition"
                        suffixes.append("_biasAddition")
                    }
                    else {
                        biasTensorName = graph.getFullName(name)!
                        suffixes.append("")
                    }
                    let additionTensor = graph.mpsgraph.addition(convolutionResult, biasTensor, name: biasTensorName)
                    addedTensors.append(additionTensor)
                    convolutionResult = additionTensor
                }
                else {
                    throw GenericMPSGraphDSLErrors.InternalError
                }
            }

            //  Get the resulting output shape
            var outputShape: [NSNumber]
            if let shape = convolutionResult.shape {
                outputShape = shape
            }
            else {
                throw GenericMPSGraphDSLErrors.UnknownShape
            }
            
            //  Reshape output - it comes in as NDHWO
            var needReshape = false
            var needTranspose = false
            var permutationIndices: [NSNumber] = [0, 1, 2, 3, 4]
            
            //  If we didn't have a batch dimension, remove it
            if (!haveBatch) {
                outputShape.removeFirst()
                permutationIndices.removeLast()
                needReshape = true
                //  Now DHWO
            }
            
            //  If we didn't have more than one filter, remove that dimension
            if (numFilters == 1) {
                outputShape.removeLast()
                permutationIndices.removeLast()
                needReshape = true
                //  Now NDHW or DHW
            }
            else {
                //  Either NDHWO or DHWO
                if (!filterDimensionLast) {
                    //  Change to NODHW or ODHW
                    if (haveBatch) {
                        //  NDHWO
                        let O = permutationIndices.last!
                        permutationIndices.removeLast()
                        permutationIndices.insert(O, at: 1)
                        //  Now NODHW
                    }
                    else {
                        //  DHWO
                        let O = permutationIndices.last!
                        permutationIndices.removeLast()
                        permutationIndices.insert(O, at: 0)
                        //  Now ODHW
                    }
                    needTranspose = true
                }
            }
            
            if (channelAsDepth) {
                //  NDHWO, DHWO, NDHW, DHW, ODHW or NODHW
                //  Last dimensions end in DHW or DHWO.  Move D after HW
                if (numFilters > 1 && filterDimensionLast) {
                    //  Last dimensions are DHWO - change to HWDO
                    let dIndex = permutationIndices.count - 4
                    let D = permutationIndices[dIndex]
                    permutationIndices.remove(at: dIndex)
                    permutationIndices.insert(D, at: dIndex+2)
                }
                else {
                    //  Last dimensions are DHW - change to HWD
                    let dIndex = permutationIndices.count - 3
                    let D = permutationIndices[dIndex]
                    permutationIndices.remove(at: dIndex)
                    permutationIndices.append(D)
                }
                needTranspose = true
            }

            shapedOutputTensor = convolutionResult
            if (needReshape) {
                haveFollowingTensors = (needTranspose || haveActivationFunction)
                let outputReshapeName: String
                if (haveFollowingTensors) {
                    outputReshapeName = graph.getFullName(name)! + "_outputReshape"
                    suffixes.append("_outputReshape")
                }
                else {
                    outputReshapeName = graph.getFullName(name)!
                    suffixes.append("")
                }
                shapedOutputTensor = graph.mpsgraph.reshape(shapedOutputTensor, shape: outputShape, name: outputReshapeName)
                addedTensors.append(shapedOutputTensor)
            }
            
            if (needTranspose) {
                let outputPermutationName: String
                if (haveActivationFunction) {
                    outputPermutationName = graph.getFullName(name)! + "_outputPermutation"
                    suffixes.append("_outputPermutation")
                }
                else {
                    outputPermutationName = graph.getFullName(name)!
                    suffixes.append("")
                }
                shapedOutputTensor = graph.mpsgraph.transpose(shapedOutputTensor, permutation: permutationIndices, name: outputPermutationName)
                addedTensors.append(shapedOutputTensor)
            }
       }
        
        //  Add any activation function
        if let activationTensor = try activationFunction.addActivation(graph: graph, inputTensor: shapedOutputTensor, name: name) {
            suffixes.append("")
            addedTensors.append(activationTensor)
        }

        targetIndices = [addedTensors.count-1]        //  Target only the last tensor
        return addedTensors
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }

    ///  Modifier to force third dimension of input tensor to be batch rather than channel.  Assumes channel unless graph built for batch processing.
    public func extraDimensionIsBatch() -> ConvolutionLayer {
        assumeExtraDimensionIsBatch = true
        return self
    }

    ///  Modifier to leave the dimension for the filters last.
    public func leaveFilterDimensionLast() -> ConvolutionLayer {
        filterDimensionLast = true
        return self
    }

    ///  Modifier to use the channel dimension as the depth dimension for 3D convolutions
    public func useChannelAsDepth() -> ConvolutionLayer {
        channelAsDepth = true
        return self
    }

    ///  Modifier to turn off addition of a bias term after the matrix multiplication
    public func noBiasTerm() -> ConvolutionLayer {
        useBias = false
        return self
    }
    
    ///  Modifier to set the initialization info for the random initialization of the weights
    public func weightInitialization(initializerInfo: WeightInitialization) -> ConvolutionLayer {
        weightInitialization = initializerInfo
        return self
    }

    ///  Modifier to set the initialization value for the initialization of the biases
    public func biasInitialValue(initialValue: Double) -> ConvolutionLayer {
        self.biasInitialValue = initialValue
        return self
    }

    ///  Modifier to set the padding for the height and width dimensions
    /// - Parameters:
    ///   - bottomPadding: The number of values to pad at the lower end of the height dimension
    ///   - topPadding: The number of values to pad at the upper end of the height dimension
    ///   - leftPadding:  The number of values to pad at the lower end of the width dimension
    ///   - rightPadding: The number of values to pad at the upper end of the width dimension
    ///   - backPadding:  (Optional)  The number of values to pad at the lower end of the depth dimension.  The default is 1
    ///   - frontPadding: (Optional)  The number of values to pad at the upper end of the depth dimension.  The default is 1
    ///   - paddingStyle: (Optional)  The type of values to pad the tensor with.  Defaults to TF_SAME
    /// - Returns: The PoolingLayer node
    public func padding(bottomPadding: Int, topPadding: Int, leftPadding: Int, rightPadding: Int, backPadding: Int = 1, frontPadding: Int = 1, paddingStyle: MPSGraphPaddingStyle = .TF_SAME) -> ConvolutionLayer {
        self.bottomPadding = bottomPadding
        self.topPadding = topPadding
        self.leftPadding = leftPadding
        self.rightPadding = rightPadding
        self.backPadding = backPadding
        self.frontPadding = frontPadding
        self.paddingStyle = paddingStyle
        return self
    }
    
    ///  Modifier to set the dilation rates - the number of indices between kernel values
    /// - Parameters:
    ///   - dilationRateH: The dilation rate for the H dimension
    ///   - dilationRateW: The dilation rate for the w dimension
    ///   - dilationRateD:  (Optional) The dilation rate for the D dimension.  Only used with 3D kernels.  Default is 1
    /// - Returns: The PoolingLayer node
    public func dilationRates(dilationRateH: Int, dilationRateW: Int, dilationRateD: Int = 1) -> ConvolutionLayer {
        self.dilationRateH = dilationRateH
        self.dilationRateW = dilationRateW
        self.dilationRateD = dilationRateD
        return self
    }

    /// Modifier to configure the layer's variables to learn
    /// - Parameter lossNode: the name of the loss calculation in the Graph
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String) -> ConvolutionLayer {
        self.lossNode = lossNode
        return self
    }
    
    internal func useTestWeights() -> ConvolutionLayer {
        testWeights = true
        return self
    }
}
