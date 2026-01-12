//
//  PoolingNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/22/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node to perform a 4-dimensional L2-pooling operation
public class L2NormPooling4D : UnaryNode {
    let descriptor: MPSGraphPooling4DOpDescriptor
    
    /// Constructor for a 4D L2-pooling operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the pooling operation, defining parameters for the operation
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, descriptor: MPSGraphPooling4DOpDescriptor, name: String? = nil) {
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let poolingResult = graph.mpsgraph.L2NormPooling4D(inputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [poolingResult]
    }
}

///   Node to perform a 2-dimensional average-pooling operation
public class AvgPooling2D : UnaryNode {
    let descriptor: MPSGraphPooling2DOpDescriptor
    
    /// Constructor for an 2D avg-pooling operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the pooling operation, defining parameters for the operation
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, descriptor: MPSGraphPooling2DOpDescriptor, name: String? = nil) {
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let poolingResult = graph.mpsgraph.avgPooling2D(withSourceTensor: inputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [poolingResult]
    }
}

///   Node to perform a 4-dimensional average-pooling operation
public class AvgPooling4D : UnaryNode {
    let descriptor: MPSGraphPooling4DOpDescriptor
    
    /// Constructor for an 4D avg-pooling operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the pooling operation, defining parameters for the operation
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, descriptor: MPSGraphPooling4DOpDescriptor, name: String? = nil) {
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let poolingResult = graph.mpsgraph.avgPooling4D(inputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [poolingResult]
    }
}

///   Node to perform a 2-dimensional max-pooling operation
public class MaxPooling2D : UnaryNode {
    let descriptor: MPSGraphPooling2DOpDescriptor
    
    /// Constructor for an 2D max-pooling operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the pooling operation, defining parameters for the operation
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, descriptor: MPSGraphPooling2DOpDescriptor, name: String? = nil) {
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let poolingResult = graph.mpsgraph.maxPooling2D(withSourceTensor: inputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [poolingResult]
    }
}

///   Node to perform a 2-dimensional max-pooling indices operation.  Returns max pool and indices used tensors (_pool and _indices suffixes)
public class MaxPooling2DReturnIndices : UnaryNode {
    let descriptor: MPSGraphPooling2DOpDescriptor
    let suffixes = ["_pool","_indices"]
    
    /// Constructor for an 2D max-pooling indices operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the pooling operation, defining parameters for the operation
    ///   - name: The name for this node and its associated tensors
    public init(input: String? = nil, descriptor: MPSGraphPooling2DOpDescriptor, name: String) {
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let poolingResults = graph.mpsgraph.maxPooling2DReturnIndices(inputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return poolingResults
    }

    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

///   Node to perform a 4-dimensional max-pooling operation
public class MaxPooling4D : UnaryNode {
    let descriptor: MPSGraphPooling4DOpDescriptor
    
    /// Constructor for an 4D max-pooling operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the pooling operation, defining parameters for the operation
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, descriptor: MPSGraphPooling4DOpDescriptor, name: String? = nil) {
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let poolingResult = graph.mpsgraph.maxPooling4D(inputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [poolingResult]
    }
}

///   Node to perform a 4-dimensional max-pooling indices operation.  Returns max pool and indices used tensors (_pool and _indices suffixes)
public class MaxPooling4DReturnIndices : UnaryNode {
    let descriptor: MPSGraphPooling4DOpDescriptor
    let suffixes = ["_pool","_indices"]
    
    /// Constructor for an 4D max-pooling indices operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input tensor.  If nil the previous node's output will be used
    ///   - descriptor: The descriptor for the pooling operation, defining parameters for the operation
    ///   - name: The name for this node and its associated tensors
    public init(input: String? = nil, descriptor: MPSGraphPooling4DOpDescriptor, name: String) {
        self.descriptor = descriptor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let poolingResults = graph.mpsgraph.maxPooling4DReturnIndices(inputTensor, descriptor: descriptor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return poolingResults
    }

    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

///   Enumeration for selecting the function of a Pooling layer
public enum PoolingFunction {
    ///  average
    case avg
    ///  maximum
    case max
    /// minimum
    case min
    /// average with zeroes for padding value
    case avgZeroPadding
}


///  Layer for a standard pooling layer.  All parameters will be inferred from input and kernal sizes if not set directly
public class PoolingLayer: UnaryNode {
    let kernelSize: [Int]
    let strides: [Int]
    let function: PoolingFunction
    
    var assumeExtraDimensionIsBatch: Bool = false

    var bottomPadding = 0
    var topPadding = 0
    var leftPadding = 0
    var rightPadding = 0
    var nLowPadding = 0
    var nHighPadding = 0
    var cLowPadding = 0
    var cHighPadding = 0
    var paddingStyle: MPSGraphPaddingStyle = .TF_SAME
    
    var dilationRateN: Int = 1
    var dilationRateH: Int = 1
    var dilationRateW: Int = 1
    var dilationRateC: Int = 1
    
    var ceilingMode: Bool = false

    /// Constructor for an Pooling layer with a 2D kernel.
    /// Kernel size is passed in.  Most of the settings will be derived from the input shape and kernel shape
    /// Default setup is used.  To change, use supplied modifiers
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used  Must be of rank 2 or higher
    ///   - function: The operation applied to the kernel mask on the tensor
    ///   - kernelHeight: The height of the pooling kernel
    ///   - kernelWidth: The width of the pooling kernel
    ///   - heightStride: (Optional) The step size for the pooling kernel in the height dimension.  If omitted all strides will be set to 1
    ///   - widthStride: (Optional) The step size for the pooling kernel in the width dimension.  If omitted all strides will be set to 1
    ///   - name: (Optional) The name for this node and its associated tensors
    public init(input: String? = nil, function: PoolingFunction, kernelHeight: Int, kernelWidth: Int, heightStride: Int? = nil, widthStride: Int? = nil, name: String? = nil) {
        self.kernelSize = [kernelHeight, kernelWidth]
        var tempStride: [Int] = []
        if let heightStride = heightStride {
            tempStride.append(heightStride)
        }
        else {
            tempStride.append(1 )
        }
        if let widthStride = widthStride {
            tempStride.append(widthStride)
        }
        else {
            tempStride.append(1 )
        }
        self.strides = tempStride
        self.function = function
        super.init(input: input, name: name)
    }

    /// Constructor for an Pooling layer with a 4D kernel.
    /// Kernel shape is passed in.  Most of the settings will be derived from the input shape and kernel shape
    /// Default setup is used.  To change, use supplied modifiers
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input.  If nil the previous node's output will be used  Must be of rank 2 or higher
    ///   - function: The operation applied to the kernel mask on the tensor
    ///   - kernelSize: The size of the pooling kernel, in all the dimensions that are to be pooled
    ///   - strides: (Optional) The step size for the pooling kernel, in all the dimensions (minimum of 2) that are to be pooled.  If omitted all strides will be set to 1
    ///   - name: (Optional) The name for this node and its associated tensors
    public init(input: String? = nil, function: PoolingFunction, kernelSize: [Int], strides: [Int]? = nil, name: String? = nil) {
        self.kernelSize = kernelSize
        if let strides = strides {
            self.strides = strides
        }
        else {
            self.strides = kernelSize.map { _ in 1 }
        }
        self.function = function
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {

        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        var inputShape: TensorShape
        if let shape = inputTensor.shape {
            inputShape = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        if (inputShape.numDimensions < 2) { throw MPSGraphNeuralNetErrors.InputTensorNot2DOrMore }
        if (inputShape.numDimensions > 4) { throw MPSGraphNeuralNetErrors.InputTensorMoreThan4D }
        if (kernelSize.count > inputShape.numDimensions) { throw MPSGraphNeuralNetErrors.KernelRankGreaterThanInput }
        if (kernelSize.count != strides.count) { throw MPSGraphNeuralNetErrors.StridesDifferentRankThanKernel }
        
        //  See if we are using 2D or 4D pooling
        let use2DPooling = (kernelSize.count == 2)

        //  Do 2D pooling
        let resultTensor: MPSGraphTensor
        if (use2DPooling) {
            
            //  Shape the input to 4D
            let shapedInputTensor: MPSGraphTensor
            if (inputShape.numDimensions == 2) {
                //  Assume HW
                //  Reshape the input to a 4D NHWC tensor
                var fourDShape = inputShape.dimensions
                fourDShape.insert(1, at: 0)     //  Add N
                fourDShape.append(1)            //  Add C
                let inputReshapeName = graph.getFullName(name)! + "_inputReshape"
                shapedInputTensor = graph.mpsgraph.reshape(inputTensor, shape: fourDShape.map { NSNumber(value: $0)}, name: inputReshapeName)
            }
            else if (inputShape.numDimensions == 3) {
                var fourDShape = inputShape.dimensions
                if (graph.batchGraph || assumeExtraDimensionIsBatch) {
                    //  Assume NWC
                    //  Reshape the input to a 4D NHWC tensor
                    fourDShape.append(1)        //  Add C
                }
                else {
                    //  Assume HWC
                    //  Reshape the input to a 4D NHWC tensor
                    fourDShape.insert(1, at: 0)        //  Add N
                }
                let inputReshapeName = graph.getFullName(name)! + "_inputReshape"
                shapedInputTensor = graph.mpsgraph.reshape(inputTensor, shape: fourDShape.map { NSNumber(value: $0)}, name: inputReshapeName)
            }
            else {
                //  Assume already 4D
                shapedInputTensor = inputTensor
            }
            
            //  Create the descriptor
            let descriptor = MPSGraphPooling2DOpDescriptor(    kernelWidth: kernelSize[1],
                                                               kernelHeight: kernelSize[0],
                                                               strideInX: strides[1],
                                                               strideInY: strides[0],
                                                               paddingStyle: paddingStyle,
                                                               dataLayout: .NHWC)!
            if (function == .avgZeroPadding) { descriptor.includeZeroPadToAverage = true }
            descriptor.ceilMode = ceilingMode
            
            //  Set the padding values
            descriptor.paddingBottom = bottomPadding
            descriptor.paddingLeft = leftPadding
            descriptor.paddingRight = rightPadding
            descriptor.paddingTop = topPadding
            
            //  Set the dilation rates
            descriptor.dilationRateInX = dilationRateW
            descriptor.dilationRateInY = dilationRateH

            //  Add to the graph itself
            let poolingName = graph.getFullName(name)! + "_pool"
            let poolingResult: MPSGraphTensor
            switch (function) {
            case .avg:
                poolingResult = graph.mpsgraph.avgPooling2D(withSourceTensor: shapedInputTensor, descriptor: descriptor, name: poolingName)
            case .avgZeroPadding:
                descriptor.includeZeroPadToAverage = true
                poolingResult = graph.mpsgraph.avgPooling2D(withSourceTensor: shapedInputTensor, descriptor: descriptor, name: poolingName)
            case .max:
                poolingResult = graph.mpsgraph.maxPooling2D(withSourceTensor: shapedInputTensor, descriptor: descriptor, name: poolingName)
            case .min:
                let inputNegate = graph.mpsgraph.negative(with: shapedInputTensor, name: graph.getFullName(name)! + "_InputNegate")
                let pooling = graph.mpsgraph.maxPooling2D(withSourceTensor: inputNegate, descriptor: descriptor, name: graph.getFullName(name)! + "_maxPooling")
                poolingResult = graph.mpsgraph.negative(with: pooling, name: poolingName)
            }
            
            var outputShape: TensorShape
            if let shape = poolingResult.shape {
                outputShape = TensorShape(fromMPS: shape)
            }
            else {
                throw GenericMPSGraphDSLErrors.UnknownShape
            }
            
            //  Reshape the output back to the input tensor rank
            if (inputShape.numDimensions == 2) {
                var twoDShape = outputShape.dimensions
                twoDShape.removeFirst()
                twoDShape.removeLast()
                let outputReshape = graph.mpsgraph.reshape(poolingResult, shape: twoDShape.map { NSNumber(value: $0)}, name: graph.getFullName(name))
                resultTensor = outputReshape
            }
            else if (inputShape.numDimensions == 3) {
                var threeDShape = outputShape.dimensions
                if (graph.batchGraph || assumeExtraDimensionIsBatch) {
                    //  Remove the C dimension
                    threeDShape.removeLast()
                }
                else {
                    //  Remove the N dimension
                    threeDShape.removeFirst()
                }
                let outputReshape = graph.mpsgraph.reshape(poolingResult, shape: threeDShape.map { NSNumber(value: $0)}, name: graph.getFullName(name))
                resultTensor = outputReshape
            }
            else {
                //  Leave as 4D
                resultTensor = poolingResult
            }
        }
        
        //  Do 4D pooling
        else {
            
            //  Create the descriptor
            let descriptor = MPSGraphPooling4DOpDescriptor(kernelSizes: kernelSize.map {NSNumber(value: $0)} , paddingStyle: paddingStyle)!
            descriptor.strides = strides.map {NSNumber(value: $0)}
            if (function == .avgZeroPadding) { descriptor.includeZeroPadToAverage = true }
            descriptor.ceilMode = ceilingMode

            //  Set the padding values (for .NHWC)
            var paddingValues: [NSNumber] = []
            paddingValues.append(NSNumber(value: nLowPadding))
            paddingValues.append(NSNumber(value: nHighPadding))
            paddingValues.append(NSNumber(value: bottomPadding))
            paddingValues.append(NSNumber(value: topPadding))
            paddingValues.append(NSNumber(value: leftPadding))
            paddingValues.append(NSNumber(value: rightPadding))
            paddingValues.append(NSNumber(value: cLowPadding))
            paddingValues.append(NSNumber(value: cHighPadding))
            descriptor.paddingValues = paddingValues
            
            //  Set the dilation rates (for .NHWC)
            var dilationRates: [NSNumber] = []
            dilationRates.append(NSNumber(value: dilationRateN))
            dilationRates.append(NSNumber(value: dilationRateH))
            dilationRates.append(NSNumber(value: dilationRateW))
            dilationRates.append(NSNumber(value: dilationRateC))
            descriptor.dilationRates = dilationRates

            //  Add to the graph itself
            let poolingName = graph.getFullName(name)! + "_pool"
            let poolingResult: MPSGraphTensor
            switch (function) {
            case .avg:
                poolingResult = graph.mpsgraph.avgPooling4D(inputTensor, descriptor: descriptor, name: poolingName)
            case .avgZeroPadding:
                descriptor.includeZeroPadToAverage = true
                poolingResult = graph.mpsgraph.avgPooling4D(inputTensor, descriptor: descriptor, name: poolingName)
            case .max:
                poolingResult = graph.mpsgraph.maxPooling4D(inputTensor, descriptor: descriptor, name: poolingName)
            case .min:
                let inputNegate = graph.mpsgraph.negative(with: inputTensor, name: graph.getFullName(name)! + "_InputNegate")
                let pooling = graph.mpsgraph.maxPooling4D(inputNegate, descriptor: descriptor, name: graph.getFullName(name)! + "_maxPooling")
                poolingResult = graph.mpsgraph.negative(with: pooling, name: poolingName)
            }
            resultTensor = poolingResult
        }

        //  Return the created MPSGraphTensor
        return [resultTensor]
    }
    
    ///  Modifier to force third dimension of input tensor to be batch rather than channel.  Assumes channel unless graph built for batch processing.
    public func extraDimensionIsBatch() -> PoolingLayer {
        assumeExtraDimensionIsBatch = true
        return self
    }

    ///  Modifier to set the padding for the height and width dimensions
    /// - Parameters:
    ///   - bottomPadding: The number of values to pad at the lower end of the height dimension
    ///   - topPadding: The number of values to pad at the upper end of the height dimension
    ///   - leftPadding:  The number of values to pad at the lower end of the width dimension
    ///   - rightPadding: The number of values to pad at the upper end of the width dimension
    ///   - paddingStyle: (Optional)  The type of values to pad the tensor with.  Defaults to TF_SAME
    /// - Returns: The PoolingLayer node
    public func HWPadding(bottomPadding: Int, topPadding: Int, leftPadding: Int, rightPadding: Int, paddingStyle: MPSGraphPaddingStyle = .TF_SAME) -> PoolingLayer {
        self.bottomPadding = bottomPadding
        self.topPadding = topPadding
        self.leftPadding = leftPadding
        self.rightPadding = rightPadding
        self.paddingStyle = paddingStyle
        return self
    }
    
    ///  Modifier to set the padding for the batch and channel dimensions.  Only used with 4D kernels
    /// - Parameters:
    ///   - nLowPadding: The number of values to pad at the lower end of the batch dimension
    ///   - nHighPadding: The number of values to pad at the upper end of the batch dimension
    ///   - cLowPadding:  The number of values to pad at the lower end of the channel dimension
    ///   - cHighPadding: The number of values to pad at the upper end of the channel dimension
    /// - Returns: The PoolingLayer node
    public func NCPadding(nLowPadding: Int, nHighPadding: Int, cLowPadding: Int, cHighPadding: Int) -> PoolingLayer {
        self.nLowPadding = nLowPadding
        self.nHighPadding = nHighPadding
        self.cLowPadding = cLowPadding
        self.cHighPadding = cHighPadding
        return self
    }
    
    ///  Modifier to set the dilation rates - the number of indices between kernel values
    /// - Parameters:
    ///   - dilationRateH: The dilation rate for the H dimension
    ///   - dilationRateW: The dilation rate for the w dimension
    ///   - dilationRateN:  (Optional) The dilation rate for the N dimension.  Only used with 4D kernels.  Default is 1
    ///   - dilationRateC: (Optional) The dilation rate for the C dimension.  Only used with 4D kernels.  Default is 1
    /// - Returns: The PoolingLayer node
    public func dilationRates(dilationRateH: Int, dilationRateW: Int, dilationRateN: Int = 1, dilationRateC: Int = 1) -> PoolingLayer {
        self.dilationRateH = dilationRateH
        self.dilationRateW = dilationRateW
        self.dilationRateN = dilationRateN
        self.dilationRateC = dilationRateC
        return self
    }
    
    ///  Modifier to turn on ceiling mode.  Output size calculated using ceiling instead of round on tensor dimension size divided by kernel (with dilation rate) size
    /// - Returns: The PoolingLayer node
    public func setCeilingMode() -> PoolingLayer {
        self.ceilingMode = true
        return self
    }
}
