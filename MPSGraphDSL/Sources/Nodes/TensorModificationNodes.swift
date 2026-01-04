//
//  TensorReshapeNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/21/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///   Node to reshape a tensor
public class Reshape : UnaryNode {
    let shape : TensorShape?
    let newShapeTensor: String?

    ///  Constructor for a reshape operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - shape: The new shape for the output tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, shape: TensorShape, name: String? = nil) {
        self.shape = shape
        self.newShapeTensor = nil
        super.init(input: input, name: name)
    }
    ///  Constructor for a reshape operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - newShapeTensor: (Optional) The tensor with the the new shape for the input tensor.  If nil, the previous node is used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, newShapeTensor: String?, name: String? = nil) {
        self.shape = nil
        self.newShapeTensor = newShapeTensor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  If we have a fixed shape, get the tensor shape and compare
        if let shape = shape {
            if let incomingShape = inputTensor.shape {
                let inputShape = TensorShape(fromMPS: incomingShape)
                if (inputShape.totalSize != shape.totalSize) {
                    throw MPSGraphDSLErrors.InputShapeError("Number of elements in input tensor shape and new shape must equal")
                }
            }
        }

        //  Add to the graph itself
        let reshapeResult: MPSGraphTensor
        if let shape = shape {
            reshapeResult = graph.mpsgraph.reshape(inputTensor, shape: shape.getMPSShape(), name: graph.getFullName(name))
        }
        else {
            if let addedNode = graph.findNamedNode(newShapeTensor!) {
                reshapeResult = graph.mpsgraph.reshape(inputTensor, shapeTensor: addedNode.mpstensor, name: graph.getFullName(name))
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(newShapeTensor!)
            }
        }
        
        //  Remember the output tensor and shape for later
        return [reshapeResult]
    }
}

///   Node to concatenate two or more tensors along a given axis
public class Concatenate : Node {
    let inputs: [String?]
    let dimension: Int
    let interleave: Bool
    
    ///  Constructor for a concatenation operation
    ///
    /// - Parameters:
    ///   - tensors: The name of the tensors that will be concatenated.  If the previous node's tensor is to be used, you can pass a nil for the name
    ///   - dimension: The index of the dimension along which the tensors will be concatenated (0 based index into the shape)
    ///   - interleave: (Optional) If true, the tensors will be interleaved, rather than just placed end-to-end
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ tensors: [String?], dimension: Int, interleave: Bool = false, name: String? = nil) {
        self.inputs = tensors
        self.dimension = dimension
        self.interleave = interleave
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        var inputTensors : [MPSGraphTensor] = []
        for tensor in inputs {
            let inputTensor = try graph.getUnaryTensor(name: tensor)
            //  Verify the shape has the dimension for concatination
            if let shape = inputTensor.shape {
                let inputShape = TensorShape(fromMPS: shape)
                if (inputShape.dimensions.count < dimension) { throw MPSGraphDSLErrors.DoesNotContainDimension }
            }
            else {
                throw GenericMPSGraphDSLErrors.UnknownShape
            }
            inputTensors.append(inputTensor)
        }
        
        //  Add to the graph itself
        let outputTensor : MPSGraphTensor
        if (inputTensors.count == 2 && !interleave) {
            outputTensor = graph.mpsgraph.concatTensor(inputTensors[0], with: inputTensors[1], dimension: dimension, name: graph.getFullName(name))
        }
        else if (interleave) {
            outputTensor = graph.mpsgraph.concatTensors(inputTensors, dimension: dimension, interleave: interleave, name: graph.getFullName(name))
        }
        else {
            outputTensor = graph.mpsgraph.concatTensors(inputTensors, dimension: dimension, name: graph.getFullName(name))
        }
        
        return [outputTensor]
    }
}

///   Node to stack two or more tensors along a given axis
public class Stack : Node {
    let inputs: [String?]
    let axis: Int
    
    ///  Constructor for a stack operation
    ///
    /// - Parameters:
    ///   - tensors: The name of the tensors that will be concatenated.  If the previous node's tensor is to be used, you can pass a nil for the name
    ///   - axis: The axis along which the tensors will be concatenated (0 based index into the shape)
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ tensors: [String?], axis: Int, name: String? = nil) {
        self.inputs = tensors
        self.axis = axis
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        var inputTensors : [MPSGraphTensor] = []
        for tensor in inputs {
            let inputTensor = try graph.getUnaryTensor(name: tensor)
            //  Verify the shape has the dimension for concatination
            if let shape = inputTensor.shape {
                let inputShape = TensorShape(fromMPS: shape)
                if (inputShape.dimensions.count < axis) { throw MPSGraphDSLErrors.DoesNotContainDimension }
            }
            else {
                throw GenericMPSGraphDSLErrors.UnknownShape
            }
            inputTensors.append(inputTensor)
        }
        
        //  Add to the graph itself
         let outputTensor = graph.mpsgraph.stack(inputTensors, axis: axis, name: graph.getFullName(name))
        
        return [outputTensor]
    }
}


///   Node to split a tensor into two or more tensors along a given axis with specified sizing
///
///   The resulting tensors nodes are given the name of the node with a suffix of "_1", "_2", etc.
public class Split : UnaryNode {
    let axis: Int
    let numberOfSplits: Int?
    let splitSizes: [Int]?
    let splitSizesTensor: String?
    var suffixes: [String] = []
    
    ///  Constructor for a split operation given a number of equal-sized partitions
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the dimension along which the tensors will be split (0 based index into the shape)
    ///   - numberOfSplits: The number of partitions made of the tensor.  These will be equal sized, if possible
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axis: Int, numberOfSplits: Int, name: String? = nil) {
        self.axis = axis
        self.numberOfSplits = numberOfSplits
        splitSizes = nil
        splitSizesTensor = nil
        super.init(input: input, name: name)
    }
    ///  Constructor for a split operation given the size of each partition
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the dimension along which the tensors will be split (0 based index into the shape)
    ///   - numberOfSplits: The size of each partition.  The total of these sizes should match the dimension size along the axis
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axis: Int, splitSizes: [Int], name: String? = nil) {
        self.axis = axis
        self.numberOfSplits = nil
        self.splitSizes = splitSizes
        splitSizesTensor = nil
        super.init(input: input, name: name)
    }
    ///  Constructor for a split operation given a tensor that contains the size of each partition
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the dimension along which the tensors will be split (0 based index into the shape)
    ///   - splitSizesTensor: (Optional) The tensor with the size of each partition.  The total of these sizes should match the dimension size along the axis.  If nil, the previous node is used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axis: Int, splitSizesTensor: String?, name: String? = nil) {
        self.axis = axis
        self.numberOfSplits = nil
        self.splitSizes = nil
        self.splitSizesTensor = splitSizesTensor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Get the input size along the specified axis
        var axisSize: Int = 0
        if let shape = inputTensor.shape {
            let inputShape = TensorShape(fromMPS: shape)
            if axis < 0 || axis >= inputShape.dimensions.count {
                throw GenericMPSGraphDSLErrors.InvalidDimension
            }
            axisSize = inputShape.dimensions[axis]
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }

        //  Get the number of partitions
        var numPartitions = 1
        if let numberOfSplits = numberOfSplits {
            numPartitions = numberOfSplits
            
            //  Make sure the number of partitions is less than the input dimension on the axis
            if (numPartitions > axisSize) {
                throw MPSGraphDSLErrors.MorePartitionsThanDimensions
            }
        }
        if let splitSizes = splitSizes {
            numPartitions = splitSizes.count
            let totalPartitionSize = splitSizes.reduce(0, +)
            
            //  Verify the total partition size matches the input size along the specified axis
            if (totalPartitionSize != axisSize) { throw MPSGraphDSLErrors.MorePartitionsThanDimensions }
        }
        if let splitSizesTensor = splitSizesTensor {
            if let addedNode = graph.findNamedNode(splitSizesTensor) {
                if let shape = addedNode.mpstensor.shape {
                    let splitSizesShape = TensorShape(fromMPS: shape)
                    if (splitSizesShape.dimensions.count != 1) { throw MPSGraphDSLErrors.TensorNot1Dimensional(splitSizesTensor) }
                    numPartitions = splitSizesShape.dimensions[0]
                }
                else {
                    throw GenericMPSGraphDSLErrors.UnknownShape
                }
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(splitSizesTensor)
            }
        }
        
        //  Set up the suffixes
        for i in 0..<numPartitions {
            let suffix = "_\(i+1)"
            suffixes.append(suffix)
        }
        
        //  Add the operation to the graph
        var partitionTensors: [MPSGraphTensor]
        if let numberOfSplits = numberOfSplits {
            partitionTensors = graph.mpsgraph.split(inputTensor, numSplits: numberOfSplits, axis: axis, name: graph.getFullName(name))
        }
        else if let splitSizes = splitSizes {
            partitionTensors = graph.mpsgraph.split(inputTensor, splitSizes: splitSizes.map { NSNumber(value: $0)}, axis: axis, name: graph.getFullName(name))
        }
        else {
            let addedNode = graph.findNamedNode(splitSizesTensor!)!
            partitionTensors = graph.mpsgraph.split(inputTensor, splitSizesTensor: addedNode.mpstensor, axis: axis, name: graph.getFullName(name))
        }
        
        return partitionTensors
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

///   Node to reverse a tensor along the specified axes
public class Reverse: UnaryNode {
    let axes: [Int]?
    let axesTensor: String?
    ///  Constructor for an reverse  operation across all axis
    ///
    /// - Parameters: 
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axes: an array of indices for the axes that the operation should be carried out for
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        self.axes = nil
        self.axesTensor = nil
        super.init(input: input, name: name)
    }
    ///  Constructor for an reverse  operation across specified axes
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: An array of indices of the dimensions along which the tensors will be reversed
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axes: [Int], name: String? = nil) {
        self.axes = axes
        self.axesTensor = nil
        super.init(input: input, name: name)
    }
    ///  Constructor for an reverse  operation across specified axes
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axisTensor: (Optional) The name of a tensor that provides an array of indices of the dimensions along which the tensors will be reversed.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axesTensor: String? = nil, name: String? = nil) {
        self.axes = nil
        self.axesTensor = axesTensor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  If we have specified axis, make sure they fit with the input tensor shape
        if let axes = axes {
            if let shape = inputTensor.shape {
                for axis in axes {
                    if (axis <= 0 || axis > shape.count) { throw MPSGraphDSLErrors.DoesNotContainDimension }
                }
            }
            else {
                throw GenericMPSGraphDSLErrors.UnknownShape
            }
        }
        
        //  Add to the graph itself
        let result: MPSGraphTensor
        
        if (axes == nil && axesTensor == nil) {
            result = graph.mpsgraph.reverse(inputTensor, name: graph.getFullName(name))
        }
        else if let axes = axes {
            result = graph.mpsgraph.reverse(inputTensor, axes: axes.map { NSNumber(value: $0)}, name: graph.getFullName(name))
        }
        else {
            if let addedNode = graph.findNamedNode(axesTensor!) {
                result = graph.mpsgraph.reverse(inputTensor, axesTensor: addedNode.mpstensor, name: graph.getFullName(name))
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(axesTensor!)
            }
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to cast a tensor to a different numerical type
public class Cast : UnaryNode {
    let newType: DataType
    let reinterpret: Bool
    
    ///  Constructor for a cast operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - newType:  The numerical type for the new tensor
    ///   - reinterpret:  (Optional)  If true the output is reinterpreted to element type passed in with the last dimension scaled by sizeof(tensor_type) / sizeof(type).  Defaults to false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, newType: DataType, reinterpret: Bool = false, name: String? = nil) {
        self.newType = newType
        self.reinterpret = reinterpret
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result: MPSGraphTensor
        if reinterpret {
            result = graph.mpsgraph.reinterpretCast(inputTensor, to: newType.getMPSDataType(), name: graph.getFullName(name))
        }
        else {
            result = graph.mpsgraph.cast(inputTensor, to: newType.getMPSDataType(), name: graph.getFullName(name))
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to transpose tensor
public class Transpose : UnaryNode {
    let dimension1: Int
    let dimension2: Int
    let permutation: [Int]
    
    ///  Constructor for a transpose operation on a 2-dimensional matrix
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        dimension1 = -1
        dimension2 = -1
        permutation = []
        super.init(input: input, name: name)
    }
    
    ///  Constructor for a transpose operation for a single dimension swap
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - swapDimension:  The zero-based index of the dimension to be swapped
    ///   - withDimension:  The zero-based index of the dimension to swap with the previous indexed dimension
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, swapDimension: Int, withDimension: Int, name: String? = nil) {
        dimension1 = swapDimension
        dimension2 = withDimension
        permutation = []
        super.init(input: input, name: name)
    }
    
    ///  Constructor for a transpose operation with a permutation array
    ///
    ///  A permutation array is an array of the tensor indices, rearranged in the manner the output tensor should be.  For example, a Tensor of shape \[3, 6, 4\] could have a permutation of \[1, 2, 0]
    ///    Note the permutation has the numbers 0,1, and 2 in it.  These are the Tensor dimension axis indices and all axis are required, they just are re-arranged.  This permutation will put axis 1 into the axis 0 position of the new Tensor (the 1 in the permutation 0 slot), axis 2 into the 1 position (the 2 in the 1 slot), and axis 0 becomes axis 2.
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, permutation: [Int], name: String? = nil) {
        dimension1 = -1
        dimension2 = -1
        self.permutation = permutation
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Make sure it is a matrix
        if let shape = inputTensor.shape {
            if (shape.count == 1) { throw MPSGraphDSLErrors.TransposeOnVectorNotSupported }
            if (shape.count > 2 && (dimension1 == -1 || dimension2 == -1) && permutation.count < 1) { throw  MPSGraphDSLErrors.TransposeNeedsDimensionSpecification }
            
            //  If there is a permutation array, check it
            if (permutation.count > 0) {
                if (permutation.count != shape.count) { throw MPSGraphDSLErrors.PermutationArrayMustHaveAllDimensions }
                for i in 0..<permutation.count {
                    if (!permutation.contains(i)) { throw MPSGraphDSLErrors.PermutationArrayMustHaveAllDimensions }
                }
            }
        }

        //  Add to the graph itself
        if (permutation.count > 1) {
            let result = graph.mpsgraph.transpose(inputTensor, permutation: permutation.map { NSNumber(value: $0) }, name: graph.getFullName(name))
            return [result]
        }
        else {
            let dim1 : Int = dimension1 == -1 ? 0 : dimension1
            let dim2 : Int = dimension2 == -1 ? 1 : dimension2
            let result = graph.mpsgraph.transposeTensor(inputTensor, dimension: dim1, withDimension: dim2, name: graph.getFullName(name))
            return [result]
        }
    }
}


///   Node to slice tensor
@available(macOS 15.2, *)
public class Slice : UnaryNode {
    enum SliceType {
        case singleDimension
        case multipleDimensions
        case multipleDimensionsWithSqueeze
        case multipleDimensionsFromTensorsWithSqueeze
        case multipleDimensionsFromTensors
    }
    
    let sliceType: SliceType
    let dimension: Int
    let start: Int
    let length: Int
    let starts: [Int]
    let ends: [Int]
    let strides: [Int]
    let startMask: UInt32
    let endMask: UInt32
    let squeezeMask: UInt32
    let startTensor: String?
    let endTensor: String?
    let strideTensor: String?
    let sizeTensor: String?

    
    ///  Constructor for a slice operation for a specified dimension, starting at a given location for the dimension, and going forwards a given length
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - dimension:  The zero-based index of the dimension to be sliced
    ///   - start:  The index within the dimension that the generated slice will start from
    ///   - length:  The size along the specified dimension that the resulting tensor will have
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, dimension: Int, start: Int, length: Int, name: String? = nil) {
        sliceType = .singleDimension
        self.dimension = dimension
        self.start = start
        self.length = length
        self.starts = []
        self.ends = []
        self.strides = []
        self.startMask = 0
        self.endMask = 0
        self.squeezeMask = 0
        self.startTensor = nil
        self.endTensor = nil
        self.strideTensor = nil
        self.sizeTensor = nil

        super.init(input: input, name: name)
    }
    
    ///  Constructor for a slice operation for a specified dimension, starting at a given location for the dimension, and going forwards a given length
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - starts:  An array of starting indices for each dimension of the tensor for the slice
    ///   - ends:  An array of ending indices for each dimension of the tensor for the slice
    ///   - strides:  An array of strides for each dimension of the tensor for the slice
    ///   - ignoreStartsOnDimension:  (Optional) An array of dimension indices that will have their start index ignored
    ///   - ignoreEndsOnDimensions:  (Optional) An array of dimension indices that will have their end index ignored
    ///   - removeDimensions:  (Optional) An array of dimension that will be removed from the slice
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, starts: [Int], ends: [Int], strides: [Int], ignoreStartsOnDimension: [Int] = [], ignoreEndsOnDimensions: [Int] = [], removeDimensions: [Int] = [], name: String? = nil) {
        self.dimension = 0
        self.start = 0
        self.length = 0
        self.starts = starts
        self.ends = ends
        self.strides = strides
        var bitMask : UInt32 = 0
        var dimensionError = false
        for dim in ignoreStartsOnDimension {
            if (dim < 0 || dim > 32) { dimensionError = true }
            else {
                bitMask |= 1 << UInt32(dim)
            }
        }
        self.startMask = bitMask
        bitMask = 0
        for dim in ignoreEndsOnDimensions {
            if (dim < 0 || dim > 32) { dimensionError = true }
            else {
                bitMask |= 1 << UInt32(dim)
            }
        }
        self.endMask = bitMask
        bitMask = 0
        for dim in removeDimensions {
            if (dim < 0 || dim > 32) { dimensionError = true }
            else {
                bitMask |= 1 << UInt32(dim)
            }
        }
        self.squeezeMask = bitMask
        self.startTensor = nil
        self.endTensor = nil
        self.strideTensor = nil
        self.sizeTensor = nil

        if (startMask == 0 && endMask == 0 && squeezeMask == 0) {
            sliceType = .multipleDimensions

        }
        else {
            sliceType = .multipleDimensionsWithSqueeze

        }
        
        super.init(input: input, name: name)
        
        if (dimensionError) {
            buildError = GenericMPSGraphDSLErrors.InvalidDimension
        }
    }
    
    ///  Constructor for a slice operation for a specified dimension, starting at a given location for the dimension, and going forwards a given length- using graph nodes as inputs
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - startTensor: (Optional)) The name of the node that will provide the array of start locations for each dimension.  If nil the previous node's output will be used
    ///   - endTensor:  (Optional)) The name of the node that will provide the array of end locations for each dimension.  If nil the previous node's output will be used
    ///   - strideNode:  (Optional)) The name of the node that will provide the array of strides for each dimension.  If nil the previous node's output will be used
    ///   - ignoreStartsOnDimension:  (Optional) An array of dimension indices that will have their start index ignored
    ///   - ignoreEndsOnDimensions:  (Optional) An array of dimension indices that will have their end index ignored
    ///   - removeDimensions:  (Optional) An array of dimension that will be removed from the slice
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, startTensor: String?, endTensor: String?, strideTensor: String?, ignoreStartsOnDimension: [Int] = [], ignoreEndsOnDimensions: [Int] = [], removeDimensions: [Int] = [], name: String? = nil) {
        sliceType = .multipleDimensionsFromTensorsWithSqueeze
        self.dimension = 0
        self.start = 0
        self.length = 0
        self.starts = []
        self.ends = []
        self.strides = []
        self.startTensor = startTensor
        self.endTensor = endTensor
        self.strideTensor = strideTensor
        var bitMask : UInt32 = 0
        var dimensionError = false
        for dim in ignoreStartsOnDimension {
            if (dim < 0 || dim > 32) { dimensionError = true }
            else {
                bitMask |= 1 << UInt32(dim)
            }
        }
        self.startMask = bitMask
        bitMask = 0
        for dim in ignoreEndsOnDimensions {
            if (dim < 0 || dim > 32) { dimensionError = true }
            else {
                bitMask |= 1 << UInt32(dim)
            }
        }
        self.endMask = bitMask
        bitMask = 0
        for dim in removeDimensions {
            if (dim < 0 || dim > 32) { dimensionError = true }
            else {
                bitMask |= 1 << UInt32(dim)
            }
        }
        self.squeezeMask = bitMask
        self.sizeTensor = nil

        super.init(input: input, name: name)
        
        if (dimensionError) {
            buildError = GenericMPSGraphDSLErrors.InvalidDimension
        }
    }
    
    ///  Constructor for a slice operation for a specified dimension, starting at a given location for the dimension, and going forwards a given length- using graph nodes as inputs
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - startTensor: (Optional)) The name of the node that will provide the array of start locations for each dimension.  If nil the previous node's output will be used
    ///   - sizeTensor:  (Optional)) The name of the node that will provide the array of size of the slice for each dimension.  If nil the previous node's output will be used
    ///   - ignoreStartsOnDimension:  (Optional) An array of dimension indices that will have their start index ignored
    ///   - ignoreEndsOnDimensions:  (Optional) An array of dimension indices that will have their end index ignored
    ///   - removeDimensions:  (Optional) An array of dimension that will be removed from the slice
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, startTensor: String?, sizeTensor: String?, ignoreStartsOnDimension: [Int] = [], ignoreEndsOnDimensions: [Int] = [], removeDimensions: [Int] = [], name: String? = nil) {
        sliceType = .multipleDimensionsFromTensors
        self.dimension = 0
        self.start = 0
        self.length = 0
        self.starts = []
        self.ends = []
        self.strides = []
        self.startTensor = startTensor
        self.sizeTensor = sizeTensor
        self.startMask = 0
        self.endMask = 0
        var bitMask : UInt32 = 0
        var dimensionError = false
        bitMask = 0
        for dim in removeDimensions {
            if (dim < 0 || dim > 32) { dimensionError = true }
            else {
                bitMask |= 1 << UInt32(dim)
            }
        }
        self.squeezeMask = bitMask
        self.endTensor = nil
        self.strideTensor = nil

        super.init(input: input, name: name)
        
        if (dimensionError) {
            buildError = GenericMPSGraphDSLErrors.InvalidDimension
        }
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Make sure the specified dimension works with the parameters
        if let shape = inputTensor.shape {
            switch (sliceType) {
            case .singleDimension:
                if dimension < 0 || dimension >= shape.count { throw MPSGraphDSLErrors.DoesNotContainDimension }
                let size = Int(truncating: shape[dimension])
                if (strides.count == 0) {
                    if (start > 0) {        //  forwards counting
                        if (length < 1 || (length + start) > size) { throw MPSGraphDSLErrors.SliceExceedsTensorDimensions }
                    }
                    else {
                        if (length > size) { throw MPSGraphDSLErrors.SliceExceedsTensorDimensions }
                    }
                }
            case .multipleDimensions,
                 .multipleDimensionsWithSqueeze:
                if starts.count != shape.count { throw MPSGraphDSLErrors.EntryRequiredForEachDimensions }
                if ends.count != shape.count { throw MPSGraphDSLErrors.EntryRequiredForEachDimensions }
                if strides.count != shape.count { throw MPSGraphDSLErrors.EntryRequiredForEachDimensions }
            case .multipleDimensionsFromTensorsWithSqueeze:
                break
            case .multipleDimensionsFromTensors:
                break
            }
       }

        //  Add to the graph itself
        let result: MPSGraphTensor
        var startMPSTensor: MPSGraphTensor? = nil
        var endMPSTensor: MPSGraphTensor? = nil
        var strideMPSTensor: MPSGraphTensor? = nil
        var sizeMPSTensor: MPSGraphTensor? = nil
        switch (sliceType) {
        case .singleDimension:
            result = graph.mpsgraph.sliceTensor(inputTensor, dimension: dimension, start:  start, length: length, name: graph.getFullName(name))
        case .multipleDimensions:
            result = graph.mpsgraph.sliceTensor(inputTensor, starts: starts.map { NSNumber(value: $0) }, ends: ends.map { NSNumber(value: $0) }, strides: strides.map { NSNumber(value: $0) }, name: graph.getFullName(name))
        case .multipleDimensionsWithSqueeze:
            result = graph.mpsgraph.sliceTensor(inputTensor, starts: starts.map { NSNumber(value: $0) }, ends: ends.map { NSNumber(value: $0) }, strides: strides.map { NSNumber(value: $0) }, startMask: startMask, endMask: endMask, squeezeMask: squeezeMask, name: graph.getFullName(name))
        case .multipleDimensionsFromTensorsWithSqueeze:
            startMPSTensor = try graph.getOptionalTensor(startTensor)
            endMPSTensor = try graph.getOptionalTensor(endTensor)
            strideMPSTensor = try graph.getOptionalTensor(strideTensor)
            result = graph.mpsgraph.sliceTensor(inputTensor, start: startMPSTensor!, end: endMPSTensor!, strideTensor: strideMPSTensor!, startMask: startMask, endMask: endMask, squeezeMask: squeezeMask, name: graph.getFullName(name))
        case .multipleDimensionsFromTensors:
            startMPSTensor = try graph.getOptionalTensor(startTensor)
            sizeMPSTensor = try graph.getOptionalTensor(sizeTensor)
            result = graph.mpsgraph.sliceTensor(inputTensor, start: startMPSTensor!, sizeTensor: sizeMPSTensor!, squeezeMask: squeezeMask, name: graph.getFullName(name))
        }
        
        return [result]
    }
}

///   Node to sort a tensor along a given dimension
public class Sort : UnaryNode {
    let axis: Int
    let axisTensor: String?
    let axisFromTensor: Bool
    let descending: Bool
    
    /// Constructor for a sort operation on a tensor with a specified axis
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The axis index along which the tensor will be sorted
    ///   - descending: (Optional) If true the tensor is sorted in descending order.  Default is false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axis: Int, descending: Bool = false, name: String? = nil) {
        self.axis = axis
        self.axisTensor = nil
        axisFromTensor = false
        self.descending = descending
        super.init(input: input, name: name)
    }
    
    /// Constructor for a sort operation on a tensor with the axis specified by another tensor
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axisTensor: (Optional) The name of the tensor that will provide the sort axis.  If nil the previous node's output will be used
    ///   - descending: (Optional) If true the tensor is sorted in descending order.  Default is false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axisTensor: String? = nil, descending: Bool = false, name: String? = nil) {
        self.axis = -100
        self.axisTensor = axisTensor
        axisFromTensor = true
        self.descending = descending
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Make sure the specified dimension works with the parameters
        if let shape = inputTensor.shape {
            if (axis < 0 || axis > shape.count) { throw MPSGraphDSLErrors.DoesNotContainDimension }
        }
        
        //  Add the node to the graph
        let result: MPSGraphTensor
        if (axisFromTensor) {
            let axisMPSTensor = try graph.getOptionalTensor(axisTensor)
            if (descending) {
                result = graph.mpsgraph.sort(inputTensor, axisTensor: axisMPSTensor, descending: descending, name: graph.getFullName(name))
            }
            else {
                result = graph.mpsgraph.sort(inputTensor, axisTensor: axisMPSTensor, name: graph.getFullName(name))
            }
        }
        else {
            if (descending) {
                result = graph.mpsgraph.sort(inputTensor, axis: axis, descending: descending, name: graph.getFullName(name))
            }
            else {
                result = graph.mpsgraph.sort(inputTensor, axis: axis, name: graph.getFullName(name))
            }
        }
        
        return [result]
    }
}

public enum ReductionOperation {
    case and
    case argmax
    case argmin
    case max
    case min
    case or
    case product
    case sum
}


///   Node to reduce a tensor along a given dimension using a specified operation
public class Reduction : UnaryNode {
    let op: ReductionOperation
    let axis: Int
    let axes: [Int]
    let multipleAxes: Bool
    let propogateNaNs: Bool

    /// Constructor for a reduction operation on a tensor along the specified axis
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - op: The reduction operation to be done
    ///   - axis: The axis index along which the tensor cumulation will be done
    ///   - propogateNaNs: (Optional) If true NaN values will be propogated, otherwise ignored.  Defaults to false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, op: ReductionOperation, axis: Int, propogateNaNs: Bool = false, name: String? = nil) {
        self.op = op
        self.axis = axis
        self.axes = []
        multipleAxes = false
        self.propogateNaNs = propogateNaNs
        super.init(input: input, name: name)
        
        //  Check NaN propogation correctness
        if (propogateNaNs) {
            if (op != .min && op != .max) {
                buildError = MPSGraphDSLErrors.NaNPropogationNotSupported
            }
        }
    }

    /// Constructor for a reduction operation on a tensor along a set of specified axes
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - op: The reduction operation to be done
    ///   - axes: An array of axis indices along which the reduction operation will be performed
    ///   - propogateNaNs: (Optional) If true NaN values will be propogated, otherwise ignored.  Defaults to false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, op: ReductionOperation, axes: [Int], propogateNaNs: Bool = false, name: String? = nil) {
        self.op = op
        self.axis = 0
        self.axes = axes
        multipleAxes = true
        self.propogateNaNs = propogateNaNs
        super.init(input: input, name: name)
        
        //  Check multi-axis operation supported
        if (op == .argmax || op == .argmin) {
            buildError = MPSGraphDSLErrors.MultipleDimensionsNotSupported
        }

        
        //  Check NaN propogation correctness
        if (propogateNaNs) {
            if (op != .min && op != .max) {
                buildError = MPSGraphDSLErrors.NaNPropogationNotSupported
            }
        }
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        let result: MPSGraphTensor
        switch (op) {
            case .and:
                if multipleAxes {
                    result = graph.mpsgraph.reductionAnd(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))

                }
                else {
                    result = graph.mpsgraph.reductionAnd(with: inputTensor, axis: axis, name: graph.getFullName(name))
                }
            case .argmax:
                result = graph.mpsgraph.reductionArgMaximum(with: inputTensor, axis: axis, name: graph.getFullName(name))
            case .argmin:
                result = graph.mpsgraph.reductionArgMinimum(with: inputTensor, axis: axis, name: graph.getFullName(name))
            case .max:
                if multipleAxes {
                    if (propogateNaNs) {
                        result = graph.mpsgraph.reductionMaximumPropagateNaN(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.reductionMaximum(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))
                    }
                }
                else {
                    if (propogateNaNs) {
                        result = graph.mpsgraph.reductionMaximumPropagateNaN(with: inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.reductionMaximum(with: inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                }
            case .min:
                if multipleAxes {
                    if (propogateNaNs) {
                        result = graph.mpsgraph.reductionMinimumPropagateNaN(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.reductionMinimum(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))
                    }
                }
                else {
                    if (propogateNaNs) {
                        result = graph.mpsgraph.reductionMinimumPropagateNaN(with: inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.reductionMinimum(with: inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                }
            case .or:
                if multipleAxes {
                    result = graph.mpsgraph.reductionOr(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))

                }
                else {
                    result = graph.mpsgraph.reductionOr(with: inputTensor, axis: axis, name: graph.getFullName(name))
                }
            case .product:
                if multipleAxes {
                    result = graph.mpsgraph.reductionProduct(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))

                }
                else {
                    result = graph.mpsgraph.reductionProduct(with: inputTensor, axis: axis, name: graph.getFullName(name))
                }
            case .sum:
                if multipleAxes {
                    result = graph.mpsgraph.reductionSum(with: inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))

                }
                else {
                    result = graph.mpsgraph.reductionSum(with: inputTensor, axis: axis, name: graph.getFullName(name))
                }
       }
        
        return [result]
    }
        
}


public enum CumulationOperation {
    case max
    case min
    case product
    case sum
}


///   Node to perform a cumulation along a specifed axis of a tensor
public class Cumulate : UnaryNode {
    let op: CumulationOperation
    let axis: Int
    let axisTensor: String?
    let useTensorForAxis: Bool
    let exclusive: Bool
    let reverse: Bool
    
    /// Constructor for a cumaltion operation on a tensor along the specified axis
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - op: The cumulation operation to be done
    ///   - axis: The axis index along which the tensor cumulation will be done
    ///   - exclusive: (Optional) If true performs the exclusive cumulative operation.  Defaults to false.  First element will be minimum value for max operation, maximum value for min opertion, 0 for sum, and 1 for product
    ///   - reverse: (Optional)  If true the cumulation operation runs in reverse.  Defaults to false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, op: CumulationOperation, axis: Int, exclusive: Bool = false, reverse: Bool = false, name: String? = nil) {
        self.op = op
        self.axis = axis
        self.axisTensor = nil
        useTensorForAxis = false
        self.exclusive = exclusive
        self.reverse = reverse
        super.init(input: input, name: name)
    }
    
    /// Constructor for a cumaltion operation on a tensor along the specified axis
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - op: The cumulation operation to be done
    ///   - axisTensor: (Optional) The node name that will provide the tensor for the axis index along which the tensor cumulation will be done.  If nil the previous node's output will be used
    ///   - exclusive: (Optional) If true performs the exclusive cumulative operation.  Defaults to false.  First element will be minimum value for max operation, maximum value for min opertion, 0 for sum, and 1 for product
    ///   - reverse: (Optional)  If true the cumulation operation runs in reverse.  Defaults to false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, op: CumulationOperation, axisTensor: String?, exclusive: Bool = false, reverse: Bool = false, name: String? = nil) {
        self.op = op
        self.axis = 0
        self.axisTensor = axisTensor
        useTensorForAxis = true
        self.exclusive = exclusive
        self.reverse = reverse
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        var axisMPSTensor: MPSGraphTensor? = nil
        if (useTensorForAxis) {
            axisMPSTensor = try graph.getOptionalTensor(axisTensor)
        }

        //  Add to the graph itself
        let result: MPSGraphTensor
        switch op {
            case .max:
                if (useTensorForAxis) {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeMaximum(inputTensor, axisTensor: axisMPSTensor!, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeMaximum(inputTensor, axisTensor: axisMPSTensor!, name: graph.getFullName(name))
                    }
                }
                else {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeMaximum(inputTensor, axis: axis, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeMaximum(inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                }
            case .min:
                if (useTensorForAxis) {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeMinimum(inputTensor, axisTensor: axisMPSTensor!, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeMinimum(inputTensor, axisTensor: axisMPSTensor!, name: graph.getFullName(name))
                    }
                }
                else {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeMinimum(inputTensor, axis: axis, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeMinimum(inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                }
            case .product:
                if (useTensorForAxis) {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeProduct(inputTensor, axisTensor: axisMPSTensor!, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeProduct(inputTensor, axisTensor: axisMPSTensor!, name: graph.getFullName(name))
                    }
                }
                else {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeProduct(inputTensor, axis: axis, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeProduct(inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                }
            case .sum:
                if (useTensorForAxis) {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeSum(inputTensor, axisTensor: axisMPSTensor!, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeSum(inputTensor, axisTensor: axisMPSTensor!, name: graph.getFullName(name))
                    }
                }
                else {
                    if (exclusive || reverse) {
                        result = graph.mpsgraph.cumulativeSum(inputTensor, axis: axis, exclusive: exclusive, reverse: reverse, name: graph.getFullName(name))
                    }
                    else {
                        result = graph.mpsgraph.cumulativeSum(inputTensor, axis: axis, name: graph.getFullName(name))
                    }
                }
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to tile a tensor
public class TileTensor : UnaryNode {
    let multipliers : [Int]

    ///  Constructor for a tiling operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - multipliers: The number of times the tensor is tiled for each dimension
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, multipliers: [Int], name: String? = nil) {
        self.multipliers = multipliers
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  If make sure the number of multipliers match the number of dimensions
        if let incomingShape = inputTensor.shape {
            if (incomingShape.count != multipliers.count) {
                throw MPSGraphDSLErrors.EntryRequiredForEachDimensions
            }
        }

        //  Add to the graph itself
        let tileResult = graph.mpsgraph.tileTensor(inputTensor, withMultiplier: multipliers.map { NSNumber(value: $0)}, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [tileResult]
    }
}

///  Mode for a padding operation on a tensor
public enum PaddingMode {
    /// Anti Periodic x\[-2\] -> -x\[L-3\]
    case antiPeriodic
    ///  ClampToEdge (PyTorch ReplicationPad)
    case clampToEdge
    ///   Constant
    case constant(Double)
    ///  Periodic x\[-2\] -> x\[L-3\], where L is size of x.
    case periodic
    ///  Reflect
    case reflect
    ///  Symmetric
    case symmetric
    ///  All zeroes
    case zero
    
    func toMPSPaddingMode() -> MPSGraphPaddingMode {
        switch self {
        case .antiPeriodic: return .antiPeriodic
        case .clampToEdge: return .clampToEdge
        case .constant: return .constant
        case .periodic: return .periodic
        case .reflect: return .reflect
        case .symmetric: return .symmetric
        case .zero: return .zero
        }
    }
}


///   Node to pad a Tensor
public class Padding : UnaryNode {
    let padMode: PaddingMode
    let leftPadding: [Int]
    let rightPadding: [Int]
    ///  Constructor for a padding operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - padMode:  The type of padding that will be performed on the tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, padMode: PaddingMode, leftPadding: [Int], rightPadding: [Int], name: String? = nil) {
        self.padMode = padMode
        self.leftPadding = leftPadding
        self.rightPadding = rightPadding
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Verify the left and right padding values are there for all dimensions
        if let incomingShape = inputTensor.shape {
            if (incomingShape.count != leftPadding.count) || (incomingShape.count != rightPadding.count) {
                throw MPSGraphDSLErrors.EntryRequiredForEachDimensions
            }
        }

        //  Add to the graph itself
        let result: MPSGraphTensor
        switch (padMode) {
            case .constant(let value):
            result = graph.mpsgraph.padTensor(inputTensor, with: .constant, leftPadding: leftPadding.map { NSNumber(value: $0)}, rightPadding: rightPadding.map { NSNumber(value: $0)}, constantValue: value, name: graph.getFullName(name))
            default:
            result = graph.mpsgraph.padTensor(inputTensor, with: padMode.toMPSPaddingMode(), leftPadding: leftPadding.map { NSNumber(value: $0)}, rightPadding: rightPadding.map { NSNumber(value: $0)}, constantValue: 0.0, name: graph.getFullName(name))
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to squeeze a tensor (remove dimensions of size 1)
public class Squeeze : UnaryNode {
    enum SqueezeType {
        case allDimensions
        case oneDimension
        case multipleDimensions
        case multipleDimensionsFromTensor
    }

    let squeezeType: SqueezeType
    let axis: Int
    let axes: [Int]
    let axesTensor: String?

    ///  Constructor for a squeezing operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, name: String? = nil) {
        squeezeType = .allDimensions
        self.axis = 0
        self.axes = []
        self.axesTensor = nil
        super.init(input: input, name: name)
    }

    ///  Constructor for a squeezing operation of a given axis
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The axis to squeeze out.  Must have size 1
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, axis: Int, name: String? = nil) {
        squeezeType = .oneDimension
        self.axis = axis
        self.axes = []
        self.axesTensor = nil
        super.init(input: input, name: name)
    }

    ///  Constructor for a squeezing operation of a multiple axes
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axes: an array of axes to squeeze out.  All must have size 1
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, axes: [Int], name: String? = nil) {
        squeezeType = .multipleDimensions
        self.axis = 0
        self.axes = axes
        self.axesTensor = nil
        super.init(input: input, name: name)
    }

    ///  Constructor for a squeezing operation of a multiple axes specified by a tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axesTensor: Name of a node that provides the array of axes to squeeze out. .  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, axesTensor: String?, name: String? = nil) {
        squeezeType = .multipleDimensionsFromTensor
        self.axis = 0
        self.axes = []
        self.axesTensor = axesTensor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let squeezeResult: MPSGraphTensor
        switch (squeezeType) {
        case .allDimensions:
            squeezeResult = graph.mpsgraph.squeeze(inputTensor, name: graph.getFullName(name))
        case .oneDimension:
            squeezeResult = graph.mpsgraph.squeeze(inputTensor, axis: axis, name: graph.getFullName(name))
        case .multipleDimensions:
            squeezeResult = graph.mpsgraph.squeeze(inputTensor, axes: axes.map { NSNumber(value: $0)}, name: graph.getFullName(name))
        case .multipleDimensionsFromTensor:
            let squeezeMPSTensor = try graph.getOptionalTensor(axesTensor)
            squeezeResult = graph.mpsgraph.squeeze(inputTensor, axesTensor: squeezeMPSTensor, name: graph.getFullName(name))
        }
        
        //  Remember the output tensor and shape for later
        return [squeezeResult]
    }
}

///   Node to perform a Bottom-K operation on a tensor
public class BottomK : UnaryNode {
    let axis : Int
    let k : Int
    let axisTensorName : String?
    let kTensorName : String?
    let useTensors: Bool
    
    var suffixes: [String] = []

    ///  Constructor for a bottom-K operation with fixed axis and size
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the axis that the operation will be performed on
    ///   - k:  The number of lowest values that will be kept
    ///   - name: (Optional) The name for this node .  The associated tensors will have "_values" and "_indices" appended
    public init(_ input: String? = nil, axis: Int, k: Int, name: String? = nil) {
        self.axis = axis
        self.k = k
        self.axisTensorName = nil
        self.kTensorName = nil
        useTensors = false
        super.init(input: input, name: name)
    }

    ///  Constructor for a bottom-K operation from tensors
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axisTensor: (Optional) The name of a node tensor that provides the index of the axis that the operation will be performed on.  If nil the previous node's output will be used
    ///   - kTensor:  (Optional) The name of a node tensor that provides the number of lowest values that will be kept.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node .  The associated tensors will have "_values" and "_indices" appended
    public init(_ input: String? = nil, axisTensor: String? = nil, kTensor: String? = nil, name: String? = nil) {
        self.axis = 0
        self.k = 0
        self.axisTensorName = axisTensor
        self.kTensorName = kTensor
        useTensors = true
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        let results: [MPSGraphTensor?]
        if (useTensors) {
            //  Get the axis tensor
            let axisMPSTensor = try graph.getOptionalTensor(axisTensorName)

            //  Get the k tensor
            let kMPSTensor = try graph.getOptionalTensor(kTensorName)
            
            //  Add to the graph itself
            results = graph.mpsgraph.bottomK(inputTensor, axisTensor: axisMPSTensor, kTensor: kMPSTensor, name: graph.getFullName(name))
        }
        else {
            //  If make sure the axis specified fits
            if let incomingShape = inputTensor.shape {
                if (axis < 0 || axis >= incomingShape.count) {
                    throw GenericMPSGraphDSLErrors.InvalidDimension
                }
            }
            
            //  Add to the graph itself
            results = graph.mpsgraph.bottomK(inputTensor, axis: axis, k: k, name: graph.getFullName(name))
        }
        
        //  Set up the suffixes
        suffixes = ["_values", "_indices"]
        
        //  Remember the output tensosr and shape for later
        return results
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

///   Node to perform a Top-K operation on a tensor
public class TopK : UnaryNode {
    let axis : Int
    let k : Int
    let axisTensorName : String?
    let kTensorName : String?
    let useTensors: Bool
    
    var suffixes: [String] = []

    ///  Constructor for a top-K operation with fixed axis and size
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the axis that the operation will be performed on
    ///   - k:  The number of lowest values that will be kept
    ///   - name: (Optional) The name for this node .  The associated tensors will have "_values" and "_indices" appended
    public init(_ input: String? = nil, axis: Int, k: Int, name: String? = nil) {
        self.axis = axis
        self.k = k
        self.axisTensorName = nil
        self.kTensorName = nil
        useTensors = false
        super.init(input: input, name: name)
    }

    ///  Constructor for a top-K operation from tensors
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axisTensor: (Optional) The name of a node tensor that provides the index of the axis that the operation will be performed on.  If nil the previous node's output will be used
    ///   - kTensor:  (Optional) The name of a node tensor that provides the number of lowest values that will be kept.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node .  The associated tensors will have "_values" and "_indices" appended
    public init(_ input: String? = nil, axisTensor: String? = nil, kTensor: String? = nil, name: String? = nil) {
        self.axis = 0
        self.k = 0
        self.axisTensorName = axisTensor
        self.kTensorName = kTensor
        useTensors = true
        super.init(input: input, name: name)
    }

    ///  Constructor for a top-K operation with fixed size on all axes
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the axis that the operation will be performed on
    ///   - name: (Optional) The name for this node .  The associated tensors will have "_values" and "_indices" appended
    public init(_ input: String? = nil, k: Int, name: String? = nil) {
        self.axis = -1
        self.k = k
        self.axisTensorName = nil
        self.kTensorName = nil
        useTensors = false
        super.init(input: input, name: name)
    }

    ///  Constructor for a top-K operation from tensors
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axisTensor: (Optional) The name of a node tensor that provides the index of the axis that the operation will be performed on.  If nil the previous node's output will be used
    ///   - kTensor:  (Optional) The name of a node tensor that provides the number of lowest values that will be kept.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node .  The associated tensors will have "_values" and "_indices" appended
    public init(_ input: String? = nil, kTensor: String? = nil, name: String? = nil) {
        self.axis = -1
        self.k = 0
        self.axisTensorName = nil
        self.kTensorName = kTensor
        useTensors = true
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        let results: [MPSGraphTensor?]
        if (useTensors) {
            //  Get the axis tensor
            let axisMPSTensor = try graph.getOptionalTensor(axisTensorName)

            //  Get the k tensor
            let kMPSTensor = try graph.getOptionalTensor(kTensorName)
            
            //  Add to the graph itself
            if (k == -1) {
                results = graph.mpsgraph.topK(inputTensor, kTensor: kMPSTensor, name: graph.getFullName(name))
            }
            else {
                results = graph.mpsgraph.topK(inputTensor, axisTensor: axisMPSTensor, kTensor: kMPSTensor, name: graph.getFullName(name))
            }
        }
        else {
            //  If make sure the axis specified fits
            if let incomingShape = inputTensor.shape {
                if (axis < 0 || axis >= incomingShape.count) {
                    throw GenericMPSGraphDSLErrors.InvalidDimension
                }
            }
            
            //  Add to the graph itself
            if (k == -1) {
                results = graph.mpsgraph.topK(inputTensor, axis: axis, k: k, name: graph.getFullName(name))
            }
            else {
                results = graph.mpsgraph.topK(inputTensor, k: k, name: graph.getFullName(name))
            }
        }
        
        //  Set up the suffixes
        suffixes = ["_values", "_indices"]
        
        //  Remember the output tensosr and shape for later
        return results
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

///   Node to perform an expand dimension operation on a tensor
public class ExpandDimension : UnaryNode {
    enum ExpandType {
        case singleDimension
        case multipleDimensions
        case multipleDimensionsFromTensor
    }
    let axis : Int
    let axes : [Int]
    let axesTensorName : String?
    let expandType: ExpandType
    
    ///  Constructor for an expand dimension operation with fixed single axis
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the axis that the operation will be performed on
    ///   - name: (Optional) The name for this node .
    public init(_ input: String? = nil, axis: Int, name: String? = nil) {
        self.axis = axis
        self.axes = []
        self.axesTensorName = nil
        expandType = .singleDimension
        super.init(input: input, name: name)
    }
    ///  Constructor for an expand dimension operation with fixed multiple axes
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The index of the axis that the operation will be performed on
    ///   - name: (Optional) The name for this node .
    public init(_ input: String? = nil, axes: [Int], name: String? = nil) {
        self.axis = 0
        self.axes = axes
        self.axesTensorName = nil
        expandType = .multipleDimensions
        super.init(input: input, name: name)
    }

    ///  Constructor for a expand dimension operation with the axes from a tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axesTensor: (Optional) The name of a node tensor that provides the indices of the axes that the operation will be performed on.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node .  The associated tensors will have "_values" and "_indices" appended
    public init(_ input: String? = nil, axesTensor: String? = nil, kTensor: String? = nil, name: String? = nil) {
        self.axis = 0
        self.axes = []
        self.axesTensorName = axesTensor
        expandType = .multipleDimensionsFromTensor
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        let result: MPSGraphTensor?
        switch expandType {
        case .singleDimension:
            //  If make sure the axis specified fits
            if let incomingShape = inputTensor.shape {
                if (axis < 0 || axis >= incomingShape.count) {
                    throw GenericMPSGraphDSLErrors.InvalidDimension
                }
            }

            result = graph.mpsgraph.expandDims(inputTensor, axis: axis, name: graph.getFullName(name))
        case .multipleDimensions:
            //  If make sure the axes specified fits
            if let incomingShape = inputTensor.shape {
                for axis in axes {
                    if (axis < 0 || axis >= incomingShape.count) {
                        throw GenericMPSGraphDSLErrors.InvalidDimension
                    }
                }
            }

            result = graph.mpsgraph.expandDims(inputTensor, axes: axes.map { NSNumber(value: $0) }, name: graph.getFullName(name))
        case .multipleDimensionsFromTensor:
            //  Get the axes tensor
            let axesMPSTensor = try graph.getOptionalTensor(axesTensorName)

            result = graph.mpsgraph.expandDims(inputTensor, axesTensor: axesMPSTensor, name: graph.getFullName(name))
        }
        
        return [result]
    }
}

///   Node to perform a bandPart on a Tensor
public class BandPart : UnaryNode {
    let numLower: Int
    let numUpper: Int
    let numLowerTensor: String?
    let numUpperTensor: String?
    let useTensors: Bool
    
    ///  Constructor for a bandPart operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - numLower:  The number of diagonals in the lower triangle to keep. If -1, the framework returns all sub diagnols
    ///   - numUpper:  The number of diagonals in the upper triangle to keep. If -1, the framework returns all super diagnols
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, numLower: Int, numUpper: Int, name: String? = nil) {
        self.numLower = numLower
        self.numUpper = numUpper
        self.numLowerTensor = nil
        self.numUpperTensor = nil
        useTensors = false
        super.init(input: input, name: name)
    }
    ///  Constructor for a bandPart operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - numLowerTensor:  (Optional) The name of a tensor that provides the number of diagonals in the lower triangle to keep.  If nil the previous node's output will be used
    ///   - numUpperTensor:  (Optional) The name of a tensor that provides the number of diagonals in the upper triangle to keep.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, numLowerTensor: String? = nil, numUpperTensor: String? = nil, name: String? = nil) {
        self.numLower = 0
        self.numUpper = 0
        self.numLowerTensor = numLowerTensor
        self.numUpperTensor = numUpperTensor
        useTensors = true
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result: MPSGraphTensor
        if (useTensors) {
            let lowerMPSTensor = try graph.getOptionalTensor(numLowerTensor)
            let upperMPSTensor = try graph.getOptionalTensor(numUpperTensor)

            result = graph.mpsgraph.bandPart(inputTensor, numLowerTensor: lowerMPSTensor, numUpperTensor: upperMPSTensor, name: graph.getFullName(name))
       }
        else {
            result = graph.mpsgraph.bandPart(inputTensor, numLower: numLower, numUpper: numUpper, name: graph.getFullName(name))
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to find the indices for sorting a Tensor along a specified axis
public class ArgSort : UnaryNode {
    let axis: Int
    let descending: Bool
    let axisTensor: String?
    let useTensors: Bool
    
    ///  Constructor for a argSort operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The axis along which the tensors sort indices will be found
    ///   - descending: (Optional) If true the sort is performed in descending order.  Defaults to false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, axis: Int, descending: Bool = false, name: String? = nil) {
        self.axis = axis
        self.descending = descending
        self.axisTensor = nil
        useTensors = false
        super.init(input: input, name: name)
    }
    ///  Constructor for a argSort operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axisTensor: (Optional) The name of a tensor that provides the axis along which the tensors sort indices will be found.  If nil the previous node's output will be used
    ///   - descending: (Optional) If true the sort is performed in descending order.  Defaults to false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, axisTensor: String? = nil, descending: Bool = false, name: String? = nil) {
        self.axis = 0
        self.descending = descending
        self.axisTensor = axisTensor
        useTensors = true
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result: MPSGraphTensor
        if (useTensors) {
            let axisMPSTensor = try graph.getOptionalTensor(axisTensor)

            result = graph.mpsgraph.argSort(inputTensor, axisTensor: axisMPSTensor, descending: descending, name: graph.getFullName(name))
       }
        else {
            result = graph.mpsgraph.argSort(inputTensor, axis: axis, descending: descending, name: graph.getFullName(name))
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to perform a broadcast on a Tensor
public class BroadCast : UnaryNode {
    let shape: [Int]
    let shapeTensor: String?
    let useShapeTensor: Bool

    /// Create a broadcast operation for the given shape
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - shape: The shape of the broadcast tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, shape: [Int], name: String? = nil) {
        self.shape = shape
        self.shapeTensor = nil
        useShapeTensor = false
        super.init(name: name)
    }

    /// Create a broadcast operation for the given shape
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - shape: The shape of the broadcast tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, shape: TensorShape, name: String? = nil) {
        self.shape = shape.dimensions
        self.shapeTensor = nil
        useShapeTensor = false
        super.init(name: name)
    }

    /// Create a broadcast operation for the given shape
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - shapeTensor: (Optional) The name of the tensor that will provide the shape of the broadcast tensor.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, shapeTensor: String? = nil, name: String? = nil) {
        self.shape = []
        self.shapeTensor = shapeTensor
        useShapeTensor = true
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result: MPSGraphTensor
        if (useShapeTensor) {
            let shapeMPSTensor = try graph.getOptionalTensor(shapeTensor)

            result = graph.mpsgraph.broadcast(inputTensor, shapeTensor: shapeMPSTensor, name: graph.getFullName(name))
       }
        else {
            result = graph.mpsgraph.broadcast(inputTensor, shape: shape.map { NSNumber(value: $0)}, name: graph.getFullName(name))
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to perform a flatten to 2D operation on a Tensor
public class Flatten2D : UnaryNode {
    let axis: Int
    let axisTensor: String?
    let useAxisTensor: Bool

    /// Create a flatten to 2D operation along the given axis
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: The axis for the operation.  Flattens dimensions before axis to dimension 0 (rows)  and dimensions starting from axis to dimension 1 (columns)
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, axis: Int, name: String? = nil) {
        self.axis = axis
        self.axisTensor = nil
        useAxisTensor = false
        super.init(name: name)
    }

    /// Create a flatten to 2D operation along the axis given by a tensor
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axisTensor: (Optional)  The name of a tensor that will provide the axis for the operation.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, axisTensor: String? = nil, name: String? = nil) {
        self.axis = 0
        self.axisTensor = axisTensor
        useAxisTensor = true
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result: MPSGraphTensor
        if (useAxisTensor) {
            let axisMPSTensor = try graph.getOptionalTensor(axisTensor)

            result = graph.mpsgraph.flatten2D(inputTensor, axisTensor: axisMPSTensor, name: graph.getFullName(name))
       }
        else {
            result = graph.mpsgraph.flatten2D(inputTensor, axis: axis, name: graph.getFullName(name))
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}


///   Node to set one-hot values on a tensor with a dimension of indices
public class OneHot : UnaryNode {
    let depth: Int
    let axis: Int?
    let dataType: DataType
    let haveDataType: Bool
    let onValue: Double
    let offValue: Double
    let haveValues: Bool
    
    /// Add a one-hot operation to a tensor with a given axis and depth.  Tensor type is same as input type
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - depth: The depth of the one-hot (number of values that create the one-hot representation
    ///   - axis: (Optional) The axis along which the indices are found - which is expanded to the depth parameter.  If nil the new axis will be the new minor dimension.  Default is nil
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, depth: Int, axis: Int? = nil, name: String? = nil) {
        self.depth = depth
        self.axis = axis
        self.dataType = .float32
        haveDataType = false
        self.onValue = 1.0
        self.offValue = 0.0
        haveValues = false
        super.init(name: name)
    }
    
    /// Add a one-hot operation to a tensor with a given axis and depth.  Tensor is changed to the type specified
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - depth: The depth of the one-hot (number of values that create the one-hot representation
    ///   - axis: (Optional) The axis along which the indices are found - which is expanded to the depth parameter.  If nil the new axis will be the new minor dimension.  Default is nil
    ///   - dataType: The element type of the resulting tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, depth: Int, axis: Int? = nil, dataType: DataType, name: String? = nil) {
        self.depth = depth
        self.axis = axis
        self.dataType = dataType
        haveDataType = false
        self.onValue = 1.0
        self.offValue = 0.0
        haveValues = false
        super.init(name: name)
    }
    
    /// Add a one-hot operation to a tensor with a given axis and depth.  Tensor is changed to the type specified
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - depth: The depth of the one-hot (number of values that create the one-hot representation
    ///   - axis: (Optional) The axis along which the indices are found - which is expanded to the depth parameter.  If nil the new axis will be the new minor dimension.  Default is nil
    ///   - dataType: The element type of the resulting tensor
    ///   - onValue: The element value for a 'on' state in the one-hot
    ///   - offValue: The element value for a 'of' state in the one-hot
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(_ input: String? = nil, depth: Int, axis: Int? = nil, dataType: DataType, onValue: Double, offValue: Double, name: String? = nil) {
        self.depth = depth
        self.axis = axis
        self.dataType = dataType
        haveDataType = false
        self.onValue = onValue
        self.offValue = offValue
        haveValues = true
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result: MPSGraphTensor
        if (haveDataType) {
            if (haveValues) {
                if let axis = axis {
                    result = graph.mpsgraph.oneHot(withIndicesTensor: inputTensor, depth: depth, axis: axis, dataType: dataType.getMPSDataType(), onValue: onValue, offValue: offValue, name: graph.getFullName(name))
                }
                else {
                    result = graph.mpsgraph.oneHot(withIndicesTensor: inputTensor, depth: depth, dataType: dataType.getMPSDataType(), onValue: onValue, offValue: offValue, name: graph.getFullName(name))
                }
            }
            else {
                if let axis = axis {
                    result = graph.mpsgraph.oneHot(withIndicesTensor: inputTensor, depth: depth, axis: axis, dataType: dataType.getMPSDataType(), name: graph.getFullName(name))
                }
                else {
                    result = graph.mpsgraph.oneHot(withIndicesTensor: inputTensor, depth: depth, dataType: dataType.getMPSDataType(), name: graph.getFullName(name))
                }
            }
        }
        else {
            if let axis = axis {
                result = graph.mpsgraph.oneHot(withIndicesTensor: inputTensor, depth: depth, axis: axis, name: graph.getFullName(name))
            }
            else {
                result = graph.mpsgraph.oneHot(withIndicesTensor: inputTensor, depth: depth, name: graph.getFullName(name))
            }
        }
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}
