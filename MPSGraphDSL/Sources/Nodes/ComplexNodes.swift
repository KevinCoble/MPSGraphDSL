//
//  ComplexNodes.swift
//  MPSGraphDSL
//
//  Nodes for complex data types
//
//  Created by Kevin Coble on 12/27/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///  Leaf Node that provides a constant complex value tensor
public class ComplexConstant : Node {
    let real: Double
    let imaginary: Double
    let shape: [Int]
    let float16: Bool
    
    /// Construct a ComplexConstant node with the specified real and imaginary values
    /// - Parameters:
    ///   - real: The real part of the complex constant
    ///   - imaginary: The imaginary part of the complex constant
    ///   - shape: (Optional) the shape of the constant Tensor.  If ommitted, becomes \[1\]
    ///   - float16: (Optional) If true the constant is of type complexFloat16, else it is of type complexFloat32.  Default is false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(real: Double, imaginary: Double, shape: [Int] = [1], float16: Bool = false, name: String? = nil) {
        self.real = real
        self.imaginary = imaginary
        self.shape = shape
        self.float16 = float16
        super.init(name: name)
    }
    
    /// Construct a ComplexConstant node with the specified real and imaginary values
    /// - Parameters:
    ///   - real: The real part of the complex constant
    ///   - imaginary: The imaginary part of the complex constant
    ///   - shape:The shape of the resultant constant Tensor
    ///   - float16: (Optional) If true the constant is of type complexFloat16, else it is of type complexFloat32.  Default is false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(real: Double, imaginary: Double, shape: TensorShape, float16: Bool = false, name: String? = nil) {
        self.real = real
        self.imaginary = imaginary
        self.shape = shape.dimensions
        self.float16 = float16
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        if (shape.count == 1 && shape[0] == 1) {
            if (float16) {
                let result = graph.mpsgraph.complexConstant(realPart: real, imaginaryPart: imaginary, dataType: .complexFloat16)
                return [result]
            }
            else {
                let result = graph.mpsgraph.complexConstant(realPart: real, imaginaryPart: imaginary)
                return [result]
            }
        }
        else {
            let result = graph.mpsgraph.complexConstant(realPart: real, imaginaryPart: imaginary, shape: shape.map { NSNumber(value: $0)}, dataType: .complexFloat16)
            return [result]
        }
    }
}


///  Leaf Node that provides a  complex value tensor from real and imaginary component tensors
public class ComplexTensor : BinaryNode {
    /// Constructor for an ConstantTensor operation
    ///
    /// - Parameters:
    ///   - real: (Optional) The name of the tensor that will provide the real part of the complex tensor.  If nil the previous node's output will be used
    ///   - imaginary: (Optional) The name of the tensor that will provide the imaginary part of the complex tensor.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(real: String? = nil, imaginary: String? = nil, name: String? = nil) {
        super.init(firstInput: real, secondInput: imaginary, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let result = graph.mpsgraph.complexTensor(realTensor: inputTensors.firstInputTensor, imaginaryTensor: inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [result]
    }
}

///   Node to return the complex conjugate of a complex tensor
public class Conjugate : UnaryNode {
    ///  Constructor for conjugation operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result = graph.mpsgraph.conjugate(tensor: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to return the imaginary part of a complex tensor
public class ImaginaryPart : UnaryNode {
    ///  Constructor for getting the imaginary part of a complex tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result = graph.mpsgraph.imaginaryPartOfTensor(tensor: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to return the real part of a complex tensor
public class RealPart : UnaryNode {
    ///  Constructor for getting the real part of a complex tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result = graph.mpsgraph.realPartOfTensor(tensor: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}
