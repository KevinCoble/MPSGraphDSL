//
//  LogicalNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 12/3/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///   Node to calculate the logical AND of two tensors of the same size together
public class LogicalAND : BinaryNode {
    /// Constructor for a logical AND operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let logicalResult = graph.mpsgraph.logicalAND(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [logicalResult]
    }
}

///   Node to calculate the logical NAND of two tensors of the same size together
public class LogicalNAND : BinaryNode {
    /// Constructor for a logical NAND operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let logicalResult = graph.mpsgraph.logicalNAND(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [logicalResult]
    }
}

///   Node to calculate the logical OR of two tensors of the same size together
public class LogicalOR : BinaryNode {
    /// Constructor for a logical OR operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let logicalResult = graph.mpsgraph.logicalOR(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [logicalResult]
    }
}

///   Node to calculate the logical NOR of two tensors of the same size together
public class LogicalNOR : BinaryNode {
    /// Constructor for a logical NOR operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let logicalResult = graph.mpsgraph.logicalNOR(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [logicalResult]
    }
}

///   Node to calculate the logical XOR of two tensors of the same size together
public class LogicalXOR : BinaryNode {
    /// Constructor for a logical XOR operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let logicalResult = graph.mpsgraph.logicalXOR(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [logicalResult]
    }
}

///   Node to calculate the logical XNOR of two tensors of the same size together
public class LogicalXNOR : BinaryNode {
    /// Constructor for a logical XNOR operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let logicalResult = graph.mpsgraph.logicalXNOR(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [logicalResult]
    }
}

///   Node to perform a logical NOT of the values of of a tensor
public class NOT : UnaryNode {
    /// Constructor for a logical not operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let logicalResult = graph.mpsgraph.not(with: inputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [logicalResult]
    }
}

///   Node to perform a bitwise AND of the values of of two tensors of the same size
public class BitwiseAND : BinaryNode {
    /// Constructor for a logical bitwise AND operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let bitwiseResult = graph.mpsgraph.bitwiseAND(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [bitwiseResult]
    }
}

///   Node to perform a bitwise left shift of the first of two tensors of the same size, by the amount specified by the second tensor
public class BitwiseLeftShift : BinaryNode {
    /// Constructor for a logical bitwise left shift operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let bitwiseResult = graph.mpsgraph.bitwiseLeftShift(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [bitwiseResult]
    }
}

///   Node to perform a bitwise NOT of the values of of a tensor
public class BitwiseNOT : UnaryNode {
    /// Constructor for a logical bitwise AND operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let bitwiseResult = graph.mpsgraph.bitwiseNOT(inputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [bitwiseResult]
    }
}

///   Node to perform a bitwise OR of the values of of two tensors of the same size
public class BitwiseOR : BinaryNode {
    /// Constructor for a logical bitwise OR operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let bitwiseResult = graph.mpsgraph.bitwiseOR(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [bitwiseResult]
    }
}

///   Node to perform a bitwise population count of the values of of a tensor
public class BitwisePopulationCount : UnaryNode {
    /// Constructor for a logical bitwise population count operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(input: String? = nil, name: String? = nil) {
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let bitwiseResult = graph.mpsgraph.bitwisePopulationCount(inputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [bitwiseResult]
    }
}

///   Node to perform a bitwise right shift of the first of two tensors of the same size, by the amount specified by the second tensor
public class BitwiseRightShift : BinaryNode {
    /// Constructor for a logical bitwise right shift operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let bitwiseResult = graph.mpsgraph.bitwiseRightShift(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [bitwiseResult]
    }
}

///   Node to perform a bitwise XOR of the values of of two tensors of the same size
public class BitwiseXOR : BinaryNode {
    /// Constructor for a logical bitwise XOR operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let bitwiseResult = graph.mpsgraph.bitwiseXOR(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [bitwiseResult]
    }
}
