//
//  ComparisonNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 12/3/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node to piecewise compare if the values of two tensors are equal
public class Equal : BinaryNode {
    /// Constructor for an Equal operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the power to raise the first operand to).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let comparisonResult = graph.mpsgraph.equal(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [comparisonResult]
    }
}

///   Node to piecewise compare if the values of the first tensor are greater than the corresponding value of the second tensor
public class GreaterThan : BinaryNode {
    /// Constructor for an GreaterThan operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the power to raise the first operand to).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let comparisonResult = graph.mpsgraph.greaterThan(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [comparisonResult]
    }
}

///   Node to piecewise compare if the values of the first tensor are greater than or equal to the corresponding value of the second tensor
public class GreaterThanOrEqualTo : BinaryNode {
    /// Constructor for an GreaterThanOrEqualTo operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the power to raise the first operand to).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let comparisonResult = graph.mpsgraph.greaterThanOrEqualTo(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [comparisonResult]
    }
}

///   Node to piecewise compare if the values of the first tensor are less than the corresponding value of the second tensor
public class LessThan : BinaryNode {
    /// Constructor for an LessThan operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the power to raise the first operand to).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let comparisonResult = graph.mpsgraph.lessThan(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [comparisonResult]
    }
}

///   Node to piecewise compare if the values of the first tensor are less than or equal to the corresponding value of the second tensor
public class LessThanOrEqualTo : BinaryNode {
    /// Constructor for an LessThanOrEqualTo operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the power to raise the first operand to).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let comparisonResult = graph.mpsgraph.lessThanOrEqualTo(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [comparisonResult]
    }
}

///   Node to piecewise compare if the values of two tensors are not equal
public class NotEqual : BinaryNode {
    /// Constructor for an NotEqual operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the power to raise the first operand to).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let comparisonResult = graph.mpsgraph.notEqual(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [comparisonResult]
    }
}
