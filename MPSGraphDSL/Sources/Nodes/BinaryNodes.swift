//
//  BinaryNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/18/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Intermediate class type for nodes with two inputs - do not instantiate
public class BinaryNode: Node {
    let firstInputName: String?
    let secondInputName: String?
    
    init(firstInput: String?, secondInput: String?, name: String?) {
        firstInputName = firstInput
        secondInputName = secondInput
        super.init(name: name)
    }
    
    func checkEqualInputShapes(inputs: (firstInputTensor: MPSGraphTensor, secondInputTensor: MPSGraphTensor)) throws {
        var inputShape1: TensorShape
        var inputShape2: TensorShape
        if let shape = inputs.firstInputTensor.shape {
            inputShape1 = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        if let shape = inputs.secondInputTensor.shape {
            inputShape2 = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        if (inputShape1 != inputShape2) {
            var string1: String = "*Unnamed - last node*"
            var string2: String = "*Unnamed - last node*"
            if (firstInputName != nil) {string1 = firstInputName!}
            if (secondInputName != nil) {string2 = secondInputName!}
            throw MPSGraphDSLErrors.BinaryShapesDontMatch(string1, string2)
        }
    }
}

///   Node to add two tensors of the same size together
public class Addition : BinaryNode {
    /// Constructor for an Addition operation
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
        let additionResult = graph.mpsgraph.addition(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [additionResult]
    }
}

///   Node to subtract a tensor from another of the same size
public class Subtraction : BinaryNode {
    /// Constructor for an Subtraction operation
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
        let subtractionResult = graph.mpsgraph.subtraction(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [subtractionResult]
    }
}

///   Node to multiply individual values in a tensor with another of the same size
public class Multiplication : BinaryNode {
    /// Constructor for an Multiplication operation
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
        let multiplicationResult = graph.mpsgraph.multiplication(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [multiplicationResult]
    }
}

///   Node to divide individual values in a tensor with values from another of the same size
public class Division : BinaryNode {
    let noNaNs: Bool
    
    /// Constructor for a Division operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    ///   - noNaNs: (Optional) If true, divisions that result in NaN will result in a zero value.  Default is false
    public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil, noNaNs: Bool = false) {
        self.noNaNs = noNaNs
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        var divisionResult: MPSGraphTensor
        if (noNaNs) {
            divisionResult = graph.mpsgraph.divisionNoNaN(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        }
        else {
            divisionResult = graph.mpsgraph.division(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        }
        
        //  Return the created MPSGraphTensor
        return [divisionResult]
    }
}

///   Node to piecewise raise individual values in a tensor to a power from values another tensor another of the same size
public class Power : BinaryNode {
    /// Constructor for an Power operation
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
        let powerResult = graph.mpsgraph.power(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [powerResult]
    }
}

///   Node to multiply two input tensors using matrix multiplication
public class MatrixMultiplication : BinaryNode {
    /// Constructor for an Power operation
    ///
    /// - Parameters:
    ///   - primary: (Optional) The name of the tensor that will provide the primary operand.  If nil the previous node's output will be used
    ///   - secondary: (Optional) The name of the tensor that will provide the secondary operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(primary: String? = nil, secondary: String? = nil, name: String? = nil) {
        super.init(firstInput: primary, secondInput: secondary, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)

        //  Get the tensor shapes
        var inputShape1: TensorShape
        var inputShape2: TensorShape
        if let shape = inputTensors.firstInputTensor.shape {
            inputShape1 = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        if let shape = inputTensors.secondInputTensor.shape {
            inputShape2 = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }

        //  Verify that the input tensor shapes match a matrix mutiplaction
        if (inputShape1.dimensions.count == 2) && (inputShape2.dimensions.count == 2) {
            if (inputShape1.dimensions[1] != inputShape2.dimensions[0]) {
                throw MPSGraphDSLErrors.InputShapeError("Matrix multiplication requires the inner dimensions of the two tensors to match.")
            }
        }
        else if (inputShape1.dimensions.count == 1) && (inputShape2.dimensions.count == 2) {    //  vector times matrix
            if (inputShape1.dimensions[0] != inputShape2.dimensions[0]) {
                throw MPSGraphDSLErrors.InputShapeError("Matrix multiplication requires the inner dimensions of the two tensors to match.")
            }
        }

        
        //  Add to the graph itself
        let matrixMultResult = graph.mpsgraph.matrixMultiplication(primary: inputTensors.firstInputTensor, secondary: inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [matrixMultResult]
    }
}

///   Node to piecewise find the maximum from values two tensors of the same size
public class Maximum : BinaryNode {
    internal let propogateNaNs : Bool
    /// Constructor for an Maximum operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - propogateNaNs: (Optional) If true, NaN values will be propogated, otherwise they will not.  Default is false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(firstInput: String? = nil, secondInput: String? = nil, propogateNaNs: Bool = false, name: String? = nil) {
        self.propogateNaNs = propogateNaNs
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        if (propogateNaNs) {
            let result = graph.mpsgraph.maximumWithNaNPropagation(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
            return [result]
        }
        else {
            let powerResult = graph.mpsgraph.maximum(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
            return [powerResult]
        }
    }
}

///   Node to piecewise find the minimum from values two tensors of the same size
public class Minimum : BinaryNode {
    internal let propogateNaNs : Bool
    /// Constructor for an Minimum operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - propogateNaNs: (Optional) If true, NaN values will be propogated, otherwise they will not.  Default is false
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(firstInput: String? = nil, secondInput: String? = nil, propogateNaNs: Bool = false, name: String? = nil) {
        self.propogateNaNs = propogateNaNs
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        if (propogateNaNs) {
            let result = graph.mpsgraph.minimumWithNaNPropagation(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
            return [result]
        }
        else {
            let powerResult = graph.mpsgraph.minimum(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
            return [powerResult]
        }
    }
}

///   Node to calculate the modulo values with two tensors another of the same size
public class Modulo : BinaryNode {
    /// Constructor for an Modulo operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the divisor).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let result = graph.mpsgraph.modulo(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [result]
    }
}

///   Node to calculate the Hamming distance between two tensors another of the same size
public class HammingDistance : BinaryNode {
    let resultType: DataType?
    /// Constructor for an Hamming Distance operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - resultType: (Optional) The data type for the result.  If nil, the data type from the first input tensor will be used.  Defaults to nil
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(firstInput: String? = nil, secondInput: String? = nil, resultType: DataType? = nil, name: String? = nil) {
        self.resultType = resultType
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        let dataType: MPSDataType
        if let type = resultType {
            dataType = type.getMPSDataType()
        }
        else {
            dataType = inputTensors.firstInputTensor.dataType
        }
        
        //  Add to the graph itself
        let result = graph.mpsgraph.HammingDistance(primary: inputTensors.firstInputTensor, secondary: inputTensors.secondInputTensor, resultDataType: dataType, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [result]
    }
}

///   Node to get the floor division remainder from two tensors
public class FloorModulo : BinaryNode {
    /// Constructor for an floor modulo operation
    ///
    /// - Parameters:
    ///   - firstInput: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondInput: (Optional) The name of the tensor that will provide the second operand (the divisor).  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    override public init(firstInput: String? = nil, secondInput: String? = nil, name: String? = nil) {
        super.init(firstInput: firstInput, secondInput: secondInput, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let result = graph.mpsgraph.floorModulo(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [result]
    }
}
