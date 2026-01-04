//
//  UnaryNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/18/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Intermediate class type for nodes with only one input - do not instantiate
public class UnaryNode: Node {
    let inputName: String?
    
    init(input: String? = nil, name: String? = nil) {
        inputName = input
        super.init(name: name)
    }
}

///   Node to calculate the absolute values the values of a tensor
public class Absolute : UnaryNode {
    ///  Constructor for an absolute value operation
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
        let absoluteResult = graph.mpsgraph.absolute(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [absoluteResult]
    }
}

///   Node to negate the values of a tensor
public class Negative : UnaryNode {
    ///  Constructor for a negative (negation) operation
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
        let negationResult = graph.mpsgraph.negative(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [negationResult]
    }
}

///   Node to square the values of a tensor
public class Square : UnaryNode {
    ///  Constructor for a square (power of two) operation
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
        let squareResult = graph.mpsgraph.square(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [squareResult]
    }
}

///   Node to calculate the square root of the values of a tensor
public class SquareRoot : UnaryNode {
    ///  Constructor for a square root operation
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
        let squareRootResult = graph.mpsgraph.squareRoot(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [squareRootResult]
    }
}

///   Node to calculate the natural exponent (eⁿ) of the values of a tensor
public class Exponent : UnaryNode {
    ///  Constructor for a natural exponent operation
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
        let exponentResult = graph.mpsgraph.exponent(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [exponentResult]
    }
}

///   Node to calculate the base 10 exponent (10ⁿ) of the values of a tensor
public class Base10Exponent : UnaryNode {
    ///  Constructor for a base 10 exponent operation
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
        let exponentResult = graph.mpsgraph.exponentBase10(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [exponentResult]
    }
}

///   Node to calculate the base 2 exponent (2ⁿ) of the values of a tensor
public class Base2Exponent : UnaryNode {
    ///  Constructor for a base 2 exponent operation
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
        let exponentResult = graph.mpsgraph.exponentBase2(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [exponentResult]
    }
}

///   Node to calculate the natural logarithm of the values of a tensor
public class Logarithm : UnaryNode {
    ///  Constructor for a natural logarithm operation
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
        let logarithmResult = graph.mpsgraph.logarithm(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [logarithmResult]
    }
}

///   Node to calculate the base 10 logarithm of the values of a tensor
public class Base10Logarithm : UnaryNode {
    ///  Constructor for a base 10 logarithm operation
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
        let logarithmResult = graph.mpsgraph.logarithmBase10(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [logarithmResult]
    }
}

///   Node to calculate the base 2 logarithm of the values of a tensor
public class Base2Logarithm : UnaryNode {
    ///  Constructor for a base 2 logarithm operation
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
        let logarithmResult = graph.mpsgraph.logarithmBase2(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [logarithmResult]
    }
}

///   Node to calculate the absolute square of the values of a tensor
public class AbsoluteSquare : UnaryNode {
    ///  Constructor for an absolute square value operation
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
        let absoluteResult = graph.mpsgraph.absoluteSquare(tensor: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [absoluteResult]
    }
}

///   Node to apply a cieling operation to the values of a tensor
public class Ceiling : UnaryNode {
    ///  Constructor for an ceiling operation
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
        let result = graph.mpsgraph.ceil(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to apply a floor operation to the values of a tensor
public class Floor : UnaryNode {
    ///  Constructor for an floor operation
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
        let result = graph.mpsgraph.floor(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to apply the error function to the values of a tensor
public class ErrorFunction : UnaryNode {
    ///  Constructor for an error function operation
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
        let result = graph.mpsgraph.erf(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to calculate the inverse (1/x) of the values of a tensor
public class Inverse : UnaryNode {
    ///  Constructor for an inverse operation
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
        let result = graph.mpsgraph.inverse(input: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to check the values of a tensor for being finite
public class isFinite : UnaryNode {
    ///  Constructor for an finite check operation
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
        let result = graph.mpsgraph.isFinite(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to check the values of a tensor for being infinite
public class isInfinite : UnaryNode {
    ///  Constructor for an infinite check operation
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
        let result = graph.mpsgraph.isInfinite(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to check the values of a tensor for being NaN (not a number)
public class isNan : UnaryNode {
    ///  Constructor for an NaN check operation
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
        let result = graph.mpsgraph.isNaN(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to calculate the reciprocal (1/x) of the values of a tensor
public class Reciprocal : UnaryNode {
    ///  Constructor for an reciprocal operation
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
        let result = graph.mpsgraph.reciprocal(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to calculate the reciprocal square root (1/sqrt(x)) of the values of a tensor
public class ReciprocalSquareRoot : UnaryNode {
    ///  Constructor for an reciprocal square root operation
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
        let result = graph.mpsgraph.reciprocalSquareRoot(inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to round the values of a tensor to the nearest even integer
public class Rint : UnaryNode {
    ///  Constructor for a round to nearest even integer operation
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
        let result = graph.mpsgraph.rint(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to round the values of a tensor to the nearest integer
public class Round : UnaryNode {
    ///  Constructor for a round to nearest integer operation
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
        let result = graph.mpsgraph.round(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to return the sign of each value of a tensor
public class Sign : UnaryNode {
    ///  Constructor for sign operation
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
        let result = graph.mpsgraph.sign(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to return the sign bit of each value of a tensor
public class SignBit : UnaryNode {
    ///  Constructor for sign bit operation
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
        let result = graph.mpsgraph.signbit(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to apply the truncation operation to each value of a tensor
public class Truncate : UnaryNode {
    ///  Constructor for a truncation operation
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
        let result = graph.mpsgraph.truncate(inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to apply the identity operation to each value of a tensor
///   This operation just returns a copy of the input tensor
public class Identity : UnaryNode {
    ///  Constructor for a identity operation
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
        let result = graph.mpsgraph.identity(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to find the non-zero element indices of the input tensor.  The return is an Int32 matrix tensor with a row for each non-zero entry, and dimension locations of the element for the columns
public class NonZeroIndices : UnaryNode {
    ///  Constructor for a non-zero indices find operation
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
        let result = graph.mpsgraph.nonZeroIndices(inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to calculate the mean of the first input along the specified axes
public class Mean : UnaryNode {
    let axes: [Int]

    ///  Constructor for a mean operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axes: an array of indices for the axes that the operation should be carried out for
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axes: [Int], name: String? = nil) {
        self.axes = axes
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let result = graph.mpsgraph.mean(of: inputTensor, axes: axes.map { NSNumber(value: $0)}, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [result]
    }
}

///   Node to calculate the variance of the first input along the specified axes
public class Variance : UnaryNode {
    let axes: [Int]
    let meanTensor: String?
    let haveMean: Bool

    ///  Constructor for a variance operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axes: an array of indices for the axes that the operation should be carried out for
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axes: [Int], name: String? = nil) {
        self.axes = axes
        self.meanTensor = nil
        haveMean = false
        super.init(input: input, name: name)
    }

    ///  Constructor for a variance operation where the mean has already been computed
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - meanTensor: (Optional) the name of the node that calculated the mean.  If nil the previous node's output will be used
    ///   - axes: an array of indices for the axes that the operation should be carried out for
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, meanTensor: String? = nil, axes: [Int], name: String? = nil) {
        self.axes = axes
        self.meanTensor = meanTensor
        haveMean = true
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        if haveMean {
            let meanMPSTensor = try graph.getOptionalTensor(meanTensor)

            let result = graph.mpsgraph.variance(of: inputTensor, mean: meanMPSTensor, axes: axes.map { NSNumber(value: $0)}, name: graph.getFullName(name))

            //  Remember the output tensor and shape for later
            return [result]
        }
        
        else {
            let result = graph.mpsgraph.variance(of: inputTensor, axes: axes.map { NSNumber(value: $0)}, name: graph.getFullName(name))
            
            //  Remember the output tensor and shape for later
            return [result]
        }
    }
}
