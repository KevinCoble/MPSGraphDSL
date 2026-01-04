//
//  TrigNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 12/3/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node to calculate the inverse cosine of the values of a tensor
public class ArcCosine : UnaryNode {
    ///  Constructor for a acos operation
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
        let acosResult = graph.mpsgraph.acos(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [acosResult]
    }
}

///   Node to calculate the inverse hyperbolic cosine of the values of a tensor
public class ArcHyperbolicCosine : UnaryNode {
    ///  Constructor for a acosh operation
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
        let acoshResult = graph.mpsgraph.acosh(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [acoshResult]
    }
}

///   Node to calculate the inverse sine of the values of a tensor
public class ArcSine : UnaryNode {
    ///  Constructor for a asin operation
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
        let asinResult = graph.mpsgraph.asin(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [asinResult]
    }
}

///   Node to calculate the inverse hyperbolic sine of the values of a tensor
public class ArcHyperbolicSine : UnaryNode {
    ///  Constructor for a asinh operation
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
        let asinhResult = graph.mpsgraph.asinh(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [asinhResult]
    }
}

///   Node to calculate the inverse tangent of the values of a tensor
public class ArcTangent : UnaryNode {
    ///  Constructor for a atan operation
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
        let atanResult = graph.mpsgraph.atan(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [atanResult]
    }
}

///   Node to calculate the inverse hyperbolic tangent of the values of a tensor
public class ArcHyperbolicTangent : UnaryNode {
    ///  Constructor for a atanh operation
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
        let atanhResult = graph.mpsgraph.atanh(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [atanhResult]
    }
}


///   Node to calculate the arctangent values in a tensor from two other same-sized tensors
public class ArcTangent2 : BinaryNode {
    /// Constructor for an atan2 operation
    ///
    /// - Parameters:
    ///   - primaryTensor: (Optional) The name of the tensor that will provide the first operand.  If nil the previous node's output will be used
    ///   - secondaryTensor: (Optional) The name of the tensor that will provide the second operand.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(primaryTensor: String? = nil, secondaryTensor: String? = nil, name: String? = nil) {
        super.init(firstInput: primaryTensor, secondInput: secondaryTensor, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let atan2Result = graph.mpsgraph.atan2(withPrimaryTensor: inputTensors.firstInputTensor, secondaryTensor: inputTensors.secondInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [atan2Result]
    }
}


///   Node to calculate the cosine of the values of a tensor
public class Cosine : UnaryNode {
    ///  Constructor for a cos operation
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
        let cosResult = graph.mpsgraph.cos(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [cosResult]
    }
}

///   Node to calculate the hyperbolic cosine of the values of a tensor
public class HyperbolicCosine : UnaryNode {
    ///  Constructor for a acosh operation
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
        let coshResult = graph.mpsgraph.cosh(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [coshResult]
    }
}

///   Node to calculate the  sine of the values of a tensor
public class Sine : UnaryNode {
    ///  Constructor for a sin operation
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
        let sinResult = graph.mpsgraph.sin(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [sinResult]
    }
}

///   Node to calculate the hyperbolic sine of the values of a tensor
public class HyperbolicSine : UnaryNode {
    ///  Constructor for a sinh operation
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
        let sinhResult = graph.mpsgraph.sinh(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [sinhResult]
    }
}


///   Node to calculate the tangent of the values of a tensor
public class Tangent : UnaryNode {
    ///  Constructor for a tan operation
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
        let tanResult = graph.mpsgraph.tan(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [tanResult]
    }
}

///   Node to calculate the hyperbolic tangent of the values of a tensor
public class HyperbolicTangent : UnaryNode {
    ///  Constructor for a tanh operation
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
        let tanhResult = graph.mpsgraph.tanh(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [tanhResult]
    }
}
