//
//  ActivationFunctionNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/20/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node to perform the ReLU (rectified linear activation unit) activation function on thei input tensor
public class ReLU : UnaryNode {
    ///  Constructor for a ReLU operation
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
        let reLUResult = graph.mpsgraph.reLU(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [reLUResult]
    }
}

///   Node to perform the leaky ReLU (rectified linear activation unit) activation function on thei input tensor
public class LeakyReLU : UnaryNode {
    let alpha: Double
    let alphaTensor: String?
    let alphaFromTensor: Bool
    
    ///  Constructor for a Leaky ReLU operation with a constant alpha
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - alpha: The alpha value for the "leak" part of the leaky ReLU
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, alpha: Double, name: String? = nil) {
        self.alpha = alpha
        self.alphaTensor = nil
        alphaFromTensor = false
        super.init(input: input, name: name)
    }
    
    ///  Constructor for a Leaky ReLU operation with alpha from another tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - alphaTensor: (Optional)  The node that provides the alpha value for the "leak" part of the leaky ReLU.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, alphaTensor: String?, name: String? = nil) {
        self.alpha = 0.0
        self.alphaTensor = alphaTensor
        alphaFromTensor = true
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        if (alphaFromTensor) {
            //  Find the node with the alpha tensor name
            let alphaMPSTensor = try graph.getOptionalTensor(alphaTensor)

            let reLUResult = graph.mpsgraph.leakyReLU(with: inputTensor, alphaTensor: alphaMPSTensor, name: graph.getFullName(name))
            
            //  Remember the output tensor and shape for later
            return [reLUResult]

        }
        else {
            let reLUResult = graph.mpsgraph.leakyReLU(with: inputTensor, alpha: alpha, name: graph.getFullName(name))
            
            //  Remember the output tensor and shape for later
            return [reLUResult]
        }
    }
}

///   Node to perform the sigmoid activation function on thei input tensor
public class Sigmoid : UnaryNode {
    ///  Constructor for a ReLU operation
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
        let sigmoidResult = graph.mpsgraph.sigmoid(with: inputTensor, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [sigmoidResult]
    }
}

///   Enumeration for selecting the activation function of a Neural Network layer
public enum ActivationFunction {
    case none
    case relu
    case tanh
    case sigmoid
    case leakyRelu(Double)
    case leakyReluFromTensor(String)
    
    func addActivation(graph: Graph, inputTensor: MPSGraphTensor, name: String?) throws -> MPSGraphTensor? {
        switch (self) {
            case .none:
                return nil
            case .relu:
                let reLUResult = graph.mpsgraph.reLU(with: inputTensor, name: graph.getFullName(name))
                return reLUResult
            case .tanh:
                let tanhResult = graph.mpsgraph.tanh(with: inputTensor, name: graph.getFullName(name))
                return tanhResult
            case .sigmoid:
                let sigmoidResult = graph.mpsgraph.sigmoid(with: inputTensor, name: graph.getFullName(name))
                return sigmoidResult
            case .leakyRelu(let alpha):
                let leakyReLUResult = graph.mpsgraph.leakyReLU(with: inputTensor, alpha: alpha, name: graph.getFullName(name))
                return leakyReLUResult
            case .leakyReluFromTensor(let alphaName):
                if let alphaNode = graph.findNamedNode(alphaName) {
                    let leakyReLUResult = graph.mpsgraph.leakyReLU(with: inputTensor, alphaTensor: alphaNode.mpstensor, name: graph.getFullName(name))
                    return leakyReLUResult
                }
                else {
                    throw MPSGraphDSLErrors.NamedTensorNotFound(alphaName)
                }
        }

    }
}
