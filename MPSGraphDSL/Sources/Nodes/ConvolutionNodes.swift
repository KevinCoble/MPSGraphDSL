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
