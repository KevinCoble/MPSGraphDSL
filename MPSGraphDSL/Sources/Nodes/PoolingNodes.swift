//
//  PoolingNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/22/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

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
