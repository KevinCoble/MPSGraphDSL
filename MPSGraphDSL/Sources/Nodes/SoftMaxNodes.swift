//
//  SoftMaxNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/23/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///   Node to perform a softmax operation
public class SoftMax : UnaryNode {
    let axis: Int
    ///  Constructor for a softMax operation across a specified axis of a tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - axis: (Optional) The axis of the tensor to perform a softMax operation on.  If nil, the tensor is assumed to be 1 dimensional and that dimension is used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, axis: Int = -1, name: String? = nil) {
        self.axis = axis
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add to the graph itself
        let softMaxResult = graph.mpsgraph.softMax(with: inputTensor, axis: axis, name: graph.getFullName(name))
        
        //  Remember the output tensor and shape for later
        return [softMaxResult]
    }
}

///   Node to perform a softmax cross-entropy operation with a labels tensor
public class SoftMaxCrossEntropy : BinaryNode {
    let axis: Int
    let reductionType: MPSGraphLossReductionType
    
    var suffixes: [String] = []
    var targetIndices: [Int] = []

    ///  Constructor for a softMax operation across a specified axis of a tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - labels: (Optional) The name of the tensor that will provide the labels tensor.  If nil the previous node's output will be used
    ///   - axis: (Optional) The axis of the tensor to perform a softMax operation on.  If nil, the tensor is assumed to be 1 dimensional and that dimension is used
    ///   - reductionType: The type of reduction used for the loss operations
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, labels: String? = nil, axis: Int = -1, reductionType: MPSGraphLossReductionType,name: String? = nil) {
        self.axis = axis
        self.reductionType = reductionType
        super.init(firstInput: input, secondInput: labels, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)

        //  Add to the graph itself
        var crossEntropyName = graph.getFullName(name)
        if (graph.batchGraph) {
            var newName: String = "softMaxCrossEntropy"
            if (crossEntropyName != nil) { newName = crossEntropyName! + "_softMaxCrossEntropy" }
            crossEntropyName = newName
            suffixes = ["_softMaxCrossEntropy"]
        }
        else {
            suffixes = [""]
        }
        let softMaxCrossEntropyResult = graph.mpsgraph.softMaxCrossEntropy(inputTensors.firstInputTensor, labels: inputTensors.secondInputTensor, axis: axis, reuctionType: reductionType, name: crossEntropyName)
        
        //  If a batch graph, divide the result by the batch size to get an actual loss mean
        if (graph.batchGraph) {
            let batchSizeTensor = graph.mpsgraph.constant(Double(graph.batchSize), shape: [1], dataType: softMaxCrossEntropyResult.dataType)
            suffixes.append("_*Unnamable constant*")
            let lossMeanTensor = graph.mpsgraph.division(softMaxCrossEntropyResult, batchSizeTensor, name: graph.getFullName(name))
            suffixes.append("")
            targetIndices = [2]
            //  Remember the output tensors for later
            return [softMaxCrossEntropyResult, batchSizeTensor, lossMeanTensor]
        }
        else {
            //  Remember the output tensor for later
            targetIndices = [0]
            return [softMaxCrossEntropyResult]
        }
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }
}
