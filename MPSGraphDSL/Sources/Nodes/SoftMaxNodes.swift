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

    ///  Constructor for a softMax cross entropy operation across a specified axis of a tensor
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - labels: (Optional) The name of the tensor that will provide the labels tensor.  If nil the previous node's output will be used
    ///   - axis: (Optional) The axis of the tensor to perform a softMax cross entropy operation on.  If nil, the tensor is assumed to be 1 dimensional and that dimension is used
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

///   Node to perform a classification  cross-entropy operation with a classification indices tensor and logits
///             Logits should be on the last dimension, and the two input tensors the same shape except the classification indices will not have the logits dimension.
public class ClassificationCrossEntropy : BinaryNode {
    let reductionType: MPSGraphLossReductionType

    var suffixes: [String] = []
    
    /// Calculate the cross entropy for a set of logits (probabilities) and associated truth value class indices
    /// - Parameters:
    ///   - logits: (Optional)  The tensor with the logits (prediction probabilities). If nil the previous node's output will be used
    ///   - classIndices: (Optional)  The tensor with the ground-truth class indices.  This tensor should have the same shape of the logits tensor minus the last dimension.    If nil the previous node's output will be used
    ///   - reductionType: The reduction type used to condense the individual entropies to a single number.
    ///   - name: (Optional) The name for this node and its associated tensors.  Three tensors are added with suffixes '_oneHot', '_crossEntropy', and the final tensor without a suffix.
    public init(logits: String? = nil, classIndices: String? = nil, reductionType: MPSGraphLossReductionType = .mean, name: String? = nil) {
        self.reductionType = reductionType
        super.init(firstInput: logits, secondInput: classIndices, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Get the shapes of the inputs
        let logitsShape: TensorShape
        let classIndicesShape: TensorShape
        if let shape = inputTensors.firstInputTensor.shape {
            logitsShape = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        if let shape = inputTensors.secondInputTensor.shape {
            classIndicesShape = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        
        //  Verify the logits shape is one dimension higher than indices, and the other dimensions match
        if (logitsShape.dimensions.count != classIndicesShape.dimensions.count + 1) {
            throw MPSGraphDSLErrors.InputShapeError("Logits must be one dimension larger than classification indices")
        }
        for i in 0..<classIndicesShape.dimensions.count {
            if (logitsShape.dimensions[i] != classIndicesShape.dimensions[i]) {
                throw MPSGraphDSLErrors.InputShapeError("Logits and classification indices must have the same shape in all dimensions except the classification indices dimension")
            }
        }
        
        //  One-hot the indices to get the same shape
        let type = inputTensors.firstInputTensor.dataType
        let depth = logitsShape.dimensions.last!
        var oneHotName: String
        if let name = graph.getFullName(name) {
            oneHotName = name + "_oneHot"
        }
        else {
            oneHotName = "*Unnamed*_oneHot"
        }
        let oneHot = graph.mpsgraph.oneHot(withIndicesTensor: inputTensors.secondInputTensor, depth: depth, dataType: type, name: oneHotName)
        suffixes.append("_oneHot")
        
        //  Calculate the cross-entropy between the one-hot and logits
        var crossEntropyName: String
        if let name = graph.getFullName(name) {
            crossEntropyName = name + "_crossEntropy"
        }
        else {
            crossEntropyName = "*Unnamed*_crossEntropy"
        }
        let crossEntropy = graph.mpsgraph.softMaxCrossEntropy(inputTensors.firstInputTensor, labels: oneHot, axis: logitsShape.dimensions.count - 1, reuctionType: reductionType, name: crossEntropyName)
        suffixes.append("_crossEntropy")
        
        //  Reshape to a scalar
        let reshape = graph.mpsgraph.reshape(crossEntropy, shape: [1 as NSNumber], name: graph.getFullName(name))
        suffixes.append("")

        return [oneHot, crossEntropy, reshape]
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return [2]  //  Only the reshape is targetted
    }
}
