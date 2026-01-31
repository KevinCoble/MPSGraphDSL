//
//  TernaryNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 12/31/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///   Intermediate class type for nodes with three inputs - do not instantiate
public class TernaryNode: Node {
    let firstInputName: String?
    let secondInputName: String?
    let thirdInputName: String?

    init(firstInput: String?, secondInput: String?, thirdInput: String?, name: String?) {
        firstInputName = firstInput
        secondInputName = secondInput
        thirdInputName = thirdInput
        super.init(name: name)
    }
    
    func checkEqualInputShapes(inputs: (firstInputTensor: MPSGraphTensor, secondInputTensor: MPSGraphTensor, thirdInputTensor: MPSGraphTensor)) throws {
        var inputShape1: TensorShape
        var inputShape2: TensorShape
        var inputShape3: TensorShape
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
        if let shape = inputs.thirdInputTensor.shape {
            inputShape3 = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }

        if (inputShape1 != inputShape2 || inputShape1 != inputShape3) {
            var string1: String = "*Unnamed - last node*"
            var string2: String = "*Unnamed - last node*"
            var string3: String = "*Unnamed - last node*"
            if (firstInputName != nil) {string1 = firstInputName!}
            if (secondInputName != nil) {string2 = secondInputName!}
            if (thirdInputName != nil) {string3 = thirdInputName!}
            throw MPSGraphDSLErrors.TernaryShapesDontMatch(string1, string2, string3)
        }
    }
}

///   Node to clamp a tensor's values to the value of two limit tensors
public class Clamp : TernaryNode {
    /// Constructor for an Clamp operation
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - minTensor: (Optional) The name of the tensor that will provide the minimum clamp value.  If nil the previous node's output will be used
    ///   - maxTensor: (Optional) The name of the tensor that will provide the maximum clamp value.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(input: String? = nil, minTensor: String? = nil, maxTensor: String? = nil, name: String? = nil) {
        super.init(firstInput: input, secondInput: minTensor, thirdInput: maxTensor, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getTernaryTensors(firstInputName, secondInputName, thirdInputName)
        
        //  Add to the graph itself
        let additionResult = graph.mpsgraph.clamp(inputTensors.firstInputTensor, min: inputTensors.secondInputTensor, max: inputTensors.thirdInputTensor, name: graph.getFullName(name))
        
        //  Return the created MPSGraphTensor
        return [additionResult]
    }
}
