//
//  ControlNodeBuilders.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 2/23/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///   Node that adds an instance of a SubGraphDefinition to another Graph or SubGraph
public class If : UnaryNode {
    let thenBlock: Then
    let elseBlock: Else
    
    var suffixes: [String] = []
    
    public init(_ predicate: String? = nil, _ thenBlock: Then, _ elseBlock: Else, name: String) {
        self.thenBlock = thenBlock
        self.elseBlock = elseBlock
        super.init(input: predicate, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the predicate tensor
        let predicateTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Put the 'then' block nodes into the graph
        try graph.processNodes(thenBlock.nodes, block: "if_then_" + name!)
        
        //  Get the 'then' return tensors
        let thenReturnTensors = try graph.getReturnTensors()
        
        //  Put the 'else' block nodes into the graph
        try graph.processNodes(elseBlock.nodes, block: "if_else_" + name!)
        
        //  Get the 'else' return tensors
        let elseReturnTensors = try graph.getReturnTensors()

        //  Add the if block
        let ifResult = graph.mpsgraph.if(predicateTensor, then: { () -> [MPSGraphTensor] in
                // Block to execute if condition is true (e.g., return Tensor A)
                return thenReturnTensors
            }, else: { () -> [MPSGraphTensor] in
                // Block to execute if condition is true (e.g., return Tensor B)
                return elseReturnTensors
            }, name: graph.getFullName(name))
        
        //  Get suffixes
        if (ifResult.count > 1) {
            for i in 0..<ifResult.count {
                suffixes.append("_\(i)")
            }
        }
        else {
            suffixes = [""]
        }

        return ifResult
    }

    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}

public class Then {
    internal var nodes : [Node]

    internal init(nodes: [Node]) {
        self.nodes = nodes
    }
    ///  Initializer for a Then block with definition nodes
    public convenience init(@ThenElseBuilder _ builtNodes: () -> [Node]) {
        let nodes = builtNodes()
        self.init(nodes: nodes)
    }

}

public class Else {
    internal var nodes : [Node]

    internal init(nodes: [Node]) {
        self.nodes = nodes
    }
    ///  Initializer for an Else block with definition nodes
    public convenience init(@ThenElseBuilder _ builtNodes: () -> [Node]) {
        let nodes = builtNodes()
        self.init(nodes: nodes)
    }

}


///  Then-Else block DSL builder
@resultBuilder
public enum ThenElseBuilder {
    public static func buildBlock(_ nodes: [Node]...) -> [Node] {
        return nodes.flatMap({$0})
    }
    public static func buildExpression(_ expression: Node) -> [Node] {
        return [expression]
    }
    public static func buildOptional(_ component: [Node]?) -> [Node] {
        return component ?? []
    }
    public static func buildEither(first component: [Node]) -> [Node] {
        return component
    }
    public static func buildEither(second component: [Node]) -> [Node] {
        return component
    }
}



//let ifTensor = mpsgraph.constant(0.0, shape: [1 as NSNumber], dataType: .int32)
//let outsideTensor = mpsgraph.constant(1.0, shape: [1 as NSNumber], dataType: .float32)
//let ifResult = mpsgraph.if(ifTensor, then: { () -> [MPSGraphTensor] in
//        // Block to execute if condition is true (e.g., return Tensor A)
//        let thenTensor = mpsgraph.constant(2.0, shape: [1 as NSNumber], dataType: .float32)
//        return [thenTensor]
//    }, else: { () -> [MPSGraphTensor] in
//        // Block to execute if condition is true (e.g., return Tensor A)
//        let elseTensor = mpsgraph.constant(3.0, shape: [1 as NSNumber], dataType: .float32)
//        return [elseTensor]
//    }, name: "if")
//let addition = mpsgraph.addition(outsideTensor, ifResult[0], name: "addition")
