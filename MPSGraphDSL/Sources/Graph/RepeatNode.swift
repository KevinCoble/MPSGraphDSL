//
//  RepeatNode.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 3/6/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///   Node that adds a repeating block of nodes to the Graph
public class Repeat : Node {
    let numTimes: Int
    let nodes: [Node]
    
    internal init(_ numTimes: Int, nodes: [Node], name: String? = nil) {
        self.numTimes = numTimes
        self.nodes = nodes
        super.init(name: name)
    }
    
    ///  Initializer for a repeat block with definition nodes
    public convenience init(_ numTimes: Int, name: String? = nil, @RepeatBuilder _ builtNodes: () -> [Node]) {
        let nodes = builtNodes()
        self.init(numTimes, nodes: nodes, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        if (numTimes < 1 || numTimes > 20) { throw GenericMPSGraphDSLErrors.InvalidInRepeatCount }
        
        //  Get the block name for the repeat
        let blockName: String
        if let name = name {
            blockName = name + "_"
        }
        else {
            blockName = "*repeat*_"
        }
        
        //  Find any RepeatTensorNames and remove from the block list
        var foundOne: Bool = true
        var filteredNodes = nodes
        var repeatReferences: [String: (String, String)] = [:]
        while foundOne {
            foundOne = false
            for i in 0..<filteredNodes.count {
                if filteredNodes[i] is RepeatTensorName {
                    foundOne = true
                    let rtn = nodes[i] as! RepeatTensorName
                    if (repeatReferences[rtn.referenceName] != nil) { throw MPSGraphDSLErrors.RepeatTensorNameReferenceNameDuplicated }
                    if (try graph.findNamedNode(rtn.initialName) == nil) { throw MPSGraphDSLErrors.RepeatTensorNameInitialNameNotFound }
                    repeatReferences[rtn.referenceName] = (rtn.initialName, rtn.repeatName)
                    filteredNodes.remove(at: i)
                    break
                }
            }
        }
        
        //  Install our repeatReference list
        let previousRepeatReferences = graph.repeatReferences
        graph.repeatReferences = repeatReferences
        
        //  Add the nodes multiple times
        let previousFirstRepeatBlock = graph.firstRepeatBlock
        for i in 0..<numTimes {
            //  Set the 'firstRepeatBlock' flag used by RepeatTensorName nodes
            graph.firstRepeatBlock = (i == 0)
            
            //  Put this node name onto the name prefix
            graph.setNewCurrentPrefix(blockName + "\(i+1)_" + graph.currentNamePrefix)
            
            //  Process the nodes
            try graph.processNodes(filteredNodes, block: blockName + "\(i+1)")
            
            //  Leave the last two prefixes in place
            if (i != 0) { graph.removeSecondFromLastFromPrefixStack() }
        }
        //  Put back the previous prefix
        graph.popLastFromPrefixStack()
        graph.firstRepeatBlock = previousFirstRepeatBlock
        graph.repeatReferences = previousRepeatReferences
        
        return []
    }
    
    override internal func isReferenced() throws {
        //  Don't throw
    }
}


///  Repeat block DSL builder
@resultBuilder
public enum RepeatBuilder {
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

///   Node that adds a circular reference to a repeat block
public class RepeatTensorName : Node {
    let initialName: String
    let repeatName: String
    let referenceName: String

    public init(initialName: String, repeatName: String, referenceName: String) {
        self.initialName = initialName
        self.repeatName = repeatName
        self.referenceName = referenceName
        super.init(name: nil)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        throw MPSGraphDSLErrors.RepeatTensorNameUsedOutsideOfRepeatBlock
    }
}
