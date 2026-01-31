//
//  SubGraph.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/19/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class SubGraphDefinition {
    internal var nodes : [Node]

    internal init(nodes: [Node]) {
        self.nodes = nodes
    }
    
    internal func clearReferencedFlags() {
        for node in nodes {
            node.clearReferencedFlag()
        }
    }
    
    internal func nodesAreReferenced() throws {
        for node in nodes {
            try node.isReferenced()
        }

    }
}


///   Node that adds an instance of a SubGraphDefinition to another Graph or SubGraph
public class SubGraph : Node {
    let subGraphDef: SubGraphDefinition
    let inputMap: [String : String?]
    let dataTensorMap: [String : Tensor]
    
    /// Modifier for a Node to indicate it is a target for the specified modes.  For a subgraph this results in an error
    /// - Parameter modes: The modes that this node is a target for
    /// - Returns: The modified Node
    override public func targetForModes(_ modes: [String]) -> Node {
        buildError = MPSGraphDSLErrors.TargetNodesMustBeNamed
        return self
    }
    
    
    override var addToNodeList: Bool { return false }

    ///  Constructor for a negative (negation) operation
    ///
    /// - Parameters:
    ///   - definition: The SubGraphDefinition to be instantiated
    ///   - name:  The name for this node and its associated tensor
    ///   - inputMap:  Dictionary with mapping of SubGraphPlaceHolder names to parent Graph or SubGraph nodes to be used as inputs to the SubGraph
    public init(definition: SubGraphDefinition, name: String, inputMap: [String : String?], dataTensorMap: [String : Tensor] = [:]) {
        subGraphDef = definition
        self.inputMap = inputMap
        self.dataTensorMap = dataTensorMap
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Remember the old input and data tensor map and name prefix so we can put them back later
        let oldInputMap = graph.currentSubGraphInputMap
        let oldDataTensorMap = graph.dataTensorMap
        
        //  Install our input map
        graph.currentSubGraphInputMap = inputMap
        
        //  Install our data tensor map
        graph.dataTensorMap = dataTensorMap

        //  Put this node name onto the name prefix
        graph.setNewCurrentPrefix(name! + "_" + graph.currentNamePrefix)
        
        //  Add these nodes to the graph
        try graph.processNodes(subGraphDef.nodes)
        
        //  Put back the previous input map and prefix
        graph.currentSubGraphInputMap = oldInputMap
        graph.popLastFromPrefixStack()
        
        //  Put back the previous data tensor map
        graph.dataTensorMap = oldDataTensorMap
        
        return [nil]
    }
    
    //  Clear the referenced flag
    override internal func clearReferencedFlag() {
        referencedByAnotherNode = false
        
        //  Clear all the sub-graph nodes
        subGraphDef.clearReferencedFlags()
    }
    
    //  Verify the subgraph nodes are referenced
    override internal func isReferenced() throws {
        try subGraphDef.nodesAreReferenced()
    }
}
