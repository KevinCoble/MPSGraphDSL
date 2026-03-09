//
//  Node.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/18/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///    Parent class for all graph nodes - do not instantiate
public class Node {
    let name: String?
    var targetModes : [String] = []
    var targetSubTensors: [String] = []
    var returnIndex : Int?          //  If not nil, return from control block index
    var buildError : Error? = nil
    var referencedByAnotherNode: Bool = false
    
    init(name: String? = nil) {
        self.name = name
    }
    
    /// Modifier for a Node to indicate it is a target for the specified modes
    /// - Parameters:
    ///   - modes: The modes that this node is a target for
    ///   - subTensors: (Optional)  The suffixes for node sub-tensors that are not automatically added to the target list, but you want targetted anyway
    /// - Returns: The modified Node
    public func targetForModes(_ modes: [String], subTensors: [String] = []) -> Node {
        if (name == nil) {
            buildError = MPSGraphDSLErrors.TargetNodesMustBeNamed
        }
        targetModes = modes
        targetSubTensors = subTensors
        return self
    }
    
    /// Modifier for a Node to indicate it is to be returned from a control block (if, for, while).  By default only the last node is returned
    /// - Parameter index: The position in the tensor return list for the node
    /// - Returns: The modified Node
    public func blockReturnIndex(_ index: Int?) -> Node {
        returnIndex = index
        return self
    }

    var addToNodeList: Bool { return true }
    
    internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        fatalError(">>> addToGraph not allowed for Node - must be subclass")
    }
    
    //  Get the suffixes for names of multiple-result output nodes
    internal func getNodeSuffixes() -> [String] {
        return [""]
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    internal func getTargetIndices() -> [Int]? {
        return nil
    }
    
    //  Clear the referenced flag
    internal func clearReferencedFlag() {
        referencedByAnotherNode = false
    }
    
    //  Verify we are referenced
    internal func isReferenced() throws {
        if (!referencedByAnotherNode && targetModes.isEmpty) {
            var nameString = "* Unnamed *"
            if let name = name {
                nameString = name
            }
            var opName = String(describing: self)
            if (opName.starts(with: "MPSGraphDSL.")) { opName = String(opName.trimmingPrefix("MPSGraphDSL.")) }
            throw MPSGraphDSLErrors.UnreferencedNode("Node: \(opName) name: \(nameString)")
        }
    }
    
    internal func getNumberOfParameters() throws -> Int {
        //  Nodes can override this to get the parameters they (or their-sub-nodes) add to the graph
        return 0
    }
}
