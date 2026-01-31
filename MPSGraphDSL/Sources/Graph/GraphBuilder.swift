//
//  GraphBuilder.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/18/25.
//

import Foundation

///  Graph DSL builder
@resultBuilder
public enum GraphBuilder {
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


extension Graph {
    ///  Initializer for a Graph with definition nodes
    public convenience init(batchSize: Int = 1, buildOptions: BuildOptions = [], @GraphBuilder _ builtNodes: () -> [Node]) {
        let nodes = builtNodes()
        self.init(batchSize: batchSize, buildOptions: buildOptions, nodes: nodes)
    }
}


///  SubGraph DSL builder
@resultBuilder
public enum SubGraphBuilder {
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


extension SubGraphDefinition {
    ///  Initializer for a SubGraph with definition nodes
    public convenience init(@SubGraphBuilder _ builtNodes: () -> [Node]) {
        let nodes = builtNodes()
        self.init(nodes: nodes)
    }
}
