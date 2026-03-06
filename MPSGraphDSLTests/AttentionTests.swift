//
//  AttentionTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 3/3/26.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct AttentionTests {

    @Test func SelfAttentionTest() async throws {
        //  Create an embedding matrix of two tokens with four embedding channels
//        let embeddingTensor = try TensorFloat32(shape: TensorShape([2, 4]), initialValues: [0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1])
        let embeddingTensor = try TensorFloat32(shape: TensorShape([6, 2, 4]), initialValues: [0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1, 0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1, 0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1, 0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1, 0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1, 0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1])

        //  Create the graph
        let graph = Graph(batchSize: 6) {
            PlaceHolder(shape: [2, 4], type: .float32, name: "embedding")
            SelfAttention(headSize: 3, numHeads: 1, name: "attention")
                .targetForModes(["attentionTest"])
        }
        
        //  Run the graph
        let results = try graph.runOne(mode: "attentionTest", inputTensors: ["embedding": embeddingTensor])
        let result = results["attention"]!
        try result.printTensor(elementWidth: 5, precision: 2)
    }

}
