//
//  ControlTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 2/24/26.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct ControlTests {

    @Test func BasicIf() async throws {
        
        let graph = Graph {
            PlaceHolder(shape: [1], type: .int32, name: "predicate")
            
            If("predicate", Then {
                Constant(shape: [1], value: Int32(3))
            },
            Else {
                Constant(shape: [1], value: Int32(5))
            },
               name: "if")
                .targetForModes(["ifTest"])
        }
        
        //  Run the graph with a false (0) predicate
        var predicateTensor = TensorInt32(shape: TensorShape([1]), initialValue: 0)
        var results = try graph.runOne(mode: "ifTest", inputTensors: ["predicate": predicateTensor])
        var result = results["if"]! as! TensorInt32
        var index: Int32 = try result.getElement(index: 0)
        #expect(index ==  5)
        
        //  Run the graph with a true (1) predicate
        predicateTensor = TensorInt32(shape: TensorShape([1]), initialValue: 1)
        results = try graph.runOne(mode: "ifTest", inputTensors: ["predicate": predicateTensor])
        result = results["if"]! as! TensorInt32
        index = try result.getElement(index: 0)
        #expect(index ==  3)
    }

    @Test func MultipleReturnIf() async throws {
        
        let graph = Graph {
            PlaceHolder(shape: [1], type: .int32, name: "predicate")
            
            If("predicate", Then {
                Constant(shape: [1], value: Int32(3))
                    .blockReturnIndex(0)
                Constant(shape: [1], value: Int32(4))
                    .blockReturnIndex(1)
           },
            Else {
                Constant(shape: [1], value: Int32(5))
                    .blockReturnIndex(1)
                Constant(shape: [1], value: Int32(6))
                    .blockReturnIndex(0)
            },
               name: "if")
                .targetForModes(["ifTest"])
        }
        
        //  Run the graph with a false (0) predicate
        var predicateTensor = TensorInt32(shape: TensorShape([1]), initialValue: 0)
        var results = try graph.runOne(mode: "ifTest", inputTensors: ["predicate": predicateTensor])
        var result = results["if_0"]! as! TensorInt32
        var index: Int32 = try result.getElement(index: 0)
        #expect(index ==  6)
        result = results["if_1"]! as! TensorInt32
        index = try result.getElement(index: 0)
        #expect(index ==  5)

        //  Run the graph with a true (1) predicate
        predicateTensor = TensorInt32(shape: TensorShape([1]), initialValue: 1)
        results = try graph.runOne(mode: "ifTest", inputTensors: ["predicate": predicateTensor])
        result = results["if_0"]! as! TensorInt32
        index = try result.getElement(index: 0)
        #expect(index ==  3)
        result = results["if_1"]! as! TensorInt32
        index = try result.getElement(index: 0)
        #expect(index ==  4)
    }

}
