//
//  MPSGraphDSLTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 11/17/25.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct MPSGraphDSLTests {

    @Test func basicGraphBuilderTest() async throws {
        do {
            let graph = Graph {
                PlaceHolder(shape: [2], type: .float32, name: "input")
                Negative(name: "result")
                    .targetForModes(["runTest"])
            }
            
            let inputTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [3.0, 4.0])

            let inputTensorList: [String: Tensor] = ["input": inputTensor]
            let results = try graph.runOne(mode: "runTest", inputTensors: inputTensorList)
            
            #expect(results.count == 1)
            let result = results["result"]
            #expect(result != nil)
            #expect(result!.shape == TensorShape([2]))
            #expect(try result!.getElement(index: 0) == -3.0)
            #expect(try result!.getElement(index: 1) == -4.0)
        }
    }
    
    @Test func namedNodeGraphBuilderTest() async throws {
        do {
            let graph = Graph {
                PlaceHolder(shape: [2], type: .float32, name: "input")
                Negative(input: "input", name: "result")
                    .targetForModes(["runTest"])
            }
            
            let inputTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [3.0, 4.0])

            let inputTensorList: [String: Tensor] = ["input": inputTensor]
            let results = try graph.runOne(mode: "runTest", inputTensors: inputTensorList)

            #expect(results.count == 1)
            let result = results["result"]
            #expect(result != nil)
            #expect(result!.shape == TensorShape([2]))
            #expect(try result!.getElement(index: 0) == -3.0)
            #expect(try result!.getElement(index: 1) == -4.0)
        }
    }
    
    @Test func binaryNodeGraphBuilderTest() async throws {
        do {
            let constantTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [-1.0, 3.0])
            let graph = Graph {
                PlaceHolder(shape: [2], type: .float32, name: "input")
                Constant(values: constantTensor, name: "constant")
                Addition(firstInput: "input", secondInput: "constant", name: "result")
                    .targetForModes(["runTest"])
            }
            
            let inputTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [6.0, 7.0])

            let inputTensorList: [String: Tensor] = ["input": inputTensor]
            let results = try graph.runOne(mode: "runTest", inputTensors: inputTensorList)

            #expect(results.count == 1)
            let result = results["result"]
            #expect(result != nil)
            #expect(result!.shape == TensorShape([2]))
            #expect(try result!.getElement(index: 0) == 5.0)
            #expect(try result!.getElement(index: 1) == 10.0)
        }
    }
    
    @Test func subGraphBuilderTest() async throws {
        do {
            let subGraph = SubGraphDefinition {
                SubGraphPlaceHolder(name: "subInput")
                Constant(shape: [2], value: Float32(3.0), name: "constant")
                Addition(firstInput: "input", secondInput: "constant", name: "result")
                    .targetForModes(["runTest"])
            }
            
            let subgraphMap : [String : String?] = ["subInput" : "input"]       //  Map "input" node of graph to "subInput" placeholder of subgraph
            let graph = Graph {
                PlaceHolder(shape: [2], type: .float32, name: "input")
                SubGraph(definition: subGraph, name: "subgraph", inputMap: subgraphMap)
            }
            
            let inputTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [6.0, 7.0])

            let inputTensorList: [String: Tensor] = ["input": inputTensor]
            let results = try graph.runOne(mode: "runTest", inputTensors: inputTensorList)

            #expect(results.count == 1)
            let result = results["subgraph_result"]
            #expect(result != nil)
            #expect(result!.shape == TensorShape([2]))
            #expect(try result!.getElement(index: 0) == 9.0)
            #expect(try result!.getElement(index: 1) == 10.0)
        }
        
        do {
            let dataTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [-1.0, 3.0])
            
            let subGraph = SubGraphDefinition {
                SubGraphPlaceHolder(name: "subInput")
                Constant(tensorReference: "data", name: "constant")
                Addition(firstInput: "input", secondInput: "constant", name: "result")
                    .targetForModes(["runTest"])
            }
            
            let subgraphMap : [String : String?] = ["subInput" : "input"]       //  Map "input" node of graph to "subInput" placeholder of subgraph
            let dataTensorMap : [String : Tensor] = ["data" : dataTensor]       //  Map the data tensor with reference string "data"
            let graph = Graph {
                PlaceHolder(shape: [2], type: .float32, name: "input")
                SubGraph(definition: subGraph, name: "subgraph", inputMap: subgraphMap, dataTensorMap: dataTensorMap)
            }
            
            let inputTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [6.0, 7.0])

            let inputTensorList: [String: Tensor] = ["input": inputTensor]
            let results = try graph.runOne(mode: "runTest", inputTensors: inputTensorList)

            #expect(results.count == 1)
            let result = results["subgraph_result"]
            #expect(result != nil)
            #expect(result!.shape == TensorShape([2]))
            #expect(try result!.getElement(index: 0) == 5.0)
            #expect(try result!.getElement(index: 1) == 10.0)
        }
    }
    
    @Test func targetTensorsTest() async throws {
        do {
            let graph = Graph {
                PlaceHolder(shape: [2], type: .float32, name: "input")
                Square(name: "setTarget")
                    .targetForModes(["runTest"])
                SquareRoot()
                Negative(name: "defaultTarget")
                    .targetForModes(["runTest"])
            }

            try graph.buildGraph()
            
            //  Verify there are two targets
            #expect(graph.targetTensors.count == 2)
        }
    }
    
    @Test func additionTest() async throws {
        //---------------------
        //  Build graph
        //---------------------
        
        //  Create the graph
        let mpsgraph = MPSGraph()
        
        //  Create the constant array
        let twos = mpsgraph.constant(2.0, shape: [5, 5, 5], dataType: .float32)
        
        //  Create a constant 5 threes
        let addTensor = try TensorFloat32(shape: TensorShape([5]), initialValues: [2, 4, 6, 8, 10])
        let data = addTensor.getData()
        let addend = mpsgraph.constant(data, shape: [5], dataType: .float32)
        
        //  add the two
        let addition = mpsgraph.addition(twos, addend, name: nil)
        
        //---------------------
        //  Get Metal device
        //---------------------
        
//        let device = MTLCreateSystemDefaultDevice()!

        //---------------------
        //  Get input data
        //---------------------
        
//        //  Get an input tensor
//        let descriptor = MPSNDArrayDescriptor(dataType: .float32, shape: [2])
//        var inputValues : [Float32] = [3.0, 4.0]
//        let inputNDArray = MPSNDArray(device: device, descriptor: descriptor)
//        inputNDArray.writeBytes(&inputValues, strideBytes: nil)
//        let inputs = MPSGraphTensorData(inputNDArray)

        //---------------------
        //  Run graph
        //---------------------
        
        // Execute the graph.
        let results = mpsgraph.run(feeds: [:],
                                targetTensors: [addition],
                                targetOperations: nil)

        let result = results[addition]

        //---------------------
        //  Extract the result values
        //---------------------
        let outputNDArray = result?.mpsndarray()
        var outputValues: [Float32] = Array.init(repeating: Float32(0), count: 125)
        outputNDArray?.readBytes(&outputValues, strideBytes: nil)
        
        #expect(outputValues.count == 125)
        #expect(outputValues[0] == 4.0)
        #expect(outputValues[1] == 6.0)
        #expect(outputValues[2] == 8.0)
        #expect(outputValues[3] == 10.0)
        #expect(outputValues[4] == 12.0)
        #expect(outputValues[5] == 4.0)
        #expect(outputValues[6] == 6.0)
        #expect(outputValues[7] == 8.0)
        #expect(outputValues[8] == 10.0)
        #expect(outputValues[9] == 12.0)

    }
    
    @Test func repeatBlockTest() async throws {
        
        //  Case where unnamed previous reference is used
        let graph = Graph {
            Constant(shape: [1], value: Float32(2), name: "two")
            Constant(shape: [1], value: Float32(3), name: "three")
            Repeat(3) {
                Addition(secondInput: "two", name: "addition")
            }
            Identity(name: "result")        //  Get the last repeat node and make it a target
                .targetForModes(["repeatTest"])
        }
        
        // Execute the graph.
        let results = try graph.runOne(mode: "repeatTest", inputTensors: [:])
        let result = results["result"]!
        
        //  Three + (2 + 2 + 2) - from repeat block, is 9
        let value = try result.getElement(index: 0)
        #expect(value == 9.0)
        
        //  Case where named circular reference is used
        let graph2 = Graph {
            Constant(shape: [1], value: Float32(2), name: "two")
            Constant(shape: [1], value: Float32(3), name: "three")
            Repeat(3) {
                RepeatTensorName(initialName: "two", repeatName: "addition", referenceName: "blockInput")
                Addition(firstInput: "blockInput", secondInput: "three", name: "addition")
            }
            Identity(name: "result")        //  Get the last repeat node and make it a target
                .targetForModes(["repeatTest"])
        }
        
        // Execute the graph.
        let results2 = try graph2.runOne(mode: "repeatTest", inputTensors: [:])
        let result2 = results2["result"]!
        
        //  two + (3 + 3 + 3) - from repeat block, is 11
        let value2 = try result2.getElement(index: 0)
        #expect(value2 == 11.0)

    }
}

