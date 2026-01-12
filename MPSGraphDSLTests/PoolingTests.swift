//
//  PoolingTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 1/11/26.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct PoolingTests {

    let initialValues: [Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                        31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
                        61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0]
    let initialChannelValues: [Float32] = [1.0, 31.0, 61.0, 2.0, 32.0, 62.0, 3.0, 33.0, 63.0, 4.0, 34.0, 64.0, 5.0, 35.0, 65.0, 6.0, 36.0, 66.0, 7.0, 37.0, 67.0, 8.0, 38.0, 68.0, 9.0, 39.0, 69.0,
                                           10.0, 40.0, 70.0, 11.0, 41.0, 71.0, 12.0, 42.0, 72.0, 13.0, 43.0, 73.0, 14.0, 44.0, 74.0, 15.0, 45.0, 75.0, 16.0, 46.0, 76.0, 17.0, 47.0, 77.0,
                                           18.0, 48.0, 78.0, 19.0, 49.0, 79.0, 20.0, 50.0, 80.0, 21.0, 51.0, 81.0, 22.0, 52.0, 82.0, 23.0, 53.0, 83.0, 24.0, 54.0, 84.0, 25.0, 55.0, 85.0]

    @Test func max_2D_HW() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: initialValues)

        let poolingGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            PoolingLayer(function: .max, kernelHeight: 3, kernelWidth: 3, name: "result")
                .targetForModes(["poolingTest"])
        }
        
        let results = try poolingGraph.runOne(mode: "poolingTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(try result.getElement(index: 0) == 7.0)
        #expect(try result.getElement(index: 6) == 13.0)
        #expect(try result.getElement(index: 12) == 19.0)
        #expect(try result.getElement(index: 18) == 25.0)
        #expect(try result.getElement(index: 24) == 25.0)
    }

    @Test func max_2D_HWC() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5, 3]), initialValues: initialChannelValues)

        let poolingGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            PoolingLayer(function: .max, kernelHeight: 3, kernelWidth: 3, name: "result")
                .targetForModes(["poolingTest"])
        }
        
        let results = try poolingGraph.runOne(mode: "poolingTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 3)
        #expect(try result.getElement(location: [0, 0, 0]) == 7.0)
        #expect(try result.getElement(location: [1, 1, 0]) == 13.0)
        #expect(try result.getElement(location: [2, 2, 0]) == 19.0)
        #expect(try result.getElement(location: [3, 3, 0]) == 25.0)
        #expect(try result.getElement(location: [4, 4, 0]) == 25.0)
        #expect(try result.getElement(location: [0, 0, 1]) == 37.0)
        #expect(try result.getElement(location: [1, 1, 1]) == 43.0)
        #expect(try result.getElement(location: [2, 2, 1]) == 49.0)
        #expect(try result.getElement(location: [3, 3, 1]) == 55.0)
        #expect(try result.getElement(location: [4, 4, 1]) == 55.0)
        #expect(try result.getElement(location: [0, 0, 2]) == 67.0)
        #expect(try result.getElement(location: [1, 1, 2]) == 73.0)
        #expect(try result.getElement(location: [2, 2, 2]) == 79.0)
        #expect(try result.getElement(location: [3, 3, 2]) == 85.0)
        #expect(try result.getElement(location: [4, 4, 2]) == 85.0)
    }

    @Test func max_2D_NHW() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([3, 5, 5]), initialValues: initialValues)

        let poolingGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            PoolingLayer(function: .max, kernelHeight: 3, kernelWidth: 3, name: "result")
                .extraDimensionIsBatch()
                .targetForModes(["poolingTest"])
        }
        
        let results = try poolingGraph.runOne(mode: "poolingTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 3)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(try result.getElement(location: [0, 0, 0]) == 7.0)
        #expect(try result.getElement(location: [0, 1, 1]) == 13.0)
        #expect(try result.getElement(location: [0, 2, 2]) == 19.0)
        #expect(try result.getElement(location: [0, 3, 3]) == 25.0)
        #expect(try result.getElement(location: [0, 4, 4]) == 25.0)
        #expect(try result.getElement(location: [1, 0, 0]) == 37.0)
        #expect(try result.getElement(location: [1, 1, 1]) == 43.0)
        #expect(try result.getElement(location: [1, 2, 2]) == 49.0)
        #expect(try result.getElement(location: [1, 3, 3]) == 55.0)
        #expect(try result.getElement(location: [1, 4, 4]) == 55.0)
        #expect(try result.getElement(location: [2, 0, 0]) == 67.0)
        #expect(try result.getElement(location: [2, 1, 1]) == 73.0)
        #expect(try result.getElement(location: [2, 2, 2]) == 79.0)
        #expect(try result.getElement(location: [2, 3, 3]) == 85.0)
        #expect(try result.getElement(location: [2, 4, 4]) == 85.0)
    }

    @Test func avg_2D_HW() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: initialValues)

        let poolingGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            PoolingLayer(function: .avg, kernelHeight: 3, kernelWidth: 3, name: "result")
                .targetForModes(["poolingTest"])
        }
        
        let results = try poolingGraph.runOne(mode: "poolingTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(abs(try result.getElement(index: 0) - 4.0) < 1.0E-4)
        #expect(abs(try result.getElement(index: 6) - 7.0) < 1.0E-4)
        #expect(abs(try result.getElement(index: 12) - 13.0) < 1.0E-4)
        #expect(abs(try result.getElement(index: 18) - 19.0) < 1.0E-4)
        #expect(abs(try result.getElement(index: 24) - 22.0) < 1.0E-4)
    }

    @Test func min_2D_HW() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: initialValues)

        let poolingGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            PoolingLayer(function: .min, kernelHeight: 3, kernelWidth: 3, name: "result")
                .targetForModes(["poolingTest"])
        }
        
        let results = try poolingGraph.runOne(mode: "poolingTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(try result.getElement(index: 0) == 1.0)
        #expect(try result.getElement(index: 6) == 1.0)
        #expect(try result.getElement(index: 12) == 7.0)
        #expect(try result.getElement(index: 18) == 13.0)
        #expect(try result.getElement(index: 24) == 19.0)
    }
}
