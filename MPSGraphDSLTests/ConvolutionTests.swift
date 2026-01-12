//
//  ConvolutionTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 1/9/26.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct ConvolutionTests {

    let initialValues: [Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    let initial3DValues: [Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                        31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0,
                        61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0]
    let initialChannelValues: [Float32] = [1.0, 31.0, 61.0, 2.0, 32.0, 62.0, 3.0, 33.0, 63.0, 4.0, 34.0, 64.0, 5.0, 35.0, 65.0, 6.0, 36.0, 66.0, 7.0, 37.0, 67.0, 8.0, 38.0, 68.0, 9.0, 39.0, 69.0,
                                           10.0, 40.0, 70.0, 11.0, 41.0, 71.0, 12.0, 42.0, 72.0, 13.0, 43.0, 73.0, 14.0, 44.0, 74.0, 15.0, 45.0, 75.0, 16.0, 46.0, 76.0, 17.0, 47.0, 77.0,
                                           18.0, 48.0, 78.0, 19.0, 49.0, 79.0, 20.0, 50.0, 80.0, 21.0, 51.0, 81.0, 22.0, 52.0, 82.0, 23.0, 53.0, 83.0, 24.0, 54.0, 84.0, 25.0, 55.0, 85.0]

    @Test func HW2D1Filter() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: initialValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 1, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(try result.getElement(index: 0) == 16.0)
        #expect(try result.getElement(index: 6) == 63.0)
        #expect(try result.getElement(index: 12) == 117.0)
        #expect(try result.getElement(index: 18) == 171.0)
        #expect(try result.getElement(index: 24) == 88.0)
    }

    @Test func HW2D2Filter() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: initialValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 2, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        var results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        var result = results["result"]!
        var resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 2)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(try result.getElement(location: [0, 0, 0]) == 16.0)
        #expect(try result.getElement(location: [0, 1, 1]) == 63.0)
        #expect(try result.getElement(location: [0, 2, 2]) == 117.0)
        #expect(try result.getElement(location: [0, 3, 3]) == 171.0)
        #expect(try result.getElement(location: [0, 4, 4]) == 88.0)
        #expect(try result.getElement(location: [1, 0, 0]) == 32.0)
        #expect(try result.getElement(location: [1, 1, 1]) == 126.0)
        #expect(try result.getElement(location: [1, 2, 2]) == 234.0)
        #expect(try result.getElement(location: [1, 3, 3]) == 342.0)
        #expect(try result.getElement(location: [1, 4, 4]) == 176.0)

        let convolutionGraph2 = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 2, name: "result")
                .noBiasTerm()
                .leaveFilterDimensionLast()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        results = try convolutionGraph2.runOne(mode: "convolutionTest", inputTensors: [:])
        result = results["result"]!
        resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 2)
        #expect(try result.getElement(location: [0, 0, 0]) == 16.0)
        #expect(try result.getElement(location: [1, 1, 0]) == 63.0)
        #expect(try result.getElement(location: [2, 2, 0]) == 117.0)
        #expect(try result.getElement(location: [3, 3, 0]) == 171.0)
        #expect(try result.getElement(location: [4, 4, 0]) == 88.0)
        #expect(try result.getElement(location: [0, 0, 1]) == 32.0)
        #expect(try result.getElement(location: [1, 1, 1]) == 126.0)
        #expect(try result.getElement(location: [2, 2, 1]) == 234.0)
        #expect(try result.getElement(location: [3, 3, 1]) == 342.0)
        #expect(try result.getElement(location: [4, 4, 1]) == 176.0)
    }

    @Test func HWC2D1Filter() async throws {
        let smallerValues = initialValues.map { $0 / 10.0 }
        var mergedValues: [Float32] = []
        for i in 0..<initialValues.count {
            mergedValues.append(initialValues[i])
            mergedValues.append(smallerValues[i])
        }
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5, 2]), initialValues: mergedValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 1, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(abs(try result.getElement(index: 0) - 17.6) < 1.0E-04)
        #expect(abs(try result.getElement(index: 6) - 69.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 12) - 128.7) < 1.0E-04)
        #expect(abs(try result.getElement(index: 18) - 188.1) < 1.0E-04)
        #expect(abs(try result.getElement(index: 24) - 96.8) < 1.0E-04)
    }

    @Test func HWC2D2Filter() async throws {
        let smallerValues = initialValues.map { $0 / 10.0 }
        var mergedValues: [Float32] = []
        for i in 0..<initialValues.count {
            mergedValues.append(initialValues[i])
            mergedValues.append(smallerValues[i])
        }
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5, 2]), initialValues: mergedValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 2, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        var results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        var result = results["result"]!
        var resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 2)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(abs(try result.getElement(location: [0, 0, 0]) - 19.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 1]) - 75.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 2, 2]) - 140.4) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 3, 3]) - 205.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 4, 4]) - 105.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 0]) - 19.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1]) - 75.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 2, 2]) - 140.4) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 3, 3]) - 205.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 4, 4]) - 105.6) < 1.0E-04)

        let convolutionGraph2 = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 2, name: "result")
                .noBiasTerm()
                .leaveFilterDimensionLast()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        results = try convolutionGraph2.runOne(mode: "convolutionTest", inputTensors: [:])
        result = results["result"]!
        resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 2)
        #expect(abs(try result.getElement(location: [0, 0, 0]) - 19.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 0]) - 75.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 2, 0]) - 140.4) < 1.0E-04)
        #expect(abs(try result.getElement(location: [3, 3, 0]) - 205.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [4, 4, 0]) - 105.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 1]) - 19.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1]) - 75.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 2, 1]) - 140.4) < 1.0E-04)
        #expect(abs(try result.getElement(location: [3, 3, 1]) - 205.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [4, 4, 1]) - 105.6) < 1.0E-04)
    }

    @Test func NHW2D1Filter() async throws {
        let smallerValues = initialValues.map { $0 / 10.0 }
        let initialTensor = try TensorFloat32(shape: TensorShape([2, 5, 5]), initialValues: initialValues + smallerValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 1, name: "result")
                .noBiasTerm()
                .extraDimensionIsBatch()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 2)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(abs(try result.getElement(location: [0, 0, 0]) - 16.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 1]) - 63.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 2, 2]) - 117.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 3, 3]) - 171.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 4, 4]) - 88.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 0]) - 1.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1]) - 6.3) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 2, 2]) - 11.7) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 3, 3]) - 17.1) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 4, 4]) - 8.8) < 1.0E-04)
    }

    @Test func NHW2D2Filter() async throws {
        let smallerValues = initialValues.map { $0 / 10.0 }
        let initialTensor = try TensorFloat32(shape: TensorShape([2, 5, 5]), initialValues: initialValues + smallerValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 2, name: "result")
                .noBiasTerm()
                .extraDimensionIsBatch()
                .leaveFilterDimensionLast()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        var results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        var result = results["result"]!
        var resultShape = result.shape
        
        #expect(resultShape.numDimensions == 4)
        #expect(resultShape.dimensions[0] == 2)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(resultShape.dimensions[3] == 2)
        #expect(abs(try result.getElement(location: [0, 0, 0, 0]) - 16.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 1, 0]) - 63.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 2, 2, 0]) - 117.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 3, 3, 0]) - 171.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 4, 4, 0]) - 88.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 0, 1]) - 32.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 1, 1]) - 126.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 2, 2, 1]) - 234.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 3, 3, 1]) - 342.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 4, 4, 1]) - 176.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 0, 0]) - 1.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1, 0]) - 6.3) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 2, 2, 0]) - 11.7) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 3, 3, 0]) - 17.1) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 4, 4, 0]) - 8.8) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 0, 1]) - 3.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1, 1]) - 12.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 2, 2, 1]) - 23.4) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 3, 3, 1]) - 34.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 4, 4, 1]) - 17.6) < 1.0E-04)

        let convolutionGraph2 = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 2, name: "result")
                .noBiasTerm()
                .extraDimensionIsBatch()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        results = try convolutionGraph2.runOne(mode: "convolutionTest", inputTensors: [:])
        result = results["result"]!
        resultShape = result.shape
        
        #expect(resultShape.numDimensions == 4)
        #expect(resultShape.dimensions[0] == 2)
        #expect(resultShape.dimensions[1] == 2)
        #expect(resultShape.dimensions[2] == 5)
        #expect(resultShape.dimensions[3] == 5)
        #expect(abs(try result.getElement(location: [0, 0, 0, 0]) - 16.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 1, 1]) - 63.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 2, 2]) - 117.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 3, 3]) - 171.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 4, 4]) - 88.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 0, 0]) - 32.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 1, 1]) - 126.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 2, 2]) - 234.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 3, 3]) - 342.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 4, 4]) - 176.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 0, 0]) - 1.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 1, 1]) - 6.3) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 2, 2]) - 11.7) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 3, 3]) - 17.1) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 4, 4]) - 8.8) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 0, 0]) - 3.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1, 1]) - 12.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 2, 2]) - 23.4) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 3, 3]) - 34.2) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 4, 4]) - 17.6) < 1.0E-04)
    }

    @Test func NHWC2D1Filter() async throws {
        let smallerValues = initialValues.map { $0 / 10.0 }
        var mergedValues: [Float32] = []
        var mergedValuesTimesThree: [Float32] = []
        for i in 0..<initialValues.count {
            mergedValues.append(initialValues[i])
            mergedValues.append(smallerValues[i])
            mergedValuesTimesThree.append(initialValues[i] * 3.0)
            mergedValuesTimesThree.append(smallerValues[i] * 3.0)
        }
        let initialTensor = try TensorFloat32(shape: TensorShape([2, 5, 5, 2]), initialValues: mergedValues+mergedValuesTimesThree)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 1, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 2)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(abs(try result.getElement(location: [0, 0, 0]) - 17.6) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 1]) - 69.3) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 2, 2]) - 128.7) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 3, 3]) - 188.1) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 4, 4]) - 96.8) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 0]) - 52.8) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1]) - 207.9) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 2, 2]) - 386.1) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 3, 3]) - 564.3) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 4, 4]) - 290.4) < 1.0E-04)
    }

    @Test func DHW3D1Filter() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([3, 5, 5]), initialValues: initial3DValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, kernelDepth: 2, numFilters: 1, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 3)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(abs(try result.getElement(location: [0, 0, 0]) - 152.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 1, 1]) - 396.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 2, 2]) - 504.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 3, 3]) - 612.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 4, 4]) - 296.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 0, 0]) - 392.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1]) - 936.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 2, 2]) - 1044.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 3, 3]) - 1152.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 4, 4]) - 536.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 0, 0]) - 256.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 1, 1]) - 603.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 2, 2]) - 657.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 3, 3]) - 711.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 4, 4]) - 328.0) < 1.0E-04)
    }

    @Test func HWC3D1Filter() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5, 3]), initialValues: initialChannelValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, kernelDepth: 2, numFilters: 1, name: "result")
                .noBiasTerm()
                .useChannelAsDepth()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 3)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 3)
        #expect(abs(try result.getElement(location: [0, 0, 0]) - 152.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 0]) - 396.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 2, 0]) - 504.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [3, 3, 0]) - 612.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [4, 4, 0]) - 296.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 1]) - 392.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 1]) - 936.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 2, 1]) - 1044.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [3, 3, 1]) - 1152.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [4, 4, 1]) - 536.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [0, 0, 2]) - 256.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [1, 1, 2]) - 603.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [2, 2, 2]) - 657.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [3, 3, 2]) - 711.0) < 1.0E-04)
        #expect(abs(try result.getElement(location: [4, 4, 2]) - 328.0) < 1.0E-04)
    }

    @Test func DHW3D2Filter() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([3, 5, 5]), initialValues: initial3DValues)
        
        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, kernelDepth: 2, numFilters: 2, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        var results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        var result = results["result"]!
        var resultShape = result.shape
        
        #expect(resultShape.numDimensions == 4)
        #expect(resultShape.dimensions[0] == 2)
        #expect(resultShape.dimensions[1] == 3)
        #expect(resultShape.dimensions[2] == 5)
        #expect(resultShape.dimensions[3] == 5)
        
        let convolutionGraph2 = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, kernelDepth: 2, numFilters: 2, name: "result")
                .noBiasTerm()
                .leaveFilterDimensionLast()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        results = try convolutionGraph2.runOne(mode: "convolutionTest", inputTensors: [:])
        result = results["result"]!
        resultShape = result.shape
        
        #expect(resultShape.numDimensions == 4)
        #expect(resultShape.dimensions[0] == 3)
        #expect(resultShape.dimensions[1] == 5)
        #expect(resultShape.dimensions[2] == 5)
        #expect(resultShape.dimensions[3] == 2)
    }

    @Test func Bias() async throws {
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: initialValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 1, name: "result")
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(abs(try result.getElement(index: 0) - 16.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 6) - 63.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 12) - 117.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 18) - 171.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 24) - 88.3) < 1.0E-04)
    }

    @Test func Activation() async throws {
        var activationValues = initialValues
        activationValues.removeLast()
        activationValues.append(-100.0)
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: activationValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 1, activationFunction: .relu, name: "result")
                .noBiasTerm()
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(abs(try result.getElement(index: 0) - 16.0) < 1.0E-04)
        #expect(abs(try result.getElement(index: 6) - 63.0) < 1.0E-04)
        #expect(abs(try result.getElement(index: 12) - 117.0) < 1.0E-04)
        #expect(abs(try result.getElement(index: 24)) < 1.0E-04)
    }

    @Test func BiasAndActivation() async throws {
        var activationValues = initialValues
        activationValues.removeLast()
        activationValues.append(-100.0)
        let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: activationValues)

        let convolutionGraph = Graph {
            Constant(values: initialTensor, name: "startTensor")
            ConvolutionLayer(kernelHeight: 3, kernelWidth: 3, numFilters: 1, activationFunction: .relu, name: "result")
                .useTestWeights()
                .targetForModes(["convolutionTest"])
        }
        
        let results = try convolutionGraph.runOne(mode: "convolutionTest", inputTensors: [:])
        let result = results["result"]!
        let resultShape = result.shape
        
        #expect(resultShape.numDimensions == 2)
        #expect(resultShape.dimensions[0] == 5)
        #expect(resultShape.dimensions[1] == 5)
        #expect(abs(try result.getElement(index: 0) - 16.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 6) - 63.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 12) - 117.3) < 1.0E-04)
        #expect(abs(try result.getElement(index: 24)) < 1.0E-04)
    }

}
