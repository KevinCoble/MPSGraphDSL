//
//  RNNTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 12/31/25.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct RNNTests {

    @Test func TestRNNLayer() async throws {
        do {
            let inputShape = TensorShape([2, 1, 1])
            let outputShape = TensorShape([1])
            let dataSet = DataSet(inputShape: inputShape, inputType: .float32, outputShape: outputShape, outputType: .float32)
            for i in 0...3 {
                for j in 0...3 {
                    let sequence: [Float32] = [Float32(i), Float32(j)]
                    let output: Float32 = (i==1 && j==2) ? 1.0 : -1.0
                    
                    let inputTensor = try TensorFloat32(shape: inputShape, initialValues: sequence)
                    let outputTensor = TensorFloat32(shape: outputShape, initialValue: output)
                    let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                    try await dataSet.appendSample(sample)
                }
            }
            
            //  Create the graph
            let graph = Graph {
                PlaceHolder(shape: [2, 1, 1], name: "input")
                RNNLayer(input: "input", stateSize: 1, name: "RNN")
                    .learnWithRespectTo("loss")
                    .targetForModes(["infer"])
                PlaceHolder(shape: [1], modes: ["learn"], name: "expectedValue")
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "RNN_lastState", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
            }
            
            //  See how many have the correct sign
            var numInitialCorrect: Int = 0
            let numSamples = await dataSet.numSamples
            for i in 0..<numSamples {
                let sample = try await dataSet.getSample(sampleIndex: i)
                let results = try graph.runOne(mode: "infer", inputTensors: ["input": sample.inputs])
                let result = results["RNN_lastState"]!
                let predictedValue = try result.getElement(index: 0)
                let expectedValue = try sample.outputs.getElement(index: 0)
                if ((predictedValue < 0.0 && expectedValue < 0.0) || (predictedValue > 0.0 && expectedValue > 0.0)) {
                    numInitialCorrect += 1
                }
            }
            
            //  Train
            for _ in 0..<200 {
                try await graph.runTraining(mode: "learn", trainingDataSet: dataSet, inputTensorName: "input", expectedValueTensorName : "expectedValue")
            }

            //  Now see how many have the correct sign
            var numFinalCorrect: Int = 0
            for i in 0..<numSamples {
                let sample = try await dataSet.getSample(sampleIndex: i)
                let results = try graph.runOne(mode: "infer", inputTensors: ["input": sample.inputs])
                let result = results["RNN_lastState"]!
                let predictedValue = try result.getElement(index: 0)
                let expectedValue = try sample.outputs.getElement(index: 0)
                if ((predictedValue < 0.0 && expectedValue < 0.0) || (predictedValue > 0.0 && expectedValue > 0.0)) {
                    numFinalCorrect += 1
                }
            }
            
            #expect(numFinalCorrect > numInitialCorrect)
        }
    }

    @Test func TestLSTMLayer() async throws {
        do {
            let inputShape = TensorShape([3, 1, 1])
            let outputShape = TensorShape([1])
            let dataSet = DataSet(inputShape: inputShape, inputType: .float32, outputShape: outputShape, outputType: .float32)
            for i in 0...6 {
                var sequence: [Float32] = []
                for j in 0..<3  {
                    sequence.append(Float32(i + j))
                }
                let output = Float32(i+3)
                
                let inputTensor = try TensorFloat32(shape: inputShape, initialValues: sequence)
                let outputTensor = TensorFloat32(shape: outputShape, initialValue: output)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await dataSet.appendSample(sample)
            }
            let testInput: [Float32] = [7.0, 8.0, 9.0]
            let testOutput: Float32 = 10.0
            let testInputTensor = try TensorFloat32(shape: inputShape, initialValues: testInput)
            
            //  Create the graph
            let graph = Graph {
                PlaceHolder(shape: [3, 1, 1], name: "input")
                LSTMLayer(input: "input", stateSize: 20, name: "LSTM")
                    .allActivationFunctions(.relu)
                    .learnWithRespectTo("loss")
                FullyConnectedLayer(input: "LSTM_lastState", outputShape: TensorShape([1]), activationFunction: .none, name: "result")
                    .learnWithRespectTo("loss")
                    .targetForModes(["infer"])
                PlaceHolder(shape: [1], modes: ["learn"], name: "expectedValue")
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
            }
            
            //  Run an input in
            var results = try graph.runOne(mode: "infer", inputTensors: ["input": testInputTensor])
            
            //  Determine the before-training error
            var testResults = results["result"]!
            var testValue = try testResults.getElement(index: 0)
            let initialError = abs(testValue - Double(testOutput))
            
            //  Train
            for _ in 0..<200 {
                try await graph.runTraining(mode: "learn", trainingDataSet: dataSet, inputTensorName: "input", expectedValueTensorName : "expectedValue")
            }

            //  Run an input in
            results = try graph.runOne(mode: "infer", inputTensors: ["input": testInputTensor])
            
            //  Determine the after-training error
            testResults = results["result"]!
            testValue = try testResults.getElement(index: 0)
            let finalError = abs(testValue - Double(testOutput))
            
            //  Verify we got better
            if (!initialError.isNaN && !finalError.isNaN) {
                #expect(initialError > (finalError * 3.0))
            }

        }
        catch let error {
            Issue.record("Issue with LSTMLayer test - \(error.localizedDescription)")
        }
    }

    @Test(.disabled("learning error with GRULayer"))
    func TestGRULayer() async throws {
        do {
            //  Create a dataset with all combinations of three numbers in range 0-4.  If there is a 1 followed by a 3, output is 1, else -1
            let inputShape = TensorShape([3, 1, 1])
            let outputShape = TensorShape([1])
            let dataSet = DataSet(inputShape: inputShape, inputType: .float32, outputShape: outputShape, outputType: .float32)
            for i in 0...4 {
                for j in 0...4 {
                    for k in 0...4 {
                        let sequence: [Float32] = [Float32(i), Float32(j), Float32(k)]
                        var output: Float32 = -1.0
                        if (i == 1 && j == 3) { output = 1.0 }
                        if (i == 1 && k == 3) { output = 1.0 }
                        if (j == 1 && k == 3) { output = 1.0 }

                        let inputTensor = try TensorFloat32(shape: inputShape, initialValues: sequence)
                        let outputTensor = TensorFloat32(shape: outputShape, initialValue: output)
                        let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                        try await dataSet.appendSample(sample)
                    }
                }
            }
            
            //  Create the graph
            let graph = Graph {
                PlaceHolder(shape: [3, 1, 1], name: "input")
                GRULayer(input: "input", stateSize: 1, name: "GRU")
                    .learnWithRespectTo("loss")
                    .targetForModes(["infer"])
                PlaceHolder(shape: [1], modes: ["learn"], name: "expectedValue")
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "GRU_lastState", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
            }
            
            //  See how many have the correct sign
            var numInitialCorrect: Int = 0
            let numSamples = await dataSet.numSamples
            for i in 0..<numSamples {
                let sample = try await dataSet.getSample(sampleIndex: i)
                let results = try graph.runOne(mode: "infer", inputTensors: ["input": sample.inputs])
                let result = results["GRU_lastState"]!
                let predictedValue = try result.getElement(index: 0)
                let expectedValue = try sample.outputs.getElement(index: 0)
                if ((predictedValue < 0.0 && expectedValue < 0.0) || (predictedValue > 0.0 && expectedValue > 0.0)) {
                    numInitialCorrect += 1
                }
            }
            
            //  Train
            for _ in 0..<20 {
                try await graph.runTraining(mode: "learn", trainingDataSet: dataSet, inputTensorName: "input", expectedValueTensorName : "expectedValue")
            }

            //  Now see how many have the correct sign
            var numFinalCorrect: Int = 0
            for i in 0..<numSamples {
                let sample = try await dataSet.getSample(sampleIndex: i)
                let results = try graph.runOne(mode: "infer", inputTensors: ["input": sample.inputs])
                let result = results["GRU_lastState"]!
                let predictedValue = try result.getElement(index: 0)
                let expectedValue = try sample.outputs.getElement(index: 0)
                if ((predictedValue < 0.0 && expectedValue < 0.0) || (predictedValue > 0.0 && expectedValue > 0.0)) {
                    numFinalCorrect += 1
                }
            }
            
            #expect(numFinalCorrect > numInitialCorrect)
        }
    }
}
