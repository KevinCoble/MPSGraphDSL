//
//  LearningTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 11/30/25.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct LearningTests {
    
    @Test func SimpleLearn() async throws {
        //  Create a graph for x * 2, where x is the variable to be learned
        do {
            let multiplicand = TensorFloat32(shape: TensorShape([1]), initialValue: Float32(2.0))
            let graph = Graph {
                Constant(shape: [1], value: Float32(3.0), name: "constant")
                Variable(values: multiplicand, name: "variable")
                    .learnWithRespectTo("loss")
                    .targetForModes(["getVar"])
                Multiplication(firstInput: "constant", secondInput: "variable", name: "result")
                    .targetForModes(["test"])
                Constant(shape: [1], value: Float32(15.0), name: "expectedValue")
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
            }
            
            //  Get the initial value before learning
            let results = try graph.runOne(mode: "test", inputTensors: [:])
            #expect(results.count == 1)
            let result = results["result"]
            #expect(result != nil)
            #expect(result!.shape == TensorShape([1]))
            #expect(try result!.getElement(index: 0).asDouble == 6.0)
            
            //  Get the loss when in learn mode
            let results2 = try graph.runOne(mode: "lossCalc", inputTensors: [:])
            #expect(results2.count == 1)
            let result2 = results2["loss"]
            #expect(result2 != nil)
            #expect(result2!.shape == TensorShape([1]))
            #expect(try result2!.getElement(index: 0).asDouble == 81.0)
            
            //  Learn
            for _ in 0..<10 {
                _ = try graph.runOne(mode: "learn", inputTensors: [:])
            }
            
            //  Get the var
            let results3 = try graph.runOne(mode: "getVar", inputTensors: [:])
            #expect(results3.count == 1)
            let result3 = results3["variable"]
            #expect(result3 != nil)
            #expect(result3!.shape == TensorShape([1]))
            #expect(try result3!.getElement(index: 0).asDouble == 5.0)
            
            //  Test the new result with the learned variable
            let results4 = try graph.runOne(mode: "test", inputTensors: [:])
            #expect(results4.count == 1)
            let result4 = results4["result"]
            #expect(result4 != nil)
            #expect(result4!.shape == TensorShape([1]))
            #expect(try result4!.getElement(index: 0).asDouble == 15.0)
        }
    }
    
    @Test func OneMatrixLearn() async throws {
        do {
            //  Get a test data set
            let testDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<100 {
                let input1 = Float32.random(in: 0...1)
                let input2 = Float32.random(in: 0...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let result: Float32 = 3.0 * input1 - 2.0 * input2 + 1.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: result)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await testDataSet.appendSample(sample)
            }
            //  Get a training data set
            let trainingDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<1000 {
                let input1 = Float32.random(in: 0...1)
                let input2 = Float32.random(in: 0...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let result: Float32 = 3.0 * input1 - 2.0 * input2 + 1.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: result)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await trainingDataSet.appendSample(sample)
            }
            
            //  Create the weight a bias tensors for a three layer 10x5x1 node network with two inputs
            let weights = TensorFloat32(shape: TensorShape([2, 1]), randomValueRange: try ParameterRange(minimum: -0.5, maximum: 0.5))
            let biases = TensorFloat32(shape: TensorShape([1]), initialValue: Float32(0.1))
            
            //  Build the graph
            let graph = Graph {
                PlaceHolder(shape: [1, 2], name: "inputs")
                PlaceHolder(shape: [1], name: "expectedValue")
                Variable(values: weights, name: "weights")
                    .learnWithRespectTo("loss")
                Variable(values: biases, name: "biases")
                    .learnWithRespectTo("loss")
                MatrixMultiplication(primary: "inputs", secondary: "weights")
                Addition(secondInput: "biases", name: "result")
                    .targetForModes(["infer"])
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
            }
            
            //  Get the initial accuracy
            var totalError : Double = 0.0
            let numTestingSamples = await testDataSet.numSamples
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                inputTensors["expectedValue"] = sample.outputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                totalError += abs(result - expected)
            }
            print("Initial error = \(totalError)")
            
            //  Train on the training set 10 times
            let numTrainingSamples = await trainingDataSet.numSamples
            for _ in 0..<10 {
                for i in 0..<numTrainingSamples {
                    let sample = try await trainingDataSet.getSample(sampleIndex: i)
                     var inputTensors : [String : Tensor] = [:]
                    inputTensors["inputs"] = sample.inputs
                    inputTensors["expectedValue"] = sample.outputs
                    _ = try graph.encodeOne(mode: "learn", inputTensors: inputTensors)
                }
            }
            
            //  Get the final accuracy
            totalError = 0.0
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                inputTensors["expectedValue"] = sample.outputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                totalError += abs(result - expected)
            }
            print("Final error = \(totalError)")
            
            #expect(totalError < 0.001)
        }
    }
    
    @Test func OneNodeNNLearn() async throws {
        do {
            //  Get a test data set
            let testDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<100 {
                let input1 = Float32.random(in: 0...1)
                let input2 = Float32.random(in: 0...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let or: Float32 = (input1 >= 0.5 || input2 >= 0.5) ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: or)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await testDataSet.appendSample(sample)
            }
            //  Get a training data set
            let trainingDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<1000 {
                let input1 = Float32.random(in: 0...1)
                let input2 = Float32.random(in: 0...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let or: Float32 = (input1 >= 0.5 || input2 >= 0.5) ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: or)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await trainingDataSet.appendSample(sample)
            }
            
            //  Create the weight a bias tensors for a single node network with two inputs
            let weights = TensorFloat32(shape: TensorShape([2, 1]), randomValueRange: try ParameterRange(minimum: -0.5, maximum: 0.5))
            let biases = TensorFloat32(shape: TensorShape([1]), initialValue: Float32(0.1))
            
            //  Build the graph
            let graph = Graph {
                PlaceHolder(shape: [1, 2], name: "inputs")
                PlaceHolder(shape: [1], name: "expectedValue")
                Variable(values: weights, name: "weights")
                    .learnWithRespectTo("loss")
                Variable(values: biases, name: "biases")
                    .learnWithRespectTo("loss")
                MatrixMultiplication(primary: "inputs", secondary: "weights")
                Addition(secondInput: "biases")
                Sigmoid(name: "result")
                    .targetForModes(["infer"])
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.02, learningModes: ["learn"])
            }
            
            //  Get the initial accuracy
            var totalCorrect = 0
            let numTestingSamples = await testDataSet.numSamples
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                inputTensors["expectedValue"] = sample.outputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Initial correct = \(totalCorrect)")
            
            //  Train on the training set 10 times
            let numTrainingSamples = await trainingDataSet.numSamples
            for _ in 0..<10 {
                for i in 0..<numTrainingSamples {
                    let sample = try await trainingDataSet.getSample(sampleIndex: i)
                    var inputTensors : [String : Tensor] = [:]
                    inputTensors["inputs"] = sample.inputs
                    inputTensors["expectedValue"] = sample.outputs
                    _ = try graph.encodeOne(mode: "learn", inputTensors: inputTensors, waitForResults: false)
                }
            }
            
            //  Get the final accuracy
            totalCorrect = 0
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                inputTensors["expectedValue"] = sample.outputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Final correct = \(totalCorrect)")
            
            #expect(totalCorrect > 60)
        }
    }
    
    @Test func ThreeNodeNNLearn() async throws {
        do {
            //  Get a test data set
            let testDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<100 {
                let input1 = Float32.random(in: 0...1)
                let input2 = Float32.random(in: 0...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let or: Float32 = (input1 >= 0.5 || input2 >= 0.5) ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: or)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await testDataSet.appendSample(sample)
            }
            //  Get a training data set
            let trainingDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<1000 {
                let input1 = Float32.random(in: 0...1)
                let input2 = Float32.random(in: 0...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let or: Float32 = (input1 >= 0.5 || input2 >= 0.5) ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: or)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await trainingDataSet.appendSample(sample)
            }
            
            //  Create the weight a bias tensors for a two layer 2x1 node network with two inputs
            let weights1 = TensorFloat32(shape: TensorShape([2, 2]), randomValueRange: try ParameterRange(minimum: -0.5, maximum: 0.5))
            let biases1 = TensorFloat32(shape: TensorShape([2]), initialValue: Float32(0.1))
            let weights2 = TensorFloat32(shape: TensorShape([2, 1]), randomValueRange: try ParameterRange(minimum: -0.5, maximum: 0.5))
            let biases2 = TensorFloat32(shape: TensorShape([1]), initialValue: Float32(0.1))
            
            //  Build the graph
            let graph = Graph {
                PlaceHolder(shape: [1, 2], name: "inputs")
                PlaceHolder(shape: [1], modes: ["learn"], name: "expectedValue")
                Variable(values: weights1, name: "weights1")
                    .learnWithRespectTo("loss")
                Variable(values: biases1, name: "biases1")
                    .learnWithRespectTo("loss")
                MatrixMultiplication(primary: "inputs", secondary: "weights1")
                Addition(secondInput: "biases1")
                Sigmoid(name: "layer1Result")
                Variable(values: weights2, name: "weights2")
                    .learnWithRespectTo("loss")
                Variable(values: biases2, name: "biases2")
                    .learnWithRespectTo("loss")
                MatrixMultiplication(primary: "layer1Result", secondary: "weights2")
                Addition(secondInput: "biases2")
                Sigmoid(name: "result")
                    .targetForModes(["infer"])
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.01, learningModes: ["learn"])
            }
            
            //  Get the initial accuracy
            var totalCorrect = 0
            let numTestingSamples = await testDataSet.numSamples
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Initial correct = \(totalCorrect)")
            
            //  Train on the training set 250 times
            let numTrainingSamples = await trainingDataSet.numSamples
            for _ in 0..<200 {
                for i in 0..<numTrainingSamples {
                    let sample = try await trainingDataSet.getSample(sampleIndex: i)
                    var inputTensors : [String : Tensor] = [:]
                    inputTensors["inputs"] = sample.inputs
                    inputTensors["expectedValue"] = sample.outputs
                    _ = try graph.encodeOne(mode: "learn", inputTensors: inputTensors, waitForResults: false)
                }
            }
            
            //  Get the final accuracy
            totalCorrect = 0
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Final correct = \(totalCorrect)")
            
            #expect(totalCorrect > 88)
        }
    }
    
    @Test func ThreeLayerNNLearn() async throws {
        do {
            //  Get a test data set
            let testDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<100 {
                //                let input1 = Float32.random(in: 0...1)
                //                let input2 = Float32.random(in: 0...1)
                //                let input1 = Float32(Int.random(in: 0...1))
                //                let input2 = Float32(Int.random(in: 0...1))
                let input1 = Int.random(in: 0...1) == 0 ? Float32.random(in: 0...0.35) : Float32.random(in: 0.65...1)
                let input2 = Int.random(in: 0...1) == 0 ? Float32.random(in: 0...0.35) : Float32.random(in: 0.65...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let xor: Float32 = ((input1 >= 0.5 && input2 >= 0.5) || (input1 < 0.5 && input2 < 0.5)) ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: xor)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await testDataSet.appendSample(sample)
            }
            //  Get a training data set
            let trainingDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<1000 {
                //                let input1 = Float32.random(in: 0...1)
                //                let input2 = Float32.random(in: 0...1)
                //                let input1 = Float32(Int.random(in: 0...1))
                //                let input2 = Float32(Int.random(in: 0...1))
                let input1 = Int.random(in: 0...1) == 0 ? Float32.random(in: 0...0.35) : Float32.random(in: 0.65...1)
                let input2 = Int.random(in: 0...1) == 0 ? Float32.random(in: 0...0.35) : Float32.random(in: 0.65...1)
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let xor: Float32 = ((input1 >= 0.5 && input2 >= 0.5) || (input1 < 0.5 && input2 < 0.5)) ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: xor)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await trainingDataSet.appendSample(sample)
            }
            
            //  Create the weight a bias tensors for a three layer 10x5x1 node network with two inputs
            let layer1weights = TensorFloat32(shape: TensorShape([2, 10]), randomValueRange: try ParameterRange(minimum: -0.5, maximum: 0.5))
            let layer1biases = TensorFloat32(shape: TensorShape([10]), initialValue: Float32(0.1))
            let layer2weights = TensorFloat32(shape: TensorShape([10, 5]), randomValueRange: try ParameterRange(minimum: -0.5, maximum: 0.5))
            let layer2biases = TensorFloat32(shape: TensorShape([5]), initialValue: Float32(0.1))
            let layer3weights = TensorFloat32(shape: TensorShape([5, 1]), randomValueRange: try ParameterRange(minimum: -0.5, maximum: 0.5))
            let layer3biases = TensorFloat32(shape: TensorShape([1]), initialValue: Float32(0.1))
            
            //  Build the graph
            let graph = Graph {
                PlaceHolder(shape: [1, 2], name: "inputs")
                PlaceHolder(shape: [1], name: "expectedValue")
                Variable(values: layer1weights, name: "weights1")
                    .learnWithRespectTo("loss")
                Variable(values: layer1biases, name: "biases1")
                    .learnWithRespectTo("loss")
                MatrixMultiplication(primary: "inputs", secondary: "weights1")
                Addition(secondInput: "biases1")
                Sigmoid(name: "layer1Outputs")
                Variable(values: layer2weights, name: "weights2")
                    .learnWithRespectTo("loss")
                Variable(values: layer2biases, name: "biases2")
                    .learnWithRespectTo("loss")
                MatrixMultiplication(primary: "layer1Outputs", secondary: "weights2")
                Addition(secondInput: "biases2")
                Sigmoid(name: "layer2Outputs")
                Variable(values: layer3weights, name: "weights3")
                    .learnWithRespectTo("loss")
                Variable(values: layer3biases, name: "biases3")
                    .learnWithRespectTo("loss")
                MatrixMultiplication(primary: "layer2Outputs", secondary: "weights3")
                Addition(secondInput: "biases3")
                Sigmoid(name: "result")
                    .targetForModes(["infer"])
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
            }
            
            //  Get the initial accuracy
            var totalCorrect = 0
            let numTestingSamples = await testDataSet.numSamples
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                inputTensors["expectedValue"] = sample.outputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Initial correct = \(totalCorrect)")
            
            //  Train on the training set 10 times
            let numTrainingSamples = await trainingDataSet.numSamples
            for _ in 0..<100 {
                for i in 0..<numTrainingSamples {
                    let sample = try await trainingDataSet.getSample(sampleIndex: i)
                    var inputTensors : [String : Tensor] = [:]
                    inputTensors["inputs"] = sample.inputs
                    inputTensors["expectedValue"] = sample.outputs
                    _ = try graph.encodeOne(mode: "learn", inputTensors: inputTensors, waitForResults: false)
                }
            }
            
            //  Get the final accuracy
            totalCorrect = 0
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                inputTensors["expectedValue"] = sample.outputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Final correct = \(totalCorrect)")
        }
    }
    
    @Test func FullyConnectedLayerLearn() async throws {
        do {
            //  Get a test data set
            let testDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<100 {
                let input1 = Float32(Int.random(in: 0...1))
                let input2 = Float32(Int.random(in: 0...1))
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let or: Float32 = (input1 >= 0.5 || input2 >= 0.5)  ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: or)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await testDataSet.appendSample(sample)
            }
            
            //  Get a training data set
            let trainingDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([1]), outputType: .float32)
            for _ in 0..<100 {
                let input1 = Float32(Int.random(in: 0...1))
                let input2 = Float32(Int.random(in: 0...1))
                let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
                let or: Float32 = (input1 >= 0.5 || input2 >= 0.5)  ? 1.0 : 0.0
                let outputTensor = TensorFloat32(shape: TensorShape([1]), initialValue: or)
                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                try await trainingDataSet.appendSample(sample)
            }
            
            //  Build the graph
            let graph = Graph {
                PlaceHolder(shape: [1, 2], name: "inputs")
                PlaceHolder(shape: [1], modes: ["learn"], name: "expectedValue")
                FullyConnectedLayer(input: "inputs", outputShape: TensorShape([1]), activationFunction: .sigmoid, name: "result")
                    .learnWithRespectTo("loss")
                    .targetForModes(["infer"])
                MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                    .targetForModes(["lossCalc", "learn"])
                Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
            }
            
            //  Get the initial accuracy
            var totalCorrect = 0
            let numTestingSamples = await testDataSet.numSamples
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Initial correct = \(totalCorrect)")
            
            //  Train on the training set 10 times
            let numTrainingSamples = await trainingDataSet.numSamples
            for _ in 0..<10 {
                for i in 0..<numTrainingSamples {
                    let sample = try await trainingDataSet.getSample(sampleIndex: i)
                    var inputTensors : [String : Tensor] = [:]
                    inputTensors["inputs"] = sample.inputs
                    inputTensors["expectedValue"] = sample.outputs
                    _ = try graph.encodeOne(mode: "learn", inputTensors: inputTensors, waitForResults: false)
                }
            }
            
            //  Get the final accuracy
            totalCorrect = 0
            for i in 0..<numTestingSamples {
                let sample = try await testDataSet.getSample(sampleIndex: i)
                var inputTensors : [String : Tensor] = [:]
                inputTensors["inputs"] = sample.inputs
                let results = try graph.runOne(mode: "infer", inputTensors: inputTensors)
                let resultTensor = results["result"]
                let result = try resultTensor!.getElement(index: 0).asDouble
                let expected = try sample.outputs.getElement(index: 0).asDouble
                if (result >= 0.5 && expected >= 0.5) { totalCorrect += 1 }
                if (result < 0.5 && expected < 0.5) { totalCorrect += 1 }
            }
            print("Final correct = \(totalCorrect)")
        }
    }
    
    @Test func BatchNormalizationTest() async throws {
        let inputTensor = try TensorFloat32(shape: TensorShape([4]), initialValues: [1.0, 2.0, 3.0, 4.0])
        let outputTensor = try TensorFloat32(shape: TensorShape([4]), initialValues: [5.0, 1.0, 4.0, 4.0])
        
        //  Build the graph
        let graph = Graph {
            PlaceHolder(shape: [4], name: "inputs")
            PlaceHolder(shape: [4], modes: ["learn"], name: "expectedValue")
            BatchNormalization(input: "inputs", name: "result")
                .learnWithRespectTo("loss")
                .targetForModes(["infer"])
            MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
                .targetForModes(["lossCalc", "learn"])
            Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
        }
        
        //  Get the initial output.  Should be the same as the input (initialization to ones and zeros)
        var results = try graph.runOne(mode: "infer", inputTensors: ["inputs": inputTensor])
        var result = results["result"]!
        #expect(try abs(result.getElement(index: 0).asDouble - 1.0) < 1.0E-04)
        #expect(try abs(result.getElement(index: 1).asDouble - 2.0) < 1.0E-04)
        #expect(try abs(result.getElement(index: 2).asDouble - 3.0) < 1.0E-04)
        #expect(try abs(result.getElement(index: 3).asDouble - 4.0) < 1.0E-04)

        //  Train
        for _ in 0..<250 {
            _ = try graph.runOne(mode: "learn", inputTensors: ["inputs": inputTensor, "expectedValue": outputTensor])
        }
        
        //  Get the post-training output.  Should match output
        results = try graph.runOne(mode: "infer", inputTensors: ["inputs": inputTensor])
        result = results["result"]!
        #expect(try abs(result.getElement(index: 0).asDouble - 5.0) < 1.0E-02)
        #expect(try abs(result.getElement(index: 1).asDouble - 1.0) < 1.0E-02)
        #expect(try abs(result.getElement(index: 2).asDouble - 4.0) < 1.0E-02)
        #expect(try abs(result.getElement(index: 3).asDouble - 4.0) < 1.0E-02)
    }

    @Test func SpeedTest() async throws {
        //  Get a test data set
        let testDataSet = DataSet(inputShape: TensorShape([1, 2]), inputType: .float32, outputShape: TensorShape([3]), outputType: .float32)
        for _ in 0..<100 {
            let input1 = Float32(Int.random(in: 0...1))
            let input2 = Float32(Int.random(in: 0...1))
            let inputTensor = try TensorFloat32(shape: TensorShape([1, 2]), initialValues: [input1, input2])
            var output: Int = 0
            if (input1 < 0.5 || input2 < 0.5) { output = 1}
            if (input1 > 0.5 || input2 > 0.5) { output = 2}
            let outputTensor = try await testDataSet.getOutputTensorForClassification(output)
            let sample = DataSample(inputs: inputTensor, outputs: outputTensor, classIndex: output)
            try await testDataSet.appendSample(sample)
        }
        
        //  Create a simple graph
        let graph = Graph {
            PlaceHolder(shape: [1, 2], name: "inputs")
            PlaceHolder(shape: [3], modes: ["learn"], name: "expectedValue")
            FullyConnectedLayer(input: "inputs", outputShape: TensorShape([3]), activationFunction: .relu, name: "result")
                .learnWithRespectTo("loss")
                .targetForModes(["infer"])
            SoftMax(name: "inferenceResult")
                .targetForModes(["infer"], )
            SoftMaxCrossEntropy(labels: "expectedValue", reductionType: .sum, name: "loss")
                .targetForModes(["learn"])
            Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
        }
        
        //  Run the data through 1000 times
        for _ in 0..<1000 {
//            let _ = try await graph.runClassifierTest(mode: "infer", testDataSet: testDataSet, inputTensorName: "inputs", resultTensorName: "inferenceResult")
            let _ = try await graph.runTraining(mode: "learn", trainingDataSet: testDataSet, inputTensorName: "inputs", expectedValueTensorName: "expectedValue")
        }

    }
}
