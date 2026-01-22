//
//  DataParserTests.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 11/20/25.
//

import Testing
import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct DataParserTests {

    @Test func testBuilder() async throws {
        var parser = DataParser {
            UnusedData(length: 16, format: .fUInt8)
        }
        #expect(parser.chunks.count == 1)
        
        parser = DataParser {
            UnusedData(length: 16, format: .fUInt8)
            InputData(length: 4, format : .fUInt8, postProcessing : .None)
            OutputData(length: 1, format : .fUInt8, postProcessing : .None)
        }
        #expect(parser.chunks.count == 3)
        
        parser = DataParser {
            UnusedData(length: 16, format: .fUInt8)
            RepeatSampleTillDone {
                InputData(length: 4, format : .fUInt8, postProcessing : .None)
                OutputData(length: 1, format : .fUInt8, postProcessing : .None)
            }
        }
        #expect(parser.chunks.count == 2)
    }


    @Test func testBinaryDataExtraction() async throws {
        //  Create some test data
        let byteArray : [UInt8] = [123, 234, 1, 2, 3, 4]
        let data = Data(byteArray)
        
        //  Create a dataset
        let singleInput = TensorShape([1])
        let singleOutput = TensorShape([1])
        var dataSet = DataSet(inputShape: singleInput, inputType: .double, outputShape: singleOutput, outputType: .double)
        
        //  Single feature extraction
        var parser = DataParser {
            UnusedData(length: 2, format: .fUInt8)
            StartNewSample()
            InputData(length: 1, format : .fUInt8, postProcessing : .None)
        }
        
        //  Parse the data
        try await parser.parseBinaryData(data, intoDataSet: dataSet)
        
        //  Check the results
        #expect(await dataSet.numSamples == 1)
        var sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0).asDouble == 1.0)
        
        //  Multiple samples extraction
        dataSet = DataSet(inputShape: singleInput, inputType: .double, outputShape: singleOutput, outputType: .double)
        parser = DataParser {
            UnusedData(length: 2, format: .fUInt8)
            StartNewSample()
            InputData(length: 1, format : .fUInt8, postProcessing : .None)
            StartNewSample()
            InputData(length: 1, format : .fUInt8, postProcessing : .None)
            StartNewSample()
            InputData(length: 1, format : .fUInt8, postProcessing : .None)
            StartNewSample()
            InputData(length: 1, format : .fUInt8, postProcessing : .None)
        }
        try await parser.parseBinaryData(data, intoDataSet: dataSet)
        #expect(await dataSet.numSamples == 4)
        sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 1.0)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(try sample.inputs.getElement(index: 0) == 2.0)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(try sample.inputs.getElement(index: 0) == 3.0)
        sample = try await dataSet.getSample(sampleIndex: 3)
        #expect(try sample.inputs.getElement(index: 0) == 4.0)
        
        //  Repeat samples extraction
        dataSet = DataSet(inputShape: singleInput, inputType: .double, outputShape: singleOutput, outputType: .double)
        parser = DataParser {
            UnusedData(length: 2, format: .fUInt8)
            RepeatSampleTillDone {
                InputData(length: 1, format : .fUInt8, postProcessing : .None)
            }
        }
        try await parser.parseBinaryData(data, intoDataSet: dataSet)
        #expect(await dataSet.numSamples == 4)
        sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 1.0)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(try sample.inputs.getElement(index: 0) == 2.0)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(try sample.inputs.getElement(index: 0) == 3.0)
        sample = try await dataSet.getSample(sampleIndex: 3)
        #expect(try sample.inputs.getElement(index: 0) == 4.0)
     }

    
    @Test func testDelineatedTextDataExtraction() async throws {
        var dataSet = DataSet(inputShape: TensorShape([2]), inputType: .double, outputShape: TensorShape([3]), outputType: .double)
        
        //  Create a test text array with all comments
        var text = """
        # comment1
        // comment2
        """
        
        //  Create a delineated text parser (whitespace separated)
        var parser = DelineatedTextParser(lineSeparator: .SpaceDelimited) {
            InputIntegerString()
        }.withCommentIndicators(["#", "//"])
        
        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 0)
        
        //  Create a test text array with some data
        text = """
        5
        7
        9
        """
        
        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 3)
        var sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 5.0)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(try sample.inputs.getElement(index: 0) == 7.0)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(try sample.inputs.getElement(index: 0) == 9.0)
        
        //  Reset the data set to empty
        dataSet = DataSet(inputShape: TensorShape([2]), inputType: .double, outputShape: TensorShape([3]), outputType: .double)

        //  Create a delineated text parser (whitespace separated) that skips the first line
        parser = DelineatedTextParser(lineSeparator: .SpaceDelimited) {
            InputIntegerString()
        }.skipInitialLines(1)
        
        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 2)
        sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 7.0)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(try sample.inputs.getElement(index: 0) == 9.0)
        
        //  Create a test text array with some floating and output data
        text = """
        5 6.2 100
        7 7.3 101
        9 8.4 102
        """
        
        //  Reset the data set to empty
        dataSet = DataSet(inputShape: TensorShape([2]), inputType: .double, outputShape: TensorShape([1]), outputType: .double)

        //  Create a delineated text parser with two inputs and an output
        parser = DelineatedTextParser(lineSeparator: .SpaceDelimited) {
            InputIntegerString()
            InputFloatString()
            OutputIntegerString()
        }
        
        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 3)
        sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 5.0)
        var value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 6.2) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 100)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(try sample.inputs.getElement(index: 0) == 7.0)
        value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 7.3) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 101)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(try sample.inputs.getElement(index: 0) == 9.0)
        value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 8.4) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 102)
        
        //  Create a test text array with some unused data, floating and output data, comma separated
        text = """
        5,123,6.2,100
        7,124,7.3,101
        9,125,8.4,102
        """
        
        //  Reset the data set to empty
        dataSet = DataSet(inputShape: TensorShape([2]), inputType: .double, outputShape: TensorShape([1]), outputType: .double)

        //  Create a delineated text parser with two inputs and an output, comma separated
        parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {
            InputIntegerString()
            UnusedTextString()
            InputFloatString()
            OutputIntegerString()
        }
        
        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 3)
        sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 5.0)
        value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 6.2) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 100)
        sample = try await dataSet.getSample(sampleIndex: 1)
         #expect(try sample.inputs.getElement(index: 0) == 7.0)
        value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 7.3) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 101)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(try sample.inputs.getElement(index: 0) == 9.0)
        value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 8.4) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 102)
    }

    
    @Test func testFixedColumnTextDataExtraction() async throws {
        //  Create a data set with 2 inputs and one output
        var dataSet = DataSet(inputShape: TensorShape([2]), inputType: .double, outputShape: TensorShape([1]), outputType: .double)
        
        //  Create a test text array with all comments
        var text = """
        # comment1
        // comment2
        """
        
        //  Create a fixed column parser
        var parser = FixedColumnTextParser() {
            InputIntegerColumns(numCharacters: 3)
        }.withCommentIndicators(["#", "//"])
        
        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 0)
        
        //  Create a test text array with some data
        text = """
          5
          7
          9
        """
        
        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 3)
        var sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 5.0)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(try sample.inputs.getElement(index: 0) == 7.0)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(try sample.inputs.getElement(index: 0) == 9.0)
        
        //  Create a test text array with some floating and output data
        text = """
          5 4 5 6.2 100
          7 324 7.3 101
          9 abc 8.4 102
        """
        
        //  Reset the data set to empty
        dataSet = DataSet(inputShape: TensorShape([2]), inputType: .double, outputShape: TensorShape([1]), outputType: .double)

        //  Create a fixed column text parser with two inputs, unused columns, and an output
        parser = FixedColumnTextParser() {
            InputIntegerColumns(numCharacters: 3)
            UnusedTextColumns(numCharacters: 4)
            InputFloatColumns(numCharacters: 4)
            OutputIntegerColumns(numCharacters: 4)
        }

        //  Parse the data
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 3)
        sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(try sample.inputs.getElement(index: 0) == 5.0)
        var value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 6.2) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 100)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(try sample.inputs.getElement(index: 0) == 7.0)
        value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 7.3) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 101)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(try sample.inputs.getElement(index: 0) == 9.0)
        value = try sample.inputs.getElement(index: 1)
        #expect(abs(value - 8.4) < 1.0e-5)
        #expect(try sample.outputs.getElement(index: 0) == 102)
    }

    
    @Test func testTensorOrder() async throws {
        //  Create some test data
        let byteArray : [UInt8] = [1, 2, 3, 4, 5, 6]
        let data = Data(byteArray)
        
        //  Create a dataset
        let singleInput = TensorShape([2, 3])
        let singleOutput = TensorShape([1])
        let dataSet = DataSet(inputShape: singleInput, inputType: .double, outputShape: singleOutput, outputType: .double)
        
        //  Single feature extraction
        let parser = DataParser {
            StartNewSample()
            RepeatDimension(count: 2, dimension: .Dimension0, affects: .input) {
                InputData(length: 3, format : .fUInt8, postProcessing : .None)
                SetDimension(dimension: .Dimension1, toValue: 0, affects: .input)
            }
        }
        
        //  Parse the data
        try await parser.parseBinaryData(data, intoDataSet: dataSet)
        let sample = try await dataSet.getSample(sampleIndex: 0)
         
        //  Check the results
        #expect(await dataSet.numSamples == 1)
        let tensor = sample.inputs
        let element = try tensor.getElement(location: [0, 1])     //  Remember, location is zero based!
        #expect(element == 2.0)
        let element3 = try tensor.getElement(location: [1, 0])     //  Remember, location is zero based!
        #expect(element3 == 4.0)
        let element2 = try tensor.getElement(location: [1, 1])     //  Remember, location is zero based!
        #expect(element2 == 5.0)
        let element4 = try tensor.getElement(location: [1, 2])     //  Remember, location is zero based!
        #expect(element4 == 6.0)

     }
    
    @Test func testDelineatedTextDimensionManagement() async throws {

        //  Create a test text array with some data
        let text = """
        #  A Slightly More Complicated Example
        6,5.0,3.4,2.0,9.61,2.3,4.87,class1
        14,5.4,3.5,107.2,8.612,4.3,9.67,class3
        11,11.8,7.4,3.1,3,2.4,11.22,class1
        7,9.6,4.8,2.2,14.74,5.6,5.3,class2
        """

        //  Create a delineated text parser (comma separated)
        var parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {
            InputIntegerString()
            RepeatDimForDelineatedText(count: 6, dimension: .Dimension0, affects: .neither) {
                InputFloatString()
            }
            OutputLabelString()
        }.withCommentIndicators(["#", "//"])
        
        //  Parse the data
        var dataSet = DataSet(inputShape: TensorShape([7]), inputType: .double, outputShape: TensorShape([3]), outputType: .double)
        try await parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(await dataSet.numSamples == 4)
        var sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(abs(try sample.inputs.getElement(index: 1) - 5.0) < 1.0E-5)
        #expect(abs(try sample.inputs.getElement(index: 5) - 2.3) < 1.0E-5)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(abs(try sample.inputs.getElement(index: 1) - 5.4) < 1.0E-5)
        #expect(abs(try sample.inputs.getElement(index: 5) - 4.3) < 1.0E-5)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(abs(try sample.inputs.getElement(index: 1) - 11.8) < 1.0E-5)
        #expect(abs(try sample.inputs.getElement(index: 5) - 2.4) < 1.0E-5)

        //  Create a delineated text parser (comma separated)
        parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {
            UnusedTextString()
            RepeatDimForDelineatedText(count: 2, dimension: .Dimension0, affects: .input) {
                InputFloatString()
                InputFloatString()
                InputFloatString()
                SetDimension(dimension: .Dimension1, toValue: 0, affects: .input)
            }
            OutputLabelString()
        }.withCommentIndicators(["#", "//"])
        
        //  Parse the data
        dataSet = DataSet(inputShape: TensorShape([2, 3]), inputType: .double, outputShape: TensorShape([3]), outputType: .double)
        try await parser.parseTextLines(dataSet: dataSet, text: text)
         
        //  Check the result
        #expect(await dataSet.numSamples == 4)
        sample = try await dataSet.getSample(sampleIndex: 0)
        #expect(abs(try sample.inputs.getElement(location: [0, 0]) - 5.0) < 1.0E-5)
        #expect(abs(try sample.inputs.getElement(location: [1, 1]) - 2.3) < 1.0E-5)
        sample = try await dataSet.getSample(sampleIndex: 1)
        #expect(abs(try sample.inputs.getElement(location: [0, 0]) - 5.4) < 1.0E-5)
        #expect(abs(try sample.inputs.getElement(location: [1, 1]) - 4.3) < 1.0E-5)
        sample = try await dataSet.getSample(sampleIndex: 2)
        #expect(abs(try sample.inputs.getElement(location: [0, 0]) - 11.8) < 1.0E-5)
        #expect(abs(try sample.inputs.getElement(location: [1, 1]) - 2.4) < 1.0E-5)
    }
    
    @Test(.disabled("Requires MNIST data files to be downloaded into the Document directory"))
    func MNISTLoad() async throws {
        //  Set URLs as needed
        let fileManager = FileManager.default
        let docsDirURL = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let MNISTTrainInputURL = docsDirURL.appendingPathComponent("MNIST/train-images.idx3-ubyte")
        let MNISTTrainOutputURL = docsDirURL.appendingPathComponent("MNIST/train-labels.idx1-ubyte")
        
        //  Load the data into memory
        let trainingInputData = try Data(contentsOf: MNISTTrainInputURL)
        let trainingOutputData = try Data(contentsOf: MNISTTrainOutputURL)
        
        //  Create the parsers
        let MNISTInputParser = DataParser {
            UnusedData(length: 16, format: .fUInt8)
            RepeatSampleTillDone {
                RepeatDimension(count: 28, dimension: .Dimension0, affects: .input) {
                    InputData(length: 28, format : .fUInt8, postProcessing : .None)
                    SetDimension(dimension: .Dimension1, toValue: 0, affects: .input)
                }
            }
        }
        let MNISTOutputParser = DataParser {
            UnusedData(length: 8, format: .fUInt8)
            RepeatSampleTillDone {
                LabelIndex(count: 1, format: .fUInt8)
            }
        }
        
        //  Parse the data concurrently from data
        let trainingDataSet_DataConcurrent = DataSet(inputShape: TensorShape([28, 28]), inputType: .float32, outputShape: TensorShape([10]), outputType: .float32)
        try await MNISTInputParser.parseBinaryData(trainingInputData, intoDataSet: trainingDataSet_DataConcurrent)
        try await MNISTOutputParser.parseBinaryData(trainingOutputData, intoDataSet: trainingDataSet_DataConcurrent)
        
        #expect(await trainingDataSet_DataConcurrent.numSamples == 60000)
        
        //  Parse the data non-concurrently from data
        let trainingDataSet_Data = DataSet(inputShape: TensorShape([28, 28]), inputType: .float32, outputShape: TensorShape([10]), outputType: .float32)
        try await MNISTInputParser.parseBinaryData(trainingInputData, intoDataSet: trainingDataSet_Data, maxConcurrency: 1)
        try await MNISTOutputParser.parseBinaryData(trainingOutputData, intoDataSet: trainingDataSet_Data, maxConcurrency: 1)
        
        #expect(await trainingDataSet_DataConcurrent.numSamples == 60000)
        #expect(try await dataSetsMatch(dataset1: trainingDataSet_DataConcurrent, dataset2: trainingDataSet_Data, tolerence: 1.0E-04))
        
        //  Parse the data concurrently from files
        let trainingDataSet_FileConcurrent = DataSet(inputShape: TensorShape([28, 28]), inputType: .float32, outputShape: TensorShape([10]), outputType: .float32)
        try await MNISTInputParser.parseBinaryFile(url: MNISTTrainInputURL, intoDataSet: trainingDataSet_FileConcurrent)
        try await MNISTOutputParser.parseBinaryFile(url: MNISTTrainOutputURL, intoDataSet: trainingDataSet_FileConcurrent)
        
        #expect(await trainingDataSet_FileConcurrent.numSamples == 60000)
        #expect(try await dataSetsMatch(dataset1: trainingDataSet_DataConcurrent, dataset2: trainingDataSet_FileConcurrent, tolerence: 1.0E-04))
        
        //  Parse the data non-concurrently from files
        let trainingDataSet_File = DataSet(inputShape: TensorShape([28, 28]), inputType: .float32, outputShape: TensorShape([10]), outputType: .float32)
        try await MNISTInputParser.parseBinaryFile(url: MNISTTrainInputURL, intoDataSet: trainingDataSet_File, maxConcurrency: 1)
        try await MNISTOutputParser.parseBinaryFile(url: MNISTTrainOutputURL, intoDataSet: trainingDataSet_File, maxConcurrency: 1)
        
        #expect(await trainingDataSet_File.numSamples == 60000)
        #expect(try await dataSetsMatch(dataset1: trainingDataSet_DataConcurrent, dataset2: trainingDataSet_File, tolerence: 1.0E-04))
    }
    
    @Test(.disabled("Requires text lines to be saved as a text file named IrisData.txt in the Document directory"))
    func TextLoad() async throws {
        let textLines = """
        sepal_length,sepal_width,petal_length,petal_width,species
        5.1,3.5,1.4,0.2,setosa
        4.9,3.0,1.4,0.2,setosa
        4.7,3.2,1.3,0.2,setosa
        4.6,3.1,1.5,0.2,setosa
        5.0,3.6,1.4,0.2,setosa
        5.4,3.9,1.7,0.4,setosa
        4.6,3.4,1.4,0.3,setosa
        5.0,3.4,1.5,0.2,setosa
        4.4,2.9,1.4,0.2,setosa
        4.9,3.1,1.5,0.1,setosa
        5.4,3.7,1.5,0.2,setosa
        4.8,3.4,1.6,0.2,setosa
        4.8,3.0,1.4,0.1,setosa
        4.3,3.0,1.1,0.1,setosa
        5.8,4.0,1.2,0.2,setosa
        5.7,4.4,1.5,0.4,setosa
        5.4,3.9,1.3,0.4,setosa
        5.1,3.5,1.4,0.3,setosa
        5.7,3.8,1.7,0.3,setosa
        5.1,3.8,1.5,0.3,setosa
        5.4,3.4,1.7,0.2,setosa
        5.1,3.7,1.5,0.4,setosa
        4.6,3.6,1.0,0.2,setosa
        5.1,3.3,1.7,0.5,setosa
        4.8,3.4,1.9,0.2,setosa
        5.0,3.0,1.6,0.2,setosa
        5.0,3.4,1.6,0.4,setosa
        5.2,3.5,1.5,0.2,setosa
        5.2,3.4,1.4,0.2,setosa
        4.7,3.2,1.6,0.2,setosa
        4.8,3.1,1.6,0.2,setosa
        5.4,3.4,1.5,0.4,setosa
        5.2,4.1,1.5,0.1,setosa
        5.5,4.2,1.4,0.2,setosa
        4.9,3.1,1.5,0.1,setosa
        5.0,3.2,1.2,0.2,setosa
        5.5,3.5,1.3,0.2,setosa
        4.9,3.1,1.5,0.1,setosa
        4.4,3.0,1.3,0.2,setosa
        5.1,3.4,1.5,0.2,setosa
        5.0,3.5,1.3,0.3,setosa
        4.5,2.3,1.3,0.3,setosa
        4.4,3.2,1.3,0.2,setosa
        5.0,3.5,1.6,0.6,setosa
        5.1,3.8,1.9,0.4,setosa
        4.8,3.0,1.4,0.3,setosa
        5.1,3.8,1.6,0.2,setosa
        4.6,3.2,1.4,0.2,setosa
        5.3,3.7,1.5,0.2,setosa
        5.0,3.3,1.4,0.2,setosa
        7.0,3.2,4.7,1.4,versicolor
        6.4,3.2,4.5,1.5,versicolor
        6.9,3.1,4.9,1.5,versicolor
        5.5,2.3,4.0,1.3,versicolor
        6.5,2.8,4.6,1.5,versicolor
        5.7,2.8,4.5,1.3,versicolor
        6.3,3.3,4.7,1.6,versicolor
        4.9,2.4,3.3,1.0,versicolor
        6.6,2.9,4.6,1.3,versicolor
        5.2,2.7,3.9,1.4,versicolor
        5.0,2.0,3.5,1.0,versicolor
        5.9,3.0,4.2,1.5,versicolor
        6.0,2.2,4.0,1.0,versicolor
        6.1,2.9,4.7,1.4,versicolor
        5.6,2.9,3.6,1.3,versicolor
        6.7,3.1,4.4,1.4,versicolor
        5.6,3.0,4.5,1.5,versicolor
        5.8,2.7,4.1,1.0,versicolor
        6.2,2.2,4.5,1.5,versicolor
        5.6,2.5,3.9,1.1,versicolor
        5.9,3.2,4.8,1.8,versicolor
        6.1,2.8,4.0,1.3,versicolor
        6.3,2.5,4.9,1.5,versicolor
        6.1,2.8,4.7,1.2,versicolor
        6.4,2.9,4.3,1.3,versicolor
        6.6,3.0,4.4,1.4,versicolor
        6.8,2.8,4.8,1.4,versicolor
        6.7,3.0,5.0,1.7,versicolor
        6.0,2.9,4.5,1.5,versicolor
        5.7,2.6,3.5,1.0,versicolor
        5.5,2.4,3.8,1.1,versicolor
        5.5,2.4,3.7,1.0,versicolor
        5.8,2.7,3.9,1.2,versicolor
        6.0,2.7,5.1,1.6,versicolor
        5.4,3.0,4.5,1.5,versicolor
        6.0,3.4,4.5,1.6,versicolor
        6.7,3.1,4.7,1.5,versicolor
        6.3,2.3,4.4,1.3,versicolor
        5.6,3.0,4.1,1.3,versicolor
        5.5,2.5,4.0,1.3,versicolor
        5.5,2.6,4.4,1.2,versicolor
        6.1,3.0,4.6,1.4,versicolor
        5.8,2.6,4.0,1.2,versicolor
        5.0,2.3,3.3,1.0,versicolor
        5.6,2.7,4.2,1.3,versicolor
        5.7,3.0,4.2,1.2,versicolor
        5.7,2.9,4.2,1.3,versicolor
        6.2,2.9,4.3,1.3,versicolor
        5.1,2.5,3.0,1.1,versicolor
        5.7,2.8,4.1,1.3,versicolor
        6.3,3.3,6.0,2.5,virginica
        5.8,2.7,5.1,1.9,virginica
        7.1,3.0,5.9,2.1,virginica
        6.3,2.9,5.6,1.8,virginica
        6.5,3.0,5.8,2.2,virginica
        7.6,3.0,6.6,2.1,virginica
        4.9,2.5,4.5,1.7,virginica
        7.3,2.9,6.3,1.8,virginica
        6.7,2.5,5.8,1.8,virginica
        7.2,3.6,6.1,2.5,virginica
        6.5,3.2,5.1,2.0,virginica
        6.4,2.7,5.3,1.9,virginica
        6.8,3.0,5.5,2.1,virginica
        5.7,2.5,5.0,2.0,virginica
        5.8,2.8,5.1,2.4,virginica
        6.4,3.2,5.3,2.3,virginica
        6.5,3.0,5.5,1.8,virginica
        7.7,3.8,6.7,2.2,virginica
        7.7,2.6,6.9,2.3,virginica
        6.0,2.2,5.0,1.5,virginica
        6.9,3.2,5.7,2.3,virginica
        5.6,2.8,4.9,2.0,virginica
        7.7,2.8,6.7,2.0,virginica
        6.3,2.7,4.9,1.8,virginica
        6.7,3.3,5.7,2.1,virginica
        7.2,3.2,6.0,1.8,virginica
        6.2,2.8,4.8,1.8,virginica
        6.1,3.0,4.9,1.8,virginica
        6.4,2.8,5.6,2.1,virginica
        7.2,3.0,5.8,1.6,virginica
        7.4,2.8,6.1,1.9,virginica
        7.9,3.8,6.4,2.0,virginica
        6.4,2.8,5.6,2.2,virginica
        6.3,2.8,5.1,1.5,virginica
        6.1,2.6,5.6,1.4,virginica
        7.7,3.0,6.1,2.3,virginica
        6.3,3.4,5.6,2.4,virginica
        6.4,3.1,5.5,1.8,virginica
        6.0,3.0,4.8,1.8,virginica
        6.9,3.1,5.4,2.1,virginica
        6.7,3.1,5.6,2.4,virginica
        6.9,3.1,5.1,2.3,virginica
        5.8,2.7,5.1,1.9,virginica
        6.8,3.2,5.9,2.3,virginica
        6.7,3.3,5.7,2.5,virginica
        6.7,3.0,5.2,2.3,virginica
        6.3,2.5,5.0,1.9,virginica
        6.5,3.0,5.2,2.0,virginica
        6.2,3.4,5.4,2.3,virginica
        5.9,3.0,5.1,1.8,virginica
        """
        
        //  Set URL as needed
        let fileManager = FileManager.default
        let docsDirURL = try fileManager.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
        let IrisDataURL = docsDirURL.appendingPathComponent("IrisData.txt")
        
        //  Create the parser
        let parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {
            InputFloatString()      //  Sepal length
            InputFloatString()      //  Sepal width
            InputFloatString()      //  Petal length
            InputFloatString()      //  Petal width
            OutputLabelString()     //  The text classification label
        }.withCommentIndicators(["s", "#", "//"])

        //  Parse the text lines concurrently
        let dataSet_Lines_Concurrent = DataSet(inputShape: TensorShape([4]), inputType: .float32, outputShape: TensorShape([3]), outputType: .float32)
        try await parser.parseTextLines(dataSet: dataSet_Lines_Concurrent, text: textLines)
        
        #expect(await dataSet_Lines_Concurrent.numSamples == 150)
        
        //  Parse the lines non-concurrently
        let dataSet_Lines_NonConcurrent = DataSet(inputShape: TensorShape([4]), inputType: .float32, outputShape: TensorShape([3]), outputType: .float32)
        try await parser.parseTextLines(dataSet: dataSet_Lines_NonConcurrent, text: textLines, maxConcurrency: 1)

        #expect(await dataSet_Lines_NonConcurrent.numSamples == 150)
        #expect(try await dataSetsMatch(dataset1: dataSet_Lines_Concurrent, dataset2: dataSet_Lines_NonConcurrent, tolerence: 1.0E-04))
        
        //  Parse the file concurrently
        let textFileReader = try #require(TextFileReader(inFileURL: IrisDataURL))
        let dataSet_File_Concurrent = DataSet(inputShape: TensorShape([4]), inputType: .float32, outputShape: TensorShape([3]), outputType: .float32)
        try await parser.parseTextFile(dataSet: dataSet_File_Concurrent, textFile: textFileReader)

        #expect(await dataSet_File_Concurrent.numSamples == 150)
        #expect(try await dataSetsMatch(dataset1: dataSet_Lines_Concurrent, dataset2: dataSet_File_Concurrent, tolerence: 1.0E-04))
        
        //  Parse the file non-concurrently
        let textFileReader2 = try #require(TextFileReader(inFileURL: IrisDataURL))
        let dataSet_File_NonConcurrent = DataSet(inputShape: TensorShape([4]), inputType: .float32, outputShape: TensorShape([3]), outputType: .float32)
        try await parser.parseTextFile(dataSet: dataSet_File_NonConcurrent, textFile: textFileReader2)

        #expect(await dataSet_File_NonConcurrent.numSamples == 150)
        #expect(try await dataSetsMatch(dataset1: dataSet_Lines_Concurrent, dataset2: dataSet_File_NonConcurrent, tolerence: 1.0E-04))
    }

    func dataSetsMatch(dataset1: DataSet, dataset2: DataSet, tolerence: Double) async throws -> Bool {
        //  Make sure the types match
        var type1 = await dataset1.inputType
        var type2 = await dataset2.inputType
        if (type1 != type2) { return false }
        type1 = await dataset1.outputType
        type2 = await dataset2.outputType
        if (type1 != type2) { return false }
        
        //  Make sure the shapes match
        var shape1 = await dataset1.inputShape
        var shape2 = await dataset2.inputShape
        if (shape1 != shape2) { return false }
        shape1 = await dataset1.outputShape
        shape2 = await dataset2.outputShape
        if (shape1 != shape2) { return false }

        //  Make sure the number of samples match
        let numSamples1 = await dataset1.numSamples
        let numSamples2 = await dataset2.numSamples
        if (numSamples1 != numSamples2) { return false }
        
        //  Make sure all the tensors match within the tolerence value
        for i in 0..<numSamples1 {
            let sample1 = try await dataset1.getSample(sampleIndex: i)
            let sample2 = try await dataset2.getSample(sampleIndex: i)
            
            //  Check the input tensors
            if (!(try sample1.inputs.compare(with: sample2.inputs, maxDifference: tolerence))) {
                print("Input tensors do not match for sample \(i)")
                return false
            }
            
            //  Check the output tensors
            if (!(try sample1.outputs.compare(with: sample2.outputs, maxDifference: tolerence))) {
                print("Output tensors do not match for sample \(i)")
                return false
            }
        }

        return true
    }
}

