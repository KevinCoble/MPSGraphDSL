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
        try parser.parseBinaryData(data, intoDataSet: dataSet)
        
        //  Check the results
        #expect(dataSet.numSamples == 1)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0).asDouble == 1.0)
        
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
        try parser.parseBinaryData(data, intoDataSet: dataSet)
        #expect(dataSet.numSamples == 4)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 1.0)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 2.0)
        #expect(try dataSet.samples[2].inputs.getElement(index: 0) == 3.0)
        #expect(try dataSet.samples[3].inputs.getElement(index: 0) == 4.0)
        
        //  Repeat samples extraction
        dataSet = DataSet(inputShape: singleInput, inputType: .double, outputShape: singleOutput, outputType: .double)
        parser = DataParser {
            UnusedData(length: 2, format: .fUInt8)
            RepeatSampleTillDone {
                InputData(length: 1, format : .fUInt8, postProcessing : .None)
            }
        }
        try parser.parseBinaryData(data, intoDataSet: dataSet)
        #expect(dataSet.numSamples == 4)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 1.0)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 2.0)
        #expect(try dataSet.samples[2].inputs.getElement(index: 0) == 3.0)
        #expect(try dataSet.samples[3].inputs.getElement(index: 0) == 4.0)
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
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 0)
        
        //  Create a test text array with some data
        text = """
        5
        7
        9
        """
        
        //  Parse the data
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 3)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 5.0)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 7.0)
        #expect(try dataSet.samples[2].inputs.getElement(index: 0) == 9.0)
        
        //  Reset the data set to empty
        dataSet = DataSet(inputShape: TensorShape([2]), inputType: .double, outputShape: TensorShape([3]), outputType: .double)

        //  Create a delineated text parser (whitespace separated) that skips the first line
        parser = DelineatedTextParser(lineSeparator: .SpaceDelimited) {
            InputIntegerString()
        }.skipInitialLines(1)
        
        //  Parse the data
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 2)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 7.0)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 9.0)
        
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
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 3)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 5.0)
        var value = try dataSet.samples[0].inputs.getElement(index: 1)
        #expect(abs(value - 6.2) < 1.0e-5)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 7.0)
        value = try dataSet.samples[1].inputs.getElement(index: 1)
        #expect(abs(value - 7.3) < 1.0e-5)
        #expect(try dataSet.samples[2].inputs.getElement(index: 0) == 9.0)
        value = try dataSet.samples[2].inputs.getElement(index: 1)
        #expect(abs(value - 8.4) < 1.0e-5)
        #expect(try dataSet.samples[0].outputs.getElement(index: 0) == 100)
        #expect(try dataSet.samples[1].outputs.getElement(index: 0) == 101)
        #expect(try dataSet.samples[2].outputs.getElement(index: 0) == 102)
        
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
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 3)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 5.0)
        value = try dataSet.samples[0].inputs.getElement(index: 1)
        #expect(abs(value - 6.2) < 1.0e-5)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 7.0)
        value = try dataSet.samples[1].inputs.getElement(index: 1)
        #expect(abs(value - 7.3) < 1.0e-5)
        #expect(try dataSet.samples[2].inputs.getElement(index: 0) == 9.0)
        value = try dataSet.samples[2].inputs.getElement(index: 1)
        #expect(abs(value - 8.4) < 1.0e-5)
        #expect(try dataSet.samples[0].outputs.getElement(index: 0) == 100)
        #expect(try dataSet.samples[1].outputs.getElement(index: 0) == 101)
        #expect(try dataSet.samples[2].outputs.getElement(index: 0) == 102)
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
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 0)
        
        //  Create a test text array with some data
        text = """
          5
          7
          9
        """
        
        //  Parse the data
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 3)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 5.0)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 7.0)
        #expect(try dataSet.samples[2].inputs.getElement(index: 0) == 9.0)
        
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
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 3)
        #expect(try dataSet.samples[0].inputs.getElement(index: 0) == 5.0)
        var value = try dataSet.samples[0].inputs.getElement(index: 1)
        #expect(abs(value - 6.2) < 1.0e-5)
        #expect(try dataSet.samples[1].inputs.getElement(index: 0) == 7.0)
        value = try dataSet.samples[1].inputs.getElement(index: 1)
        #expect(abs(value - 7.3) < 1.0e-5)
        #expect(try dataSet.samples[2].inputs.getElement(index: 0) == 9.0)
        value = try dataSet.samples[2].inputs.getElement(index: 1)
        #expect(abs(value - 8.4) < 1.0e-5)
        #expect(try dataSet.samples[0].outputs.getElement(index: 0) == 100)
        #expect(try dataSet.samples[1].outputs.getElement(index: 0) == 101)
        #expect(try dataSet.samples[2].outputs.getElement(index: 0) == 102)
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
        try parser.parseBinaryData(data, intoDataSet: dataSet)
        try dataSet.samples[0].inputs.print2D(elementWidth: 5, precision: 1)
        
        //  Check the results
        #expect(dataSet.numSamples == 1)
        let tensor = dataSet.samples[0].inputs
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
        try parser.parseTextLines(dataSet: dataSet, text: text)
        
        //  Check the result
        #expect(dataSet.numSamples == 4)
        #expect(abs(try dataSet.samples[0].inputs.getElement(index: 1) - 5.0) < 1.0E-5)
        #expect(abs(try dataSet.samples[0].inputs.getElement(index: 5) - 2.3) < 1.0E-5)
        #expect(abs(try dataSet.samples[1].inputs.getElement(index: 1) - 5.4) < 1.0E-5)
        #expect(abs(try dataSet.samples[1].inputs.getElement(index: 5) - 4.3) < 1.0E-5)
        #expect(abs(try dataSet.samples[2].inputs.getElement(index: 1) - 11.8) < 1.0E-5)
        #expect(abs(try dataSet.samples[2].inputs.getElement(index: 5) - 2.4) < 1.0E-5)

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
        try parser.parseTextLines(dataSet: dataSet, text: text)
         
        //  Check the result
        #expect(dataSet.numSamples == 4)
        #expect(abs(try dataSet.samples[0].inputs.getElement(location: [0, 0]) - 5.0) < 1.0E-5)
        #expect(abs(try dataSet.samples[0].inputs.getElement(location: [1, 1]) - 2.3) < 1.0E-5)
        #expect(abs(try dataSet.samples[1].inputs.getElement(location: [0, 0]) - 5.4) < 1.0E-5)
        #expect(abs(try dataSet.samples[1].inputs.getElement(location: [1, 1]) - 4.3) < 1.0E-5)
        #expect(abs(try dataSet.samples[2].inputs.getElement(location: [0, 0]) - 11.8) < 1.0E-5)
        #expect(abs(try dataSet.samples[2].inputs.getElement(location: [1, 1]) - 2.4) < 1.0E-5)
    }
}

