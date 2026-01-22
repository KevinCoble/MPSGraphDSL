//
//  DataParser.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 6/23/21.
//

import Foundation


///  Format of file to be parsed
public enum DataFileFormat : Int, Sendable {
    ///  Text file with fixed columns for data portions
    case FixedColumns = 1
    ///  Text file with comma separated data pieces
    case CommaSeparated = 2
    ///  Text file with whitespace separated data pieces
    case SpaceDelimited = 3
    ///  Folder with individual image files
    case ImagesInFolders = 4
    ///  Binary encoded file
    case Binary = 5
}


//  Parsing data
actor ParsingData {
    let inputShape : [Int]
    let outputShape : [Int]
    let inputSize : Int
    let outputSize : Int
    var currentSampleIndex = -1
    var currentSample: DataSample? = nil
    var currentInputLocation : [Int]
    var currentOutputLocation : [Int]
    var inputLocationOffsets : [Int]     //  Offsets into array for each dimension
    var outputLocationOffsets : [Int]     //  Offsets into array for each dimension
    var inputNormalizationMap : [Int]
    var outputNormalizationMap : [Int]
    
    init(forDataSet: DataSet) {
        inputShape = forDataSet.inputShape.dimensions
        outputShape = forDataSet.outputShape.dimensions
        inputSize = forDataSet.inputShape.dimensions.reduce(1, *)
        outputSize = forDataSet.outputShape.dimensions.reduce(1, *)

        //  Set up the data location arrays
        //!!  change to get input and output shapes
        currentInputLocation = [Int](repeating: 0, count: forDataSet.inputShape.numDimensions)
        currentOutputLocation = [Int](repeating: 0, count: forDataSet.outputShape.numDimensions)
        inputLocationOffsets = [Int](repeating: 1, count: forDataSet.inputShape.numDimensions)
        outputLocationOffsets = [Int](repeating: 1, count: forDataSet.outputShape.numDimensions)
        if (forDataSet.inputShape.numDimensions > 1) {
            for index in stride(from: forDataSet.inputShape.numDimensions-2, through: 0, by: -1) {
                inputLocationOffsets[index] = inputLocationOffsets[index+1] * forDataSet.inputShape.dimensions[index+1]
            }
        }
        if (forDataSet.outputShape.numDimensions > 1) {
            for index in stride(from: forDataSet.outputShape.numDimensions-2, through: 0, by: -1) {
                outputLocationOffsets[index] = outputLocationOffsets[index+1] * forDataSet.outputShape.dimensions[index+1]
            }
        }
        inputNormalizationMap = [Int](repeating: 0, count: inputSize)
        outputNormalizationMap = [Int](repeating: 0, count: outputSize)
    }
    
    //  Copy constructor
    init(copyFrom: ParsingData) async {
        //  Get all the parameters at once to avoid concurrency issues
        let parameters = await copyFrom.getAllParameters()
        inputShape = parameters.inputShape
        outputShape = parameters.outputShape
        inputSize = parameters.inputSize
        outputSize  = parameters.outputSize
        currentSampleIndex = parameters.currentSampleIndex
        currentSample = parameters.currentSample
        currentInputLocation = parameters.currentInputLocation
        currentOutputLocation = parameters.currentOutputLocation
        inputLocationOffsets = parameters.inputLocationOffsets
        outputLocationOffsets = parameters.outputLocationOffsets
        inputNormalizationMap = parameters.inputNormalizationMap
        outputNormalizationMap = parameters.outputNormalizationMap
    }
    
    func getAllParameters() -> (
        inputShape : [Int],
        outputShape : [Int],
        inputSize : Int,
        outputSize : Int,
        currentSampleIndex: Int,
        currentSample: DataSample?,
        currentInputLocation : [Int],
        currentOutputLocation : [Int],
        inputLocationOffsets : [Int],
        outputLocationOffsets : [Int],
        inputNormalizationMap : [Int],
        outputNormalizationMap : [Int]
    )
    {
        return (
            inputShape : inputShape,
            outputShape : outputShape,
            inputSize : inputSize,
            outputSize : outputSize,
            currentSampleIndex: currentSampleIndex,
            currentSample: currentSample,
            currentInputLocation : currentInputLocation,
            currentOutputLocation : currentOutputLocation,
            inputLocationOffsets : inputLocationOffsets,
            outputLocationOffsets : outputLocationOffsets,
            inputNormalizationMap : inputNormalizationMap,
            outputNormalizationMap : outputNormalizationMap
        )
    }
    
    // MARK: - Parsing

    func inputLocationInRange() -> Bool {
        for i in 0..<inputShape.count {
            if (currentInputLocation[i] >= inputShape[i]) {
                return false
            }
        }
        return true
    }
    
    func outputLocationInRange() -> Bool {
        for i in 0..<outputShape.count {
            if (currentOutputLocation[i] >= outputShape[i]) {
                return false
            }
        }
        return true
    }

    //  Get input storage index for parsing
    func getInputSampleStorageIndex() -> Int {
        var index = 0
        for dimension in 0..<inputShape.count {
            index += currentInputLocation[dimension] * inputLocationOffsets[dimension]
        }

        return index
    }
    
    //  Get output storage index for parsing
    func getOutputSampleStorageIndex() -> Int {
        var index = 0
        for dimension in 0..<outputShape.count {
            index += currentOutputLocation[dimension] * outputLocationOffsets[dimension]
        }

        return index
    }
    
    func haveAddedData() -> Bool {
        let inputSum = currentInputLocation.reduce(0, +)
        let outputSum = currentOutputLocation.reduce(0, +)
        if (inputSum == 0 && outputSum == 0) { return false }
        return true
    }
    
    func putSampleBackIntoDataSet(dataSet: DataSet) async throws
    {
        //  Make sure we have a sample
        if (currentSampleIndex < 0) { return }
        
        if let currentSample = currentSample {
            try await dataSet.setSample(currentSample, sampleIndex: currentSampleIndex)
        }
        else {
            throw DataParsingErrors.AttemptToPutNilSampleInDataSet
        }
    }
    
    func dropCurrentSample() {
        currentSampleIndex = -1
        currentSample = nil
    }
    
    func getSampleFromDataSet(sampleIndex: Int, dataSet: DataSet) async throws
    {
        currentSampleIndex = sampleIndex
        let sample = try await dataSet.getSample(sampleIndex: sampleIndex)
        currentSample = sample

        //  Store location starts at the beginning for the sample
        currentInputLocation = [Int](repeating: 0, count: inputShape.count)
        currentOutputLocation = [Int](repeating: 0, count: outputShape.count)
    }

    func appendOutputClass(_ sampleClass : Int) throws {
        //  Validate the class fits with the output size
        if (sampleClass < 0 || ((sampleClass != 1 || outputSize != 1) && sampleClass >= outputSize)) {
            throw GenericMPSGraphDSLErrors.ClassificationValueOutOfRange
        }
        //  Set the class and output data
        currentSample!.outputClass = sampleClass
        try currentSample!.outputs.setOneHot(hot: sampleClass)
    }
    
    func appendInputData(_ inputArray : [Double], normalizationIndex : Int?) throws
    {
        //  Store the values
        let startIndex = getInputSampleStorageIndex()
        
        //  Verify locations and calculate the normalization map
        var index = startIndex
        let lastDimension = inputShape.count - 1
        for _ in inputArray {
            //  Verify we are in range
            if (!inputLocationInRange()) { throw DataParsingErrors.InputLocationOutOfRange }
            
            //  Calculate the normalization map
            if (currentSampleIndex == 0 && inputNormalizationMap[index] == 0) {
                if let normIndex = normalizationIndex {
                    if (normIndex > 0) {
                        inputNormalizationMap[index] = normIndex
                    }
                    else {
                        for i in 0..<inputSize { inputNormalizationMap[i] = normIndex }
                    }
                }
            }
            //  Increment the location
            currentInputLocation[lastDimension] += 1
            index += 1
        }
        
        //  Store the values
        try currentSample!.inputs.setElements(startIndex: startIndex, values: inputArray)
    }
    
    func appendColorData(_ inputArray : [Double], channel: ColorChannel, normalizationIndex : Int?) throws
    {
        for newValue in inputArray {
            //  Set the color as the third dimension (X-Y pixel grid)
            currentInputLocation[2] = channel.rawValue
            if (currentInputLocation[2] >= inputShape[2]) {
                throw DataParsingErrors.ColorChannelOutOfRange
            }
            //  Verify we are in range
            if (!inputLocationInRange()) { throw DataParsingErrors.InputLocationOutOfRange }
            //  Store the value
            let index = getInputSampleStorageIndex()
            try currentSample!.inputs.setElement(index: index, value: newValue)
            if (currentSampleIndex == 0 && inputNormalizationMap[index] == 0) {
                if let normIndex = normalizationIndex {
                    if (normIndex > 0) {
                        inputNormalizationMap[index] = normIndex
                    }
                    else {
                        for i in 0..<inputSize { inputNormalizationMap[i] = normIndex }
                    }
                }
            }
            //  Increment the location
            currentInputLocation[0] += 1
        }
    }
    
    func appendOutputData(_ outputArray : [Double], normalizationIndex : Int?) throws
    {
        //  Store the values
        let startIndex = getOutputSampleStorageIndex()
        
        //  Verify locations and calculate the normalization map
        var index = startIndex
        for _ in outputArray {
            //  Verify we are in range
            if (!outputLocationInRange()) { throw DataParsingErrors.OutputLocationOutOfRange }
            
            //  Calculate the normalization map
            if (currentSampleIndex == 0 && outputNormalizationMap[index] == 0) {
                if let normIndex = normalizationIndex {
                    if (normIndex > 0) {
                        outputNormalizationMap[index] = normIndex
                    }
                    else {
                        for i in 0..<outputSize { outputNormalizationMap[i] = normIndex }
                    }
                }
            }
            
            //  Increment the location
            currentOutputLocation[0] += 1
            index += 1
        }
        
        //  Store the values
        try currentSample!.outputs.setElements(startIndex: startIndex, values: outputArray)
    }
    
    func appendOutputLabel(_ labelIndex : Int, normalizationIndex : Int?) throws
    {
        //  Get the label index
        if (labelIndex >= outputSize) { throw DataParsingErrors.MoreUniqueLabelsThanOutputDimension }
        
        //  Store the values
        try appendOutputClass(labelIndex)
    }
    
    func incrementInputDimension(dimension: Int)
    {
        if (dimension < currentInputLocation.count) { currentInputLocation[dimension] += 1 }
    }
    
    func incrementOutputDimension(dimension: Int)
    {
        if (dimension < currentOutputLocation.count) { currentOutputLocation[dimension] += 1 }
    }
    
    func setOrIncrementInputDimension(dimension: Int, toValue: Int)
    {
        if (dimension < currentInputLocation.count) {
            if (toValue < 0) {
                currentInputLocation[dimension] += 1
            }
            else {
                currentInputLocation[dimension] = toValue
            }
        }
    }
    
    func setOrIncrementOutputDimension(dimension: Int, toValue: Int)
    {
        if (dimension < currentOutputLocation.count) {
            if (toValue < 0) {
                currentOutputLocation[dimension] += 1
            }
            else {
                currentOutputLocation[dimension] = toValue
            }
        }
    }
}

///  Binary data parser and parent class for the text parsers
public class DataParser {
    var dataFormat : DataFileFormat
    var chunks : [DataChunk]
    var numSkipLines : Int
    var commentIndicators : [String]
    var maxConcurrent = 4  //  Set less than 1 to turn off concurrency

    ///  Initializer to create a DataParser with a trailing closure of a list of DataChunks
    public convenience init(@DataParserBuilder _ buildChunks: () -> [DataChunkCreator]) {
        self.init()
        chunks = buildChunks().map { $0.createDataChunk()}
    }

    ///  Initializer for creating a parser
    public init(dataFormat : DataFileFormat = .Binary) {
        self.dataFormat = dataFormat
        chunks = []
        numSkipLines = 0
        commentIndicators = []
    }

    ///  Initializer for creating a text parser
    public init(dataFormat : DataFileFormat, chunks: [DataChunk], numSkipLines : Int, commentIndicators : [String]) {
        self.dataFormat = dataFormat
        self.chunks = chunks
        self.numSkipLines = numSkipLines
        self.commentIndicators = commentIndicators
    }


    // MARK: - Parsing
    
    //  Determine if there is a top-level sample repeat node
    internal func hasSampleRepeatTillDone() -> Bool {
        for chunk in chunks {
            if (chunk.format == .rSample && chunk.length > 999999) { return true }
        }
        return false
    }
    
    /// Parse a binary file that is read from the given URL
    /// - Parameters:
    ///   - url: The URL of the file to be parsed
    ///   - intoDataSet: the DataSet to parse the file into
    ///   - maxConcurrency: (Optional) the maximum number of concurrent processing tasks.  Set to 1 or below to turn off concurrency.  Default is 4
    public func parseBinaryFile(url: URL, intoDataSet: DataSet, maxConcurrency : Int = 4) async throws
    {
        guard let inputStream = InputStream(url: url) else {
            throw PersistanceErrors.UnableToOpenFile(url.absoluteString)
        }
        if let error = inputStream.streamError {
            throw PersistanceErrors.InputStreamError("Unable to open file - " + error.localizedDescription)
        }

        try await parseBinaryFile(inputFile : inputStream, intoDataSet: intoDataSet, maxConcurrency : maxConcurrency)
    }

    /// Parse a binary file that is read from the given InputStream
    /// - Parameters:
    ///   - inputFile: The InputStream of the file to be parsed
    ///   - intoDataSet: the DataSet to parse the file into
    ///   - maxConcurrency: (Optional) the maximum number of concurrent processing tasks.  Set to 1 or below to turn off concurrency.  Default is 4
    public func parseBinaryFile(inputFile : InputStream, intoDataSet: DataSet, maxConcurrency : Int = 4) async throws
    {
        //  Create and initialize the parsing data
        maxConcurrent = maxConcurrency
        let parsingData = ParsingData(forDataSet: intoDataSet)
        
        //  Open the file
        inputFile.open()
        defer { inputFile.close() }

        //  Process each chunk
        var didConcurrentParse: Bool = false
        for chunk in chunks {
            //  If chunk is a repeat sample, see if we should parse concurrently
            if (maxConcurrent > 1 && chunk.format == .rSample && chunk.length > 999999) {
                if let byteLength = chunk.getRequiredBytes() {
                    try await chunk.repeatBinarySample(ofLength: byteLength, intoDataSet: intoDataSet, inputFile: inputFile, parsingData : parsingData, maxConcurrent: maxConcurrent)
                    didConcurrentParse = true
                }
                else {
                    //  Indeterminate length of contents - parse sequentially
                    try await chunk.parseBinaryChunk(dataSet: intoDataSet, inputFile: inputFile, parsingData: parsingData)
                }
            }
            else {
                try await chunk.parseBinaryChunk(dataSet: intoDataSet, inputFile: inputFile, parsingData: parsingData)
            }
        }
        
        //  Put back any last data
        if (!didConcurrentParse) {
            try await parsingData.putSampleBackIntoDataSet(dataSet: intoDataSet)
        }
    }
    
    /// Parse a byte array that's in the form of a Data object
    /// - Parameters:
    ///   - data: The Data object to be parsed
    ///   - intoDataSet: the DataSet to parse the binary data into
    ///   - maxConcurrency: (Optional) the maximum number of concurrent processing tasks.  Set to 1 or below to turn off concurrency.  Default is 4
    public func parseBinaryData(_ data: Data, intoDataSet: DataSet, maxConcurrency : Int = 4) async throws
    {
        maxConcurrent = maxConcurrency
        let parsingData = ParsingData(forDataSet: intoDataSet)
        
        //  Start at the beginning of the data
         var offset = 0
        
        //  Process each chunk
        var didConcurrentParse: Bool = false
        for chunk in chunks {
            //  If chunk is a repeat sample, see if we should parse concurrently
            if (maxConcurrent > 1 && chunk.format == .rSample && chunk.length > 999999) {
                if let byteLength = chunk.getRequiredBytes() {
                    try await chunk.repeatBinarySample(ofLength: byteLength, intoDataSet: intoDataSet, data: data, offset: &offset, parsingData : parsingData, maxConcurrent: maxConcurrent)
                    didConcurrentParse = true
                }
                else {
                    //  Indeterminate length of contents - parse sequentially
                    try await chunk.parseBinaryChunk(intoDataSet: intoDataSet, data: data, offset: &offset, parsingData : parsingData)
                }
            }
            else {
                try await chunk.parseBinaryChunk(intoDataSet: intoDataSet, data: data, offset: &offset, parsingData : parsingData)
            }
        }
        //  Put back any last data
        if (!didConcurrentParse) {
            try await parsingData.putSampleBackIntoDataSet(dataSet: intoDataSet)
        }
    }
    
    /// Parse a set of text lines in String format into a DataSet
    /// - Parameters:
    ///   - dataSet: The DataSet to receive the parsed data
    ///   - text: The text lines to parse
    ///   - maxConcurrency: (Optional) the maximum number of concurrent processing tasks.  Set to 1 or below to turn off concurrency.  Default is 4
    public func parseTextLines(dataSet: DataSet, text : String, maxConcurrency : Int = 4) async throws
    {
        maxConcurrent = maxConcurrency
        var lines = text.components(separatedBy: .newlines)
        
        //  Bypass any skip lines
        if (numSkipLines > 0) {
            if (numSkipLines >= lines.count) { throw DataParsingErrors.CannotReadEnoughSkipLines }
            lines = Array(lines[numSkipLines...])
        }
        
        //  See if we should parse concurrently
        if (maxConcurrent > 1) {
            try await parseTextLinesConcurrently(dataSet: dataSet, lines: lines)
        }
        else {
            let parsingData = ParsingData(forDataSet: dataSet)
            
            //  Process each line
            for line in lines {
                //  Trim characters
                var textTine: String
                if (dataFormat == .FixedColumns) {
                    textTine = line.trimmingCharacters(in: .newlines)
                }
                else {
                    textTine = line.trimmingCharacters(in: .whitespaces)
                }
                
                //  See if the line is a comment line
                if (!lineIsComment(textTine))  {
                    //  Add a sample
                    let sampleIndex = try await dataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                    
                    try await parseTextLine(textLine: line, dataSet: dataSet, parsingData : parsingData)
                }
            }
        }
    }
    
    internal func parseTextLinesConcurrently(dataSet: DataSet, lines : [String]) async throws
    {
        let parsingData = ParsingData(forDataSet: dataSet)
        
        let format = dataFormat
        let localChunks = chunks

        var numSubmitted = 0
        try await withThrowingTaskGroup(of: Void.self) { taskGroup in            
            //  Process each line
            for line in lines {
                //  Trim characters
                var textTine: String
                if (dataFormat == .FixedColumns) {
                    textTine = line.trimmingCharacters(in: .newlines)
                }
                else {
                    textTine = line.trimmingCharacters(in: .whitespaces)
                }
                
                //  See if the line is a comment line
                if (!lineIsComment(textTine))  {
                    //  Add a sample
                    let sampleIndex = try await dataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                    
                    //  Make a copy of the parsingData
                    let parsingDataCopy = await ParsingData(copyFrom: parsingData)
                    
                    //  If less than max concurrency submitted, add it immediately.  Otherwise wait for one to finish
                    if (numSubmitted >= maxConcurrent) {
                        try await taskGroup.next()
                    }

                    let added = taskGroup.addTaskUnlessCancelled {
                        //  Parse the line based on the format
                        switch (format) {
                        case .CommaSeparated:
                            let components = line.components(separatedBy: CharacterSet(charactersIn: ","))
                            //  Process each chunk
                            var componentOffset = 0
                            for chunk in localChunks {
                                let usedComponents = try await chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingDataCopy)
                                componentOffset += usedComponents
                            }

                        case .SpaceDelimited:
                            let components = line.components(separatedBy: .whitespaces)
                            //  Process each chunk
                            var componentOffset = 0
                            for chunk in localChunks {
                                let usedComponents = try await chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingDataCopy)
                                componentOffset += usedComponents
                            }

                        case .FixedColumns:
                            //  Process each chunk
                            var startIndex = line.startIndex
                            for chunk in localChunks {
                                let finalIndex = try await chunk.parseFixedWidthTextChunk(dataSet: dataSet, string: line, index : startIndex, parsingData: parsingDataCopy)
                                startIndex = finalIndex!
                            }
                            
                        default:
                            throw DataParsingErrors.UnsupportedTextFormat
                        }
                        
                        //  Put the sample we just parsed into the dataset
                        try await parsingDataCopy.putSampleBackIntoDataSet(dataSet: dataSet)
                    }
                    if (added) { numSubmitted += 1 }
                }
            }
        }
    }

    /// Parse a set of text lines from a TextFileReader object
    /// - Parameters:
    ///   - dataSet: The DataSet to receive the parsed data
    ///   - textFile: the TextFileReader object for the text file that will be parsed
    ///   - maxConcurrency: (Optional) the maximum number of concurrent processing tasks.  Set to 1 or below to turn off concurrency.  Default is 4
    public func parseTextFile(dataSet: DataSet, textFile : TextFileReader, maxConcurrency : Int = 4) async throws
    {
        maxConcurrent = maxConcurrency

        //  Bypass any skip lines
        if (numSkipLines > 0) {
            for _ in 0..<numSkipLines {
                let line = textFile.readLine()
                if (line == nil) { throw DataParsingErrors.CannotReadEnoughSkipLines }
            }
        }
        
        //  See if we should parse concurrently
        if (maxConcurrent > 1) {
            try await parseTextFileConcurrently(dataSet: dataSet, textFile: textFile)
        }
        else {
            let parsingData = ParsingData(forDataSet: dataSet)

            //  Process lines till the end of the file
            while (true) {
                var line = textFile.readTrimmedLine()
                if (line == nil) { break }
                
                //  Trim characters
                if (dataFormat == .FixedColumns) {
                    line = line!.trimmingCharacters(in: .newlines)
                }
                else {
                    line = line!.trimmingCharacters(in: .whitespaces)
                }
                
                //  See if the line is a comment line
                if (!lineIsComment(line!))  {
                    //  Add a sample
                    let sampleIndex = try await dataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                    
                    try await parseTextLine(textLine: line!, dataSet: dataSet, parsingData : parsingData)
                }
            }
        }
    }
    
    internal func parseTextFileConcurrently(dataSet: DataSet, textFile : TextFileReader) async throws
    {
        let parsingData = ParsingData(forDataSet: dataSet)
        
        let format = dataFormat
        let localChunks = chunks

        var numSubmitted = 0
        try await withThrowingTaskGroup(of: Void.self) { taskGroup in
            //  Process each line
            while true {
                let line = textFile.readTrimmedLine()
                if (line == nil) { break }
                //  Trim characters
                var textTine: String
                if (dataFormat == .FixedColumns) {
                    textTine = line!.trimmingCharacters(in: .newlines)
                }
                else {
                    textTine = line!.trimmingCharacters(in: .whitespaces)
                }
                
                //  See if the line is a comment line
                if (!lineIsComment(textTine))  {
                    //  Add a sample
                    let sampleIndex = try await dataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                    
                    //  Make a copy of the parsingData
                    let parsingDataCopy = await ParsingData(copyFrom: parsingData)
                    
                    //  If less than max concurrency submitted, add it immediately.  Otherwise wait for one to finish
                    if (numSubmitted >= maxConcurrent) {
                        try await taskGroup.next()
                    }

                    let added = taskGroup.addTaskUnlessCancelled {
                        //  Parse the line based on the format
                        switch (format) {
                        case .CommaSeparated:
                            let components = line!.components(separatedBy: CharacterSet(charactersIn: ","))
                            //  Process each chunk
                            var componentOffset = 0
                            for chunk in localChunks {
                                let usedComponents = try await chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingDataCopy)
                                componentOffset += usedComponents
                            }

                        case .SpaceDelimited:
                            let components = line!.components(separatedBy: .whitespaces)
                            //  Process each chunk
                            var componentOffset = 0
                            for chunk in localChunks {
                                let usedComponents = try await chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingDataCopy)
                                componentOffset += usedComponents
                            }

                        case .FixedColumns:
                            //  Process each chunk
                            var startIndex = line!.startIndex
                            for chunk in localChunks {
                                let finalIndex = try await chunk.parseFixedWidthTextChunk(dataSet: dataSet, string: line!, index : startIndex, parsingData: parsingDataCopy)
                                startIndex = finalIndex!
                            }
                            
                        default:
                            throw DataParsingErrors.UnsupportedTextFormat
                        }
                        
                        //  Put the sample we just parsed into the dataset
                        try await parsingDataCopy.putSampleBackIntoDataSet(dataSet: dataSet)
                    }
                    if (added) { numSubmitted += 1 }
                }
            }
        }
    }

    func parseTextLine(textLine : String, dataSet: DataSet, parsingData : ParsingData) async throws
    {
        //  Parse the line based on the format
        switch (dataFormat) {
        case .CommaSeparated:
            let components = textLine.components(separatedBy: CharacterSet(charactersIn: ","))
            //  Process each chunk
            var componentOffset = 0
            for chunk in chunks {
                let usedComponents = try await chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingData)
                componentOffset += usedComponents
            }

        case .SpaceDelimited:
            let components = textLine.components(separatedBy: .whitespaces)
            //  Process each chunk
            var componentOffset = 0
            for chunk in chunks {
                let usedComponents = try await chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingData)
                componentOffset += usedComponents
            }

        case .FixedColumns:
            //  Process each chunk
            var startIndex = textLine.startIndex
            for chunk in chunks {
                let finalIndex = try await chunk.parseFixedWidthTextChunk(dataSet: dataSet, string: textLine, index : startIndex, parsingData: parsingData)
                startIndex = finalIndex!
            }
            
        default:
            throw DataParsingErrors.UnsupportedTextFormat
        }
        
        //  Put the sample we just parsed into the dataset
        try await parsingData.putSampleBackIntoDataSet(dataSet: dataSet)
    }
    
    /// Determine if a text line meets the specified comment definition
    /// - Parameter line: The text line to be looked at
    /// - Returns: true if the text line meets the comment line specification, else false
    public func lineIsComment(_ line: String) -> Bool {
        for indicator in commentIndicators {
            if (line.hasPrefix(indicator)) { return true }
        }
        return false
    }
}

// MARK: - TextFileReader

///  Helper class for reading and parsing text fiels
public class TextFileReader
{
    let fileHandle : FileHandle?
    var lineDelimiter : String
    let fileURL : URL
    var currentOffset : UInt64
    var chunkSize : Int
    var totalFileLength : UInt64 = 0
    
    /// Create a TextFileReader from a URL
    /// - Parameter inFileURL: The URL to open and parse
    public init?(inFileURL: URL)
    {
        
        lineDelimiter = "\n"
        fileURL = inFileURL
        currentOffset = 0
        chunkSize = 16
        
        do {
            fileHandle = try FileHandle(forReadingFrom: fileURL)
            if (fileHandle == nil) {
                return nil;
            }
            fileHandle!.seekToEndOfFile()
            totalFileLength = fileHandle!.offsetInFile
            //we don't need to seek back, since readLine will do that.
        }
        catch {
            return nil
        }
    }
    
    deinit
    {
        if (fileHandle != nil) {fileHandle!.closeFile()}
        currentOffset = 0
    }
    
    /// Read and return the next text line
    /// - Returns: The next line, or nil if at the end of file or another error occurred
    public func readLine() -> String?
    {
        if (fileHandle == nil) { return nil }
        if (currentOffset >= totalFileLength) { return nil }
        
        let newLineData = lineDelimiter.data(using: String.Encoding.utf8)
        if (newLineData == nil) { return nil }
        fileHandle!.seek(toFileOffset: currentOffset)
        var currentData = Data()
        var shouldReadMore = true
        
        while (shouldReadMore) {
            if (self.currentOffset >= self.totalFileLength) { break; }
            var chunk = fileHandle!.readData(ofLength: chunkSize)
            if let delimiterRange = chunk.range(of: newLineData!) {
                //  Include the length so we can include the delimiter in the string
                chunk.removeSubrange((delimiterRange.lowerBound + newLineData!.count)..<chunk.count)
                shouldReadMore = false
            }
            currentData.append(chunk)
            currentOffset += UInt64(chunk.count)
        }
        
        let line = String(data: currentData, encoding: .utf8)
        return line;
    }
    
    func getOffsetOfLineDelimiter(data : Data, delimiter : Data) -> Int?
    {
        return nil
    }
    
    /// Read and return the next text line from a file assumed to be encoded in ASCII
    /// - Returns: The next line, or nil if at the end of file or another error occurred
    public func readASCIILine() -> String?
    {
        if (fileHandle == nil) { return nil }
        if (currentOffset >= totalFileLength) { return nil }
        
        let newLineData = lineDelimiter.data(using: String.Encoding.ascii)
        if (newLineData == nil) { return nil }
        fileHandle!.seek(toFileOffset: currentOffset)
        var currentData = Data()
        var shouldReadMore = true
        
        while (shouldReadMore) {
            if (self.currentOffset >= self.totalFileLength) { break; }
            var chunk = fileHandle!.readData(ofLength: chunkSize)
            if let delimiterRange = chunk.range(of: newLineData!) {
                //  Include the length so we can include the delimiter in the string
                chunk.removeSubrange((delimiterRange.lowerBound + newLineData!.count)..<chunk.count)
                shouldReadMore = false
            }
            currentData.append(chunk)
            currentOffset += UInt64(chunk.count)
        }
        
        let line = String(data: currentData, encoding: .ascii)
        return line;
    }

    /// Read and return the next text line after trimming leading whitespace and newline characters
    /// - Returns: The next line, or nil if at the end of file or another error occurred
    public func readTrimmedLine() -> String?
    {
        let line = readLine()
        if let string = line {
            return string.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return line
    }
    
    /// Read each line and pass it to the passed in closure
    /// - Parameter closure: The closure to perform whatever operation is needed to be done on each line of the file
    public func enumerateLinesUsingBlock(closure: (String) -> Bool)
    {
        var bStop = false
        while (!bStop) {
            let line = readLine()
            if (line == nil) {return}
            bStop = closure(line!)
        }
    }
}
