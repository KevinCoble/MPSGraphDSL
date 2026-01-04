//
//  DataParser.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 6/23/21.
//

import Foundation


///  Format of file to be parsed
public enum DataFileFormat : Int {
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
class ParsingData {
    var inputSize : Int
    var outputSize : Int
    var currentSample = -1
    var currentInputLocation : [Int]
    var currentOutputLocation : [Int]
    var inputLocationOffsets : [Int]     //  Offsets into array for each dimension
    var outputLocationOffsets : [Int]     //  Offsets into array for each dimension
    var inputNormalizationMap : [Int]
    var outputNormalizationMap : [Int]
    
    init(forDataSet: DataSet) {
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
}

///  Binary data parser and parent class for the text parsers
public class DataParser {
    var dataFormat : DataFileFormat
    var chunks : [DataChunk]
    var numSkipLines : Int
    var commentIndicators : [String]
    
    ///  Initializer to create a DataParser with a trailing closure of a list of DataChunks
    public convenience init(@DataParserBuilder _ buildChunks: () -> [DataChunk]) {
        self.init()
        chunks = buildChunks()
    }

    ///  Initializer for creating a parser
    public init(dataFormat : DataFileFormat = .Binary) {
        self.dataFormat = dataFormat
        chunks = []
        numSkipLines = 0
        commentIndicators = []
    }

    ///  Initializer for createing a text parser
    public init(dataFormat : DataFileFormat, chunks: [DataChunk], numSkipLines : Int, commentIndicators : [String]) {
        self.dataFormat = dataFormat
        self.chunks = chunks
        self.numSkipLines = numSkipLines
        self.commentIndicators = commentIndicators
    }
    
    ///  Copy constructor
    public init(copyFrom: DataParser) {
        self.dataFormat = copyFrom.dataFormat
        self.chunks = copyFrom.chunks
        self.numSkipLines = copyFrom.numSkipLines
        self.commentIndicators = copyFrom.commentIndicators
    }


    // MARK: - Parsing
    
    /// Parse a binary file that is read from the given URL
    /// - Parameters:
    ///   - url: The URL of the file to be parsed
    ///   - intoDataSet: the DataSet to parse the file into
    public func parseBinaryFile(url: URL, intoDataSet: DataSet) throws
    {
        guard let inputStream = InputStream(url: url) else {
            throw PersistanceErrors.UnableToOpenFile(url.absoluteString)
        }
        if let error = inputStream.streamError {
            throw PersistanceErrors.InputStreamError("Unable to open file - " + error.localizedDescription)
        }

        try parseBinaryFile(inputFile : inputStream, intoDataSet: intoDataSet)
    }

    /// Parse a binary file that is read from the given InputStream
    /// - Parameters:
    ///   - inputFile: The InputStream of the file to be parsed
    ///   - intoDataSet: the DataSet to parse the file into
    public func parseBinaryFile(inputFile : InputStream, intoDataSet: DataSet) throws
    {
        //  Create and initialize the parsing data
        let parsingData = ParsingData(forDataSet: intoDataSet)
        
        //  Open the file
        inputFile.open()
        defer { inputFile.close() }

        //  Process each chunk
        for chunk in chunks {
            try chunk.parseBinaryChunk(dataSet: intoDataSet, inputFile: inputFile, parsingData: parsingData)
        }
    }
    
    /// Parse a byte array thats in the form of a Data object
    /// - Parameters:
    ///   - data: The Data object to be parsed
    ///   - intoDataSet: the DataSet to parse the binary data into
    public func parseBinaryData(_ data: Data, intoDataSet: DataSet) throws
    {
        let parsingData = ParsingData(forDataSet: intoDataSet)
        
        //  Start at the beginning of the data
         var offset = 0
        
        //  Process each chunk
        for chunk in chunks {
            try chunk.parseBinaryChunk(intoDataSet: intoDataSet, data: data, offset: &offset, parsingData : parsingData)
        }
    }
    
    /// Parse a set of text lines in String format into a DataSet
    /// - Parameters:
    ///   - dataSet: The DataSet to receive the parsed data
    ///   - text: The text lines to parse
    public func parseTextLines(dataSet: DataSet, text : String) throws
    {
        let parsingData = ParsingData(forDataSet: dataSet)
        
        var lines = text.components(separatedBy: .newlines)
        
        //  Bypass any skip lines
        if (numSkipLines > 0) {
            if (numSkipLines >= lines.count) { throw DataParsingErrors.CannotReadEnoughSkipLines }
            lines = Array(lines[numSkipLines...])
        }
        
        //  Process each line
        for line in lines {
            try parseTextLine(textLine: line, dataSet: dataSet, parsingData : parsingData)
        }
    }
    
    /// Parse a set of text lines from a TextFileReader object
    /// - Parameters:
    ///   - dataSet: The DataSet to receive the parsed data
    ///   - textFile: the TextFileReader object for the text file that will be parsed
    public func parseTextFile(dataSet: DataSet, textFile : TextFileReader) throws
    {
        let parsingData = ParsingData(forDataSet: dataSet)
        
        //  Bypass any skip lines
        if (numSkipLines > 0) {
            for _ in 0..<numSkipLines {
                let line = textFile.readLine()
                if (line == nil) { throw DataParsingErrors.CannotReadEnoughSkipLines }
            }
        }
        
        //  Process lines till the end of the file
        while (true) {
            let line = textFile.readTrimmedLine()
            if (line == nil) { break }
            
            try parseTextLine(textLine: line!, dataSet: dataSet, parsingData : parsingData)
        }
    }

    func parseTextLine(textLine : String, dataSet: DataSet, parsingData : ParsingData) throws
    {
        var line : String
        if (dataFormat == .FixedColumns) {
            line = textLine.trimmingCharacters(in: .newlines)
        }
        else {
            line = textLine.trimmingCharacters(in: .whitespaces)
        }
        
        //  See if the line is a comment line
        if (lineIsComment(line))  { return }
        
        //  Add a sample
        dataSet.incrementSample(parsingData : parsingData)
        
        //  Parse the line based on the format
        switch (dataFormat) {
        case .CommaSeparated:
            let components = line.components(separatedBy: CharacterSet(charactersIn: ","))
            //  Process each chunk
            var componentOffset = 0
            for chunk in chunks {
                let usedComponents = try chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingData)
                componentOffset += usedComponents
            }

        case .SpaceDelimited:
            let components = line.components(separatedBy: .whitespaces)
            //  Process each chunk
            var componentOffset = 0
            for chunk in chunks {
                let usedComponents = try chunk.parseTextChunk(dataSet: dataSet, components: components, offset : componentOffset, parsingData: parsingData)
                componentOffset += usedComponents
            }

        case .FixedColumns:
            //  Process each chunk
            var startIndex = line.startIndex
            for chunk in chunks {
                let finalIndex = try chunk.parseFixedWidthTextChunk(dataSet: dataSet, string: line, index : startIndex, parsingData: parsingData)
                startIndex = finalIndex!
            }
            
        default:
            throw DataParsingErrors.UnsupportedTextFormat
        }
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
