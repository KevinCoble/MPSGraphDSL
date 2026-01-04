//
//  TextParserBuilder.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 7/13/21.
//

import Foundation


@resultBuilder
///  The Result Builder for building deliniated text parsers
public enum TextParserBuilder {
    public static func buildBlock(_ dataChunks: [DataChunk]...) -> [DataChunk] {
        return dataChunks.flatMap({$0})
    }
    public static func buildExpression(_ expression: DelineatedTextChunk) -> [DataChunk] {
        return [expression as! DataChunk]
    }
    public static func buildOptional(_ component: [DataChunk]?) -> [DataChunk] {
        return component ?? []
    }
    public static func buildEither(first component: [DataChunk]) -> [DataChunk] {
        return component
    }
    public static func buildEither(second component: [DataChunk]) -> [DataChunk] {
        return component
    }
}

///   Type of formatting used in a text line being parsed
public enum TextLineSeparator {
    ///  Text file with comma separated data pieces
    case CommaSeparated
    ///  Text file with whitespace separated data pieces
    case SpaceDelimited
}

///  Data parser for delineated text
public class DelineatedTextParser : DataParser {
    
    /// Initializer for a DelineatedTextParser with the result builder providing data chunks for line processing
    /// - Parameters:
    ///   - lineSeparator: The dilineator in the text that separates the items
    ///   - buildChunks: An array of DelineatedTextChunk chunks that determine how the items on the line will be processed
    public init(lineSeparator: TextLineSeparator, @TextParserBuilder _ buildChunks: () -> [DataChunk]) {
        var format : DataFileFormat
        switch (lineSeparator) {
        case .CommaSeparated:
            format = .CommaSeparated
        case .SpaceDelimited:
            format = .SpaceDelimited
        }
        super.init(dataFormat: format)
        chunks = buildChunks()
    }
    
    fileprivate override init(dataFormat : DataFileFormat, chunks: [DataChunk], numSkipLines : Int, commentIndicators : [String]) {
        super.init(dataFormat : dataFormat, chunks: chunks, numSkipLines : numSkipLines, commentIndicators : commentIndicators)
    }
    
    /// Modifier for DelineatedTextParser to skip initial lines in the text source (such as for headers, etc.)
    /// - Parameter numLinesToSkip: The number of initial lines to skip before looking for data items
    /// - Returns: The modified DelineatedTextParser
    public func skipInitialLines(_ numLinesToSkip : Int) -> DelineatedTextParser {
        return DelineatedTextParser(dataFormat: dataFormat, chunks: chunks, numSkipLines: numLinesToSkip, commentIndicators: commentIndicators)
    }
    
    /// Modifier for DelineatedTextParser to define what is a comment line in the text source
    /// - Parameter commentIndicators: An array of strings that can mark the start of a comment line
    /// - Returns: The modified DelineatedTextParser
    public func withCommentIndicators(_ commentIndicators : [String]) -> DelineatedTextParser {
        return DelineatedTextParser(dataFormat: dataFormat, chunks: chunks, numSkipLines: numSkipLines, commentIndicators: commentIndicators)
    }
}

///  Protocol to mark a DataChunk subclass as includable in a delineated text data parser
public protocol DelineatedTextChunk {
    
}

///  Delineated Text Parser chunk for a delineated string is thrown away
public class UnusedTextString : DataChunk, DelineatedTextChunk {
    /// Create an UnusedTextString chunk for a DelineatedTextParser
    public init() {
        super.init(type: .Unused, length: 1, format: .fTextString, postProcessing: .None, affects: .neither)
    }
}

///  Delineated Text Parser chunk where the delineated string is assumed to be a class label for the input tensor
public class LabelTextString : DataChunk, DelineatedTextChunk {
    /// Create an LabelTextString chunk for a DelineatedTextParser
    public init() {
        super.init(type: .Label, length: 1, format: .fTextString, postProcessing: .None, affects: .input)
    }
}

///  Delineated Text Parser chunk where the delineated string is taken as a class index integer and stored in the input tensor
public class LabelIndexString : DataChunk, DelineatedTextChunk {
    /// Create an LabelIndexString chunk for a DelineatedTextParser
    public init() {
        super.init(type: .LabelIndex, length: 1, format: .fTextInt, postProcessing: .None, affects: .input)
    }
}

///  Delineated Text Parser chunk where the delineated string is taken as an integer and stored in the input tensor
public class InputIntegerString : DataChunk, DelineatedTextChunk {
    /// Create an InputIntegerString chunk for a DelineatedTextParser
    /// - Parameter postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(postProcessing : PostReadProcessing = .None) {
        super.init(type: .Feature, length: 1, format: .fTextInt, postProcessing: postProcessing, affects: .input)
    }
}

///  Delineated Text Parser chunk where the delineated string is taken as a floating value and stored in the input tensor
public class InputFloatString : DataChunk, DelineatedTextChunk {
    /// Create an InputFloatString chunk for a DelineatedTextParser
    /// - Parameter postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(postProcessing : PostReadProcessing = .None) {
        super.init(type: .Feature, length: 1, format: .fTextFloat, postProcessing: postProcessing, affects: .input)
    }
}

///  Delineated Text Parser chunk where the delineated string is taken as an integer and stored in the output tensor
public class OutputIntegerString : DataChunk, DelineatedTextChunk {
    /// Create an OutputIntegerString chunk for a DelineatedTextParser
    /// - Parameter postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(postProcessing : PostReadProcessing = .None) {
        super.init(type: .OutputValues, length: 1, format: .fTextInt, postProcessing: postProcessing, affects: .output)
    }
}

///  Delineated Text Parser chunk where the delineated string is taken as an floating value and stored in the output tensor
public class OutputFloatString : DataChunk, DelineatedTextChunk {
    /// Create an OutputFloatString chunk for a DelineatedTextParser
    /// - Parameter postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(postProcessing : PostReadProcessing = .None) {
        super.init(type: .OutputValues, length: 1, format: .fTextFloat, postProcessing: postProcessing, affects: .output)
    }
}

///  Delineated Text Parser chunk where the delineated string is assumed to be a class label for the output tensor
public class OutputLabelString : DataChunk, DelineatedTextChunk {
    /// Create an OutputLabelString chunk for a DelineatedTextParser
    /// - Parameter postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(postProcessing : PostReadProcessing = .None) {
        super.init(type: .OutputLabel, length: 1, format: .fTextString, postProcessing: postProcessing, affects: .output)
    }
}

///  Delineated Text Parser data chunk that allows you to have a repeating block of DataChunks, where a storage dimension can be updated at the end of each loop
public class RepeatDimForDelineatedText : DataChunk, DelineatedTextChunk  {
    /// Create a RepeatDimForDelineatedText chunk for a DelineatedTextParser
    /// - Parameters:
    ///   - count: The number of times the contents are executed with the storage location updated
    ///   - dimension: The storage location dimension that gets incremented
    ///   - affects: which Tensors storage location is modified, input, output, or both
    public init(count: Int, dimension : ParserDimension, affects: SampleTensorAffect) {
        super.init(type: .Repeat, length: count, format: dimension.getFormatType(), postProcessing: .None, affects: affects)
    }
    
    /// Initializer for RepeatDimForDelineatedText that takes a list of data chunks for processing the sample
    /// - Parameters:
    ///   - count: The number of times the contents are executed with the storage location updated
    ///   - dimension: The storage location dimension that gets incremented
    ///   - affects: which Tensors storage location is modified, input, output, or both
    ///   - repeatChunks: the delineated text chunks that get repeated
    public convenience init(count: Int, dimension : ParserDimension, affects: SampleTensorAffect, @TextParserBuilder _ repeatChunks: () -> [DataChunk]) {
        self.init(count: count, dimension : dimension, affects: affects)
        self.repeatChunks = repeatChunks()
    }
}



@resultBuilder
///  The Result Builder for building fixed-column text parsers
public enum FixedTextParserBuilder {
    public static func buildBlock(_ dataChunks: [DataChunk]...) -> [DataChunk] {
        return dataChunks.flatMap({$0})
    }
    public static func buildExpression(_ expression: FixedColumnChunk) -> [DataChunk] {
        return [expression as! DataChunk]
    }
    public static func buildOptional(_ component: [DataChunk]?) -> [DataChunk] {
        return component ?? []
    }
    public static func buildEither(first component: [DataChunk]) -> [DataChunk] {
        return component
    }
    public static func buildEither(second component: [DataChunk]) -> [DataChunk] {
        return component
    }
}

///  Data parser for fixed-column text
public class FixedColumnTextParser : DataParser {
    /// Initializer for a FixedColumnTextParser with the result builder providing data chunks for line processing
    ///   - buildChunks: An array of FixedColumnChunk chunks that determine how the items on the line will be processed
    public init(@FixedTextParserBuilder _ buildChunks: () -> [DataChunk]) {
        super.init(dataFormat: .FixedColumns)
        chunks = buildChunks()
    }
    
    fileprivate override init(dataFormat : DataFileFormat, chunks: [DataChunk], numSkipLines : Int, commentIndicators : [String]) {
        super.init(dataFormat : dataFormat, chunks: chunks, numSkipLines : numSkipLines, commentIndicators : commentIndicators)
    }
    
    /// Modifier for FixedColumnTextParser to skip initial lines in the text source (such as for headers, etc.)
    /// - Parameter numLinesToSkip: The number of initial lines to skip before looking for data items
    /// - Returns: The modified FixedColumnTextParser
    public func skipInitialLines(_ numLinesToSkip : Int) -> FixedColumnTextParser {
        return FixedColumnTextParser(dataFormat: dataFormat, chunks: chunks, numSkipLines: numLinesToSkip, commentIndicators: commentIndicators)
    }
    
    /// Modifier for FixedColumnTextParser to define what is a comment line in the text source
    /// - Parameter commentIndicators: An array of strings that can mark the start of a comment line
    /// - Returns: The modified FixedColumnTextParser
    public func withCommentIndicators(_ commentIndicators : [String]) -> FixedColumnTextParser {
        return FixedColumnTextParser(dataFormat: dataFormat, chunks: chunks, numSkipLines: numSkipLines, commentIndicators: commentIndicators)
    }
}

///  Protocol to mark a DataChunk class as includable in a fixed column text data parser
public protocol FixedColumnChunk {
    
}


///  Fixed-Column Text Parser chunk for a specified amount of columns to be thrown away
public class UnusedTextColumns : DataChunk, FixedColumnChunk {
    /// Create an UnusedTextColumns chunk for a FixedColumnTextParser
    /// - Parameter numCharacters: The number of columns to be parsed by this chunk (size of data item)
    public init(numCharacters : Int) {
        super.init(type: .Unused, length: numCharacters, format: .fTextString, postProcessing: .None, affects: .neither)
    }
}

///  Fixed-Column Text Parser chunk where the fixed-column string is assumed to be a class label for the input tensor
public class LabelTextColumns : DataChunk, FixedColumnChunk {
    /// reate an LabelTextColumns chunk for a FixedColumnTextParser
    /// - Parameter numCharacters: The number of columns to be parsed by this chunk (size of data item)
    public init(numCharacters : Int) {
        super.init(type: .Label, length: numCharacters, format: .fTextString, postProcessing: .None, affects: .input)
    }
}

///  Fixed-Column Text Parser chunk where the fixed-column string is taken as a class index integer and stored in the input tensor
public class LabelIndexColumns : DataChunk, FixedColumnChunk {
    /// Create an LabelIndexColumns chunk for a FixedColumnTextParser
    /// - Parameter numCharacters: The number of columns to be parsed by this chunk (size of data item)
    public init(numCharacters : Int) {
        super.init(type: .LabelIndex, length: numCharacters, format: .fTextInt, postProcessing: .None, affects: .input)
    }
}

///  Fixed-Column Text Parser chunk where the fixed-column string is taken as an integer and stored in the input tensor
public class InputIntegerColumns : DataChunk, FixedColumnChunk {
    /// Create an InputIntegerColumns chunk for a FixedColumnTextParser
    /// - Parameters:
    ///   - numCharacters: The number of columns to be parsed by this chunk (size of data item)
    ///   - postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(numCharacters : Int, postProcessing : PostReadProcessing = .None) {
        super.init(type: .Feature, length: numCharacters, format: .fTextInt, postProcessing: postProcessing, affects: .input)
    }
}

///  Fixed-Column Text Parser chunk where the fixed-column string is taken as a floating value and stored in the input tensor
public class InputFloatColumns : DataChunk, FixedColumnChunk {
    /// Create an InputFloatColumns chunk for a FixedColumnTextParser
    /// - Parameters:
    ///   - numCharacters: The number of columns to be parsed by this chunk (size of data item)
    ///   - postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(numCharacters : Int, postProcessing : PostReadProcessing = .None) {
        super.init(type: .Feature, length: numCharacters, format: .fTextFloat, postProcessing: postProcessing, affects: .input)
    }
}

///  Fixed-Column Text Parser chunk where the fixed-column string is taken as an integer and stored in the output tensor
public class OutputIntegerColumns : DataChunk, FixedColumnChunk {
    /// Create an OutputIntegerColumns chunk for a FixedColumnTextParser
    /// - Parameters:
    ///   - numCharacters: The number of columns to be parsed by this chunk (size of data item)
    ///   - postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(numCharacters : Int, postProcessing : PostReadProcessing = .None) {
        super.init(type: .OutputValues, length: numCharacters, format: .fTextInt, postProcessing: postProcessing, affects: .output)
    }
}

///  Fixed-Column Text Parser chunk where the fixed-column string is taken as a floating value and stored in the output tensor
public class OutputFloatColumns : DataChunk, FixedColumnChunk {
    /// Create an OutputFloatColumns chunk for a FixedColumnTextParser
    /// - Parameters:
    ///   - numCharacters: The number of columns to be parsed by this chunk (size of data item)
    ///   - postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(numCharacters : Int, postProcessing : PostReadProcessing = .None) {
        super.init(type: .OutputValues, length: numCharacters, format: .fTextFloat, postProcessing: postProcessing, affects: .output)
    }
}

///  Fixed-Column Text Parser chunk where the fixed-column string is assumed to be a class label for the output tensor
public class OutputLabelColumns : DataChunk, FixedColumnChunk {
    /// Create an OutputLabelColumns chunk for a FixedColumnTextParser
    /// - Parameters:
    ///   - numCharacters: The number of columns to be parsed by this chunk (size of data item)
    ///   - postProcessing: (Optional)  The processing for the value that will be done after the value is read, or in some cases all values are read and stored
    public init(numCharacters : Int, postProcessing : PostReadProcessing = .None) {
        super.init(type: .OutputLabel, length: numCharacters, format: .fTextString, postProcessing: postProcessing, affects: .output)
    }
}

///  Fixed-Column  Text Parser data chunk that allows you to have a repeating block of DataChunks, where a storage dimension can be updated at the end of each loop
public class RepeatDimForFixedText : DataChunk, FixedColumnChunk  {
    /// Create an RepeatDimForFixedText chunk for a FixedColumnTextParser
    /// - Parameters:
    ///   - count: The number of times the contents are executed with the storage location updated
    ///   - dimension: The storage location dimension that gets incremented
    ///   - affects: which Tensors storage location is modified, input, output, or both
    public init(count: Int, dimension : ParserDimension, affects: SampleTensorAffect) {
        super.init(type: .Repeat, length: count, format: dimension.getFormatType(), postProcessing: .None, affects: affects)
    }
    
    /// Initializer for RepeatDimForFixedText that takes a list of data chunks for processing the sample
    /// - Parameters:
    ///   - count: The number of times the contents are executed with the storage location updated
    ///   - dimension: The storage location dimension that gets incremented
    ///   - affects: which Tensors storage location is modified, input, output, or both
    ///   - repeatChunks: the fixed-column text chunks that get repeated
    public convenience init(count: Int, dimension : ParserDimension, affects: SampleTensorAffect, @FixedTextParserBuilder _ repeatChunks: () -> [DataChunk]) {
        self.init(count: count, dimension : dimension, affects: affects)
        self.repeatChunks = repeatChunks()
    }
}
