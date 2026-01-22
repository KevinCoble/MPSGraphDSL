//
//  DataParserBuilder.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 6/23/21.
//

import Foundation


//  Class that can be sub-classed for parser DSL entries.  DataChunk itself needs to be sendable for concurrency
public class DataChunkCreator {
    let type : DataChunkType
    let length : Int
    let format : DataFormatType
    let repeatChunks : [DataChunk]?
    let postProcessing : PostReadProcessing
    let tensorAffect: SampleTensorAffect
    
    init(type: DataChunkType, length: Int, format : DataFormatType, repeatChunks: [DataChunk]? = nil, postProcessing : PostReadProcessing, affects: SampleTensorAffect)
    {
        self.type = type
        self.length = length
        self.format = format
        self.repeatChunks = repeatChunks
        self.postProcessing = postProcessing
        self.tensorAffect = affects
    }
    
    func createDataChunk() -> DataChunk
    {
        return DataChunk(type: type, length: length, format: format, repeatChunks: repeatChunks, postProcessing: postProcessing, affects: tensorAffect)
    }
}


///  Protocol to mark a DataChunkCreator class as includable in a binary data parser
public protocol BinaryDataChunk {
    
}

@resultBuilder
///  The Result Builder for building binary data parsers
public enum DataParserBuilder {
    public static func buildBlock(_ dataChunks: [DataChunkCreator]...) -> [DataChunkCreator] {
        return dataChunks.flatMap({$0})
    }
    public static func buildExpression(_ expression: BinaryDataChunk) -> [DataChunkCreator] {
        return [expression as! DataChunkCreator]
    }
    public static func buildOptional(_ component: [DataChunkCreator]?) -> [DataChunkCreator] {
        return component ?? []
    }
    public static func buildEither(first component: [DataChunkCreator]) -> [DataChunkCreator] {
        return component
    }
    public static func buildEither(second component: [DataChunkCreator]) -> [DataChunkCreator] {
        return component
    }
}

///  The allowable dimensions for storage location manipulations like repeat and set
public enum ParserDimension {
    case Dimension0
    case Dimension1
    case Dimension2
    case Dimension3
    case Dimension4
    case Dimension5
    case Dimension6
    case Dimension7
    case Dimension8
    case Dimension9
    case Dimension10
    case Dimension11
    case Dimension12
    case Dimension13
    case Dimension14
    case Dimension15
    
    internal func getFormatType() -> DataFormatType {
        switch (self) {
        case .Dimension0:
            return .rDimension0
        case .Dimension1:
            return .rDimension1
        case .Dimension2:
            return .rDimension2
        case .Dimension3:
            return .rDimension3
        case .Dimension4:
            return .rDimension4
        case .Dimension5:
            return .rDimension5
        case .Dimension6:
            return .rDimension6
        case .Dimension7:
            return .rDimension7
        case .Dimension8:
            return .rDimension8
        case .Dimension9:
            return .rDimension9
        case .Dimension10:
            return .rDimension10
        case .Dimension11:
            return .rDimension11
        case .Dimension12:
            return .rDimension12
        case .Dimension13:
            return .rDimension13
        case .Dimension14:
            return .rDimension14
        case .Dimension15:
            return .rDimension15
        }
    }
}

//!!  error checking format types for some of the DataChunkCreator sublclasses!

///  Unused Binary Data for a Binary DataParser
public class UnusedData : DataChunkCreator, BinaryDataChunk {
    /// Create an UnusedData chunk for a binary parser
    /// - Parameters:
    ///   - length: The number of items of the specified format to be read and discarded
    ///   - format: The format of the data items to be discarded
    public init(length: Int, format : DataFormatType) {
        super.init(type: .Unused, length: length, format: format, postProcessing: .None, affects: .neither)
    }
}

///  A string of a specified size is read and treated as an output classification label for a Binary DataParser
public class LabelString : DataChunkCreator, BinaryDataChunk  {
    /// Create a LabelString chunk for a binary parser
    /// - Parameter length: The UTF8 length of the string that makes up the label
    public init(length: Int) {
        super.init(type: .Label, length: length, format: .fTextString, postProcessing: .None, affects: .output)
    }
}

/// A data chunk where the data is read and treated as an output classification index for a Binary DataParser
public class LabelIndex : DataChunkCreator, BinaryDataChunk  {
    /// Create the LabelIndex chunk for a binary parser
    /// - Parameters:
    ///   - count: The number of data elements to be read and converted into a Label Index (should generally be one)
    ///   - format: The format of the data items to be read
    public init(count: Int, format : DataFormatType) {
        super.init(type: .LabelIndex, length: count, format: format, postProcessing: .None, affects: .output)
    }
}

///  A data chunk where the specified data is read and stored in the input tensor at the current input storage location, which is incremented
public class InputData : DataChunkCreator, BinaryDataChunk  {
    /// Create the InputData chunk for a binary parser
    /// - Parameters:
    ///   - length: The number of parameters to read and add to the input Tensor
    ///   - format: The format of the parameters to be parsed
    ///   - postProcessing: a specification for any post-parsing processing done with the data
    public init(length: Int, format : DataFormatType, postProcessing : PostReadProcessing) {
        super.init(type: .Feature, length: length, format: format, postProcessing: postProcessing, affects: .input)
    }
}

///  A data chunk where the specified data is read and stored in the first location (index zero) of the last dimension of an input tensor
public class RedPixelData : DataChunkCreator, BinaryDataChunk  {
    /// Create the RedPixelData chunk for a binary parser
    /// - Parameters:
    ///   - length: The number of parameters read and added to the input tensor (usually one)
    ///   - format: The format of the parameters to be parsed
    ///   - postProcessing: a specification for any post-parsing processing done with the data
    public init(length: Int, format : DataFormatType, postProcessing : PostReadProcessing) {
        super.init(type: .RedValue, length: length, format: format, postProcessing: postProcessing, affects: .input)
    }
}

///  A data chunk where the specified data is read and stored in the second location (index one) of the last dimension of an input tensor
public class GreenPixelData : DataChunkCreator, BinaryDataChunk  {
    /// Create the GreenPixelData chunk for a binary parser
    /// - Parameters:
    ///   - length: The number of parameters read and added to the input tensor (usually one)
    ///   - format: The format of the parameters to be parsed
    ///   - postProcessing: a specification for any post-parsing processing done with the data
    public init(length: Int, format : DataFormatType, postProcessing : PostReadProcessing) {
        super.init(type: .GreenValue, length: length, format: format, postProcessing: postProcessing, affects: .input)
    }
}

///  A data chunk where the specified data is read and stored in the third location (index two) of the last dimension of an input tensor
public class BluePixelData : DataChunkCreator, BinaryDataChunk  {
    /// Create the BluePixelData chunk for a binary parser
    /// - Parameters:
    ///   - length: The number of parameters read and added to the input tensor (usually one)
    ///   - format: The format of the parameters to be parsed
    ///   - postProcessing: a specification for any post-parsing processing done with the data
    public init(length: Int, format : DataFormatType, postProcessing : PostReadProcessing) {
        super.init(type: .BlueValue, length: length, format: format, postProcessing: postProcessing, affects: .input)
    }
}


///  A data chunk where the specified data is read and stored in the output tensor at the current output storage location, which is incremented
public class OutputData : DataChunkCreator, BinaryDataChunk  {
    /// Create the OutputData chunk for a binary parser
    /// - Parameters:
    ///   - length: The number of parameters read and added to the output tensor
    ///   - format: The format of the parameters to be parsed
    ///   - postProcessing: a specification for any post-parsing processing done with the data
    public init(length: Int, format : DataFormatType, postProcessing : PostReadProcessing) {
        super.init(type: .OutputValues, length: length, format: format, postProcessing: postProcessing, affects: .output)
    }
}

/// A data chunk that starts and stops each sample in the data file.  The chunks in the loop should define each sample
public class RepeatSampleTillDone : DataChunkCreator, BinaryDataChunk  {
    /// Create a RepeatSampleTillDone chunk for a binary parser
    public init(chunks: [DataChunk]) {
        super.init(type: .Repeat, length: Int.max, format: .rSample, repeatChunks: chunks, postProcessing: .None, affects: .both)
    }
    
    ///  Initializer for RepeatSampleTillDone that takes a list of data chunks for processing the sample
    public convenience init(@DataParserBuilder _ repeatChunks: () -> [DataChunkCreator]) {
        let chunks = repeatChunks().map { $0.createDataChunk()}
        self.init(chunks: chunks)
    }
}

///  A data chunk that allows you to have a repeating block of DataChunks, where a storage dimension can be updated at the end of each loop
public class RepeatDimension : DataChunkCreator, BinaryDataChunk  {
    /// Create a RepeatDimension chunk for a binary parser
    /// - Parameters:
    ///   - count: The number of times the contents are executed with the storage location updated
    ///   - dimension: The storage location dimension that gets incremented
    ///   - affects: which Tensors storage location is modified, input, output, or both
    public init(count: Int, dimension : ParserDimension, affects: SampleTensorAffect, repeatChunks: [DataChunk]) {
        super.init(type: .Repeat, length: count, format: dimension.getFormatType(), repeatChunks: repeatChunks, postProcessing: .None, affects: affects)
    }
    
    /// Initializer for RepeatDimension that takes a list of data chunks for processing the sample
    /// - Parameters:
    ///   - count: The number of times the contents are executed with the storage location updated
    ///   - dimension: The storage location dimension that gets incremented
    ///   - affects: which Tensors storage location is modified, input, output, or both
    ///   - repeatChunks: the binary data chunks that get repeated
    public convenience init(count: Int, dimension : ParserDimension, affects: SampleTensorAffect, @DataParserBuilder _ repeatChunks: () -> [DataChunkCreator]) {
        let chunks = repeatChunks().map { $0.createDataChunk()}
        self.init(count: count, dimension : dimension, affects: affects, repeatChunks: chunks)
    }
}

///  A data chunk that allows you to set the storage location dimension for input and/or output tensors
public class SetDimension : DataChunkCreator, BinaryDataChunk, DelineatedTextChunk, FixedColumnChunk  {
    /// Create a SetDimension chunk for a parser
    /// - Parameters:
    ///   - dimension: The storage location dimension that gets set to the specified value
    ///   - toValue: The value the storage location dimension gets set to
    ///   - affects: which Tensors storage location is modified, input, output, or both
    public init(dimension : ParserDimension, toValue: Int, affects: SampleTensorAffect) {
        super.init(type: .SetDimension, length: toValue, format: dimension.getFormatType(), postProcessing: .None, affects: affects)
    }
}

///  A data chunk that ends the current sample, creates a new one, and resets the storage locations for the sample
public class StartNewSample : DataChunkCreator, BinaryDataChunk  {
    /// Create a StartNewSample chunk for a binary parser
    public init() {
        super.init(type: .SetDimension, length: -1, format: .rSample, postProcessing: .None, affects: .both)
    }
}

///  A data chunk that increments a specified storage location dimension for input and/or output tensors
public class IncrementDimension : DataChunkCreator, BinaryDataChunk, DelineatedTextChunk, FixedColumnChunk  {
    /// Create an IncrementDimension chunk for a parser
    /// - Parameters:
    ///   - dimension: The storage location dimension that gets incremented
    ///   - affects: which Tensors storage location is modified, input, output, or both
    public init(dimension : DataFormatType, affects: SampleTensorAffect) {
        super.init(type: .SetDimension, length: -1, format: dimension, postProcessing: .None, affects: affects)
    }
}
