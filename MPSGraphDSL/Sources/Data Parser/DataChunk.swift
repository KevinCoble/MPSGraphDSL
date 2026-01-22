//
//  DataChunk.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 6/23/21.
//

import Foundation


///   Enumeration for the DataParserChunk Type
internal enum DataChunkType : Int, Sendable  {
    ///  Data in chunk is unused
    case Unused = 1
    ///  Data in chunk is a classification label
    case Label = 2
    ///  Data in chunk is a classification index integer
    case LabelIndex = 3
    ///  Data in chunk is a input feature value
    case Feature = 4
    ///  Data in chunk is a input feature value that goes into the red channel (index 0 of dimension 3)
    case RedValue = 5
    ///  Data in chunk is a input feature value that goes into the green channel (index 1 of dimension 3)
    case GreenValue = 6
    ///  Data in chunk is a input feature value that goes into the blue channel (index 2 of dimension 3)
    case BlueValue = 7
    ///  Data in chunk is an output value
    case OutputValues = 8
    ///  Data in chunk is an output value in the form of a text label
    case OutputLabel = 9
    ///  Chunk is a repeat operation on a given dimension or sample index
    case Repeat = 100
    ///  Chunk is a dimension set operation (set a dimension to a specified constant)
    case SetDimension = 101
    
    ///  Get a string representation of the Chunk type
    public var typeString : String
    {
        get {
            switch (self)
            {
            case .Unused:
                return "Unused"
            case .Label:
                return "Label"
            case .LabelIndex:
                return "Label #"
            case .Feature:
                return "Feature"
            case .RedValue:
                return "Red"
            case .GreenValue:
                return "Green"
            case .BlueValue:
                return "Blue"
            case .OutputValues:
                return "Output"
            case .OutputLabel:
                return "Label"
            case .Repeat:
                return "Repeat"
            case .SetDimension:
                return "Set Dim."
            }
        }
    }
}

///  Enumeration for the data type read by a chunk
public enum DataFormatType : Int, Sendable {
    ///  Signed bytes
    case fInt8 = 1
    ///  Unsigned bytes
    case fUInt8 = 2
    ///  Signed 16-bit integer
    case fInt16 = 3
    ///  Unsigned 16-bit integer
    case fUInt16 = 4
    ///  Signed 32-bit integer
    case fInt32 = 5
    ///  Unsigned 32-bit integer
    case fUInt32 = 6
    ///  16-bit floating value
    case fFloat16 = 7
    ///  Standard 32-bit floating value
    case fFloat32 = 8
    ///  Standard 64-bit floating value
    case fDouble = 9
    ///  UTF-8 encoded text string
    case fTextString = 10
    ///  UTF-8 encoded text string that will be read into an integer value
    case fTextInt = 11
    ///  UTF-8 encoded text string that will be read into an floating value
    case fTextFloat = 12
    ///  Dimension 0 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension0 = 100
    ///  Dimension 1 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension1 = 101
    ///  Dimension 2 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension2 = 102
    ///  Dimension 3 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension3 = 103
    ///  Dimension 4 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension4 = 104
    ///  Dimension 5 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension5 = 105
    ///  Dimension 6 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension6 = 106
    ///  Dimension 7 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension7 = 107
    ///  Dimension 8 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension8 = 108
    ///  Dimension 9 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension9 = 109
    ///  Dimension 10 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension10 = 110
    ///  Dimension 11 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension11 = 111
    ///  Dimension 12 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension12 = 112
    ///  Dimension 13 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension13 = 113
    ///  Dimension 14 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension14 = 114
    ///  Dimension 15 indicator - used with Repeat and Set Dimension Chunk types
    case rDimension15 = 115
    ///  Sample index indicator - used with Repeat Chunk types
    case rSample = 120

    ///  Get a string representation of the format type
    public var typeString : String
    {
        get {
            switch (self)
            {
            case .fInt8:
                return "Int8"
            case .fUInt8:
                return "UInt8"
            case .fInt16:
                return "Int16"
            case .fUInt16:
                return "UInt16"
            case .fInt32:
                return "Int32"
            case .fUInt32:
                return "UInt32"
            case .fFloat16:
                return "Float16"
            case .fFloat32:
                return "Float32"
            case .fDouble:
                return "Double"
            case .fTextString:
                return "Text String"
            case .fTextInt:
                return "Text Integer"
            case .fTextFloat:
                return "Text Float"
            case .rDimension0:
                return "Dim. 0"
            case .rDimension1:
                return "Dim. 1"
            case .rDimension2:
                return "Dim. 2"
            case .rDimension3:
                return "Dim. 3"
            case .rDimension4:
                return "Dim. 4"
            case .rDimension5:
                return "Dim. 5"
            case .rDimension6:
                return "Dim. 6"
            case .rDimension7:
                return "Dim. 7"
            case .rDimension8:
                return "Dim. 8"
            case .rDimension9:
                return "Dim. 9"
            case .rDimension10:
                return "Dim. 10"
            case .rDimension11:
                return "Dim. 11"
            case .rDimension12:
                return "Dim. 12"
            case .rDimension13:
                return "Dim. 13"
            case .rDimension14:
                return "Dim. 14"
            case .rDimension15:
                return "Dim. 15"
            case .rSample:
                return "Sample"
           }
        }
    }

    ///  Gets the byte length of a single format type item
    public var byteLength : Int?
    {
        get {
            switch (self)
            {
            case .fInt8:
                return MemoryLayout<Int8>.size
            case .fUInt8:
                return MemoryLayout<Int8>.size
            case .fInt16:
                return MemoryLayout<Int16>.size
            case .fUInt16:
                return MemoryLayout<UInt16>.size
            case .fInt32:
                return MemoryLayout<Int32>.size
            case .fUInt32:
                return MemoryLayout<UInt32>.size
            case .fFloat16:
                #if os(iOS)
                return MemoryLayout<Float16>.size
                #else
                return MemoryLayout<Float32>.size
                #endif
            case .fFloat32:
                return MemoryLayout<Float32>.size
            case .fDouble:
                return MemoryLayout<Double>.size
            case .fTextString, .fTextInt, .fTextFloat:
                return nil       //  Indeterminate
            case .rDimension0, .rDimension1, .rDimension2, .rDimension3, .rDimension4, .rDimension5, .rDimension6, .rDimension7, .rDimension8,
                 .rDimension9, .rDimension10, .rDimension11, .rDimension12, .rDimension13, .rDimension14, .rDimension15, .rSample:
                return 0
            }
        }
    }
}


///  Enumeration for the post-processing performed by the parser on the chunk data
public enum PostReadProcessing : Int, Sendable  {
    ///  No post-processing performed
    case None = 1
    ///  Each value is scaled to be between 0 an 1, based on format type value range
    case Scale_0_1 = 2
    ///  Each value is scaled to be between -1 an 1, based on format type value range
    case Scale_M1_1 = 7
    ///  Each value in the sample is normalized to be between 0 an 1, based on actual range of data in the sample
    case Normalize_0_1 = 3
    ///  Each value in the sample is normalized to be between -1 an 1, based on actual range of data in the sample
    case Normalize_M1_1 = 4
    ///  Each value in the data set is normalized to be between 0 an 1, based on actual range of data in the data set
    case Normalize_All_0_1 = 5
    ///  Each value in the data set is normalized to be between -1 an 1, based on actual range of data in the data set
    case Normalize_All_M1_1 = 6

    ///  Get a string representation of the post-processing type
    public var typeString : String
    {
        get {
            switch (self)
            {
            case .None:
                return "None"
            case .Scale_0_1:
                return "Scale 0 to 1"
            case .Scale_M1_1:
                return "Scale -1 to 1"
            case .Normalize_0_1:
                return "Norm. 0 to 1"
            case .Normalize_M1_1:
                return "Norm. -1 to 1"
            case .Normalize_All_0_1:
                return "Norm. All 0 to 1"
            case .Normalize_All_M1_1:
                return "Norm. All -1 to 1"
            }
        }
    }
}

///  Enumeration for the color channel indices (assumed to be dimension 3)
public enum ColorChannel : Int, Sendable  {
    case red = 0
    case green = 1
    case blue = 2
    case alpha = 3
    
    ///  Get a string representation of the color channel reference
    public var typeString : String
    {
        get {
            switch (self)
            {
            case .red:
                return "Red"
            case .green:
                return "Green"
            case .blue:
                return "Blue"
            case .alpha:
                return "Alpha"
            }
        }
    }
}

///  Enumeration to select which tensor dimensions are affected by repeat, setDimension, IncrementDimension, etc. parser chunks
public enum SampleTensorAffect: Sendable  {
    case neither
    case input
    case output
    case both
}



public struct DataChunk : Sendable {
    let type : DataChunkType
    let length : Int
    let format : DataFormatType
    let repeatChunks : [DataChunk]?     //  If a repeating chunk, these are the chunks to repeat
    let postProcessing : PostReadProcessing
    let tensorAffect: SampleTensorAffect
    
    internal var normalizationIndex : Int?
    
    init(type: DataChunkType, length: Int, format : DataFormatType, repeatChunks: [DataChunk]?, postProcessing : PostReadProcessing, affects: SampleTensorAffect)
    {
        self.type = type
        self.length = length
        self.format = format
        self.repeatChunks = repeatChunks
        self.postProcessing = postProcessing
        self.tensorAffect = affects
    }
    
    //  Get the bytes required by the chunk (binary parsing use).  Return nil if indeterminate
    func getRequiredBytes() -> Int?
    {
        if let repeatChunks = repeatChunks {
            //  Get the length of the repeat chunks
            var totalByteLength = 0
            for chunk in repeatChunks {
                let byteLength = chunk.getRequiredBytes()
                if byteLength == nil { return nil }
                totalByteLength += byteLength!
            }
            if (format == .rSample) {
                return totalByteLength
            }
            else {
                return totalByteLength * length
            }
        }
        else {
            let dataByteLength = format.byteLength
            if dataByteLength == nil { return nil }
            let byteLength = dataByteLength! * length
            return byteLength
        }
    }

    
    // MARK: - Parsing
    func parseBinaryChunk(dataSet: DataSet, inputFile : InputStream, parsingData : ParsingData) async throws
    {
        //  Get the data for the types that can use it
        var data : [Double]?
        if (type != .OutputLabel && type != .Repeat && type != .SetDimension) {
            data = await readBinaryData(inputFile : inputFile, dataSet: dataSet)
            if (data == nil) {
                throw DataParsingErrors.ErrorReadingBinaryData
            }
        }
        
        //  Processing varies depending on the chunk type
        switch (type) {
        case .Unused:
            //  Throw away the unused data
            return
        case .Label, .LabelIndex:
            //  Store the label for this data set
            try await parsingData.appendOutputClass(Int(data![0]))
        case .Feature:
            //  Store the feature values for the current input location
            try await parsingData.appendInputData(data!, normalizationIndex: normalizationIndex)
        case .RedValue:
            //  Store the red values for the current input location
            try await parsingData.appendColorData(data!, channel: .red, normalizationIndex: normalizationIndex)
        case .GreenValue:
            //  Store the green values for the current input location
            try await parsingData.appendColorData(data!, channel: .green, normalizationIndex: normalizationIndex)
        case .BlueValue:
            //  Store the blue values for the current input location
            try await parsingData.appendColorData(data!, channel: .blue, normalizationIndex: normalizationIndex)
        case .OutputValues:
            //  Store the output values for the current output location
            try await parsingData.appendOutputData(data!, normalizationIndex: normalizationIndex)
        case .OutputLabel:
            if let string = readBinaryString(inputFile: inputFile, dataSet: dataSet) {
                let labelIndex = await dataSet.getLabelIndex(label: string)
                try await parsingData.appendOutputLabel(labelIndex, normalizationIndex: normalizationIndex)
            }
            else {
                throw DataParsingErrors.ErrorReadingBinaryData
            }
        case .Repeat:
            //  Set the stage for reading this dimension
            let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
            //  Iterate
            for _ in 0..<length {
                //  If a sample iterator, increment up front so new sample can be created when needed
                if (format == .rSample) {
                    let sampleIndex = try await dataSet.incrementIndexAppendEmptySample(oldIndex: parsingData.currentSampleIndex)
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                }
                //  Process each chunk
                for chunk in repeatChunks! {
                    do {
                        try await chunk.parseBinaryChunk(dataSet: dataSet, inputFile : inputFile, parsingData: parsingData)
                    }
                    catch DataParsingErrors.ErrorReadingBinaryData {
                        //  If a sample repeat, and and the current location is 0, delete last (empty) sample and return true - we are done
                        if (format == .rSample) {
                            let inputSum = await parsingData.currentInputLocation.reduce(0, +)
                            let outputSum = await parsingData.currentOutputLocation.reduce(0, +)
                            if (inputSum == 0 && outputSum == 0) {
                                try await dataSet.removeFinalSample()
                                await parsingData.dropCurrentSample()
                            }
                            return
                        }
                        else {
                            //  Not a sample repeat, rethrow the error
                            throw DataParsingErrors.ErrorReadingBinaryData
                        }
                    }
                }
                
                //  Increment the dimension
                if (format != .rSample) {
                    if (tensorAffect == .input || tensorAffect == .both) {
                        await parsingData.incrementInputDimension(dimension: dimension)
                    }
                    if (tensorAffect == .output || tensorAffect == .both) {
                        await parsingData.incrementOutputDimension(dimension: dimension)
                     }
                }
                else {
                    //  Sample repeat - put the sample back
                    try await parsingData.putSampleBackIntoDataSet(dataSet: dataSet)
                }
           }
        case .SetDimension:
            if (format == .rSample) {
                try await parsingData.putSampleBackIntoDataSet(dataSet: dataSet)
                if (length < 0) {
                    let sampleIndex = try await dataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                }
                else {
                    try await parsingData.getSampleFromDataSet(sampleIndex: length, dataSet: dataSet)
                }
            }
            else {
                let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
                if (tensorAffect == .input || tensorAffect == .both) {
                    await parsingData.setOrIncrementInputDimension(dimension: dimension, toValue: length)
                }
                if (tensorAffect == .output || tensorAffect == .both) {
                    await parsingData.setOrIncrementOutputDimension(dimension: dimension, toValue: length)
                }
            }
        }
    }

    func readBinaryData(inputFile : InputStream, dataSet : DataSet) async -> [Double]?
    {
        //  Read the data bytes
        let formatByteLength = format.byteLength
        let dataByteLength: Int
        let byteLength: Int
        if let formatByteLength = formatByteLength {
            dataByteLength = formatByteLength
            byteLength = dataByteLength * length
        }
        else {
            dataByteLength = 1
            byteLength = 1
        }
        var bytes = [UInt8](repeating: 0, count: byteLength)
        let numRead = inputFile.read(&bytes, maxLength: byteLength)
        if (numRead < byteLength) {
            return nil
        }
        
        //  Create the double array
        var floats = [Double](repeating: 0, count: length)

        //  Convert the bytes to the expected type, then to a double
        switch (format) {
        case .fInt8:
            let scaleFactor = 1.0 / Double(127)
            for index in 0..<length {
                var integer = Int(bytes[index])
                if (integer > Int8.max) { integer -= Int(UInt8.max) }
                let x : Int8 = Int8(integer)
                if (postProcessing == .Scale_0_1) {
                    floats[index] = Double(x) * scaleFactor
                }
                else {
                    floats[index] = Double(x)
                }
            }
        case .fUInt8:
            let scaleFactor = 1.0 / Double(255)
            for index in 0..<length {
                floats[index] = Double(bytes[index])
                if (postProcessing == .Scale_0_1) {
                    floats[index] = Double(bytes[index]) * scaleFactor
                }
                else {
                    floats[index] = Double(bytes[index])
                }
           }
        case .fInt16:
            let scaleFactor = 1.0 / Double(32767)
            for index in 0..<length {
                var intValue : Int16 = 0
                let data = NSData(bytes: bytes, length: byteLength)
                data.getBytes(&intValue, range: NSRange(location: index * dataByteLength, length: dataByteLength))
                if (postProcessing == .Scale_0_1) {
                    floats[index] = Double(intValue) * scaleFactor
                }
                else {
                    floats[index] = Double(intValue)
                }
           }
        case .fUInt16:
            let scaleFactor = 1.0 / Double(65535)
            for index in 0..<length {
                var uintValue : UInt16 = 0
                let data = NSData(bytes: bytes, length: byteLength)
                data.getBytes(&uintValue, range: NSRange(location: index * dataByteLength, length: dataByteLength))
                if (postProcessing == .Scale_0_1) {
                    floats[index] = Double(uintValue) * scaleFactor
                }
                else {
                    floats[index] = Double(uintValue)
                }
            }
        case .fInt32:
            for index in 0..<length {
                var intValue : Int32 = 0
                let data = NSData(bytes: bytes, length: byteLength)
                data.getBytes(&intValue, range: NSRange(location: index * dataByteLength, length: dataByteLength))
                floats[index] = Double(intValue)
                if (postProcessing == .Scale_0_1) { floats[index] /= Double(Int32.max) }
            }
        case .fUInt32:
            for index in 0..<length {
                var uintValue : UInt32 = 0
                let data = NSData(bytes: bytes, length: byteLength)
                data.getBytes(&uintValue, range: NSRange(location: index * dataByteLength, length: dataByteLength))
                floats[index] = Double(uintValue)
                if (postProcessing == .Scale_0_1) { floats[index] /= Double(UInt32.max) }
            }
        case .fFloat32:
            for index in 0..<length {
                var uintValue : UInt32 = 0
                let data = NSData(bytes: bytes, length: byteLength)
                data.getBytes(&uintValue, range: NSRange(location: index * dataByteLength, length: dataByteLength))
                floats[index] = Double(Float(bitPattern: uintValue))
            }
        case .fDouble:
            for index in 0..<length {
                var uintValue : UInt64 = 0
                let data = NSData(bytes: bytes, length: byteLength)
                data.getBytes(&uintValue, range: NSRange(location: index * dataByteLength, length: dataByteLength))
                floats[index] = Double(bitPattern: uintValue)
            }
        case .fTextString:
            if let string = String(bytes: bytes, encoding: .utf8) {
                //  Convert the string to an index based on the known labels
                let labelIndex = await dataSet.getLabelIndex(label: string)
                if (labelIndex >= 0) {
                    return [Double(labelIndex)]
                }
            }
            return nil
        case .fTextInt:
            if let string = String(bytes: bytes, encoding: .utf8) {
                if let x = Int(string) {
                    return [Double(x)]
                }
            }
            return nil
        case .fTextFloat:
            if let string = String(bytes: bytes, encoding: .utf8) {
                if let x = Double(string) {
                    return [x]
                }
            }
            return nil
        default:
            fatalError("invalid format type on data read")
        }
        
        return floats
    }

    func readBinaryString(inputFile : InputStream, dataSet : DataSet) -> String?
    {
        //  Read the data bytes
        let byteLength = length
        var bytes = [UInt8](repeating: 0, count: byteLength)
        let numRead = inputFile.read(&bytes, maxLength: byteLength)
        if (numRead < byteLength) {
            return nil
        }
        
        //  Convert the bytes into a string
        if let string = String(bytes: bytes, encoding: .utf8) {
            return string
        } else {
            return nil
        }
    }

    func parseBinaryChunk(intoDataSet: DataSet, data: Data, offset : inout Int, parsingData: ParsingData) async throws
    {
        //  Get the data for the types that can use it
        var dataValues : [Double]?
        if (type != .OutputLabel && type != .Repeat && type != .SetDimension) {
            dataValues = await extractBinaryData(data : data, offset: &offset, dataSet: intoDataSet)
            if (dataValues == nil) {
                throw DataParsingErrors.ErrorReadingBinaryData
            }
        }
        
        //  Processing varies depending on the chunk type
        switch (type) {
        case .Unused:
            //  Throw away the unused data
            return
        case .Label, .LabelIndex:
            //  Store the label for this data set
            try await parsingData.appendOutputClass(Int(dataValues![0]))
        case .Feature:
            //  Store the feature values for the current input location
            try await parsingData.appendInputData(dataValues!, normalizationIndex: normalizationIndex)
        case .RedValue:
            //  Store the red values for the current input location
            try await parsingData.appendColorData(dataValues!, channel: .red, normalizationIndex: normalizationIndex)
        case .GreenValue:
            //  Store the green values for the current input location
            try await parsingData.appendColorData(dataValues!, channel: .green, normalizationIndex: normalizationIndex)
        case .BlueValue:
            //  Store the blue values for the current input location
            try await parsingData.appendColorData(dataValues!, channel: .blue, normalizationIndex: normalizationIndex)
        case .OutputValues:
            //  Store the output values for the current output location
            try await parsingData.appendOutputData(dataValues!, normalizationIndex: normalizationIndex)
        case .OutputLabel:
            if let string = extractBinaryString(data : data, offset: &offset, dataSet: intoDataSet) {
                let labelIndex = await intoDataSet.getLabelIndex(label: string)
                try await parsingData.appendOutputLabel(labelIndex, normalizationIndex: normalizationIndex)
            }
            else {
                throw DataParsingErrors.ErrorReadingBinaryData
            }
        case .Repeat:
            //  Set the stage for repeating this dimension
            let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
            //  Iterate
            for _ in 0..<length {
                //  If a sample iterator, increment up front so new sample can be created when needed
                if (format == .rSample) {
                    let sampleIndex = try await intoDataSet.incrementIndexAppendEmptySample(oldIndex: parsingData.currentSampleIndex)
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : intoDataSet)
                }
                //  Process each chunk
                for chunk in repeatChunks! {
                    do {
                        try await chunk.parseBinaryChunk(intoDataSet: intoDataSet, data : data, offset: &offset, parsingData: parsingData)
                    }
                    catch DataParsingErrors.ErrorReadingBinaryData {
                        //  If a sample repeat, and the current location is 0, delete last (empty) sample and return true - we are done
                        if (format == .rSample) {
                            let haveAddedData = await parsingData.haveAddedData()
                            if (!haveAddedData) {
                                try await intoDataSet.removeFinalSample()
                                await parsingData.dropCurrentSample()
                            }
                            return
                       }
                        else {
                            //  Not a sample repeat, rethrow the error
                            throw DataParsingErrors.ErrorReadingBinaryData
                        }
                    }
                }
                
                //  Increment the dimension
                if (format != .rSample) {
                    if (tensorAffect == .input || tensorAffect == .both) {
                        await parsingData.incrementInputDimension(dimension: dimension)
                    }
                    if (tensorAffect == .output || tensorAffect == .both) {
                        await parsingData.incrementOutputDimension(dimension: dimension)
                    }
                }
                else {
                    //  Sample repeat - put the sample back
                    try await parsingData.putSampleBackIntoDataSet(dataSet: intoDataSet)
                }
           }
        case .SetDimension:
            if (format == .rSample) {
                try await parsingData.putSampleBackIntoDataSet(dataSet: intoDataSet)
                if (length < 0) {
                    let sampleIndex = try await intoDataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : intoDataSet)
                }
                else {
                    try await parsingData.getSampleFromDataSet(sampleIndex: length, dataSet: intoDataSet)
                }
            }
            else {
                let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
                if (tensorAffect == .input || tensorAffect == .both) {
                    await parsingData.setOrIncrementInputDimension(dimension: dimension, toValue: length)
                }
                if (tensorAffect == .output || tensorAffect == .both) {
                    await parsingData.setOrIncrementOutputDimension(dimension: dimension, toValue: length)
                }
            }
        }
    }
    
    func repeatBinarySample(ofLength: Int, intoDataSet: DataSet, inputFile : InputStream, parsingData: ParsingData, maxConcurrent: Int) async throws
    {
        var numSubmitted = 0
        var bytes: [UInt8] = [UInt8](repeating: 0, count: ofLength)
        try await withThrowingTaskGroup(of: Void.self) { taskGroup in
            //  Iterate
            while true {
                //  Get the data for this sample
                let numRead = inputFile.read(&bytes, maxLength: ofLength)
                if (numRead < ofLength) {
                    return
                }
                
                //  Increment up front so new sample can be created when needed
                let sampleIndex = try await intoDataSet.incrementIndexAppendEmptySample(oldIndex: parsingData.currentSampleIndex)
                try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : intoDataSet)
                
                //  Make a copy of the parsingData
                let parsingDataCopy = await ParsingData(copyFrom: parsingData)
                
                //  If less than max concurrency submitted, add it immediately.  Otherwise wait for one to finish
                if (numSubmitted >= maxConcurrent) {
                    try await taskGroup.next()
                }
                
                //  Create the data object
                let sampleData = Data(bytes)

                let added = taskGroup.addTaskUnlessCancelled {
                    //  Process each chunk
                    var sampleOffset = 0
                    for chunk in self.repeatChunks! {
                        do {
                            try await chunk.parseBinaryChunk(intoDataSet: intoDataSet, data : sampleData, offset: &sampleOffset, parsingData: parsingDataCopy)
                        }
                        catch DataParsingErrors.ErrorReadingBinaryData {
                            //  Rethrow the error
                            throw DataParsingErrors.ErrorReadingBinaryData
                        }
                    }
                    
                    //  Put back the  data
                    try await parsingDataCopy.putSampleBackIntoDataSet(dataSet: intoDataSet)
                }
                if (added) { numSubmitted += 1 }
            }
        }

    }

    func repeatBinarySample(ofLength: Int, intoDataSet: DataSet, data: Data, offset : inout Int, parsingData: ParsingData, maxConcurrent: Int) async throws
    {
        var localOffset = offset
        var numSubmitted = 0
        try await withThrowingTaskGroup(of: Void.self) { taskGroup in
            //  Iterate
            while true {
                //  Get the data for this sample
                let sampleEnd = localOffset+ofLength
                if (data.count < sampleEnd) { break }   //  Not enough data for another one
                let range = localOffset..<(sampleEnd)
                let sampleData = data[range]
                localOffset += ofLength
                
                //  Increment up front so new sample can be created when needed
                let sampleIndex = try await intoDataSet.incrementIndexAppendEmptySample(oldIndex: parsingData.currentSampleIndex)
                try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : intoDataSet)
                
                //  Make a copy of the parsingData
                let parsingDataCopy = await ParsingData(copyFrom: parsingData)
                
                //  If less than max concurrency submitted, add it immediately.  Otherwise wait for one to finish
                if (numSubmitted >= maxConcurrent) {
                    try await taskGroup.next()
                }
                
                let added = taskGroup.addTaskUnlessCancelled {
                    //  Process each chunk
                    var sampleOffset = 0
                    for chunk in self.repeatChunks! {
                        do {
                            try await chunk.parseBinaryChunk(intoDataSet: intoDataSet, data : sampleData, offset: &sampleOffset, parsingData: parsingDataCopy)
                        }
                        catch DataParsingErrors.ErrorReadingBinaryData {
                            //  Rethrow the error
                            throw DataParsingErrors.ErrorReadingBinaryData
                        }
                    }
                    
                    //  Put back the  data
                    try await parsingDataCopy.putSampleBackIntoDataSet(dataSet: intoDataSet)
                }
                if (added) { numSubmitted += 1 }
             }
        }
        offset = localOffset
    }
    
    func extractBinaryData(data : Data, offset: inout Int, dataSet: DataSet) async -> [Double]? {
        //  Create the double array
        var floats = [Double](repeating: 0, count: length)

        //  Extract the bytes as the expected type, then convert to doubles
        switch (format) {
        case .fInt8:
            let scaleFactor = 1.0 / Double(Int8.max)
            let itemSize = MemoryLayout<Int8>.size
            let totalLength = itemSize * length
            let endIndex = offset + totalLength
            if (endIndex > data.count) { return nil }
            let span: Span<UInt8> = data.span.extracting(offset..<endIndex)
            span.withUnsafeBytes { ptr in
                let bufferPointer: UnsafeBufferPointer<Int8> = ptr.bindMemory(to: Int8.self)
                for index in 0..<length {
                    floats[index] = Double(bufferPointer[index])
                    if (postProcessing == .Scale_0_1) {
                        floats[index] *= scaleFactor
                    }
                }
            }
            offset += totalLength
        case .fUInt8:
            let scaleFactor = 1.0 / Double(UInt8.max)
            let itemSize = MemoryLayout<UInt8>.size
            let totalLength = itemSize * length
            let endIndex = offset + totalLength
            if (endIndex > data.count) { return nil }
            let span: Span<UInt8> = data.span.extracting(offset..<endIndex)
            for index in 0..<length {
                floats[index] = Double(span[index])
                if (postProcessing == .Scale_0_1) {
                    floats[index] *= scaleFactor
                }
            }
            offset += totalLength

        case .fInt16:
            let scaleFactor = 1.0 / Double(Int16.max)
            let itemSize = MemoryLayout<Int16>.size
            let totalLength = itemSize * length
            let endIndex = offset + totalLength
            if (endIndex > data.count) { return nil }
            let span: Span<UInt8> = data.span.extracting(offset..<endIndex)
            span.withUnsafeBytes { ptr in
                let bufferPointer: UnsafeBufferPointer<Int16> = ptr.bindMemory(to: Int16.self)
                for index in 0..<length {
                    floats[index] = Double(bufferPointer[index])
                    if (postProcessing == .Scale_0_1) {
                        floats[index] *= scaleFactor
                    }
                }
            }
            offset += totalLength
        case .fUInt16:
            let scaleFactor = 1.0 / Double(UInt16.max)
            let itemSize = MemoryLayout<UInt16>.size
            let totalLength = itemSize * length
            let endIndex = offset + totalLength
            if (endIndex > data.count) { return nil }
            let span: Span<UInt8> = data.span.extracting(offset..<endIndex)
            span.withUnsafeBytes { ptr in
                let bufferPointer: UnsafeBufferPointer<UInt16> = ptr.bindMemory(to: UInt16.self)
                for index in 0..<length {
                    floats[index] = Double(bufferPointer[index])
                    if (postProcessing == .Scale_0_1) {
                        floats[index] *= scaleFactor
                    }
                }
            }
            offset += totalLength
        case .fInt32:
            let itemSize = MemoryLayout<Int32>.size
            let totalLength = itemSize * length
            let endIndex = offset + totalLength
            if (endIndex > data.count) { return nil }
            let span: Span<UInt8> = data.span.extracting(offset..<endIndex)
            span.withUnsafeBytes { ptr in
                let bufferPointer: UnsafeBufferPointer<Int32> = ptr.bindMemory(to: Int32.self)
                for index in 0..<length {
                    floats[index] = Double(bufferPointer[index])
                    if (postProcessing == .Scale_0_1) {
                        floats[index] /= Double(Int32.max)
                    }
                }
            }
            offset += totalLength
        case .fUInt32:
            let itemSize = MemoryLayout<UInt32>.size
            let totalLength = itemSize * length
            let endIndex = offset + totalLength
            if (endIndex > data.count) { return nil }
            let span: Span<UInt8> = data.span.extracting(offset..<endIndex)
            span.withUnsafeBytes { ptr in
                let bufferPointer: UnsafeBufferPointer<UInt32> = ptr.bindMemory(to: UInt32.self)
                for index in 0..<length {
                    floats[index] = Double(bufferPointer[index])
                    if (postProcessing == .Scale_0_1) {
                        floats[index] /= Double(UInt32.max)
                    }
                }
            }
            offset += totalLength
        case .fFloat32:
            let itemSize = MemoryLayout<Float32>.size
            let totalLength = itemSize * length
            let endIndex = offset + totalLength
            if (endIndex > data.count) { return nil }
            let span: Span<UInt8> = data.span.extracting(offset..<endIndex)
            span.withUnsafeBytes { ptr in
                let bufferPointer: UnsafeBufferPointer<Float32> = ptr.bindMemory(to: Float32.self)
                for index in 0..<length {
                    floats[index] = Double(bufferPointer[index])
                }
            }
            offset += totalLength
        case .fDouble:
            for index in 0..<length {
                guard let x : Double = data.extractValue(offset: &offset) else { return nil }
                floats[index] = x
            }
        case .fTextString:
            guard let string = data.extractUnsizedString(length: length, offset: &offset) else { return nil }
            //  Convert the string to an index based on the known labels
            let labelIndex = await dataSet.getLabelIndex(label: string)
            if (labelIndex >= 0) {
                return [Double(labelIndex)]
            }
            return nil
        case .fTextInt:
            guard let string = data.extractUnsizedString(length: length, offset: &offset) else { return nil }
            if let x = Int(string) {
                return [Double(x)]
            }
            return nil
        case .fTextFloat:
            guard let string = data.extractUnsizedString(length: length, offset: &offset) else { return nil }
            if let x = Double(string) {
                return [x]
            }
            return nil
        default:
            fatalError("invalid format type on data read")
        }

        return floats
    }

    
    func extractBinaryString(data : Data, offset: inout Int, dataSet: DataSet) -> String? {
        guard let string = data.extractUnsizedString(length: length, offset: &offset) else { return nil }
        return string
    }
    

    func parseTextChunk(dataSet: DataSet, components: [String], offset : Int, parsingData: ParsingData) async throws -> Int
    {
        //  Get the data for types that can use it
        var numUsed = 0
        var data : [Double] = []
        if (type != .OutputLabel && type != .Repeat && type != .SetDimension) {
            data = [Double](repeating: 0.0, count: length)
            for i in 0..<length {
                let index = offset + i
                if (index >= components.count) {
                    throw DataParsingErrors.NotEnoughComponentsOnLine
                }
                if (type == .Unused) {
                    numUsed += 1
                }
                else {
                    if let value = await getFloatData(component: components[index], dataSet: dataSet) {
                        data[i] = value
                        numUsed += 1
                    }
                    else {
                        throw DataParsingErrors.InvalidStringValue(components[index])
                    }
                }
            }
        }

        //  Processing varies depending on the chunk type
        switch (type) {
        case .Unused:
            //  Throw away the unused data
            return numUsed
        case .Label, .LabelIndex:
            //  Store the label for this data set
            try await parsingData.appendOutputClass(Int(data[0]))
        case .Feature:
            //  Store the feature values for the current input location
            try await parsingData.appendInputData(data, normalizationIndex: normalizationIndex)
        case .RedValue:
            //  Store the red values for the current input location
            try await parsingData.appendColorData(data, channel: .red, normalizationIndex: normalizationIndex)
        case .GreenValue:
            //  Store the green values for the current input location
            try await parsingData.appendColorData(data, channel: .green, normalizationIndex: normalizationIndex)
        case .BlueValue:
            //  Store the blue values for the current input location
            try await parsingData.appendColorData(data, channel: .blue, normalizationIndex: normalizationIndex)
        case .OutputValues:
            //  Store the output values for the current output location
            try await parsingData.appendOutputData(data, normalizationIndex: normalizationIndex)
        case .OutputLabel:
            let labelIndex = await dataSet.getLabelIndex(label: components[offset])
            try await parsingData.appendOutputLabel(labelIndex, normalizationIndex: normalizationIndex)
            return 1
        case .Repeat:
            //  Set the stage for reading this dimension
            let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
            if (dimension > 15) { break }        //  Skip sample iterators
            //  Iterate
            for _ in 0..<length {
                //  Process each chunk
                for chunk in repeatChunks! {
                    let usedComponents = try await chunk.parseTextChunk(dataSet: dataSet, components: components, offset : offset + numUsed, parsingData: parsingData)
                    numUsed += usedComponents
                }

                //  Increment the dimension
                if (format != .rSample) {
                    if (tensorAffect == .input || tensorAffect == .both) {
                        await parsingData.incrementInputDimension(dimension: dimension)
                    }
                    if (tensorAffect == .output || tensorAffect == .both) {
                        await parsingData.incrementOutputDimension(dimension: dimension)
                    }
                }
            }
        case .SetDimension:
            if (format == .rSample) {
                try await parsingData.putSampleBackIntoDataSet(dataSet: dataSet)
                if (length < 0) {
                    let sampleIndex = try await dataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                }
                else {
                    try await parsingData.getSampleFromDataSet(sampleIndex: length, dataSet: dataSet)
                }
            }
            else {
                let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
                if (tensorAffect == .input || tensorAffect == .both) {
                    await parsingData.setOrIncrementInputDimension(dimension: dimension, toValue: length)
                }
                if (tensorAffect == .output || tensorAffect == .both) {
                    await parsingData.setOrIncrementOutputDimension(dimension: dimension, toValue: length)
                }
            }
        }

        return numUsed
    }

    func getFloatData(component: String, dataSet: DataSet) async -> Double?
    {
        switch (format) {
        case .fTextString:
            //  Convert the string to an index based on the known labels
            let labelIndex = await dataSet.getLabelIndex(label: component)
            if (labelIndex >= 0) {
                return Double(labelIndex)
            }
        case .fTextInt, .fTextFloat:
            return Double(component)
         default:
            return nil      //  Unsupported text type
        }

        return nil
    }

    func parseFixedWidthTextChunk(dataSet: DataSet, string: String, index : String.Index, parsingData : ParsingData) async throws -> String.Index?
    {
        //  Get the data for types that can use it
        var data : Double = 0.0
        var startIndex = index
        var endIndex = startIndex
        if (type != .OutputLabel && type != .Repeat && type != .SetDimension) {
            endIndex = string.index(startIndex, offsetBy: length)
            if (type != .Unused) {
                let substring = string[startIndex..<endIndex].trimmingCharacters(in: .whitespacesAndNewlines)
                if let value = Double(substring) {
                    data = value
                 }
                else {
                    throw DataParsingErrors.InvalidStringValue(substring)
                }
            }
        }

        //  Processing varies depending on the chunk type
        switch (type) {
        case .Unused:
            //  Throw away the unused data
            return endIndex
        case .Label, .LabelIndex:
            //  Store the label for this data set
            try await parsingData.appendOutputClass(Int(data))
        case .Feature:
            //  Store the feature values for the current input location
            try await parsingData.appendInputData([data], normalizationIndex: normalizationIndex)
        case .RedValue:
            //  Store the red values for the current input location
            try await parsingData.appendColorData([data], channel: .red, normalizationIndex: normalizationIndex)
        case .GreenValue:
            //  Store the green values for the current input location
            try await parsingData.appendColorData([data], channel: .green, normalizationIndex: normalizationIndex)
        case .BlueValue:
            //  Store the blue values for the current input location
            try await parsingData.appendColorData([data], channel: .blue, normalizationIndex: normalizationIndex)
        case .OutputValues:
            //  Store the output values for the current output location
            try await parsingData.appendOutputData([data], normalizationIndex: normalizationIndex)
        case .OutputLabel:
            let labelIndex = await dataSet.getLabelIndex(label: string)
            try await parsingData.appendOutputLabel(labelIndex, normalizationIndex: normalizationIndex)
        case .Repeat:
            //  Set the stage for reading this dimension
            let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
            if (dimension > 3) { break }        //  Skip sample iterators
            //  Iterate
            for _ in 0..<length {
                //  Process each chunk
                for chunk in repeatChunks! {
                    let finalIndex = try await chunk.parseFixedWidthTextChunk(dataSet: dataSet, string: string, index : startIndex, parsingData : parsingData)
                    if (finalIndex == nil) {
                        return nil
                    }
                    startIndex = finalIndex!
                }
            }

            //  Increment the dimension
            if (format != .rSample) {
                if (tensorAffect == .input || tensorAffect == .both) {
                    await parsingData.incrementInputDimension(dimension: dimension)
                }
                if (tensorAffect == .output || tensorAffect == .both) {
                    await parsingData.incrementOutputDimension(dimension: dimension)
                 }
            }
        case .SetDimension:
            if (format == .rSample) {
                try await parsingData.putSampleBackIntoDataSet(dataSet: dataSet)
                if (length < 0) {
                    let sampleIndex = try await dataSet.appendEmptySample()
                    try await parsingData.getSampleFromDataSet(sampleIndex: sampleIndex, dataSet : dataSet)
                }
                else {
                    try await parsingData.getSampleFromDataSet(sampleIndex: length, dataSet: dataSet)
                }
            }
            else {
                let dimension = format.rawValue - DataFormatType.rDimension0.rawValue
                if (tensorAffect == .input || tensorAffect == .both) {
                    await parsingData.setOrIncrementInputDimension(dimension: dimension, toValue: length)
                }
                if (tensorAffect == .output || tensorAffect == .both) {
                    await parsingData.setOrIncrementOutputDimension(dimension: dimension, toValue: length)
                }
            }
        }

        return endIndex
    }
}
