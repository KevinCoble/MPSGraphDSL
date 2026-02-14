//
//  DataType.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/17/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Type of the data elements in a tensor
///
public enum DataType: Sendable {
    ///  UInt8
    case uInt8
    //    case int8
    //    case uInt16
    //    case int16
    //    case uInt32
    ///  Int32
    case int32
    //    case uInt64
    //    case int64
    ///  Float16
    case float16
    ///  Float32
    case float32
    ///  Double
    case double
    
    /// Initialize a data type from an MPSDataType enumeration
    /// - Parameter mpsDataType: the MPSDataType enumeration to be converted
    public init(from mpsDataType: MPSDataType) {
        switch (mpsDataType) {
        case .uInt8:
            self = .uInt8
        case .int32:
            self = .int32
        case .float16:
            self = .float16
        case .float32:
            self = .float32
        default:
            self = .float32
        }
    }
    
    public func usableInGraph() -> Bool
    {
        switch (self) {
        case .double:
            return false
        default:
            return true
        }

    }
    
    ///  Verify an ``DataElement`` is of this  type
    public func elementIsOfThisType(_ element: DataElement) -> Bool {
        switch (self) {
        case .uInt8:
            return (element is UInt8)
        case .int32:
            return (element is Int32)
        case .float16:
            return (element is Float16)
        case .float32:
            return (element is Float32)
        case .double:
            return (element is Double)
        }
    }
    
    func zeroElementOfType() -> DataElement
    {
        switch (self) {
        case .uInt8:
            return UInt8(0)
        case .int32:
            return Int32(0)
        case .float16:
            return Float16(0)
        case .float32:
            return Float32(0)
        case .double:
            return Double(0.0)
        }
    }

    ///  Get typed value from int
    public func typedFromInt(_ value : Int) -> DataElement {
        switch (self) {
        case .uInt8:
            if (value < Int(UInt8.min)) { return UInt8.min }
            if (value > Int(UInt8.max)) { return UInt8.max }
            return UInt8(value)
        case .int32:
            if (value < Int32.min) { return Int32.min }
            if (value > Int32.max) { return Int32.max }
            return Int32(value)
        case .float16:
            return Float16(value)
        case .float32:
            return Float(value)
        case .double:
            return Double(value)
        }
    }

    ///  Get typed value from double
    public func typedFromDouble(_ value : Double) -> DataElement {
        switch (self) {
        case .uInt8:
            if (value < Double(UInt8.min)) { return UInt8.min }
            if (value > Double(UInt8.max)) { return UInt8.max }
            return UInt8(value)
        case .int32:
            if (value < Double(Int32.min)) { return Int32.min }
            if (value > Double(Int32.max)) { return Int32.max }
            return Int32(value)
        case .float16:
            return Float16(value)
        case .float32:
            return Float(value)
        case .double:
            return value
        }
    }

    ///  Get the full range for the data type
    public func getFullRange() -> ParameterRange {
        switch (self) {
        case .uInt8:
            return UInt8.defaultRange
        case .int32:
            return Int32.defaultRange
        case .float16:
            return Float16.defaultRange
        case .float32:
            return Float.defaultRange
        case .double:
            return Double.defaultRange
        }
    }

    ///  Get the MPSGraph data type
    public func getMPSDataType() -> MPSDataType {
        switch (self) {
        case .uInt8:
            return .uInt8
        case .int32:
            return .int32
        case .float16:
            return .float16
        case .float32:
            return .float32
        default:
            return .invalid
        }
    }
}


///  Protocol for a single data item in a tensor
public protocol DataElement
{
    ///  Gets the data type of the element
    static var dataType : DataType { get }
    var dataType: DataType { get }
    
    ///  Gets the default (maximum) range of the type
    static var defaultRange : ParameterRange { get }
    
    ///  Gets or sets the value as an Integer
    var asInteger : Int { get set}
    
    ///  Gets or sets the value as a Double
    var asDouble : Double { get set}
    static func fromDouble(_ value: Double) -> DataElement
    
    ///  Gets or sets the value as a Float32
    var asFloat32 : Float { get set}

    ///  Gets the value as an unsigned byte, using the provided scaling range for the source value
    ///  If no range given, full positive range of data type is used (0 to maxValue)
    ///    (often used for converting to image pixel values for visualization)
    func asUnsignedByte(range: ParameterRange?) -> UInt8

    /// Get a random number in a specified range
    static func getRandomNumberInRange(range: ParameterRange) -> DataElement
}

extension UInt8 : DataElement {
    ///  Class function to get the DataType of the DataElement
    public static var dataType : DataType { get { return .uInt8 }}
    ///  Instance function to get the DataType of the DataElement
    public var dataType : DataType { get { return .uInt8 }}
    
    /// Get the default range of the data type
    public static var defaultRange : ParameterRange {
        get {
            do {
                return try ParameterRange(minimum: UInt8.min, maximum: UInt8.max)
            }
            catch { fatalError() } //  Should never be reached}
        }
    }
    
    ///  Get the data type as an integer
    public var asInteger : Int {
        get { return Int(self) }
        set(newValue) {
            if (newValue < Int(UInt8.min)) { self = UInt8.min }
            if (newValue > Int(UInt8.max)) { self = UInt8.max }
            self = UInt8(newValue)
        }
    }

    ///  Get the data type as a Double floating point value
    public var asDouble : Double {
        get { return Double(self) }
        set(newValue) {
            if (newValue < Double(UInt8.min)) { self = UInt8.min }
            if (newValue > Double(UInt8.max)) { self = UInt8.max }
            self = UInt8(newValue)
        }
    }
    ///  Return the value of a Double as the type of this DataElement
    public static func fromDouble(_ value: Double) -> DataElement {
        var setValue: UInt8
        if (value < Double(UInt8.min)) { setValue = UInt8.min }
        else if (value > Double(UInt8.max)) { setValue = UInt8.max }
        else { setValue = UInt8(value) }
        return setValue

    }

    ///  Gets or sets the value as a Float32
    public var asFloat32 : Float32 {
        get { return Float32(self) }
        set(newValue) {
            if (newValue < Float32(UInt8.min)) { self = UInt8.min }
            if (newValue > Float32(UInt8.max)) { self = UInt8.max }
            self = UInt8(newValue)
        }
    }

    /// Convert the DataElement value to an unsigned byte.  If a range is passed in, that range is used to scale the value into the 0-255 value.  Otherwise the default positive scale of the DataElement type is used
    /// - Parameter range: The range of the DataElement to scale to teh UInt8 range
    /// - Returns: The DataElement value converted to a UInt8 value
    public func asUnsignedByte(range: ParameterRange?) -> UInt8 {
        if let scale = range {
            let minimum = scale.min.asDouble
            let maximum = scale.max.asDouble
            var fraction = (Double(self) - minimum) / (maximum - minimum)
            if (fraction < 0.0) { fraction = 0.0 }
            if (fraction > 1.0) { fraction = 1.0 }
            fraction *= 255.0
            return UInt8(fraction + 0.5)
        }
        else {
            return self
        }
    }
    
    /// Get a random number of the DataElement type within a passed-in range
    /// - Parameter range: the range for the random value
    /// - Returns: a random value in the specified range
    public static func getRandomNumberInRange(range: ParameterRange) -> DataElement {
        let intRange = Int(range.min.asInteger)...Int(range.max.asInteger)
        let randomValue = Int.random(in: intRange)
        return UInt8(randomValue)
    }
}

extension Int32 : DataElement {
    ///  Class function to get the DataType of the DataElement
    public static var dataType : DataType { get { return .uInt8 }}
    ///  Instance function to get the DataType of the DataElement
    public var dataType : DataType { get { return .uInt8 }}
    
    /// Get the default range of the data type
    public static var defaultRange : ParameterRange {
        get {
            do {
                return try ParameterRange(minimum: UInt8.min, maximum: UInt8.max)
            }
            catch { fatalError() } //  Should never be reached}
        }
    }
    
    ///  Get the data type as an integer
    public var asInteger : Int {
        get { return Int(self) }
        set(newValue) {
            if (newValue < Int(Int32.min)) { self = Int32.min }
            if (newValue > Int(Int32.max)) { self = Int32.max }
            self = Int32(newValue)
        }
    }
    
    ///  Get the data type as a Double floating point value
    public var asDouble : Double {
        get { return Double(self) }
        set(newValue) {
            if (newValue < Double(Int32.min)) { self = Int32.min }
            if (newValue > Double(Int32.max)) { self = Int32.max }
            self = Int32(newValue)
        }
    }
    ///  Return the value of a Double as the type of this DataElement
    public static func fromDouble(_ value: Double) -> DataElement {
        var setValue: Int32
        if (value < Double(Int32.min)) { setValue = Int32.min }
        else if (value > Double(Int32.max)) { setValue = Int32.max }
        else { setValue = Int32(value) }
        return setValue
        
    }
    
    ///  Gets or sets the value as a Float32
    public var asFloat32 : Float32 {
        get { return Float32(self) }
        set(newValue) {
            if (newValue < Float32(Int32.min)) { self = Int32.min }
            if (newValue > Float32(Int32.max)) { self = Int32.max }
            self = Int32(newValue)
        }
    }
    
    /// Convert the DataElement value to an unsigned byte.  If a range is passed in, that range is used to scale the value into the 0-255 value.  Otherwise the default positive scale of the DataElement type is used
    /// - Parameter range: The range of the DataElement to scale to teh UInt8 range
    /// - Returns: The DataElement value converted to a UInt8 value
    public func asUnsignedByte(range: ParameterRange?) -> UInt8 {
        if let scale = range {
            let minimum = scale.min.asDouble
            let maximum = scale.max.asDouble
            var fraction = (Double(self) - minimum) / (maximum - minimum)
            if (fraction < 0.0) { fraction = 0.0 }
            if (fraction > 1.0) { fraction = 1.0 }
            fraction *= 255.0
            return UInt8(fraction + 0.5)
        }
        else {
            if (self < 0) { return UInt8.min }
            if (self < UInt8.max) { return UInt8.max }
            return UInt8(self)
        }
    }
    
    /// Get a random number of the DataElement type within a passed-in range
    /// - Parameter range: the range for the random value
    /// - Returns: a random value in the specified range
    public static func getRandomNumberInRange(range: ParameterRange) -> DataElement {
        let intRange = Int(range.min.asInteger)...Int(range.max.asInteger)
        let randomValue = Int.random(in: intRange)
        return Int32(randomValue)
    }
}

extension Float16 : DataElement {
    public static var dataType : DataType { get { return .float16 }}
    public var dataType : DataType { get { return .float16 }}

    /// Get the default range of the data type
    public static var defaultRange : ParameterRange {
        get {
            do {
                return try ParameterRange(minimum: -Float16.greatestFiniteMagnitude, maximum: Float16.greatestFiniteMagnitude)
            }
            catch { fatalError() } //  Should never be reached}
        }
    }
    
    ///  Get the data type as an integer
    public var asInteger : Int {
        get {
            if (self < Float16(Int.min)) { return Int.min }
            if (self > Float16(Int.max)) { return Int.max }
            return Int(self + 0.5)
        }
        set(newValue) {
            self = Float16(newValue)
        }
    }

    ///  Get the data type as a Double floating point value
    public var asDouble : Double {
        get { return Double(self) }
        set(newValue) {
            self = Float16(newValue)
        }
    }
    
    ///  Return the value of a Double as the type of this DataElement
    public static func fromDouble(_ value: Double) -> DataElement {
        return Float16(value)
    }

    ///  Gets or sets the value as a Float32
    public var asFloat32 : Float32 {
        get { return Float32(self) }
        set(newValue) {
            self = Float16(newValue)
        }
    }
    
    /// Convert the DataElement value to an unsigned byte.  If a range is passed in, that range is used to scale the value into the 0-255 value.  Otherwise the default positive scale of the DataElement type is used
    /// - Parameter range: The range of the DataElement to scale to teh UInt8 range
    /// - Returns: The DataElement value converted to a UInt8 value
    public func asUnsignedByte(range: ParameterRange?) -> UInt8 {
        var minimum: Double
        var maximum: Double
        if let scale = range {
            minimum = scale.min.asDouble
            maximum = scale.max.asDouble
        }
        else {
            minimum = 0.0
            maximum = Double(Float16.greatestFiniteMagnitude)
        }
        var fraction = (Double(self) - minimum) / (maximum - minimum)
        if (fraction < 0.0) { fraction = 0.0 }
        if (fraction > 1.0) { fraction = 1.0 }
        fraction *= 255.0
        return UInt8(fraction + 0.5)
    }
    
    /// Get a random number of the DataElement type within a passed-in range
    /// - Parameter range: the range for the random value
    /// - Returns: a random value in the specified range
    public static func getRandomNumberInRange(range: ParameterRange) -> DataElement {
        let doubleRange = range.min.asDouble...range.max.asDouble
        return Float(Double.random(in: doubleRange))
    }
}

extension Float : DataElement {
    public static var dataType : DataType { get { return .float32 }}
    public var dataType : DataType { get { return .float32 }}

    /// Get the default range of the data type
    public static var defaultRange : ParameterRange {
        get {
            do {
                return try ParameterRange(minimum: -Float.greatestFiniteMagnitude, maximum: Float.greatestFiniteMagnitude)
            }
            catch { fatalError() } //  Should never be reached}
        }
    }
    
    ///  Get the data type as an integer
    public var asInteger : Int {
        get {
            if (self < Float(Int.min)) { return Int.min }
            if (self > Float(Int.max)) { return Int.max }
            return Int(self + 0.5)
        }
        set(newValue) {
            self = Float(newValue)
        }
    }

    ///  Get the data type as a Double floating point value
    public var asDouble : Double {
        get { return Double(self) }
        set(newValue) {
            self = Float(newValue)
        }
    }
    
    ///  Return the value of a Double as the type of this DataElement
    public static func fromDouble(_ value: Double) -> DataElement {
        return Float(value)
    }

    ///  Gets or sets the value as a Float32
    public var asFloat32 : Float32 {
        get { return Float32(self) }
        set(newValue) {
            self = Float(newValue)
        }
    }
    
    /// Convert the DataElement value to an unsigned byte.  If a range is passed in, that range is used to scale the value into the 0-255 value.  Otherwise the default positive scale of the DataElement type is used
    /// - Parameter range: The range of the DataElement to scale to teh UInt8 range
    /// - Returns: The DataElement value converted to a UInt8 value
    public func asUnsignedByte(range: ParameterRange?) -> UInt8 {
        var minimum: Double
        var maximum: Double
        if let scale = range {
            minimum = scale.min.asDouble
            maximum = scale.max.asDouble
        }
        else {
            minimum = 0.0
            maximum = Double(Float.greatestFiniteMagnitude)
        }
        var fraction = (Double(self) - minimum) / (maximum - minimum)
        if (fraction < 0.0) { fraction = 0.0 }
        if (fraction > 1.0) { fraction = 1.0 }
        fraction *= 255.0
        return UInt8(fraction + 0.5)
    }
    
    /// Get a random number of the DataElement type within a passed-in range
    /// - Parameter range: the range for the random value
    /// - Returns: a random value in the specified range
    public static func getRandomNumberInRange(range: ParameterRange) -> DataElement {
        let doubleRange = range.min.asDouble...range.max.asDouble
        return Float(Double.random(in: doubleRange))
    }
}


extension Double : DataElement {
    public static var dataType : DataType { get { return .double }}
    public var dataType : DataType { get { return .double }}

    /// Get the default range of the data type
    public static var defaultRange : ParameterRange {
        get {
            do {
                return try ParameterRange(minimum: -Double.greatestFiniteMagnitude, maximum: Double.greatestFiniteMagnitude)
            }
            catch { fatalError() } //  Should never be reached}
        }
    }
    
    ///  Get the data type as an integer
    public var asInteger : Int {
        get {
            if (self < Double(Int.min)) { return Int.min }
            if (self > Double(Int.max)) { return Int.max }
            return Int(self + 0.5)
        }
        set(newValue) {
            self = Double(newValue)
        }
    }

    //  Get the data type as a Double floating point value
    public var asDouble : Double {
        get { return self }
        set(newValue) {
            self = newValue
        }
    }
    ///  Return the value of a Double as the type of this DataElement
    public static func fromDouble(_ value: Double) -> DataElement {
        return value
    }

    ///  Gets or sets the value as a Float32
    public var asFloat32 : Float32 {
        get { return Float32(self) }
        set(newValue) {
            self = Double(newValue)
        }
    }

    /// Convert the DataElement value to an unsigned byte.  If a range is passed in, that range is used to scale the value into the 0-255 value.  Otherwise the default positive scale of the DataElement type is used
    /// - Parameter range: The range of the DataElement to scale to teh UInt8 range
    /// - Returns: The DataElement value converted to a UInt8 value
    public func asUnsignedByte(range: ParameterRange?) -> UInt8 {
        var minimum: Double
        var maximum: Double
        if let scale = range {
            minimum = scale.min.asDouble
            maximum = scale.max.asDouble
        }
        else {
            minimum = 0.0
            maximum = Double.greatestFiniteMagnitude
        }
        var fraction = (self - minimum) / (maximum - minimum)
        if (fraction < 0.0) { fraction = 0.0 }
        if (fraction > 1.0) { fraction = 1.0 }
        fraction *= 255.0
        return UInt8(fraction + 0.5)
    }

    /// Get a random number of the DataElement type within a passed-in range
    /// - Parameter range: the range for the random value
    /// - Returns: a random value in the specified range
    public static func getRandomNumberInRange(range: ParameterRange) -> DataElement {
        let doubleRange = range.min.asDouble...range.max.asDouble
        return Double.random(in: doubleRange)
    }
}
