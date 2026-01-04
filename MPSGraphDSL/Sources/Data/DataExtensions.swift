//
//  DataExtensions.swift
//  
//
//  Created by Kevin Coble on 7/4/20.
//

import Foundation
import CoreGraphics

//  From  https://stackoverflow.com/questions/38023838/round-trip-swift-number-types-to-from-data
///  Protocol to assist in converting values (Int, Float, etc.) to and from Data objects
public protocol DataConvertible {
    init?(data: Data)
    var data: Data { get }
}

extension DataConvertible where Self: ExpressibleByIntegerLiteral{

    ///  Initializeer for a value from a set of bytes in a Data object
    public init?(data: Data) {
        var value: Self = 0
        guard data.count == MemoryLayout.size(ofValue: value) else { return nil }
        _ = withUnsafeMutableBytes(of: &value, { data.copyBytes(to: $0)} )
        self = value
    }

    /// Function to get the data bytes from a value
    public var data: Data {
        return withUnsafeBytes(of: self) { Data($0) }
    }
}

extension Int8 : DataConvertible { }
extension UInt8 : DataConvertible { }
extension Int16 : DataConvertible { }
extension UInt16 : DataConvertible { }
extension Int32 : DataConvertible { }
extension UInt32 : DataConvertible { }
extension Int : DataConvertible { }
extension UInt64 : DataConvertible { }
#if os(iOS)
extension Float16 : DataConvertible { }
#endif
extension Float32 : DataConvertible { }
extension Double : DataConvertible { }
extension CGFloat : DataConvertible { }

extension String: DataConvertible {
    /// Initialize a string from bytes in a Data object
    /// - Parameter data: The data object with the string bytes in UTF8 encoding
    public init?(data: Data) {
        self.init(data: data, encoding: .utf8)
    }
    /// Computed parameter that returns a Data object with the bytes from the string (in UTF8 encoding)
    public var data: Data {
        // Note: a conversion to UTF-8 cannot fail.
        return Data(self.utf8)
    }
}

extension Data {
    /// Append the byte representation of a value to the bytes in the Data object
    /// - Parameter value: The value to be converted to bytes and added
    public mutating func appendValue<T : DataConvertible>(_ value : T) {
        append(value.data)
    }
    
    /// Append the byte representation of a string to the bytes in the Data object
    /// - Parameter string: The string to be converted to bytes and added
    public mutating func appendString(_ string : String) {
        let temp = Data(string.utf8)
        let size = temp.count
        append(size.data)
        append(Data(temp))
    }
    
    /// Append the byte representation of a boolean value to the bytes in the Data object
    /// - Parameter flag: The boolean value to be converted to bytes and added
    public mutating func appendBool(_ flag : Bool) {
        append(flag ? UInt8(1) : UInt8(0))
    }
    
    /// Append the byte represention of an array of values to the bytes in the Data object
    /// - Parameter values: The array of values to be converted to bytes and added
    public mutating func appendValueArray<T : DataConvertible>(_ values : [T]) {
        let count = values.count
        appendValue(count)
        for value in values {
            appendValue(value)
        }
    }
    
    /// Append the byte represention of an array of strings to the bytes in the Data object
    /// - Parameter strings: The array of strings to be converted to bytes and added
    public mutating func appendStringArray(_ strings : [String]) {
        let count = strings.count
        appendValue(count)
        for string in strings {
            appendString(string)
        }
    }
    
    /// Append the byte represention of an optional URL to the bytes in the Data object
    /// - Parameter url: The optional URL to be converted to bytes and added
    public mutating func appendOptionalURL(_ url : URL?) {
        //  Put one byte in for the optional
        if let u = url {
            append(1)   //  Some indicator
            let data = u.dataRepresentation
            let length = data.count
            self.appendValue(length)
            append(data)
        }
        else {
            append(0)   //  Nil indicator
        }
    }
    
    /// Extract a value type from the Data byte stream at the supplied and increment the offset by the length of the value
    /// - Parameter offset: The offset in the byte stream for the value
    /// - Returns: The value extracted
    public func extractValue<T : DataConvertible>(offset : inout Int) -> T? {
        let size = MemoryLayout<T>.size
        if (offset + size > count) { return nil }
        let range = offset..<(offset+size)
        offset += size
        return T(data: self[range])
    }
    
    /// Extract a boolean value from the Data byte stream at the supplied and increment the offset by the length of the value
    /// - Parameter offset: The offset in the byte stream for the value
    /// - Returns: The boolean value extracted
    public func extractBool(offset : inout Int) -> Bool {
        let flag = self[offset]
        offset += 1
        return (flag != 0)
    }

    /// Extract a string from the Data byte stream at the supplied and increment the offset by the length of the string representation
    /// - Parameter offset: The offset in the byte stream for the string
    /// - Returns: The string extracted.  nil if the string was not a standard UTF8 encoding
    public func extractString(offset : inout Int) -> String?
    {
        let size = MemoryLayout<Int>.size
        let range = offset..<(offset+size)
        offset += size
        let length = Int(data : self[range])!
        let stringRange = offset..<(offset+length)
        offset += length
        return String(data : self[stringRange])
    }
    
    /// Extract a string from the Data byte stream at the supplied and increment the offset by the length of the string, which is passed in
    /// - Parameters:
    ///   - length: the length of the string to be extracted
    ///   - offset: The offset in the byte stream for the string
    /// - Returns: The string extracted.  nil if the string was not a standard UTF8 encoding
    public func extractUnsizedString(length: Int, offset : inout Int) -> String?
    {
        let stringRange = offset..<(offset+length)
        offset += length
        return String(data : self[stringRange])
    }
    
    /// Extract an array of values from the Data byte stream at the supplied and increment the offset by the length of the array representation
    /// - Parameter offset: The offset in the byte stream for the value array
    /// - Returns: the value array, or nil if an error occured
    public func extractValueArray<T : DataConvertible>(offset : inout Int) -> [T]? {
        if let count : Int = extractValue(offset: &offset) {
            var array : [T] = []
            for _ in 0..<count {
                let value : T? = extractValue(offset: &offset)
                if (value == nil) { return nil }
                array.append(value!)
            }
            return array
        }
        return nil
    }
    
    /// Extract an array of strings from the Data byte stream at the supplied and increment the offset by the length of the array representation
    /// - Parameter offset: The offset in the byte stream for the string array
    /// - Returns: the string array, or nil if an error occured
    public func extractStringArray(offset : inout Int) -> [String]? {
        if let count : Int = extractValue(offset: &offset) {
            var array : [String] = []
            for _ in 0..<count {
                let string : String? = extractString(offset: &offset)
                if (string == nil) { return nil }
                array.append(string!)
            }
            return array
        }
        return nil
    }
    
    /// Extract an optional URL from the Data byte stream at the supplied and increment the offset by the length of the optional URL representation
    /// - Parameter offset: The offset in the byte stream for the start of the optional URL
    /// - Returns: the optional URL, or nil if an error occurs
    public func extractOptionalURL(offset : inout Int) -> URL? {
        //  Get the some/nil indicator
        let optional = self[offset]
        offset += 1
        if (optional == 0) { return nil }
        let size = MemoryLayout<Int>.size
        let range = offset..<(offset+size)
        offset += size
        let length = Int(data : self[range])!
        let urlRange = offset..<(offset+length)
        offset += length
        return URL(dataRepresentation: self[urlRange], relativeTo: nil, isAbsolute: true)
    }
}
