//
//  ParameterRange.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 6/10/21.
//

import Foundation

///  Struct that defines the valid range for a parameter (a single dimension in an input tensor)
public struct ParameterRange
{
    let min : DataElement
    let max : DataElement
    
    ///  Constructor with minimum and maximum values
    ///  
    /// - Throws: `GenericMPSGraphDSLErrors.InvalidDimension` if the dimension is outside if the input tensor dimension count
    public init<T : DataElement>(minimum : T, maximum: T) throws {
        if (minimum.asDouble > maximum.asDouble) { throw GenericMPSGraphDSLErrors.InvalidValue }
        min = minimum
        max = maximum
    }
    
    ///  Check if a value is within the parameter range
    ///
    /// - Parameters:
    ///   - value: The value to be checked
    ///
    /// - Returns: a Bool result, true if value is within the parameter range
    public func valueIsInRange(value : DataElement) -> Bool {
        if (value.asDouble < min.asDouble) { return false }
        if (value.asDouble > max.asDouble) { return false }
        return true
    }
    
    ///  Add a uniform random error value that is a +/- a fraction of the range to the supplied value, clipping to result to the specified range
    ///
    /// - Parameters:
    ///   - value: The value to be add the error offset to
    ///   - errorFraction:  The fraction of the current parameter range that defines the (absoute) amount the error value added can be
    ///
    /// - Returns: an ``DataElement`` of the same data type as the value parameter, with the modified value
    public func addRandomError(value : DataElement, errorFraction: Double) -> DataElement {
        //  Get the limit range as doubles
        let minimum = min.asDouble
        let maximum = max.asDouble
        let fraction = (maximum - minimum) * errorFraction
        
        //  Get a random value
        let errorValue = Double.random(in: -fraction...fraction)
        
        //  Add the error value to the given value
        let modifiedValue = value.asDouble + errorValue
        
        //  Convert the value back to the correct type
        return value.dataType.typedFromDouble(modifiedValue)
    }
}
