//
//  TensorShape.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 6/9/21.
//

import Foundation

///  Struct to define the shape of a Tensor, including number of dimensions and size of a Tensor in each of those dimensions
public struct TensorShape : Equatable, Sendable
{
    ///  The array of dimensions sizes for the shape of the Tensor
    public var dimensions : [Int]
    
    
    ///  Constructor when tensor dimensions are known
    ///
    /// - Parameters:
    ///   - dimensionSizes: The size of each dimension used in the tensor
    public init(_ dimensionSizes : [Int])
    {
        dimensions = dimensionSizes
    }
    
    ///  Constructor from an MPS shape array
    ///
    /// - Parameters:
    ///   - dimensionSizes: The size of each dimension used in the tensor
    public init(fromMPS : [NSNumber])
    {
        dimensions = []
        for dimension in fromMPS {
            dimensions.append(Int(truncating: dimension))
        }
    }

    ///  Constructor for shape after concatenation of two tensors
    ///
    /// - Parameters:
    ///   - tensor1Shape: The shape of the first tensor being concatenated
    ///   - tensor2Shape: The shape of the second tensor being concatenated
    ///   - concatenationDimension: The dimension along which the two tensors are being concatenated
    public init?(tensor1Shape : TensorShape, tensor2Shape : TensorShape, concatenationDimension : Int)
    {
        //  Verify the concatenation dimension is in range
        if (concatenationDimension < 0 && concatenationDimension >= tensor1Shape.numDimensions) { return nil }

        //  Verify both tensors have the same dimensions in all but the concatenation dimension
        if (tensor1Shape.numDimensions != tensor2Shape.numDimensions) { return nil }
        for dimension in 0..<tensor1Shape.numDimensions {
            if (dimension != concatenationDimension) {
                if (tensor1Shape.dimensions[dimension] != tensor2Shape.dimensions[dimension]) { return nil }
            }
        }
        var concatenatedDimensions = tensor1Shape.dimensions
        concatenatedDimensions[concatenationDimension] += tensor2Shape.dimensions[concatenationDimension]
        
        dimensions = concatenatedDimensions
    }
    
    ///  Get the number of dimensions in the Tensor Shape
    public var numDimensions : Int {
        get {
            return dimensions.count
        }
    }
    
    /// Determine if the shape starts with the passed in batch size
    /// - Parameter batchSize: The batch size to check the shape agains
    /// - Returns: true if the shape starts with the specified size, else false
    public func firstDimensionIsBatchSize(_ batchSize: Int) -> Bool {
            if (dimensions[0] ==  batchSize) { return true }
            return false
    }
    
    public func shapeWithRemovedBatchDimension() -> TensorShape {
        if (dimensions.count <= 1) { return self }
        var newDimensions = dimensions
        newDimensions.removeFirst()
        return TensorShape(newDimensions)
    }
    
    public func shapeWithAddedBatchDimension(_ batchSize: Int) -> TensorShape {
        var newDimensions = dimensions
        newDimensions.insert(batchSize, at: 0)
        return TensorShape(newDimensions)
    }

    ///  Get the total number of elements specified by the Tensor Shape
    public var totalSize : Int {
        get {
            return dimensions.reduce(1, *)
        }
    }
    
    ///  Get the storage location given a location within the dimension space
    ///  This assumes row-major storage
    public func storageIndex(location : [Int]) throws -> Int {
        //  Check the location matches the shape
        if (location.count != dimensions.count) { throw GenericMPSGraphDSLErrors.InvalidShape }
        
        var sliceSize = 1
        var index = 0
        for dimension in stride(from: dimensions.count-1, through: 0, by: -1) {
            if (location[dimension] < 0 || location[dimension] >= dimensions[dimension]) { throw GenericMPSGraphDSLErrors.InvalidShape }
            index += location[dimension] * sliceSize
            sliceSize *= dimensions[dimension]
        }
        
        return index
    }
    
    /// Get a shape array used by MPSGraph objects
    /// - Returns: an array of NSNumbers with the shape of the Tensor
    public func getMPSShape() -> [NSNumber]
    {
        return dimensions.map { NSNumber(value: $0) }
    }
    
    /// Verify the TensorShape matches a shape array used by MPSGraph objects
    /// - Returns: an array of NSNumbers with the shape of the Tensor
    public func matchesMPSShape(_ shape: [NSNumber]) -> Bool
    {
        if (dimensions.count != shape.count) { return false }
        for i in 0..<dimensions.count {
            if (dimensions[i] != Int(truncating: shape[i] as NSNumber)) { return false }
        }
        return true
    }

    // MARK: - Persistence
    
    ///  Create a TensorShape object from a portion of a Data object, starting at a given offset within the data bytes
    ///
    /// - Parameters:
    ///   - fromData: The Data object that has the bytes to extract the object from
    ///   - atOffset:  The offset within the byte array to start extracting this object.  This parameter is returned updated to point to the first byte that follows this object
    ///
    /// - Throws:
    ///   - CocoaError(fileReadCorruptFile) if the data in the bytes indicated do not match what was expected
    ///   - `PersistanceAIToolboxErrors.VersionAboveKnown` if the version of the object written to the bytes is greater than the version known by this code
    public init(fromData: Data, atOffset: inout Int) throws {
        //  Get the version
        guard let version : Int = fromData.extractValue(offset: &atOffset) else { throw CocoaError(.fileReadCorruptFile) }
        if (version > 1) {
            throw PersistanceErrors.VersionAboveKnown
        }

        //  Get the dimensions
        guard let tempdimensions : [Int] = fromData.extractValueArray(offset: &atOffset) else { throw CocoaError(.fileReadCorruptFile) }
        dimensions = tempdimensions
    }
    
    ///  Routine to get the bytes that need to be written for persistance storage for this object
    ///
    /// - Returns: A Data object with the bytes that should be stored
    public func getData() -> Data {
        var data = Data()
        
        //  Add the version
        let version = 1
        data.appendValue(version)
                
        //  Add the dimensions
        data.appendValueArray(dimensions)
       
        return data
    }

    ///   Routine to determine if another shape matches in all dimensions except the specified one
    ///
    /// - Returns: true if all non-indicated dimensions match
    public func shapesMatchExcept(_ dimension : Int, _ otherShape: TensorShape) -> Bool
    {
        if (numDimensions != otherShape.numDimensions) { return false }
        for index in 0..<dimensions.count {
            if (index != dimension) {
                if (dimensions[index] != otherShape.dimensions[index]) { return false }
            }
        }
        return true
    }

}
