//
//  Tensor.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/17/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Accelerate

/// Protocol that defines the operations that can be used on a Tensor
public protocol Tensor : Sendable {
    var shape : TensorShape { get }
    var type : DataType { get }

    func getElement(index : Int) throws -> Double
    func getElement(location: [Int]) throws -> Double
    func getElements() -> [Double]
    func getMPSGraphTensorData(forGraph: Graph) throws -> MPSGraphTensorData
    func getClassification() -> Int

    //  Parsing store functions
    mutating func setElement(index : Int, value: Double) throws
    mutating func setElement(location : [Int], value: Double) throws
    mutating func setElements(startIndex :Int, values: [Double]) throws
    mutating func setOneHot(hot: Int) throws
    
    //  Batch building/access functions
    mutating func setBatchSample(tensor: Tensor, batchIndex: Int) throws
    func getValuesForBatch(_ batchIndex: Int) throws -> [Double]
    func getClassificationForBatch(_ batchIndex: Int) throws -> Int
    func getTensorForBatch(_ batchIndex: Int) throws -> Tensor

    //  Run functions
    func getData() -> Data
}

extension Tensor {
    ///  Compares the tensor with another one.  If shape matches and all values are withing allowed difference, true is returned
    /// - Parameters:
    ///   - with: The other tensor to compare with this one
    ///   - maxDifference: The maximum difference on value compares
    public func compare(with: Tensor, maxDifference: Double) throws -> Bool {
        if (with.shape != shape) { return false }
        let numElements = shape.totalSize
        for i in 0..<numElements {
            let thisValue: Double = try getElement(index: i)
            let otherValue: Double = try with.getElement(index: i)
            if (abs(thisValue - otherValue) > maxDifference) { return false }
        }
        return true
    }
    
    /// Get the total difference (sum of absolute value of difference between each element) of this tensor with another of the same shape and size
    /// - Parameter with: The tensor to calculate the difference value with
    /// - Returns: The total difference value, or infinite if the tensor shapes do not match
    public func totalDifference(with: Tensor) -> Double {
        if (self.shape != with.shape) { return .infinity }
        
        var totalDifference = 0.0
        let ourElements : [Double] = getElements()
        let otherElements : [Double] = with.getElements()
        for i in 0..<ourElements.count {
            totalDifference += abs(ourElements[i] - otherElements[i])
        }
        
        return totalDifference
    }
    
    
    /// Print out a 1-dimensional (vector) tensor as a string of values with specified width and precision, enclosed in square brackets
    /// - Parameters:
    ///   - elementWidth: The width each value will be.  Left-side spaces added as needed
    ///   - precision: The precision (number of digits to the right of the decimal point
    /// - Throws: `GenericMPSGraphDSLErrors.InvalidShape` if the Tensor is not 1-dimensional
    public func print1D(elementWidth: Int, precision : Int) throws {
        if (shape.dimensions.count != 1) { throw GenericMPSGraphDSLErrors.InvalidShape }
        let numCols = shape.dimensions[0]
        var line = "["
        for col in 0..<numCols {
            let element = try getElement(index: col)
            let formattedNumber = String(format: "%\(elementWidth).\(precision)f", element)
            if (col > 0) { line += "," }
            line += formattedNumber
        }
        line += "]"
        print(line)
    }
    
    /// Print out a 2-dimensional (matrix) tensor as a string of values with specified width and precision, enclosed in square brackets for each row, and and additional bracket for the entire matrix
    /// - Parameters:
    ///   - elementWidth: The width each value will be.  Left-side spaces added as needed
    ///   - precision: The precision (number of digits to the right of the decimal point
    /// - Throws: `GenericMPSGraphDSLErrors.InvalidShape` if the Tensor is not 2-dimensional
    public func print2D(elementWidth: Int, precision : Int) throws {
        if (shape.dimensions.count != 2) { throw GenericMPSGraphDSLErrors.InvalidShape }
        let numRows = shape.dimensions[0]
        let numCols = shape.dimensions[1]
        var line : String = "["
        for row in 0..<numRows {
            line += "["
            for col in 0..<numCols {
                let index = row*numCols + col       //  row-major storage
                let element = try getElement(index: index)
                let formattedNumber = String(format: "%\(elementWidth).\(precision)f", element)
                if (col > 0) { line += "," }
                line += formattedNumber
            }
            line += "]"
            if (row == (numRows-1)) { line += "]" }
            print(line)
            line = " "
        }
    }
    
    /// Print out a 3-dimensional (row-column-channel or batch-row-column) tensor as a string of values with specified width and precision, as a row of 2D matrixes
    /// - Parameters:
    ///   - elementWidth: The width each value will be.  Left-side spaces added as needed
    ///   - precision: The precision (number of digits to the right of the decimal point
    ///   - splitLastDimension:(Optional) If true (the default), tensor is treated as row-column channel and each matrix printed is different in the last dimension.  Otherwise batch-row-column and each matrix is different in first dimension
    /// - Throws: `GenericMPSGraphDSLErrors.InvalidShape` if the Tensor is not 3-dimensional
    public func print3D(elementWidth: Int, precision : Int, splitLastDimension: Bool = true) throws {
        if (shape.dimensions.count != 3) { throw GenericMPSGraphDSLErrors.InvalidShape }
        let numRows: Int
        let numCols: Int
        let numMatrices: Int
        if (splitLastDimension) {
            numRows = shape.dimensions[0]
            numCols = shape.dimensions[1]
            numMatrices = shape.dimensions[2]
        }
        else {
            numRows = shape.dimensions[1]
            numCols = shape.dimensions[2]
            numMatrices = shape.dimensions[0]
        }
        var lines = Array(repeating: "", count: numRows)
        var index = 0
        for matrix in 0..<numMatrices {
            var line : String = "["
            for row in 0..<numRows {
                line += "["
                for col in 0..<numCols {
                    if (splitLastDimension) {
                        index = row*numCols*numMatrices + col*numMatrices + matrix
                    }
                    else {
                        index = matrix*numRows*numCols + row*numCols + col
                    }
                    let element = try getElement(index: index)
                    let formattedNumber = String(format: "%\(elementWidth).\(precision)f", element)
                    if (col > 0) { line += "," }
                    line += formattedNumber
                }
                line += "]"
                if (row == (numRows-1)) {
                    line += "]  "
                }
                else {
                    line += "   "
                }
                lines[row] += line
                line = " "
            }
        }
        
        for row in 0..<numRows {
            print(lines[row])
        }
    }
    
    /// Print out a 4-dimensional batch-row-column-channel) tensor as a string of values with specified width and precision, as a multiple rows of 2D matrixes
    /// - Parameters:
    ///   - elementWidth: The width each value will be.  Left-side spaces added as needed
    ///   - precision: The precision (number of digits to the right of the decimal point
    /// - Throws: `GenericMPSGraphDSLErrors.InvalidShape` if the Tensor is not 4-dimensional
    public func print4D(elementWidth: Int, precision : Int) throws {
        if (shape.dimensions.count != 4) { throw GenericMPSGraphDSLErrors.InvalidShape }
        let batchSize = shape.dimensions[0]
        let numRows = shape.dimensions[1]
        let numCols = shape.dimensions[2]
        let numMatrices = shape.dimensions[3]

        for batch in 0..<batchSize {
            var lines = Array(repeating: "", count: numRows)
            var index = 0
            for matrix in 0..<numMatrices {
                var line : String = "["
                for row in 0..<numRows {
                    line += "["
                    for col in 0..<numCols {
                        index = batch*numRows*numCols*numMatrices + row*numCols*numMatrices + col*numMatrices + matrix
                        let element = try getElement(index: index)
                        let formattedNumber = String(format: "%\(elementWidth).\(precision)f", element)
                        if (col > 0) { line += "," }
                        line += formattedNumber
                    }
                    line += "]"
                    if (row == (numRows-1)) {
                        line += "]  "
                    }
                    else {
                        line += "   "
                    }
                    lines[row] += line
                    line = " "
                }
            }
            
            for row in 0..<numRows {
                print(lines[row])
            }
            
            if (batch < batchSize-1) { print("") }
        }
    }
    
    /// Print out a  tensor as a string of values with specified width and precision
    /// - Parameters:
    ///   - elementWidth: The width each value will be.  Left-side spaces added as needed
    ///   - precision: The precision (number of digits to the right of the decimal point
    /// - Throws: `GenericMPSGraphDSLErrors.InvalidShape` if the Tensor is not 4-dimensional
    public func printTensor(elementWidth: Int, precision : Int) throws {
        switch shape.dimensions.count {
        case 1:
            try print1D(elementWidth: elementWidth, precision: precision)
        case 2:
            try print2D(elementWidth: elementWidth, precision: precision)
        case 3:
            try print3D(elementWidth: elementWidth, precision: precision)
        case 4:
            try print4D(elementWidth: elementWidth, precision: precision)
        default:
            throw GenericMPSGraphDSLErrors.InvalidShape
        }
    }
    
    /// Get a gaussian random value using the Box-Muller transform
    /// - Parameters:
    ///   - mean: mean value for the distribution
    ///   - standardDeviation: standard deviation of the distribution
    /// - Returns: random gaussian value
    public static func getGaussianRandom(mean: Double, standardDeviation: Double) -> Double {
        let u1 = Double.random(in: 0.0...1.0)
        let u2 = Double.random(in: 0.0...1.0)
        
        // Box-Muller transform
        let randStdNormal = sqrt(-2.0 * log(u1)) * sin(2.0 * Double.pi * u2)
        
        // Return with custom mean and standard deviation
        return mean + standardDeviation * randStdNormal
    }
}


/// Class with static methods for creating a tensor of a specific type, or with specific values
public final class CreateTensor {
    /// Create a Tensor of the specified type and shape, with all elements set to the passed in constant value
    /// - Parameters:
    ///   - type: The data type of the elements of the Tensor
    ///   - shape: The shape of the resulting Tensor
    ///   - initialValue: The initial value for each element of the Tensor
    /// - Returns: The created Tensor
    public static func constantValues(type: DataType, shape: TensorShape, initialValue: Double) -> Tensor {
        switch (type) {
        case .uInt8:
            return TensorUInt8(shape: shape, initialValue: UInt8(initialValue))
        case .int32:
            return TensorInt32(shape: shape, initialValue: Int32(initialValue))
        case .float16:
            return TensorFloat16(shape: shape, initialValue: Float16(initialValue))
        case .float32:
            return TensorFloat32(shape: shape, initialValue: Float32(initialValue))
        case .double:
            return TensorDouble(shape: shape, initialValue: initialValue)
        }
    }
    
    /// Create a Tensor of the specified type and shape, with all elements set to a uniform random value in the range specified
    /// - Parameters:
    ///   - type: The data type of the elements of the Tensor
    ///   - shape: The shape of the resulting Tensor
    ///   - range: The range for random values for the initial value for each element of the Tensor
    /// - Returns: The created Tensor
    public static func randomUniformValues(type: DataType, shape: TensorShape, range: ParameterRange) -> Tensor {
        switch (type) {
        case .uInt8:
            return TensorUInt8(shape: shape, randomValueRange: range)
        case .int32:
            return TensorInt32(shape: shape, randomValueRange: range)
        case .float16:
            return TensorFloat16(shape: shape, randomValueRange: range)
        case .float32:
            return TensorFloat32(shape: shape, randomValueRange: range)
        case .double:
            return TensorDouble(shape: shape, randomValueRange: range)
        }
    }
    
    /// Create a Tensor of the specified type and shape, with all elements set to a gaussian random value with the mean and standard deviation specified
    /// - Parameters:
    ///   - type: The data type of the elements of the Tensor
    ///   - shape: The shape of the resulting Tensor
    ///   - range: The range for random values for the initial value for each element of the Tensor
    /// - Returns: The created Tensor
    public static func randomNormalValues(type: DataType, shape: TensorShape, mean: Double, standardDeviation: Double) -> Tensor {
        switch (type) {
        case .uInt8:
            return TensorUInt8(shape: shape, mean: mean, standardDeviation: standardDeviation)
        case .int32:
            return TensorInt32(shape: shape, mean: mean, standardDeviation: standardDeviation)
        case .float16:
            return TensorFloat16(shape: shape, mean: mean, standardDeviation: standardDeviation)
        case .float32:
            return TensorFloat32(shape: shape, mean: mean, standardDeviation: standardDeviation)
        case .double:
            return TensorDouble(shape: shape, mean: mean, standardDeviation: standardDeviation)
        }
    }

    /// Create a Tensor from an MPSGraphTensorData object
    /// - Parameter from: The MPSGraphTensorData object to create the Tensor from.  The data type and shape are extracted from this object
    /// - Returns: The created Tensor
    public static func fromMPSTensorData(_ from: MPSGraphTensorData) -> Tensor {
        let type = DataType(from: from.dataType)
        switch (type) {
            case .uInt8:
                return TensorUInt8(fromMPSTensorData: from)
            case .int32:
                return TensorInt32(fromMPSTensorData: from)
            case .float16:
                return TensorFloat16(fromMPSTensorData: from)
            case .float32:
                return TensorFloat32(fromMPSTensorData: from)
            case .double:
                return TensorDouble(fromMPSTensorData: from)
        }
    }
    
    /// Create a weight initialization tensor with the given parameters
    /// - Parameters:
    ///   - type: The data type for the resulting tensor
    ///   - shape: The shape of the weight tensor
    ///   - initializationInfo: The weight initialization description with the initialization type and parameters
    ///   - numInputs: The number of inputs to the weight calculation
    ///   - numOutput: The number of outpus from the weight calcuation
    /// - Returns: A tensor of the specified type and size, initialized based in the initialization info given
    public static func createWeightInitializationTensor(type: DataType, shape: TensorShape, initializationInfo: WeightInitialization, numInputs: Int, numOutput: Int) throws -> Tensor {
        switch initializationInfo {
        case .uniform(let min, let max):
            let range = try ParameterRange(minimum: min, maximum: max)
            return CreateTensor.randomUniformValues(type: type, shape: shape, range: range)
        case .normal(let mean, let standardDeviation):
            return CreateTensor.randomNormalValues(type: type, shape: shape, mean: mean, standardDeviation: standardDeviation)
        case .XavierGlorotUniform:
            let max = sqrt(6.0 / Double(numInputs + numOutput))
            let min = -max
            let range = try ParameterRange(minimum: min, maximum: max)
            return CreateTensor.randomUniformValues(type: type, shape: shape, range: range)
        case .HeUniform:
            let max = sqrt(6.0 / Double(numInputs))
            let min = -max
            let range = try ParameterRange(minimum: min, maximum: max)
            return CreateTensor.randomUniformValues(type: type, shape: shape, range: range)
        case .XavierGlorotNormal:
            let standardDeviation = sqrt(2.0 / Double(numInputs + numOutput))
            return CreateTensor.randomNormalValues(type: type, shape: shape, mean: 0.0, standardDeviation: standardDeviation)
        case .HeNormal:
            let standardDeviation = sqrt(2.0 / Double(numInputs))
            return CreateTensor.randomNormalValues(type: type, shape: shape, mean: 0.0, standardDeviation: standardDeviation)
        }
    }
    
    /// Create a weight initialization tensor with the given parameters
    /// - Parameters:
    ///   - type: The data type for the resulting tensor.  Must be a floating type
    ///   - shape: The shape of the weight tensor.  Must be a square matrix of dimension 2.  The number of inputs and outputs are assumed to be the given by the dimensions of the square
    ///   - initializationInfo: The weight initialization description with the initialization type and parameters
    ///   - numGates: The number of square matrices concatenated together.  For example, a 4 gate 5x5 square shape will result in a 20x5 matrix returned, with each 5x5 square an orthogonal matrix
    /// - Returns: A tensor of the specified type and size, initialized based in the initialization info given
    public static func createOrthogonalWeightInitializationTensor(type: DataType, shape: TensorShape, initializationInfo: WeightInitialization, numGates: Int) throws -> Tensor {
        //  Verify it is of an appropriate type
        if (type != .float32 && type != .float16) { throw MPSGraphNeuralNetErrors.OrthogonalizationOnNonFloatingTensor }
        if (shape.numDimensions != 2) { throw MPSGraphNeuralNetErrors.InputTensorNotSquare2D }
        let numInputs = shape.dimensions[0]

        //  Get each gate's orthogonal tensor
        var gateTensors: [TensorDouble] = []
        for _ in 0..<numGates {
            var tensor = try CreateTensor.createWeightInitializationTensor(type: .double, shape: shape, initializationInfo: initializationInfo, numInputs: numInputs, numOutput: numInputs) as! TensorDouble
            try tensor.orthogonalize()
            gateTensors.append(tensor)
        }
        
        //  Concatenate all the gate tensors along the row axis
        var concatenatedElements: [Double]
        if (numGates == 1) {
            concatenatedElements = gateTensors[0].getElements()
        }
        else {
            concatenatedElements = []
            var offset = 0
            for _ in 0..<numInputs {
                for gate in 0..<numGates {
                    let elements = gateTensors[gate].getElements()
                    concatenatedElements += elements[offset..<(offset+numInputs)]
                }
                offset += numInputs
            }
        }
        
        //  Convert to the required tensor type
        if (type == .float16) {
            let float16Elements : [Float16] = concatenatedElements.map { Float16($0) }
            let tensor = try TensorFloat16(shape: shape, initialValues: float16Elements)
            return tensor
        }
        else {
            let float32Elements : [Float32] = concatenatedElements.map { Float32($0) }
            let tensor = try TensorFloat32(shape: shape, initialValues: float32Elements)
            return tensor
        }
    }
    
    /// Create a tensor of the same type with the same values, but a different shape.  The new shape must have the same number of elements as the old shape
    /// - Parameters:
    ///   - from: The Tensor to create the new tensor from
    ///   - newShape: The new shape of the tensor
    /// - Returns: The new tensor with the old tensors' type and elements, but the new shape
    public static func createReshapedTensor(from: Tensor, newShape: TensorShape) throws -> Tensor {
        //  Make sure they are of the same total size
        if (from.shape.totalSize != newShape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidShape }
        
        switch (from.type) {
        case .uInt8:
            let elements: [UInt8] = (from as! TensorUInt8).getElements()
            return try TensorUInt8(shape: newShape, initialValues: elements)
        case .int32:
            let elements: [Int32] = (from as! TensorInt32).getElements()
            return try TensorInt32(shape: newShape, initialValues: elements)
        case .float16:
            let elements: [Float16] = (from as! TensorFloat16).getElements()
            return try TensorFloat16(shape: newShape, initialValues: elements)
        case .float32:
            let elements: [Float32] = (from as! TensorFloat32).getElements()
            return try TensorFloat32(shape: newShape, initialValues: elements)
        case .double:
            let elements: [Double] = (from as! TensorDouble).getElements()
            return try TensorDouble(shape: newShape, initialValues: elements)
        }
    }
}

///  A Tensor with elements of type Double (Not usable by MPSGraph)
public struct TensorDouble : Tensor {
    ///  The shape of the Tensor
    public let shape : TensorShape
    var elements : [Double]
    
    ///  The data type of the Tensor
    public var type : DataType {
        return .double
    }

    ///  Construct a constant value tensor of a given type and shape
    public init(shape : TensorShape, initialValue : Double) {
        self.shape = shape
        let totalElements = shape.totalSize
        
        //  Create the array
        elements = Array(repeating: initialValue, count: totalElements)
    }

    ///  Construct a value tensor of a given type and shape with the given data
    ///
    ///  This function assumes a single dimension input vector, sized to the total size of the TensorShape, mapping to the tensor using row-major format
    public init(shape : TensorShape, initialValues : [Double]) throws {
        self.shape = shape
        
        if (shape.totalSize > initialValues.count) { throw GenericMPSGraphDSLErrors.NotEnoughValues }
        
        //  Create the array
    
        elements = []
        for value in initialValues {
            elements.append(value)
        }
     }

    ///  Construct a value tensor of a given type and shape with uniform random data
    public init(shape : TensorShape, randomValueRange: ParameterRange) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        let doubleRange = randomValueRange.min.asDouble...randomValueRange.max.asDouble
        for _ in 0..<elementCount {
            elements.append(Double.random(in: doubleRange))
        }
    }

    ///  Construct a value tensor of a given type and shape with normal random data
    public init(shape : TensorShape, mean: Double, standardDeviation: Double) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        for _ in 0..<elementCount {
            elements.append(TensorDouble.getGaussianRandom(mean: mean, standardDeviation: standardDeviation))
        }
    }

    ///  Construct a Tensor from an MPSGraphTensorData object
    public init(fromMPSTensorData tensorData: MPSGraphTensorData) {
        self.shape = TensorShape(fromMPS: tensorData.shape)
        
        let NDArray = tensorData.mpsndarray()
        let totalSize = shape.totalSize
        
            var values : [Double] = Array(repeating: 0, count: totalSize)
            NDArray.readBytes(&values, strideBytes: nil)
            elements = values
     }
    
    /// Create a MPSGraphTensorData object for the specified graph, using the data in the Tensor
    /// - Parameter forGraph: The ``Graph`` object that the MPSGraphTensorData will be used with
    /// - Returns: The create MPSGraphTensorData object
    public func getMPSGraphTensorData(forGraph: Graph) throws -> MPSGraphTensorData {
        throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph
    }
    
    public mutating func orthogonalize() throws {
        //  Make sure the tensor is 2-dimensional and square
        if (shape.numDimensions != 2) { throw MPSGraphNeuralNetErrors.InputTensorNotSquare2D }
        if (shape.dimensions[0] != shape.dimensions[1]) { throw MPSGraphNeuralNetErrors.InputTensorNotSquare2D }
        
        //  Perform a Gram-Schmidt process to make an orthogonal basis from the matrix vectors
        let numRows = shape.dimensions[0]
        let numColumns = shape.dimensions[1]
        let length = vDSP_Length(numColumns)
        var dp1: Double = 0.0
        var dp2: Double = 0.0
        var orthogonalBasis: [Double] = []
        var numOrthogonalVectorsAdded = 0
        for row in 0..<numRows {
            let rowOffset = row * numColumns
            let unProjectedOriginalVector = Array(elements[rowOffset..<(rowOffset + numColumns)])
            //  Repeatedly subtract projections of all the already orthogonal vectors to get this one
            var newOrthogonalVector = unProjectedOriginalVector
            for ov in 0..<numOrthogonalVectorsAdded {
                let ovOffset = ov * numColumns
                let orthogonalBasisVector = Array(orthogonalBasis[ovOffset..<(ovOffset + numColumns)])
                // Calculate projection: (v . b) / (b . b) * b
                vDSP_dotprD(unProjectedOriginalVector, 1, orthogonalBasisVector, 1, &dp1, length)  //  (v . b)
                vDSP_dotprD(orthogonalBasisVector, 1, orthogonalBasisVector, 1, &dp2, length)  //  (b . b)
                let projScalar = dp1 / dp2
                let projection = vDSP.multiply(projScalar, orthogonalBasisVector)
                newOrthogonalVector = vDSP.subtract(newOrthogonalVector, projection)    //  Subtract projection from vector
            }
            orthogonalBasis += newOrthogonalVector
            numOrthogonalVectorsAdded += 1
        }
        elements = orthogonalBasis
    }
    
    // MARK: - Getting elements
    
    ///  Get a specified storage index element
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Double {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }
    
    ///  Get a specified tensor element by location, as a Double
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of offsets within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Double {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }
    
    ///  Get all the tensor elements as an array of the Tensor's type
    ///
    /// - Returns: the elements as an array of the type of the Tensor
    public func getElements() -> [Double] {
        return elements
    }
    
    /// Get the classification value (index of highest element in Tensor)
    /// - Returns: The classification value
    public func getClassification() -> Int {
        let dataSize = shape.totalSize
        var argmax = 0
        var maxValue = -Double.greatestFiniteMagnitude
        for i in 0..<dataSize {
            if (elements[i] > maxValue) {
                maxValue = elements[i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    // MARK: - Setting elements

    ///  Set a specified index element to the specified Double value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElement(index : Int, value: Double) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }
    
    /// Set a specified location of the Tensor to the specified Double value
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to Float32
    public mutating func setElement(location : [Int], value: Double) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }

    ///  Store a specified array of elements into the tensor, starting at a given location index
    ///
    /// - Parameters:
    ///   - index: The starting index into the tensor storage.  The count of the values parameter determines the length of the store
    ///   - values: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElements(startIndex :Int, values: [Double]) throws {
        if (startIndex < 0 || (startIndex + values.count) > shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        for i in 0..<values.count {
            elements[startIndex + i] = values[i]
        }
    }

    ///  Set the tensor values to all zeros, except the hot index, which gets set to 1
       public mutating func setOneHot(hot: Int) throws {
        //  Check the index
        let totalElements = shape.totalSize
        if (hot < 0 || hot >= totalElements) {
            throw GenericMPSGraphDSLErrors.ClassificationValueOutOfRange
        }

        //  Set the one-hot
           elements = Array(repeating: 0.0, count: totalElements)
           elements[hot] = 1.0
    }

    
// MARK: - Batch functions

    ///  Put a passed in tensor into the batch index provided
    ///
    /// - Parameters:
    ///   - tensor: The sample tensor to be added to this batch tensor
    ///   - batchIndex: The location within the batch that the sample goes into
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the input tensor type does not match the batch tensor type
    ///   - `MPSGraphRunErrors.SampleDoesntMatchBatchShape` if the input tensor type does not match the batch tensor shape minus the first (batch index) dimension
    public mutating func setBatchSample(tensor: Tensor, batchIndex: Int) throws {
        //  Verify the tensor is of the right type
        if (tensor.type != .float32) { throw GenericMPSGraphDSLErrors.InvalidType }
        
        //  Verify the tensor matches our size, minus the batch index
        let sampleShape = tensor.shape.dimensions
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        var expectedShape = ourShape
        expectedShape.removeFirst()
        if (sampleShape != expectedShape) { throw MPSGraphRunErrors.SampleDoesntMatchBatchShape }
       
        //  Calculate a start index for the sample
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let sampleSize = sampleShape.reduce(1, *)
        let startIndex = batchIndex * sampleSize
        
        //  Move the data in
        let tensorDouble = tensor as! TensorDouble
        for i in 0..<sampleSize {
            elements[startIndex + i] = tensorDouble.elements[i]
        }
    }
    
    ///  Retrieve the element values for a specified batch.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the elements for the batch, as a Double array`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getValuesForBatch(_ batchIndex: Int) throws -> [Double] {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var values: [Double] = []
        for i in 0..<dataSize {
            values.append(elements[startIndex + i])
        }
        return values
    }

    ///  Parses the element values s\for a specified batch, taking the argmax to find the classification value.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the index of the highest value for the batch - the classification value`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getClassificationForBatch(_ batchIndex: Int) throws -> Int {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var argmax = 0
        var maxValue = -Double.greatestFiniteMagnitude
        for i in 0..<dataSize {
            if (elements[startIndex + i]) > maxValue {
                maxValue = elements[startIndex + i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    /// Create a tensor from extracted batch data.
    /// - Parameter batchIndex: The index within the batch (first) dimension to extract the data from
    /// - Returns: A tensor of the size of this tensor minus the batch dimension, filled with the data for the specified batch index
    public func getTensorForBatch(_ batchIndex: Int) throws -> Tensor {
        if (shape.numDimensions < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > shape.dimensions[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let outputShape = shape.shapeWithRemovedBatchDimension()
        let dataSize = outputShape.totalSize
        let startIndex = dataSize * batchIndex
        let batchValues = Array(elements[startIndex..<startIndex+dataSize])
        let resultTensor = try TensorDouble(shape: outputShape, initialValues: batchValues)
        return resultTensor
    }


    // MARK: - Persistance functions

    /// Get a Data object filled with the bytes from the Tensor
    ///
    ///  - Returns: the Data object with the bytes
    public func getData() -> Data {
        let data: Data
        let elementCount = shape.totalSize
        data = Data(bytes: elements, count: elementCount * MemoryLayout<Double>.size)
        
        return data
    }
}


///  A Tensor with elements of type Float32
public struct TensorFloat32 : Tensor {
    ///  The shape of the Tensor
    public let shape : TensorShape
    var elements : [Float32]
    
    ///  The data type of the Tensor
    public var type : DataType {
        return .float32
    }

    ///  Construct a constant value tensor of a given type and shape
    public init(shape : TensorShape, initialValue : Float32) {
        self.shape = shape
        let totalElements = shape.totalSize
        
        //  Create the array
        elements = Array(repeating: initialValue, count: totalElements)
    }

    ///  Construct a value tensor of a given type and shape with the given data
    ///
    ///  This function assumes a single dimension input vector, sized to the total size of the TensorShape, mapping to the tensor using row-major format
    public init(shape : TensorShape, initialValues : [Float32]) throws {
        self.shape = shape
        
        if (shape.totalSize > initialValues.count) { throw GenericMPSGraphDSLErrors.NotEnoughValues }
        
        //  Create the array
    
        elements = []
        for value in initialValues {
            elements.append(value)
        }
     }

    ///  Construct a value tensor of a given type and shape with the random data
    public init(shape : TensorShape, randomValueRange: ParameterRange) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        let doubleRange = randomValueRange.min.asDouble...randomValueRange.max.asDouble
        for _ in 0..<elementCount {
            elements.append(Float32(Double.random(in: doubleRange)))
        }
    }

    ///  Construct a value tensor of a given type and shape with normal random data
    public init(shape : TensorShape, mean: Double, standardDeviation: Double) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        for _ in 0..<elementCount {
            elements.append(Float32(TensorFloat32.getGaussianRandom(mean: mean, standardDeviation: standardDeviation)))
        }
    }

    ///  Construct a Tensor from an MPSGraphTensorData object
    public init(fromMPSTensorData tensorData: MPSGraphTensorData) {
        self.shape = TensorShape(fromMPS: tensorData.shape)
        
        let NDArray = tensorData.mpsndarray()
        let totalSize = shape.totalSize
        
            var values : [Float32] = Array(repeating: 0, count: totalSize)
            NDArray.readBytes(&values, strideBytes: nil)
            elements = values
     }
    
    /// Create a MPSGraphTensorData object for the specified graph, using the data in the Tensor
    /// - Parameter forGraph: The ``Graph`` object that the MPSGraphTensorData will be used with
    /// - Returns: The create MPSGraphTensorData object
    public func getMPSGraphTensorData(forGraph: Graph) throws -> MPSGraphTensorData {
        let descriptor = MPSNDArrayDescriptor(dataType: DataType.float32.getMPSDataType(), shape: shape.getMPSShape())
        let NDArray = MPSNDArray(device: forGraph.device, descriptor: descriptor)
        var array = elements
        NDArray.writeBytes(&array, strideBytes: nil)
        return MPSGraphTensorData(NDArray)
    }
    
    // MARK: - Getting elements

    ///  Get a specified storage index element
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as an Float32
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Float32 {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }
    
    ///  Get a specified storage index element
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Double {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }
    
    ///  Get a specified tensor element by location, as a Float32
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of offsets within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a Float32
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Float {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }
    
    ///  Get a specified tensor element by location, as a Double
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of offsets within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Double {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }
    
    ///  Get all the tensor elements as an array of the Tensor's type
    ///
    /// - Returns: the elements as an array of the type of the Tensor
    public func getElements() -> [Float32] {
        return elements
    }

    ///  Get all the tensor elements as a double array
    ///
    /// - Returns: the elements as a double array
    public func getElements() -> [Double] {
        var array: [Double] = []
        for element in elements {
            array.append(element.asDouble)
        }
        return array
    }
    
    /// Get the classification value (index of highest element in Tensor)
    /// - Returns: The classification value
    public func getClassification() -> Int {
        let dataSize = shape.totalSize
        var argmax = 0
        var maxValue = -Float32.greatestFiniteMagnitude
        for i in 0..<dataSize {
            if (elements[i] > maxValue) {
                maxValue = elements[i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    // MARK: - Setting elements

    ///  Set a specified index element to the specified Float32 value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the value type does not match the tensor type
    public mutating func setElement(index : Int, value: Float32) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }

    ///  Set a specified index element to the specified Double value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElement(index : Int, value: Double) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = Float32(value)
    }
    
    /// Set a specified location of the Tensor to the specified Float32 value
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to Float32
    public mutating func setElement(location : [Int], value: Float32) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }
    
    /// Set a specified location of the Tensor to the specified Double value
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to Float32
    public mutating func setElement(location : [Int], value: Double) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = Float32(value)
    }

    ///  Store a specified array of elements into the tensor, starting at a given location index
    ///
    /// - Parameters:
    ///   - index: The starting index into the tensor storage.  The count of the values parameter determines the length of the store
    ///   - values: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElements(startIndex :Int, values: [Double]) throws {
        if (startIndex < 0 || (startIndex + values.count) > shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        for i in 0..<values.count {
            elements[startIndex + i] = Float32(values[i])
        }
    }

    ///  Set the tensor values to all zeros, except the hot index, which gets set to 1
       public mutating func setOneHot(hot: Int) throws {
        //  Check the index
        let totalElements = shape.totalSize
        if (hot < 0 || hot >= totalElements) {
            throw GenericMPSGraphDSLErrors.ClassificationValueOutOfRange
        }

        //  Set the one-hot
        elements = Array(repeating: Float32(0), count: totalElements)
        elements[hot] = Float32(1)
    }

    
// MARK: - Batch functions

    ///  Put a passed in tensor into the batch index provided
    ///
    /// - Parameters:
    ///   - tensor: The sample tensor to be added to this batch tensor
    ///   - batchIndex: The location within the batch that the sample goes into
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the input tensor type does not match the batch tensor type
    ///   - `MPSGraphRunErrors.SampleDoesntMatchBatchShape` if the input tensor type does not match the batch tensor shape minus the first (batch index) dimension
    public mutating func setBatchSample(tensor: Tensor, batchIndex: Int) throws {
        //  Verify the tensor is of the right type
        if (tensor.type != .float32) { throw GenericMPSGraphDSLErrors.InvalidType }
        
        //  Verify the tensor matches our size, minus the batch index
        let sampleShape = tensor.shape.dimensions
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        var expectedShape = ourShape
        expectedShape.removeFirst()
        if (sampleShape != expectedShape) { throw MPSGraphRunErrors.SampleDoesntMatchBatchShape }
       
        //  Calculate a start index for the sample
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let sampleSize = sampleShape.reduce(1, *)
        let startIndex = batchIndex * sampleSize
        
        //  Move the data in
        let tensorFloat32 = tensor as! TensorFloat32
        for i in 0..<sampleSize {
            elements[startIndex + i] = tensorFloat32.elements[i]
        }
    }
    
    ///  Retrieve the element values for a specified batch.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the elements for the batch, as a Double array`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getValuesForBatch(_ batchIndex: Int) throws -> [Double] {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var values: [Double] = []
        for i in 0..<dataSize {
            values.append(Double(elements[startIndex + i]))
        }
        return values
    }

    ///  Parses the element values s\for a specified batch, taking the argmax to find the classification value.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the index of the highest value for the batch - the classification value`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getClassificationForBatch(_ batchIndex: Int) throws -> Int {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var argmax = 0
        var maxValue = -Float32.greatestFiniteMagnitude
        for i in 0..<dataSize {
            if (elements[startIndex + i]) > maxValue {
                maxValue = elements[startIndex + i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    /// Create a tensor from extracted batch data.
    /// - Parameter batchIndex: The index within the batch (first) dimension to extract the data from
    /// - Returns: A tensor of the size of this tensor minus the batch dimension, filled with the data for the specified batch index
    public func getTensorForBatch(_ batchIndex: Int) throws -> Tensor {
        if (shape.numDimensions < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > shape.dimensions[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let outputShape = shape.shapeWithRemovedBatchDimension()
        let dataSize = outputShape.totalSize
        let startIndex = dataSize * batchIndex
        let batchValues = Array(elements[startIndex..<startIndex+dataSize])
        let resultTensor = try TensorFloat32(shape: outputShape, initialValues: batchValues)
        return resultTensor
    }


    // MARK: - Persistance functions

    /// Get a Data object filled with the bytes from the Tensor
    ///
    ///  - Returns: the Data object with the bytes
    public func getData() -> Data {
        let data: Data
        let elementCount = shape.totalSize
        data = Data(bytes: elements, count: elementCount * MemoryLayout<Float32>.size)
        
        return data
    }
}


///  A Tensor with elements of type Float16
public struct TensorFloat16 : Tensor {
    ///  The shape of the Tensor
    public let shape : TensorShape
    var elements : [Float16]
    
    ///  The data type of the Tensor
    public var type : DataType {
        return .float16
    }

    ///  Construct a constant value tensor of a given type and shape
    public init(shape : TensorShape, initialValue : Float16) {
        self.shape = shape
        let totalElements = shape.totalSize
        
        //  Create the array
        elements = Array(repeating: initialValue, count: totalElements)
    }

    ///  Construct a value tensor of a given type and shape with the given data
    ///
    ///  This function assumes a single dimension input vector, sized to the total size of the TensorShape, mapping to the tensor using row-major format
    public init(shape : TensorShape, initialValues : [Float16]) throws {
        self.shape = shape
        
        if (shape.totalSize > initialValues.count) { throw GenericMPSGraphDSLErrors.NotEnoughValues }
        
        //  Create the array
    
        elements = []
        for value in initialValues {
            elements.append(value)
        }
     }

    ///  Construct a value tensor of a given type and shape with the random data
    public init(shape : TensorShape, randomValueRange: ParameterRange) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        let doubleRange = randomValueRange.min.asDouble...randomValueRange.max.asDouble
        for _ in 0..<elementCount {
            elements.append(Float16(Double.random(in: doubleRange)))
        }
    }

    ///  Construct a value tensor of a given type and shape with normal random data
    public init(shape : TensorShape, mean: Double, standardDeviation: Double) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        for _ in 0..<elementCount {
            elements.append(Float16(TensorFloat32.getGaussianRandom(mean: mean, standardDeviation: standardDeviation)))
        }
    }

    ///  Construct a Tensor from an MPSGraphTensorData object
    public init(fromMPSTensorData tensorData: MPSGraphTensorData) {
        self.shape = TensorShape(fromMPS: tensorData.shape)
        
        let NDArray = tensorData.mpsndarray()
        let totalSize = shape.totalSize
        
            var values : [Float16] = Array(repeating: 0, count: totalSize)
            NDArray.readBytes(&values, strideBytes: nil)
            elements = values
     }
    
    /// Create a MPSGraphTensorData object for the specified graph, using the data in the Tensor
    /// - Parameter forGraph: The ``Graph`` object that the MPSGraphTensorData will be used with
    /// - Returns: The create MPSGraphTensorData object
    public func getMPSGraphTensorData(forGraph: Graph) throws -> MPSGraphTensorData {
        let descriptor = MPSNDArrayDescriptor(dataType: DataType.float16.getMPSDataType(), shape: shape.getMPSShape())
        let NDArray = MPSNDArray(device: forGraph.device, descriptor: descriptor)
        var array = elements
        NDArray.writeBytes(&array, strideBytes: nil)
        return MPSGraphTensorData(NDArray)
    }
    
    // MARK: - Getting elements

    ///  Get a specified storage index element
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as an Float32
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Float16 {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }
    
    ///  Get a specified storage index element
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Double {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }
    
    ///  Get a specified tensor element by location, as a Float32
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of offsets within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a Float32
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Float16 {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }
    
    ///  Get a specified tensor element by location, as a Double
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of offsets within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Double {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }
    
    ///  Get all the tensor elements as an array of the Tensor's type
    ///
    /// - Returns: the elements as an array of the type of the Tensor
    public func getElements() -> [Float16] {
        return elements
    }

    ///  Get all the tensor elements as a double array
    ///
    /// - Returns: the elements as a double array
    public func getElements() -> [Double] {
        var array: [Double] = []
        for element in elements {
            array.append(element.asDouble)
        }
        return array
    }
    
    /// Get the classification value (index of highest element in Tensor)
    /// - Returns: The classification value
    public func getClassification() -> Int {
        let dataSize = shape.totalSize
        var argmax = 0
        var maxValue = -Float16.greatestFiniteMagnitude
        for i in 0..<dataSize {
            if (elements[i] > maxValue) {
                maxValue = elements[i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    // MARK: - Setting elements

    ///  Set a specified index element to the specified Float32 value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the value type does not match the tensor type
    public mutating func setElement(index : Int, value: Float16) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }

    ///  Set a specified index element to the specified Double value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElement(index : Int, value: Double) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = Float16(value)
    }
    
    /// Set a specified location of the Tensor to the specified Float32 value
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to Float32
    public mutating func setElement(location : [Int], value: Float16) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }
    
    /// Set a specified location of the Tensor to the specified Double value
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to Float32
    public mutating func setElement(location : [Int], value: Double) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = Float16(value)
    }

    ///  Store a specified array of elements into the tensor, starting at a given location index
    ///
    /// - Parameters:
    ///   - index: The starting index into the tensor storage.  The count of the values parameter determines the length of the store
    ///   - values: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElements(startIndex :Int, values: [Double]) throws {
        if (startIndex < 0 || (startIndex + values.count) > shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        for i in 0..<values.count {
            elements[startIndex + i] = Float16(values[i])
        }
    }

    ///  Set the tensor values to all zeros, except the hot index, which gets set to 1
       public mutating func setOneHot(hot: Int) throws {
        //  Check the index
        let totalElements = shape.totalSize
        if (hot < 0 || hot >= totalElements) {
            throw GenericMPSGraphDSLErrors.ClassificationValueOutOfRange
        }

        //  Set the one-hot
        elements = Array(repeating: Float16(0), count: totalElements)
        elements[hot] = Float16(1)
    }

    
// MARK: - Batch functions

    ///  Put a passed in tensor into the batch index provided
    ///
    /// - Parameters:
    ///   - tensor: The sample tensor to be added to this batch tensor
    ///   - batchIndex: The location within the batch that the sample goes into
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the input tensor type does not match the batch tensor type
    ///   - `MPSGraphRunErrors.SampleDoesntMatchBatchShape` if the input tensor type does not match the batch tensor shape minus the first (batch index) dimension
    public mutating func setBatchSample(tensor: Tensor, batchIndex: Int) throws {
        //  Verify the tensor is of the right type
        if (tensor.type != .float32) { throw GenericMPSGraphDSLErrors.InvalidType }
        
        //  Verify the tensor matches our size, minus the batch index
        let sampleShape = tensor.shape.dimensions
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        var expectedShape = ourShape
        expectedShape.removeFirst()
        if (sampleShape != expectedShape) { throw MPSGraphRunErrors.SampleDoesntMatchBatchShape }
       
        //  Calculate a start index for the sample
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let sampleSize = sampleShape.reduce(1, *)
        let startIndex = batchIndex * sampleSize
        
        //  Move the data in
        let tensorFloat16 = tensor as! TensorFloat16
        for i in 0..<sampleSize {
            elements[startIndex + i] = tensorFloat16.elements[i]
        }
    }
    
    ///  Retrieve the element values for a specified batch.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the elements for the batch, as a Double array`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getValuesForBatch(_ batchIndex: Int) throws -> [Double] {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var values: [Double] = []
        for i in 0..<dataSize {
            values.append(Double(elements[startIndex + i]))
        }
        return values
    }

    ///  Parses the element values s\for a specified batch, taking the argmax to find the classification value.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the index of the highest value for the batch - the classification value`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getClassificationForBatch(_ batchIndex: Int) throws -> Int {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var argmax = 0
        var maxValue = -Float16.greatestFiniteMagnitude
        for i in 0..<dataSize {
            if (elements[startIndex + i]) > maxValue {
                maxValue = elements[startIndex + i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    /// Create a tensor from extracted batch data.
    /// - Parameter batchIndex: The index within the batch (first) dimension to extract the data from
    /// - Returns: A tensor of the size of this tensor minus the batch dimension, filled with the data for the specified batch index
    public func getTensorForBatch(_ batchIndex: Int) throws -> Tensor {
        if (shape.numDimensions < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > shape.dimensions[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let outputShape = shape.shapeWithRemovedBatchDimension()
        let dataSize = outputShape.totalSize
        let startIndex = dataSize * batchIndex
        let batchValues = Array(elements[startIndex..<startIndex+dataSize])
        let resultTensor = try TensorFloat16(shape: outputShape, initialValues: batchValues)
        return resultTensor
    }


    // MARK: - Persistance functions

    /// Get a Data object filled with the bytes from the Tensor
    ///
    ///  - Returns: the Data object with the bytes
    public func getData() -> Data {
        let data: Data
        let elementCount = shape.totalSize
        data = Data(bytes: elements, count: elementCount * MemoryLayout<Float16>.size)
        
        return data
    }
}

///  A Tensor with elements of type Int32
public struct TensorInt32 : Tensor {
    ///  The shape of the Tensor
    public let shape : TensorShape
    var elements : [Int32]
    
    ///  The data type of the Tensor
    public var type : DataType {
        return .int32
    }

    ///  Construct a constant value tensor of a given type and shape
    public init(shape : TensorShape, initialValue : Int32) {
        self.shape = shape
        let totalElements = shape.totalSize
        
        //  Create the array
        elements = Array(repeating: initialValue, count: totalElements)
    }

    ///  Construct a value tensor of a given type and shape with the given data
    ///
    ///  This function assumes a single dimension input vector, sized to the total size of the TensorShape, mapping to the tensor using row-major format
    public init(shape : TensorShape, initialValues : [Int32]) throws {
        self.shape = shape
        
        if (shape.totalSize > initialValues.count) { throw GenericMPSGraphDSLErrors.NotEnoughValues }
        
        //  Create the array
    
        elements = []
        for value in initialValues {
            elements.append(value)
        }
     }

    ///  Construct a value tensor of a given type and shape with the random data
    ///
    ///  This function assumes a single dimension input vector
    public init(shape : TensorShape, randomValueRange: ParameterRange) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        let doubleRange = randomValueRange.min.asDouble...randomValueRange.max.asDouble
        for _ in 0..<elementCount {
            elements.append(Int32(Double.random(in: doubleRange)))
        }
    }

    ///  Construct a value tensor of a given type and shape with normal random data
    public init(shape : TensorShape, mean: Double, standardDeviation: Double) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        for _ in 0..<elementCount {
            var value = Double.infinity
            while (value < Double(UInt8.min) || value > Double(UInt8.max)) {        //  Make sure distribution doesn't extend past UInt8 range
                value = TensorUInt8.getGaussianRandom(mean: mean, standardDeviation: standardDeviation)
            }
            elements.append(Int32(value))
        }
    }

    ///  Construct a value tensor of a given type and shape with the given data
    ///
    public init(fromMPSTensorData tensorData: MPSGraphTensorData) {
        self.shape = TensorShape(fromMPS: tensorData.shape)
        
        let NDArray = tensorData.mpsndarray()
        let totalSize = shape.totalSize
        
            var values : [Int32] = Array(repeating: 0, count: totalSize)
            NDArray.readBytes(&values, strideBytes: nil)
            elements = values
     }

    /// Create a MPSGraphTensorData object for the specified graph, using the data in the Tensor
    /// - Parameter forGraph: The ``Graph`` object that the MPSGraphTensorData will be used with
    /// - Returns: The create MPSGraphTensorData object
    public func getMPSGraphTensorData(forGraph: Graph) throws -> MPSGraphTensorData {
        let descriptor = MPSNDArrayDescriptor(dataType: DataType.int32.getMPSDataType(), shape: shape.getMPSShape())
        let NDArray = MPSNDArray(device: forGraph.device, descriptor: descriptor)
        var array = elements
        NDArray.writeBytes(&array, strideBytes: nil)
        return MPSGraphTensorData(NDArray)
    }
    
// MARK: - Getting elements

    ///  Get a specified storage index element as a UInt8
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as an UInt8
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Int32 {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }

    ///  Get a specified storage index element as a Double
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as an Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Double {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }

    ///  Get a specified tensor element from a location, as a UInt8
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a UInt8
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Int32 {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }

    ///  Get a specified tensor element from a location, as a Double
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Double {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }
    
    ///  Get all the tensor elements as an array of the Tensor's type
    ///
    /// - Returns: the elements as an array of the type of the Tensor
    public func getElements() -> [Int32] {
        return elements
    }

    ///  Get all the tensor elements as a double array
    ///
    /// - Returns: the elements as a double array
    public func getElements() -> [Double] {
        var array: [Double] = []
        for element in elements {
            array.append(element.asDouble)
        }
        return array
    }
    
    /// Get the classification value (index of highest element in Tensor)
    /// - Returns: The classification value
    public func getClassification() -> Int {
        let dataSize = shape.totalSize
        var argmax = 0
        var maxValue = Int32.min
        for i in 0..<dataSize {
            if (elements[i] >= maxValue) {
                maxValue = elements[i]
                argmax = i
            }
        }
        
        return argmax
    }

    // MARK: - Setting elements


    ///  Set a specified index element to the specified UInt8 value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the value type does not match the tensor type
    public mutating func setElement(index : Int, value: Int32) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }

    ///  Set a specified index element to the specified Double value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElement(index : Int, value: Double) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = Int32(value)
    }
        
    /// Set a specified location of the Tensor to the specified UInt8 value
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to UInt8
    public mutating func setElement(location : [Int], value: Int32) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }
    
    /// Set a specified location of the Tensor to the specified Doublevalue
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to UInt8
    public mutating func setElement(location : [Int], value: Double) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = Int32(value)
    }

    ///  Store a specified array of Double elements into the tensor, starting at a given location index
    ///
    /// - Parameters:
    ///   - index: The starting index into the tensor storage.  The count of the values parameter determines the length of the store
    ///   - values: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElements(startIndex :Int, values: [Double]) throws {
        if (startIndex < 0 || (startIndex + values.count) > shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        for i in 0..<values.count {
            elements[startIndex + i] = Int32(values[i])
        }
    }

    ///  Set the tensor values to all zeros, except the hot index, which gets set to 1
       public mutating func setOneHot(hot: Int) throws {
        //  Check the index
        let totalElements = shape.totalSize
        if (hot < 0 || hot >= totalElements) {
            throw GenericMPSGraphDSLErrors.ClassificationValueOutOfRange
        }

        //  Set the one-hot
        elements = Array(repeating: Int32(0), count: totalElements)
        elements[hot] = Int32(1)
    }
    
// MARK: - Batch functions

    //  Batch building functions
    ///  Put a passed in tensor into the batch index provided
    ///
    /// - Parameters:
    ///   - tensor: The sample tensor to be added to this batch tensor
    ///   - batchIndex: The location within the batch that the sample goes into
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the input tensor type does not match the batch tensor type
    ///   - `MPSGraphRunErrors.SampleDoesntMatchBatchShape` if the input tensor type does not match the batch tensor shape minus the first (batch index) dimension
    public mutating func setBatchSample(tensor: Tensor, batchIndex: Int) throws {
        //  Verify the tensor is of the right type
        if (tensor.type != .float32) { throw GenericMPSGraphDSLErrors.InvalidType }
        
        //  Verify the tensor matches our size, minus the batch index
        let sampleShape = tensor.shape.dimensions
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        var expectedShape = ourShape
        expectedShape.removeFirst()
        if (sampleShape != expectedShape) { throw MPSGraphRunErrors.SampleDoesntMatchBatchShape }
       
        //  Calculate a start index for the sample
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let sampleSize = sampleShape.reduce(1, *)
        let startIndex = batchIndex * sampleSize
        
        //  Move the data in
        let tensorInt32 = tensor as! TensorInt32
        for i in 0..<sampleSize {
            elements[startIndex + i] = tensorInt32.elements[i]
        }
    }
    
    ///  Retrieve the element values for a specified batch.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the elements for the batch, as a Double array`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getValuesForBatch(_ batchIndex: Int) throws -> [Double] {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var values: [Double] = []
        for i in 0..<dataSize {
            values.append(Double(elements[startIndex + i]))
        }
        return values
    }
    
    
    ///  Parses the element values s\for a specified batch, taking the argmax to find the classification value.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the index of the highest value for the batch - the classification value`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getClassificationForBatch(_ batchIndex: Int) throws -> Int {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var argmax = 0
        var maxValue = Int32.min
        for i in 0..<dataSize {
            if (elements[startIndex + i]) > maxValue {
                maxValue = elements[startIndex + i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    /// Create a tensor from extracted batch data.
    /// - Parameter batchIndex: The index within the batch (first) dimension to extract the data from
    /// - Returns: A tensor of the size of this tensor minus the batch dimension, filled with the data for the specified batch index
    public func getTensorForBatch(_ batchIndex: Int) throws -> Tensor {
        if (shape.numDimensions < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > shape.dimensions[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let outputShape = shape.shapeWithRemovedBatchDimension()
        let dataSize = outputShape.totalSize
        let startIndex = dataSize * batchIndex
        let batchValues = Array(elements[startIndex..<startIndex+dataSize])
        let resultTensor = try TensorInt32(shape: outputShape, initialValues: batchValues)
        return resultTensor
    }

    // MARK: - Persistance functions

    /// get a Data object filled with the bytes from the Tensor
    ///
    ///  - Returns: the Data object with the bytes
    public func getData() -> Data {
        let data: Data
        let elementCount = shape.totalSize
        data = Data(bytes: elements, count: elementCount * MemoryLayout<Int32>.size)
        
        return data
    }
}

///  A Tensor with elements of type UInt8
public struct TensorUInt8 : Tensor {
    ///  The shape of the Tensor
    public let shape : TensorShape
    var elements : [UInt8]
    
    ///  The data type of the Tensor
    public var type : DataType {
        return .uInt8
    }

    ///  Construct a constant value tensor of a given type and shape
    public init(shape : TensorShape, initialValue : UInt8) {
        self.shape = shape
        let totalElements = shape.totalSize
        
        //  Create the array
        elements = Array(repeating: initialValue, count: totalElements)
    }

    ///  Construct a value tensor of a given type and shape with the given data
    ///
    ///  This function assumes a single dimension input vector, sized to the total size of the TensorShape, mapping to the tensor using row-major format
    public init(shape : TensorShape, initialValues : [UInt8]) throws {
        self.shape = shape
        
        if (shape.totalSize > initialValues.count) { throw GenericMPSGraphDSLErrors.NotEnoughValues }
        
        //  Create the array
    
        elements = []
        for value in initialValues {
            elements.append(value)
        }
     }

    ///  Construct a value tensor of a given type and shape with the random data
    ///
    ///  This function assumes a single dimension input vector
    public init(shape : TensorShape, randomValueRange: ParameterRange) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        let doubleRange = randomValueRange.min.asDouble...randomValueRange.max.asDouble
        for _ in 0..<elementCount {
            elements.append(UInt8(Double.random(in: doubleRange)))
        }
    }

    ///  Construct a value tensor of a given type and shape with normal random data
    public init(shape : TensorShape, mean: Double, standardDeviation: Double) {
        self.shape = shape
                
        //  Create the array
        let elementCount = shape.totalSize
        elements = []
        for _ in 0..<elementCount {
            var value = Double.infinity
            while (value < Double(UInt8.min) || value > Double(UInt8.max)) {        //  Make sure distribution doesn't extend past UInt8 range
                value = TensorUInt8.getGaussianRandom(mean: mean, standardDeviation: standardDeviation)
            }
            elements.append(UInt8(value))
        }
    }

    ///  Construct a value tensor of a given type and shape with the given data
    ///
    public init(fromMPSTensorData tensorData: MPSGraphTensorData) {
        self.shape = TensorShape(fromMPS: tensorData.shape)
        
        let NDArray = tensorData.mpsndarray()
        let totalSize = shape.totalSize
        
            var values : [UInt8] = Array(repeating: 0, count: totalSize)
            NDArray.readBytes(&values, strideBytes: nil)
            elements = values
     }

    /// Create a MPSGraphTensorData object for the specified graph, using the data in the Tensor
    /// - Parameter forGraph: The ``Graph`` object that the MPSGraphTensorData will be used with
    /// - Returns: The create MPSGraphTensorData object
    public func getMPSGraphTensorData(forGraph: Graph) throws -> MPSGraphTensorData {
        let descriptor = MPSNDArrayDescriptor(dataType: DataType.uInt8.getMPSDataType(), shape: shape.getMPSShape())
        let NDArray = MPSNDArray(device: forGraph.device, descriptor: descriptor)
        var array = elements
        NDArray.writeBytes(&array, strideBytes: nil)
        return MPSGraphTensorData(NDArray)
    }
    
// MARK: - Getting elements

    ///  Get a specified storage index element as a UInt8
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as an UInt8
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> UInt8 {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }

    ///  Get a specified storage index element as a Double
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///
    /// - Returns: the element at the specified storage index, as an Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public func getElement(index : Int) throws -> Double {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }

    ///  Get a specified tensor element from a location, as a UInt8
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a UInt8
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> UInt8 {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index]
    }

    ///  Get a specified tensor element from a location, as a Double
    ///
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///
    /// - Returns: the element at the specified location, as a Double
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidShape` if the location does not match dimension size to the tensor shape, or any location is outside the shape dimensions
    public func getElement(location: [Int]) throws -> Double {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return elements[index].asDouble
    }
    
    ///  Get all the tensor elements as an array of the Tensor's type
    ///
    /// - Returns: the elements as an array of the type of the Tensor
    public func getElements() -> [UInt8] {
        return elements
    }

    ///  Get all the tensor elements as a double array
    ///
    /// - Returns: the elements as a double array
    public func getElements() -> [Double] {
        var array: [Double] = []
        for element in elements {
            array.append(element.asDouble)
        }
        return array
    }
    
    /// Get the classification value (index of highest element in Tensor)
    /// - Returns: The classification value
    public func getClassification() -> Int {
        let dataSize = shape.totalSize
        var argmax = 0
        var maxValue = UInt8.min
        for i in 0..<dataSize {
            if (elements[i] >= maxValue) {
                maxValue = elements[i]
                argmax = i
            }
        }
        
        return argmax
    }

    // MARK: - Setting elements


    ///  Set a specified index element to the specified UInt8 value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the value type does not match the tensor type
    public mutating func setElement(index : Int, value: UInt8) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }

    ///  Set a specified index element to the specified Double value
    ///
    /// - Parameters:
    ///   - index: The index into the tensor storage
    ///   - value: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElement(index : Int, value: Double) throws {
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = UInt8(value)
    }
        
    /// Set a specified location of the Tensor to the specified UInt8 value
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to UInt8
    public mutating func setElement(location : [Int], value: UInt8) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = value
    }
    
    /// Set a specified location of the Tensor to the specified Doublevalue
    /// - Parameters:
    ///   - location: The location in the tensor, given as an array of locatons within each dimension of the tensor
    ///   - value: value to store.  It will be cast to UInt8
    public mutating func setElement(location : [Int], value: Double) throws {
        //  Get the storage index from the location
        let index = try shape.storageIndex(location: location)
        if (index < 0 || index >= shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        elements[index] = UInt8(value)
    }

    ///  Store a specified array of Double elements into the tensor, starting at a given location index
    ///
    /// - Parameters:
    ///   - index: The starting index into the tensor storage.  The count of the values parameter determines the length of the store
    ///   - values: The new value to be stored at the the specified index
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside if the  tensor storage array size
    public mutating func setElements(startIndex :Int, values: [Double]) throws {
        if (startIndex < 0 || (startIndex + values.count) > shape.totalSize) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        for i in 0..<values.count {
            elements[startIndex + i] = UInt8(values[i])
        }
    }

    ///  Set the tensor values to all zeros, except the hot index, which gets set to 1
       public mutating func setOneHot(hot: Int) throws {
        //  Check the index
        let totalElements = shape.totalSize
        if (hot < 0 || hot >= totalElements) {
            throw GenericMPSGraphDSLErrors.ClassificationValueOutOfRange
        }

        //  Set the one-hot
        elements = Array(repeating: UInt8(0), count: totalElements)
        elements[hot] = UInt8(1)
    }
    
// MARK: - Batch functions

    //  Batch building functions
    ///  Put a passed in tensor into the batch index provided
    ///
    /// - Parameters:
    ///   - tensor: The sample tensor to be added to this batch tensor
    ///   - batchIndex: The location within the batch that the sample goes into
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    ///   - `GenericMPSGraphDSLErrors.InvalidType` if the input tensor type does not match the batch tensor type
    ///   - `MPSGraphRunErrors.SampleDoesntMatchBatchShape` if the input tensor type does not match the batch tensor shape minus the first (batch index) dimension
    public mutating func setBatchSample(tensor: Tensor, batchIndex: Int) throws {
        //  Verify the tensor is of the right type
        if (tensor.type != .float32) { throw GenericMPSGraphDSLErrors.InvalidType }
        
        //  Verify the tensor matches our size, minus the batch index
        let sampleShape = tensor.shape.dimensions
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        var expectedShape = ourShape
        expectedShape.removeFirst()
        if (sampleShape != expectedShape) { throw MPSGraphRunErrors.SampleDoesntMatchBatchShape }
       
        //  Calculate a start index for the sample
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let sampleSize = sampleShape.reduce(1, *)
        let startIndex = batchIndex * sampleSize
        
        //  Move the data in
        let tensorUInt8 = tensor as! TensorUInt8
        for i in 0..<sampleSize {
            elements[startIndex + i] = tensorUInt8.elements[i]
        }
    }
    
    ///  Retrieve the element values for a specified batch.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the elements for the batch, as a Double array`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getValuesForBatch(_ batchIndex: Int) throws -> [Double] {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var values: [Double] = []
        for i in 0..<dataSize {
            values.append(Double(elements[startIndex + i]))
        }
        return values
    }
    
    
    ///  Parses the element values s\for a specified batch, taking the argmax to find the classification value.  Assumes the tensor is a batch tensor (batch index as first dimension)
    ///
    /// - Parameters:
    ///   - batchIndex: The location within the batch that the values come frin
    ///
    /// - Returns: the index of the highest value for the batch - the classification value`
    ///
    /// - Throws:
    ///   - `MPSGraphRunErrors.NotABatchTensor` if this tensor does not have enough dimensions to be a batch tensor (minimum of 2)
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the batch index is outside if the  tensor's first dimension (batch size)
    public func getClassificationForBatch(_ batchIndex: Int) throws -> Int {
        let ourShape = shape.dimensions
        if (ourShape.count < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > ourShape[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        var outputShape = ourShape
        outputShape.removeFirst()
        let dataSize = outputShape.reduce(1, *)
        let startIndex = dataSize * batchIndex
        var argmax = 0
        var maxValue = UInt8.min
        for i in 0..<dataSize {
            if (elements[startIndex + i]) > maxValue {
                maxValue = elements[startIndex + i]
                argmax = i
            }
        }
        
        return argmax
    }
    
    /// Create a tensor from extracted batch data.
    /// - Parameter batchIndex: The index within the batch (first) dimension to extract the data from
    /// - Returns: A tensor of the size of this tensor minus the batch dimension, filled with the data for the specified batch index
    public func getTensorForBatch(_ batchIndex: Int) throws -> Tensor {
        if (shape.numDimensions < 2) { throw MPSGraphRunErrors.NotABatchTensor }
        if (batchIndex < 0 || batchIndex > shape.dimensions[0]) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        let outputShape = shape.shapeWithRemovedBatchDimension()
        let dataSize = outputShape.totalSize
        let startIndex = dataSize * batchIndex
        let batchValues = Array(elements[startIndex..<startIndex+dataSize])
        let resultTensor = try TensorUInt8(shape: outputShape, initialValues: batchValues)
        return resultTensor
    }

    // MARK: - Persistance functions

    /// get a Data object filled with the bytes from the Tensor
    ///
    ///  - Returns: the Data object with the bytes
    public func getData() -> Data {
        let data: Data
        let elementCount = shape.totalSize
        data = Data(bytes: elements, count: elementCount * MemoryLayout<UInt8>.size)
        
        return data
    }
}
