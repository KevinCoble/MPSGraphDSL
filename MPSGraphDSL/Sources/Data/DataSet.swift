//
//  DataSet.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/17/25.
//

import Foundation

///  Struct for single data sample
///
///  Must be part of ``DataSet`` class to get it's data type and shape
public struct DataSample
{
    ///  The input Tensor
    public var inputs : Tensor
    ///  The output Tensor
    public var outputs : Tensor
    ///  The output classification index
    public var outputClass: Int
    
    /// Initializer to create a DataSample
    /// - Parameters:
    ///   - inputs: the tensor that becomes the input Tensor for the sample.  If the sample is to be added to a DataSet, it must match the type and shape for the DataSet
    ///   - outputs: the tensor that becomes the output Tensor for the sample.  If the sample is to be added to a DataSet, it must match the type and shape for the DataSet
    ///   - classIndex: the classification index stored into the sample
    public init(inputs: Tensor, outputs: Tensor, classIndex : Int = 0)
    {
        self.inputs = inputs
        self.outputs = outputs
        outputClass = classIndex
    }
}



///  Class for a set of training or testing data
open class DataSet
{
    let inputShape : TensorShape
    let inputType : DataType
    let outputShape : TensorShape
    let outputType : DataType
    var samples : [DataSample] = []
    var labels : [String]? = nil
    
    /// Constructor for creating with known data parameters
    /// 
    /// - Parameters:
    ///   - inputShape: The shape of the input tensor for each data sample
    ///   - inputType: The data type for the input tensor elementes
    ///   - outputShape: The shape of the output tensor for each data sample
    ///   - outputType: The data type for the output tensor elementes

    public init(inputShape : TensorShape, inputType : DataType, outputShape : TensorShape, outputType : DataType)
    {
        self.inputShape = inputShape
        self.inputType = inputType
        self.outputShape = outputShape
        self.outputType = outputType
    }
    
    // MARK: Value Conversion
    
    /// Convert a DataElement to the type needed by the input tensor
    /// - Parameter value: the DataElement to be converted
    /// - Returns: a new DataElement of the input tensor data type, with the passed in value converted
    public func convertValueToInputType(_ value : DataElement) -> DataElement {
        if (inputType == value.dataType) { return value }
        switch (inputType) {
        case .uInt8:
            return value.asUnsignedByte(range: nil)
        case .float32:
            return value.asFloat32
        case .double:
            return value.asDouble
        }
    }
    
    /// Convert a DataElement to the type needed by the output tensor
    /// - Parameter value: the DataElement to be converted
    /// - Returns: a new DataElement of the output tensor data type, with the passed in value converted
    public func convertValueToOutputType(_ value : DataElement) -> DataElement {
        if (outputType == value.dataType) { return value }
        switch (outputType) {
        case .uInt8:
            return value.asUnsignedByte(range: nil)
        case .float32:
            return value.asFloat32
        case .double:
            return value.asDouble
        }
    }
    
    /// Convert an array DataElements to the type needed by the input tensor
    /// - Parameter array: the array of DataElement to be converted
    /// - Returns: a new array of DataElement of the input tensor data type, with the passed in values converted
    public func convertArrayToInputType(_ array : [DataElement]) -> [DataElement] {
        if (array.isEmpty) { return array }
        if (inputType == array[0].dataType) { return array }
        var newArray: [DataElement] = []
        switch (inputType) {
        case .uInt8:
            for element in array {
                newArray.append(element.asUnsignedByte(range: nil))
            }
        case .float32:
            for element in array {
                newArray.append(element.asFloat32)
            }
        case .double:
            for element in array {
                newArray.append(element.asDouble)
            }
        }
        return newArray
    }
    
    /// Convert an array DataElements to the type needed by the output tensor
    /// - Parameter array: the array of DataElement to be converted
    /// - Returns: a new array of DataElement of the output tensor data type, with the passed in values converted
    public func convertArrayToOutputType(_ array : [DataElement]) -> [DataElement] {
        if (array.isEmpty) { return array }
        if (outputType == array[0].dataType) { return array }
        var newArray: [DataElement] = []
        switch (outputType) {
        case .uInt8:
            for element in array {
                newArray.append(element.asUnsignedByte(range: nil))
            }
        case .float32:
            for element in array {
                newArray.append(element.asFloat32)
            }
        case .double:
            for element in array {
                newArray.append(element.asDouble)
            }
        }
        return newArray
    }

    
    // MARK: Properties
    ///  Get the number of samples in the set
    /// - Returns: the current number of samples for the data set
    open var numSamples : Int {
        get { return samples.count }
    }

    
    func getLabelIndex(label: String) -> Int {
        //  Try to find it in existing labels
        var labelIndex = -1
        if let labels = labels {
            for index in 0..<labels.count {
                if(labels[index].caseInsensitiveCompare(label) == .orderedSame) {
                    labelIndex = index
                    break
                }
            }
        }
        
        //  If not found, see if we should add the label
        if (labelIndex < 0) {
            //  Add the label
            if (labels == nil) { labels = [] }
            labels!.append(label)
            labelIndex = labels!.count - 1
        }

        return labelIndex
    }
    
    /// Get the label for the passed in label index
    /// - Parameter labelIndex: the index into the classification label list
    /// - Returns: the classification label string, or nil if no classification data is present or the index is out of range
    public func getLabel(labelIndex: Int) -> String? {
        guard let labels else { return nil }
        guard labelIndex >= 0, labelIndex < labels.count else { return nil }
        return labels[labelIndex]
    }
    
    
    // MARK: - Parsing

    func inputLocationInRange(parsingData: ParsingData) -> Bool {
        for i in 0..<inputShape.numDimensions {
            if (parsingData.currentInputLocation[i] >= inputShape.dimensions[i]) {
                return false
            }
        }
        return true
    }
    func outputLocationInRange(parsingData: ParsingData) -> Bool {
        for i in 0..<outputShape.numDimensions {
            if (parsingData.currentOutputLocation[i] >= outputShape.dimensions[i]) {
                return false
            }
        }
        return true
    }

    //  Get input storage index for parsing
    func getInputSampleStorageIndex(parsingData: ParsingData) -> Int {
        var index = 0
        for dimension in 0..<inputShape.numDimensions {
            index += parsingData.currentInputLocation[dimension] * parsingData.inputLocationOffsets[dimension]
        }

        return index
    }
    
    //  Get output storage index for parsing
    func getOutputSampleStorageIndex(parsingData: ParsingData) -> Int {
        var index = 0
        for dimension in 0..<outputShape.numDimensions {
            index += parsingData.currentOutputLocation[dimension] * parsingData.outputLocationOffsets[dimension]
        }

        return index
    }
    
    func incrementSample(parsingData: ParsingData)
    {
        parsingData.currentSample += 1
        
        //  If needed, create a new sample
        if (parsingData.currentSample >= samples.count) { addEmptySample(parsingData: parsingData) }
    }
    
    //  Add an empty sample for data parsing
    func addEmptySample(parsingData: ParsingData)
    {
        let inputTensor = CreateTensor.constantValues(type: inputType, shape: inputShape, initialValue: 0)
        let outputTensor = CreateTensor.constantValues(type: outputType, shape: outputShape, initialValue: 0)
        let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
        samples.append(sample)
        
        //  Store location starts at the beginning for the sample
        parsingData.currentInputLocation = [Int](repeating: 0, count: inputShape.numDimensions)
        parsingData.currentOutputLocation = [Int](repeating: 0, count: outputShape.numDimensions)
    }

    func appendOutputClass(_ sampleClass : Int, parsingData: ParsingData) throws {
        //  Validate the class fits with the output size
        if (sampleClass < 0 || ((sampleClass != 1 || parsingData.outputSize != 1) && sampleClass >= parsingData.outputSize)) {
            throw GenericMPSGraphDSLErrors.ClassificationValueOutOfRange
        }
        //  Set the class and output data
        samples[parsingData.currentSample].outputClass = sampleClass
        try samples[parsingData.currentSample].outputs.setOneHot(hot: sampleClass)
    }
    
    func appendInputData(_ inputArray : [Double], normalizationIndex : Int?, parsingData: ParsingData) throws
    {
        //  Store the values
        let startIndex = getInputSampleStorageIndex(parsingData : parsingData)
        
        //  Verify locations and calculate the normalization map
        var index = startIndex
        let lastDimension = inputShape.dimensions.count - 1
        for _ in inputArray {
            //  Verify we are in range
            if (!inputLocationInRange(parsingData: parsingData)) { throw DataParsingErrors.InputLocationOutOfRange }
            
            //  Calculate the normalization map
            if (parsingData.currentSample == 0 && parsingData.inputNormalizationMap[index] == 0) {
                if let normIndex = normalizationIndex {
                    if (normIndex > 0) {
                        parsingData.inputNormalizationMap[index] = normIndex
                    }
                    else {
                        for i in 0..<parsingData.inputSize { parsingData.inputNormalizationMap[i] = normIndex }
                    }
                }
            }
            //  Increment the location
            parsingData.currentInputLocation[lastDimension] += 1
            index += 1
        }
        
        //  Store the values
        try samples[parsingData.currentSample].inputs.setElements(startIndex: startIndex, values: inputArray)
    }
    
    func appendColorData(_ inputArray : [Double], channel: ColorChannel, normalizationIndex : Int?, parsingData: ParsingData) throws
    {
        for newValue in inputArray {
            //  Set the color as the third dimension (X-Y pixel grid)
            parsingData.currentInputLocation[2] = channel.rawValue
            if (parsingData.currentInputLocation[2] >= inputShape.dimensions[2]) {
                throw DataParsingErrors.ColorChannelOutOfRange
            }
            //  Verify we are in range
            if (!inputLocationInRange(parsingData: parsingData)) { throw DataParsingErrors.InputLocationOutOfRange }
            //  Store the value
            let index = getInputSampleStorageIndex(parsingData: parsingData)
            try samples[parsingData.currentSample].inputs.setElement(index: index, value: newValue)
            if (parsingData.currentSample == 0 && parsingData.inputNormalizationMap[index] == 0) {
                if let normIndex = normalizationIndex {
                    if (normIndex > 0) {
                        parsingData.inputNormalizationMap[index] = normIndex
                    }
                    else {
                        for i in 0..<parsingData.inputSize { parsingData.inputNormalizationMap[i] = normIndex }
                    }
                }
            }
            //  Increment the location
            parsingData.currentInputLocation[0] += 1
        }
    }
    
    func appendOutputData(_ outputArray : [Double], normalizationIndex : Int?, parsingData: ParsingData) throws
    {
        //  Store the values
        let startIndex = getOutputSampleStorageIndex(parsingData : parsingData)
        
        //  Verify locations and calculate the normalization map
        var index = startIndex
        for _ in outputArray {
            //  Verify we are in range
            if (!outputLocationInRange(parsingData: parsingData)) { throw DataParsingErrors.OutputLocationOutOfRange }
            
            //  Calculate the normalization map
            if (parsingData.currentSample == 0 && parsingData.outputNormalizationMap[index] == 0) {
                if let normIndex = normalizationIndex {
                    if (normIndex > 0) {
                        parsingData.outputNormalizationMap[index] = normIndex
                    }
                    else {
                        for i in 0..<parsingData.outputSize { parsingData.outputNormalizationMap[i] = normIndex }
                    }
                }
            }
            
            //  Increment the location
            parsingData.currentOutputLocation[0] += 1
            index += 1
        }
        
        //  Store the values
        try samples[parsingData.currentSample].outputs.setElements(startIndex: startIndex, values: outputArray)
    }
    
    func appendOutputLabel(_ label : String, normalizationIndex : Int?, parsingData: ParsingData) throws
    {
        //  Get the label index
        let labelIndex = getLabelIndex(label: label)
        if (labelIndex >= outputShape.totalSize) { throw DataParsingErrors.MoreUniqueLabelsThanOutputDimension }
        
        //  Store the values
        try appendOutputClass(labelIndex, parsingData: parsingData)
    }

    // MARK: - Getting samples
    
    /// Get the specified sample from the supplied index
    /// - Parameter sampleIndex: the index into the sample array to retrieve
    /// - Returns: the DataSample at the specified index
    /// - Throws: `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside of the sample count range
    public func getSample(sampleIndex: Int) throws -> DataSample {
        if (sampleIndex < 0 || sampleIndex >= samples.count) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        return samples[sampleIndex]
    }
    
    ///  Get a pair of tensors (input and output) that corrospond to a batch input.
    ///     The tensor sizes are increased by a dimension, with the batch size as the first dimension.  i.e. - a \[5,5\] data  tensor results in a \[batchSize, 5, 5\] resulting tensor
    ///
    /// - Parameters:
    ///   - sampleIndices: An array of sample indices that will make up the batch.  The count of this array becomes the batch size
    ///
    /// - Returns: a tuple containing the input and output tensor
    ///
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if a sample index is out of range
    public func getBatch(sampleIndices: [Int]) throws -> (inputTensor: Tensor, outputTensor: Tensor) {
        let batchSize = sampleIndices.count
        
        //  Create the input tensor
        var inputShapeArray = inputShape.dimensions
        inputShapeArray.insert(batchSize, at: 0)
        let inputShape = TensorShape(inputShapeArray)
        var inputTensor = CreateTensor.constantValues(type: inputType, shape: inputShape, initialValue: 0.0)
        
        //  Create the output tensor
        var outputShapeArray = outputShape.dimensions
        outputShapeArray.insert(batchSize, at: 0)
        let outputShape = TensorShape(outputShapeArray)
        var outputTensor = CreateTensor.constantValues(type: outputType, shape: outputShape, initialValue: 0.0)
        
        //  Concatenate the data for the tensors
        for i in 0..<batchSize {
            let sampleIndex = sampleIndices[i]
            if (sampleIndex < 0 || sampleIndex >= samples.count) { throw GenericMPSGraphDSLErrors.InvalidIndex }
            try inputTensor.setBatchSample(tensor: samples[sampleIndex].inputs, batchIndex: i)
            try outputTensor.setBatchSample(tensor: samples[sampleIndex].outputs, batchIndex: i)
        }
        
        //  Return the tensors
        return (inputTensor: inputTensor, outputTensor: outputTensor)
    }
    
    // MARK: - Setting samples
    
    /// Add a sample to the DataSet
    /// - Parameter sample: the sample to be added
    /// - Throws:
    ///   - `DataSetErrors.SampleInputShapeMismatch` if a sample input tensor does not match the shape for the DataSet
    ///   - `DataSetErrors.SampleOutputShapeMismatch` if a sample output tensor does not match the shape for the DataSet
    ///   - `DataSetErrors.SampleInputTypeMismatch` if a sample input tensor does not match the type for the DataSet
    ///   - `DataSetErrors.SampleOutputTypeMismatch` if a sample output tensor does not match the type for the DataSet
    public func appendSample(_ sample: DataSample) throws {
        //  Verify the sample matches
        if (sample.inputs.shape != inputShape) { throw DataSetErrors.SampleInputShapeMismatch }
        if (sample.outputs.shape != outputShape) { throw DataSetErrors.SampleOutputShapeMismatch }
        if (sample.inputs.type != inputType) { throw DataSetErrors.SampleInputTypeMismatch }
        if (sample.outputs.type != outputType) { throw DataSetErrors.SampleOutputTypeMismatch }
        
        //  Everything good, add it to the set
        samples.append(sample)
    }
    
    
    // MARK: - Splitting the set
    
    /// Split the DataSet into two DataSets, with a specified number of samples randomly selected for the second DataSet and the remainder of samples put into the first one
    /// - Parameter secondSetCount: The number of samples to put into the second DataSet
    /// - Returns: a tuple (set1: DataSet, set2: DataSet), with the two new DataSets
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidValue` if the number of samples for the second DataSet is out of range
    public func splitSetRandomly(secondSetCount: Int) throws -> (set1: DataSet, set2: DataSet) {
        //  Check that we have enough samples
        if (secondSetCount >= numSamples) { throw GenericMPSGraphDSLErrors.InvalidValue }
        
        //  Create the new DataSets
        let set1 = DataSet(inputShape: inputShape, inputType: inputType, outputShape: outputShape, outputType: outputType)
        let set2 = DataSet(inputShape: inputShape, inputType: inputType, outputShape: outputShape, outputType: outputType)
        
        //  Get the random sample list
        let sampleIndices = Array(0..<numSamples).shuffled()
        
        for i in 0..<numSamples {
            let sampleIndex = sampleIndices[i]
            let sample = samples[sampleIndex]
            
            if (i < secondSetCount) {
                try set2.appendSample(sample)
            }
            else {
                try set1.appendSample(sample)
            }
        }
        
        //  Copy any labels to the sets
        set1.labels = labels
        set2.labels = labels
        
        return (set1: set1, set2: set2)
    }
}
