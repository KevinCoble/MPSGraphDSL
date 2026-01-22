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
public struct DataSample : Sendable
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



///  Actor for a set of training or testing data
public actor DataSet
{
    let inputShape : TensorShape
    let inputType : DataType
    let outputShape : TensorShape
    let outputType : DataType
    var samples : [DataSample] = []
    var labels : [String]? = nil
    var inUseByGraph : Bool = false
    
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
    
    ///  Locking for Graph runs (or other multi-sample uses)
   public func lockForMultiSampleUse() throws {
       if (inUseByGraph) {
           throw DataSetErrors.InUseByGraph
       }
       inUseByGraph = true
   }
    
    ///  Locking for Graph runs (or other multi-sample uses)
   public func releaseLock() throws {
       if (!inUseByGraph) {
           throw DataSetErrors.NotLocked
       }
       inUseByGraph = false
   }
    
    ///  Create an output tensor based on the classification index provided
    /// - Parameter classification: the classification index
    /// - Returns: a tensor of the required type and shape, set to a one-hot value of the classification
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.ClassificationValueOutOfRange` if the classification index is outside the size of the output tensor
    public func getOutputTensorForClassification(_ classification: Int) throws -> Tensor {
        var tensor = CreateTensor.constantValues(type: outputType, shape: outputShape, initialValue: 0.0)
        try tensor.setOneHot(hot: classification)
        return tensor
    }

    // MARK: Properties
    ///  Get the number of samples in the set
    /// - Returns: the current number of samples for the data set
    public var numSamples : Int {
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
    
    
    /// Set the labels for the data set
    /// - Parameter labels: the new label list
    public func setLabels(_ labels: [String]?) {
        self.labels = labels
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
    
    /// Function to create an empty DataSample with all zeroes
    /// - Returns: DataSample of appropriate size and type tensors with all values set to zero
    public func getEmptySample() throws -> DataSample
    {
        if (inUseByGraph) {
            throw DataSetErrors.InUseByGraph
        }

        let inputs = CreateTensor.constantValues(type: inputType, shape: inputShape, initialValue: 0)
        let outputs = CreateTensor.constantValues(type: outputType, shape: outputShape, initialValue: 0)
        let sample = DataSample(inputs: inputs, outputs: outputs)
        return sample
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
    
    //  If incremented index in range return the index, else append empty and return that index
    internal func incrementIndexAppendEmptySample(oldIndex: Int) throws -> Int {
        if (inUseByGraph) {
            throw DataSetErrors.InUseByGraph
        }

        //  Increment the index
        let proposedIndex = oldIndex + 1
        if (proposedIndex < samples.count) { return proposedIndex }
        
        //  Create and add an empty sample
        try samples.append(getEmptySample())
        return samples.count - 1
    }

    
    /// Add an empty sample to the DataSet
    /// - Returns: index that sample was added at
    public func appendEmptySample() throws -> Int {
        if (inUseByGraph) {
            throw DataSetErrors.InUseByGraph
        }

        //  Create and add an empty sample
        try samples.append(getEmptySample())
        return samples.count - 1
    }

    /// Add a sample to the DataSet
    /// - Parameter sample: the sample to be added
    /// - Throws:
    ///   - `DataSetErrors.SampleInputShapeMismatch` if a sample input tensor does not match the shape for the DataSet
    ///   - `DataSetErrors.SampleOutputShapeMismatch` if a sample output tensor does not match the shape for the DataSet
    ///   - `DataSetErrors.SampleInputTypeMismatch` if a sample input tensor does not match the type for the DataSet
    ///   - `DataSetErrors.SampleOutputTypeMismatch` if a sample output tensor does not match the type for the DataSet
    ///   - `DataSetErrors.InUseByGraph` if DataSet is in use
    public func appendSample(_ sample: DataSample) throws {
        if (inUseByGraph) {
            throw DataSetErrors.InUseByGraph
        }

        //  Verify the sample matches
        if (sample.inputs.shape != inputShape) { throw DataSetErrors.SampleInputShapeMismatch }
        if (sample.outputs.shape != outputShape) { throw DataSetErrors.SampleOutputShapeMismatch }
        if (sample.inputs.type != inputType) { throw DataSetErrors.SampleInputTypeMismatch }
        if (sample.outputs.type != outputType) { throw DataSetErrors.SampleOutputTypeMismatch }
        
        //  Everything good, add it to the set
        samples.append(sample)
    }
    
    /// Set a sample in the DataSet at a specified index
    /// - Parameters:
    ///   - sample: the sample to be added
    ///   - sampleIndex: the index to store the sample
    /// - Throws:
    ///   - `DataSetErrors.SampleInputShapeMismatch` if a sample input tensor does not match the shape for the DataSet
    ///   - `DataSetErrors.SampleOutputShapeMismatch` if a sample output tensor does not match the shape for the DataSet
    ///   - `DataSetErrors.SampleInputTypeMismatch` if a sample input tensor does not match the type for the DataSet
    ///   - `DataSetErrors.SampleOutputTypeMismatch` if a sample output tensor does not match the type for the DataSet
    ///   - `GenericMPSGraphDSLErrors.InvalidIndex` if the index is outside of the sample count range
    ///   - `DataSetErrors.InUseByGraph` if DataSet is in use
    public func setSample(_ sample: DataSample, sampleIndex: Int) throws {
        if (inUseByGraph) {
            throw DataSetErrors.InUseByGraph
        }

        //  Verify the sample matches
        if (sample.inputs.shape != inputShape) { throw DataSetErrors.SampleInputShapeMismatch }
        if (sample.outputs.shape != outputShape) { throw DataSetErrors.SampleOutputShapeMismatch }
        if (sample.inputs.type != inputType) { throw DataSetErrors.SampleInputTypeMismatch }
        if (sample.outputs.type != outputType) { throw DataSetErrors.SampleOutputTypeMismatch }
        if (sampleIndex < 0 || sampleIndex >= samples.count) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        
        //  Everything good, put it in the set
        samples[sampleIndex] = sample
    }
    
    func removeFinalSample() throws {
        if (inUseByGraph) {
            throw DataSetErrors.InUseByGraph
        }

        if (samples.count < 1) { throw GenericMPSGraphDSLErrors.InvalidIndex }
        samples.removeLast()
    }

    
    // MARK: - Splitting the set
    
    /// Split the DataSet into two DataSets, with a specified number of samples randomly selected for the second DataSet and the remainder of samples put into the first one
    /// - Parameter secondSetCount: The number of samples to put into the second DataSet
    /// - Returns: a tuple (set1: DataSet, set2: DataSet), with the two new DataSets
    /// - Throws:
    ///   - `GenericMPSGraphDSLErrors.InvalidValue` if the number of samples for the second DataSet is out of range
    public func splitSetRandomly(secondSetCount: Int) async throws -> (set1: DataSet, set2: DataSet) {
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
                try await set2.appendSample(sample)
            }
            else {
                try await set1.appendSample(sample)
            }
        }
        
        //  Copy any labels to the sets
        await set1.setLabels(labels)
        await set2.setLabels(labels)
        
        return (set1: set1, set2: set2)
    }
}
