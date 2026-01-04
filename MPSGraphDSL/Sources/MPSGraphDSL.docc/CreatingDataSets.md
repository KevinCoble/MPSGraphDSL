# Creating and Using DataSets

This article describes some of the methods used to create DataSets, and some of the common usage patterns

## Overview

A DataSet consists of an array of DataSamples, along with any output classification labels associated with it.  Each ``DataSample`` has an input and an output ``Tensor``, with the type and shape of each controlled by the DataSet, and an optional classification index.

### Creating a DataSet

To create a ``DataSet`` you use the single initializer for the class, which requires you to specify the type and shape of both the input and the output tensors.  The following code creates a ``DataSet`` with an input Tensor of type Float32 and a shape of a 4x3 matrix and an output Tensor also of type Float32, but a shape of a vector with 10 elements:

```swift
let inputShape = TensorShape([4, 3])
let outputShape = TensorShape([10])

let dataSet = DataSet(inputShape: inputShape, inputType: .float32,
                      outputShape: outputShape, outputType: .float32)
```

The ``DataSet`` is created with no initial samples.

### Adding Samples to a DataSet

You can create and add samples manually from whatever data source you may have, or you can use one of the data parsers that are part of the MPSGraphDSL library.

To manually add a sample, you first create the input and output tensors with the required data.  They **must** match that of the DataSet the sample will be going into, or errors will be thrown.  Then, if it is a known classification sample (a training or testing sample, rather than one with unknown results), get the classification index (there is a method on the Tensor class to do that from the data if needed).  Once you have the tensors and the classification index, make a ``DataSample`` using the single initializer and add it to the ``DataSet``

```swift
let inputTensor = TensorFloat32(...)
let outputTensor = TensorFloat32(...)
let classificationIndex = outputTensor.getClassification()

let sample = DataSample(inputs: inputTensor, outputs: outputTensor, classIndex : classificationIndex)

dataSet.appendSample(sample)
```

The classification index is an optional parameter of the ``DataSample`` initializer, and does not need to be passed for unknown inference samples.  Classification inference on a ``Graph``  will set the classification index along with the graph's' output.

To learn how have a data parser add samples to a DataSet, see the article on Data Parsers


### Splitting a DataSet

When training a neural network, you need both a training and a testing data set.  The training is used to update the parameters of the network, while the testing allows you to determine how the network is doing as training progresses.  The testing data should **never** be used for training.

Some data sets come in two pieces, a training and testings set.  The MNIST and CIFAR data sets are examples of these.  Some only come as one big set, and it is your responsibility to split out some of the data for testing.  The ``DataSet`` class has a function to make this easy.  You just need to specify how many of the samples go into a second set.  The following code shows this in action

```swift
let numTestSamples = Int(testingPercentage * dataSet.numSamples)
let splitSets = try dataSet.splitSetRandomly(secondSetCount: numTestSamples)
let trainingDataSet = splitSets.set1
let testDataSet = splitSets.set2
```

