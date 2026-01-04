# Creating Data Parsers

This article describes how to create and use data parsers to extract the information from a data file or other source into a ``DataSet``

## Overview

There are three types of data parsers available in the MPSGraphDSL library:  Binary, Fixed Column Text, and Delineated Text parsers.  We will start with Delineated Text parsers

## Delineated Text Parsers

Delineated Text Parsers are used to parse text files where each line is a sample, and the data pieces are seperated with known delineators, such as white-space or commas.  CSV files (comma separated value) fall into this category.

###  A Simple Example

Let's start with a very simple example - a regression problem that takes two floating-point inputs and results in a single value for output

```swift
#  Extremely simple regression data set - what can you do?
1.0    2.0    5.0
4.02 -1.6  5.64
2 4 10
...
```

As you can see, the data is not evenly spaced out, so a Fixed Column Text parser would not work.  The data is separated by spaces, although a variable number.  This works fine for a Delineated Text Parser - the delineation is the whitespace.  A parser for this data could look like the following:

```swift
let parser = DelineatedTextParser(lineSeparator: .SpaceDelimited) {
    InputFloatString()
    InputFloatString()
    OutputFloatString()
}.withCommentIndicators(["#", "//"])
```

Let's unpack what is in that code, line by line:

**let parser = DelineatedTextParser(lineSeparator: .SpaceDelimited) {**

This line creates a Delineated Text Parser that uses white-space for delineation, and assigns it to the variable 'parser'.  Currently only two delineation styles are supported, .SpaceDelimited and .CommaSeparated

**InputFloatString()**

This line says the first piece of delineated text should be read as a floating point number, and that it goes into the input tensor of the current sample.  Since no dimension management has been added to the parser for this example, the floating point value will go into the first element of the input tensor

**InputFloatString()**

This line says the next (second) piece of delineated text should be read as a floating point number, and that it goes into the input tensor of the current sample.  Again, no dimension management has occurred, so this value will go into the next element of the input tensor - the second element

**OutputFloatString()**

This line says the next (third) piece of delineated text should be read as a floating point number, and that it goes into the output tensor of the current sample.  Again, no dimension management has occurred, so this value will go into the first element of the output tensor

**}.withCommentIndicators(["#", "//"])**

This line ends the parser (end squirly-brace), but adds a modify to declare that all lines that start with a hash-tag ('#') or two slashes ('//') should be treated as comment lines and not be parsed for data

###  Using the Parser to Extract Data

The above parser is now ready to be used with a text source and a ``DataSet``.  This parser is expecting a DataSet with float values, two in the input tensor, and one in the output tensor.  The following code creates an appropriate ``DataSet`` and instructs the parser to read lines from the multi-line string 'text' into that DataSet:

```swift
var dataSet = DataSet(inputShape: TensorShape([2]), inputType: .float32,
                    outputShape: TensorShape([1]), outputType: .float32)
try parser.parseTextLines(dataSet: dataSet, text: text)
```

Each non-comment line is assumed to be another sample for the DataSet.

###  A Slightly More Complicated Example

```swift
#  A Slightly More Complicated Example
6,5.0,3.4,2.0,9.61,2.3,4.87,class1
14,5.4,3.5,107.2,8.612,4.3,9.67,class3
11,11.8,7.4,3.1,3,2.4,11.22,class1
7,9.6,4.8,2.2,14.74,5.6,5.3,class2
...
```

This data is a little harder to read.  It has an initial integer value, followed by 6 floating point values, and then a string giving a class name label for a catagorization problem.  It is separated by commas, but has the same comment identifier.  A possible parser definition for this data file is:

```swift
let parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {
    InputIntegerString()
    RepeatDimForDelineatedText(count: 6, dimension: .Dimension0, affects: .neither) {
        InputFloatString()
    }
    OutputLabelString()
}.withCommentIndicators(["#", "//"])
```

**let parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {**

This is the same as the previous example, except we state that the data elements are separated by commas.

**InputIntegerString()**

The first parameter on each line is an integer, so we read it as such and send it to the input tensor.

**RepeatDimForDelineatedText(count: 6, dimension: .Dimension0) {**

This starts a 'repeat a tensor dimension' block.  The contents are repeated 6 times, with the dimension 0 indicated for increment on the loop, but the 'affects' parameter set to '.neither' removes that issue.  All the examples so far have been reading into a 1-dimensional tensor (a vector, actually), and so only dimension 0 (the dimension on the left side [0 index] of a shape) is used for the data.

**InputFloatString()**

This line says the next (second) piece of delineated text should be read as a floating point number, and that it goes into the input tensor of the current sample.  This will increment the right-most (in this case, being a 1-dimensional tensor, dimension index 0) location for parsing.  Being inside of the Repeat block, this operation will be performed 6 times

**}**

This is the end of the Repeat block.  It is at this point any parsing locations (dimension indices) will be modified as specified in the Repeat block.

**OutputLabelString()**

This line states that the next delineated piece of text (number 8, after one from the integer and 6 from the repeat block) is a classification label string.  The string is read and compared to existing labels.  If it is new the label is added to the list.  The index of the label is then stored in the classfication index of the sample, and the output tensor is set to a 'one-hot' rendition of the value.  Since we have three unique labels in the shown data set (assuming there aren't more later in the file), the output tensor needs to be sized with three elements.

**}.withCommentIndicators(["#", "//"])**

Again, this line ends the parser (end squirly-brace), but adds a modify to declare that all lines that start with a hash-tag ('#') or two slashes ('//') should be treated as comment lines and not be parsed for data

The ``DataSet`` for this parser would need to be something like the following:

```swift
let dataSet = DataSet(inputShape: TensorShape([7]), inputType: .float32,
                    outputShape: TensorShape([3]), outputType: .float32)
try parser.parseTextLines(dataSet: dataSet, text: text)
```

###  Dimension Management

If you started with the previous example but changed your mind about the format of the input, thinking that beginning integer is a waste of time and the 6 floating values should be in a 2x3 matrix - then you have to do something a little extra.  You have to deal with dimension management, as you are putting the read data into two different dimensions.  The following parser could be used:

```swift
parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {
    UnusedTextString()
    RepeatDimForDelineatedText(count: 2, dimension: .Dimension0, affects: .input) {
        InputFloatString()
        InputFloatString()
        InputFloatString()
        SetDimension(dimension: .Dimension1, toValue: 0, affects: .input)
    }
    OutputLabelString()
}.withCommentIndicators(["#", "//"])
```

On a text parser, each line is assumed to be a sample (although there are ways to split a line into multiple samples).  At the start of each sample the parser resets the storage locations to the beginning of the tensor - one zero for each dimension.  The input tensor has two dimensions so gets set to \[0, 0\], while the single dimension output gets a storage location of \[0\].  Here is a discussion of the things in this parser that will affect or use those storage locations:

**UnusedTextString()**

This data chunk indicates the current text string component is unused and should not be stored in the sample.  No dimension storage locations are affected.

**RepeatDimForDelineatedText(count: 2, dimension: .Dimension0, affects: .input) {**

The repeat this time is for dimension zero (rows in a two-dimensional matrix tensor) and will update the storage location of the input tensor.  It will loop two times (we want two rows).  The location update is of course performed at the end of the repeat loop

**InputFloatString()**

This line says the next piece of delineated text should be read as a floating point number, and that it goes into the input tensor of the current sample.  This will increment the right-most (in this case, being a 2-dimensional tensor, dimension index 1 or columns) location for parsing.  Being inside of the Repeat block, this operation will be executed twice, one for each row.  We could have put it inside of another nested repeat loop for the columns (3 times), but it was just as easy to put in three of these InputFloatString chunks.

**SetDimension(dimension: .Dimension1, toValue: 0, affects: .input)**

This line is right before the end of the repeat loop, where we will advance to the next line.  Before it are 3 InputFloatStrings, each of which advanced the right-most dimension (dimension 1 - columns in this case) after storing in the input tensor.  Therefore the store location for dimension 1 is now at 3.  We need to reset it to 0 to start the next line, and that is exactly what this line is doing.

**}**

The end of the repeat loop.  At this point the repeat dimension (0) for the input tensor will be incremented, taking us to the next row.  The 'affects' parameter says only do the input tensor, so the output tensor is unchanged.  This is good as it is a single-dimension tensor for this case!

###  DataChunks Used by DelineatedTextParsers

The following classes can be used in a DelineatedTextParser construction:
| DataChunk            | Description |
| -------------------- | ----------- |
| ``UnusedTextString``           | The delineated string is thrown away                                                                                     |
| ``LabelTextString``            | The delineated string is assumed to be a class label (see note below) for the input tensor                               |
| ``LabelIndexString``           | The delineated string is taken as a class index integer and stored in the input tensor                                   |
| ``InputIntegerString``         | The delineated string is taken as an integer and stored in the input tensor                                              |
| ``InputFloatString``           | The delineated string is taken as a floating value and stored in the input tensor                                        |
| ``OutputIntegerString``        | The delineated string is taken as an integer and stored in the output tensor                                            |
| ``OutputFloatString``          | The delineated string is taken as a floating value and stored in the output tensor                                       |
| ``OutputLabelString``          | The delineated string is assumed to be a class label (see note below) for the output tensor                              |
| ``RepeatDimForDelineatedText`` | This allows you to put a repeating block of DataChunks, where a storage dimension can be updated at the end of each loop |
| ``SetDimension``               | This allows you to set the storage location dimension for input and/or output tensors                                    |
| ``IncrementDimension``         | This increments a specified storage location dimension for input and/or output tensors                                   |

All storage into tensors is done at the current storage location, after which the right-most (highest numbered) dimension of that storage location is incremented.

Note: Class labels, whether for the input tensor or output tensor are a list of strings associated with a classification index.  The classification value will be stored in the sample (if an OutputLabelString), and the output tensor values will be set to a 'one-hot' representation of the classification index.  Only one set of labels will be stored by a DataSet, and it is usually assumed to be the output classification label.

## Fixed Column Text Parsers

A fixed-column text parser is very similar to the delineated text parser, except there are no markers like whitespace or commas to indicate where data on the line starts or stops.  As a result you must give the character width of each piece of data on each chunk that reads a value.  Let's look at another example.  Here is some text data:

```swift
text = """
  5 4 5 6.2 100
  7 324 7.3 101
  9 abc 8.4 102
 14 del17.6 103
"""
```

The data has three pieces of information we want, a starting integer that is followed by some garbage, a floating value, and an output integer.  Note that the floating value on the last line runs into the unneeded text before it, but this isn't a problem because we are using text columns rather than delineation.  The columns for the data can be seen below:

```swift
  5 4 5 6.2 100
^  ^   ^    ^  
012345678901234
```

The first piece of data is assumed to start at column 0.  It is 3 columns wide.  The next (unneeded) piece starts at the next column, at 4, and is 4 columns wide.  The float starts after that and is also 4 columns wide.  Finaly the output integer starts after that and is 4 columns wide (the leading space is included).  The A parser for this data could then be written as following:

```swiftparser = FixedColumnTextParser() {
    InputIntegerColumns(numCharacters: 3)
    UnusedTextColumns(numCharacters: 4)
    InputFloatColumns(numCharacters: 4)
    OutputIntegerColumns(numCharacters: 4)
}
```

As the chunks do similar functions as the delineated versions, we are not going to describe the in detail here.  Just note the column widths and how they lay out against the text data.

###  DataChunks Used by FixedColumnTextParsers

The following classes can be used in a FixedColumnTextParser construction:
| DataChunk            | Description |
| -------------------- | ----------- |
| ``UnusedTextColumns``     | The specified columns string is thrown away                                                                              |
| ``LabelTextColumns``      | The string from the specified columns is assumed to be a class label (see note below) for the input tensor               |
| ``LabelIndexColumns``     | The string from the specified columns is taken as a class index integer and stored in the input tensor                   |
| ``InputIntegerColumns``   | The string from the specified columns is taken as an integer and stored in the input tensor                              |
| ``InputFloatColumns``     | The string from the specified columns is taken as a floating value and stored in the input tensor                        |
| ``OutputIntegerColumns``  | The string from the specified columns is taken as an integer and stored in the output tensor                             |
| ``OutputFloatColumns``    | The string from the specified columns is taken as a floating value and stored in the output tensor                       |
| ``OutputLabelColumns``    | The string from the specified columns is assumed to be a class label (see note below) for the output tensor              |
| ``RepeatDimForFixedText`` | This allows you to put a repeating block of DataChunks, where a storage dimension can be updated at the end of each loop |
| ``SetDimension``          | This allows you to set the storage location dimension for input and/or output tensors                                    |
| ``IncrementDimension``    | This increments a specified storage location dimension for input and/or output tensors                                   |


## Binary Parsers

Binary parsers are used to parse information from a byte stream either a ``Data`` object or a binary file.  Text files have line ends, which are a convenient marker for starting a new sample.  Binary data does not have this, so you have to deal with sample start/stop in a binary parser.  Don't worry, it's fairly easy and works the same as the storage location dimension wrangling we did earlier.

For our example, let's work with the MNIST data set.  It is a well-known data set that consists of 28x28 grey-scale images with a classification output of a single integer in the range of 0-9.  All data are unsigned bytes.  The input and output data are in separate files, with a few bytes of header before the actual data.  We'll read the input data into a two-dimensional tensor, and the output data into a one-hot vector.  Since the input and output are separate, we will need separate parsers for them:  They look like the following:

```swift
let MNISTInputParser = DataParser {
    UnusedData(length: 16, format: .fUInt8)
    RepeatSampleTillDone {
        RepeatDimension(count: 28, dimension: .Dimension0) {
            InputData(length: 28, format : .fUInt8, postProcessing : .None)
            SetDimension(dimension: .Dimension1, toValue: 0)
        }
    }
}
let MNISTOutputParser = DataParser {
    UnusedData(length: 8, format: .fUInt8)
    RepeatSampleTillDone {
        LabelIndex(count: 1, format: .fUInt8)
    }
}
```

Let's go through each line again:

**let MNISTInputParser = DataParser {**

Create a binary data parser for the input files

**UnusedData(length: 16, format: .fUInt8)**

Skip some bytes.  In this case 16 elements of UInt8 - or 16 bytes.  This is the header of the input file before the data samples start.

**RepeatSampleTillDone {**

There are no explicit sample markers in the data, so we have to tell the parser when one sample is full and we should go on to the next.  And we want to do this until we have read all the samples.  This chunk will repeat the enclosed chunk list until the file is depleted.  When the parser is started an empty sample is added to the DataSet.  Data read goes into that sample.  For text parsers, the end-of-line indicates time to close out the sample and create a new one.  For binary parsers, repeat or SetDimension chunks will be doing that.  At the end of the file (for both binary and text parsers), any empty created sample (ready for more data that won't be coming) is removed.  When the end of this chunk is reached, the current sample is stopped and the new empty one is created.

**RepeatDimension(count: 28, dimension: .Dimension0) {**

The data is stored as 28 rows of 28 pixels.  A two-dimfensional tensor has rows (index 0) and column (index 1) dimensions.  Data read goes into the right-most or highest index dimension (due to row-major storage) and increments that dimension axes storage location.  We need to read 28 rows of those 28 columns, and this repeat does that by iterating on the row axes

**InputData(length: 28, format : .fUInt8, postProcessing : .None)**

This input is configured to read 28 elements of type UInt8 (doing no processing on the values (post-processing is discussed later).  These 28 values will be stored into the input tensor (chunk is 'InputData'), with the largest dimension (dimension 1 in this case) being incremented after each store.

**SetDimension(dimension: .Dimension1, toValue: 0)**

The 28 input values moved dimension 1 storage location to 28.  We need to get it back to 0 so we can start the next row.  Hence, this chunk

**} - end of RepeatDimension**

When this iterator end is reached the specified dimension, in this case dimension 0 (row) is incremented.

**} - end of RepeatSampleTillDone**

When this iterator end is reached the sample is assumed done, a new one is created and added to the DataSet, and the storage locations (for both input and output tensors) are reset to 0s.  Since we have a 2-dimensional input, it is set to [0,0], while the single dimension output is reset to [0]

Note that we haven't read any output data yet.  That comes from another file, using another parser.  This is explained below

**} - end of DataParser construction**

That's all for the input parser

**let MNISTOutputParser = DataParser {**

Create a binary data parser for the input files

**UnusedData(length: 8, format: .fUInt8)**

The output (labels) file only has 8 bytes of header information we don't need

**RepeatSampleTillDone {**

We want to read all the outputs into the samples so we loop until the end of the file.  Since the samples will already have been created by the input parser (assuming that is run first), this chunk doesn't create new samples, it just increments the sample index it is working with when the end-bracket is reached

**LabelIndex(count: 1, format: .fUInt8)**

This chunk is configured to read 1 UInt8 byte and use the value as a classification label index.  The output tensor will be set as a one-hot representation of the integer and the classification index set on the sample.

**} - end of RepeatSampleTillDone**

When this iterator end is reached the sample index is incremented.  Since there will already be a sample at the new index (assuming the input file was run first), no new sample is created.  The storage location is reset to all zeros.

**} - end of DataParser construction**

And that's all for the output parser

### Running a two-step parser set

Since the MNIST input and output data is in two files, we have to run the DataSet through each parser.  Here is the code that creates the ``DataSet`` and uses the parsers to read the training data.  It is assumed you have two URLs to the two files needed, MNISTInputURL and MNISTOutputURL:

```swift
let inputShape = TensorShape([28, 28])
let outputShape = TensorShape([10])
var trainingDataSet = DataSet(inputShape: inputShape, inputType: .float32, outputShape: outputShape, outputType: .float32)

try MNISTInputParser.parseBinaryFile(url: MNISTInputURL, intoDataSet: trainingDataSet)
try MNISTOutputParser.parseBinaryFile(url: MNISTOutputURL, intoDataSet: trainingDataSet)
```

The first run with the input parser will create the samples in the DataSet, while the output parser run will fill in the output tensors of each sample.

### Post-Processing Data

The bytes read by the MNISTInputParser are converted to floats in the DataSet input tensor with a straight cast, as we have specified that no processing should be done after reading with the InputData(length: 28, format : .fUInt8, postProcessing : .None) data chunk.  However, we may want them to be in a different range for a neural network.  To this end binary input data can be processed after reading.  The following post-processing options are available:
| Post-Processing Option | Description                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------|
| None                   | No post-processing performed                                                                                  |
| Scale_0_1              | Each value is scaled to be between 0 an 1, based on format type value range                                   |
| Scale_M1_1             | Each value is scaled to be between -1 an 1, based on format type value range                                  |
| Normalize_0_1          | Each value in the sample is normalized to be between 0 an 1, based on actual range of data in the sample      |
| Normalize_M1_1         | Each value in the sample is normalized to be between -1 an 1, based on actual range of data in the sample     |
| Normalize_All_0_1      | Each value in the data set is normalized to be between 0 an 1, based on actual range of data in the data set  |
| Normalize_All_M1_1     | Each value in the data set is normalized to be between -1 an 1, based on actual range of data in the data set |

###  DataChunks Used by Binary DataParsers

The following classes can be used in a DataParser construction:
| DataChunk                | Description                                                                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| ``UnusedData``           | The specified data is read and discarded                                                                                 |
| ``LabelString``          | A string of a specified size is read and treated as an output classification label                                       |
| ``LabelIndex``           | The data is read and treated as an output classification index                                                           |
| ``InputData``            | The specified data is read and stored in the input tensor at the current input storage location, which is incremented    |
| ``RedPixelData``         | The specified data is read and stored in the first location (index zero) of the last dimension of an input tensor        |
| ``GreenPixelData``       | The specified data is read and stored in the second location (index one) of the last dimension of an input tensor        |
| ``BluePixelData``        | The specified data is read and stored in the third location (index two) of the last dimension of an input tensor         |
| ``OutputData``           | The specified data is read and stored in the output tensor at the current output storage location, which is incremented  |
| ``RepeatSampleTillDone`` | This starts and stops each sample in the data file.  The chunks in the loop should define each sample                    |
| ``RepeatDimension``      | This allows you to put a repeating block of DataChunks, where a storage dimension can be updated at the end of each loop |
| ``SetDimension``         | This allows you to set the storage location dimension for input and/or output tensors                                    |
| ``StartNewSample``       | This ends the current sample, creates a new one, and resets the storage locations for the sample                         |
| ``IncrementDimension``   | This increments a specified storage location dimension for input and/or output tensors                                   |

