# ``MPSGraphDSL``

The MPSGraphDSL framework is a tool for parsing data from input sources and running that data through a Metal Performance Shader Graph calculation.  It has an emphasis on machine learning, but can be used for any compute-intensive operation that requires the GPU or Neural Engine to be engaged.  Description of the data format and the calculation graph are both done using a DSL (Domain Specific Language) addition to Swift, similar to how SwiftUI is used.
 
## Overview

There are four top-level components of the MPSGraphDSL framework:  Tensors, DataSets, Data Parsers, and Graphs

### Tensors
Tensors are a collection of data arranged into a multi-dimensional.  A one-dimensional ``Tensor`` is a vector.  A two-dimensional one is a matrix.  MPSGraphDSL Tensors support up to 16 dimensions.  They can be converted between MPSGraphTensors when needed - and this is automatically done when sending them to a ``Graph``.  A ``Tensor`` is a protocol that is conformed to by specific classes that have a defined data type, like ``TensorFloat32`` and ``TensorUInt8``

For more information, see the Article <doc:CreatingTensors>

### DataSets
DataSets are a collection of samples.  Each sample has the input and output ``Tensor`` for a single case within the ``DataSet``.  The complete dataset can be ran through a ``Graph`` for testing or learning

For more information, see the Article <doc:CreatingDataSets>

### Data Parsers
MPSGraphDSL has three types of data parsers to extract data from files or other sources:  Binary, Fixed-Column Text, and Delineated Test.  After describing the format of the data using one of these Parsers, they can be used to read a file or other source into a complete ``DataSet``

#### Binary Data Parsers
Binary data parsers are used to extract data from non-textual files or Data objects.  The format of the file is specified by describing the order of the data types (float, int, text, etc.) in the file using a DSL.

#### Fixed-Column Text Data Parsers
Fixed-Column Text data parsers are used to extract data from text files when the text is formatted such that data pieces on each line align to specific columns.  The format of the file is specified by describing the order of the data types (float, int, text, etc.) in the file along with the columns they are on inside the line using a DSL.

#### Delineated Text Data Parsers
Dilineated Text data parsers are used to extract data from text files when the text is formatted such that data pieces on each line are separated by known symbols (often a comma).  The format of the file is specified by describing the order of the data types (float, int, text, etc.) in the file as they appear in the line using a DSL.

For more information, see the Article <doc:CreatingParsers>

### Graphs
Graphs are computation definitions that can run on the GPU or Neural Engine.  They are defined using a DSL that has an extensive collection of 'Nodes' that perform a single operation on a Tensor - like addition, or matrix multiplication, or even a fully connected neural network layer.  The defined graph can then be used to process Tensors or complete DataSets

For more information, see the Article <doc:CreatingGraphs>

### Neural Networks
The highest-level operations of MPSGraphDSL is creation of neural networks using just a few lines of code.

For more information, see the Article <doc:NeuralNetworks>

##  Where to Start

Reading the five articles referenced above will definitely get you well on your way.  The next easiest way to start is probably with an example.  To this end the project contains a Swift Playground called **TutorialPlayground**.  There is also a command-line tool called **TensorOperations** that perform many of the Tensor modification operations so the results can be seen.  Looking at the source code for that tool can give you an idea of how to perform simple operations.  Lastly, you can look at the Test suite, as most of the tests perform either data parsing or graph calculations.

## Topics

### Tensors

- ``TensorShape``
- ``Tensor``
- ``CreateTensor``

### DataSets

- ``DataSet``
- ``DataType``

### Data Parsers

- ``DataParser``
- ``DelineatedTextParser``
- ``FixedColumnTextParser``

### Graphs

- ``Graph``
- ``SubGraph``
- ``Node``

### Neural Networks

- ``FullyConnectedLayer``
- ``LSTMLayer``
