# Creating and Using Graphs

This article describes how you use the DSL to define a ``Graph`` object, and how to use that object to perform operations on Metal devices, including training and testing neural networks

## Overview

A ``Graph`` is a set of calculation (or other Tensor manipulation) nodes connected together to form a graph structure.  A tree graph is a common example of this, but an MPS Graph often has more than one input and more than one output, so the shape of the graph is not generally described as a tree.  Calculations in the Graph are performed from the leaf nodes (nodes that start a process line like inputs, constants, etc.) and proceed to designated result nodes.  Data can flow backwards through the graph with learning and other assignment operations.

### Concepts to Know

The following concepts are needed for a discussion on creating and using a Graph:

#### Modes

The calculations performed by a Graph can follow different paths with the node graph depending on what you want to do.  A common use of this is inference and learning modes for a neural network, where inference stops after the results of the network are determined but learning requires a loss function to be calculated with that network result.  Some nodes and modifiers used to create a Graph can be mode dependent, and therefore need to be configured with the modes they should operate on.

Modes are identified by a (case-sensitive) string.  This string is used to configure nodes and is passed to the 'run' functions of the Graph to specify which mode of the Graph calculations should be performed.

#### Nodes

The ``Graph`` is composed of a set of nodes with connections between them that represent the data flow for the calculation.  Nodes can be named so they can be referenced as inputs to other nodes.  This reference is what makes the 'lines' between nodes.  MPSGraphDSL has the nodes part of a result builder, allowing declarative definitions of the Graphs.

MPSGraphDSL Graphs are mostly just a wrapper for Apple's MPSGraph objects.  The MPSGraph object has hundreds of different nodes that can be added to the graph.  Most of these are directly represented in MPSGraphDSL, but may be referenced slightly differently to make them easier to use.  For example, there are over a dozen different reduction nodes in MPSGraph, while a Graph only has one - with options in the initialization parameters to specify the functionality of all the different operations MPSGraph supports.

Additionally there are some MPSGraphDSL specific nodes that are added to make common operations simpler.  An example of this is the ``FullyConnectedLayer``, a fully-connected neural network layer with weights, matrix multiplication, bias terms and activation functions.  While all of these parts can be manually put in, or a ``SubGraph`` created (explained below), it was deemed common enough to make a single node for ease of use.

Nodes are checked for use.  If a node is not referenced by another node, and is not a target for an operation (described below), an error will be thrown

#### Targets

At least one node in a Graph must be designated as a target, although many can be.  Targets are the outputs of the Graph, so any data coming out of the system comes through a target node.  MPSGraph uses the designated targets to determine the calculation paths that will be performed.  A neural network likely has an inference output that would be a target, but also a loss calculation when learning that would also be a target.  Since which one of these these is the result desired depends on what you are doing, training with learning operations or infering an output for testing or after training is complete, targets are mode dependent.

Any node can be designated as a target.  To do so use a 'targetForModes' modifier on the node to specify which modes the node should be a target for.  Nodes designated as targets must be named (often a name is optional), as the output data Tensors will be returned in a dictionary keyed by the name of the target node.  The 'targetForModes' modifier should be last of all modifiers for a node, as it is generic to all nodes and can stop any other modifiers that come after it from being recognized as valid for the node.

#### PlaceHolders

A ``PlaceHolder`` is a 'leaf' node, a node with no inputs from other nodes in the Graph so data only flows out of it.  A PlaceHolder, as the name implies, holds the place in the Graph where external inputs will be present when the Graph is performing calculations.  PlaceHolders also are required to be named.  When you 'run' the Graph you provide a dictionary of Tensors keyed by the PlaceHolder names for the inputs needed.

Some inputs to a Graph may not be needed for all modes.  For example, the expected value of a neural network is needed for the loss calculation, but not for the inference result.  Therefore you can limit what PlaceHolders need to be filled for each mode.  By default a PlaceHolder is required to be filled for all modes, but you can use an optional parameter on initialization to specify what modes it is actually required for.

When creating a Graph to take batch input (described below) you don't add the batch dimension to the PlaceHolder shape, it will be done automatically for you by the Graph build system.  The Tensors passed in will be concatenated together for a batch and passed to the auto-modified placeholder.  There are times where you don't want a PlaceHolder to be automatically adjusted for handling batch input, such as when adding adjustment constants and the like.  For these cases there is a 'isBatchExempt' modifier that tells the Graph build system to not extend the shape to cover a batch dimension.

##  Creating a Graph

Let's start by creating a simple graph - one that takes an input TensorFloat32 with shape \[3, 4\] and multiplies each element by two.  The following code creates a Graph that does that:

```swift
let graph = Graph {
    PlaceHolder(shape: [3, 4], name: "inputs")
    Constant(shape: [3, 4], value: Float32(2.0), name: "constant")
    Multiplication(firstInput: "inputs", secondInput: "constant", name: "multiplicationResult")
        .targetForModes(["runGraph"])
}
```

Let's walk through each line of that code and explain what is going on:

**let graph = Graph {**

Create a ``Graph`` object and set up the declarative construction with a set of nodes inbetween the braces

**PlaceHolder(shape: [3, 4], name: "inputs")**

This ``PlaceHolder`` is the stand-in for the input.  When this Graph is run a Tensor of the specified shape (3 rows of 4 columns) must be provided for the inputs with key 'inputs' (the name of the PlaceHolder).  Note the data type was not specified in the PlaceHolder.  Most nodes can take any type of data and will adapt base on what they are passed - but you must be consistent with types being passed into operations to maintain data integrity.

**Constant(shape: [3, 4], value: Float32(2.0), name: "constant")**

This node is another leaf node - one that does not take inputs from another node.  It defines a constant value source shaped [3, 4], with a constant Float32 value of 2.0 for each element.

**Multiplication(firstInput: "inputs", secondInput: "constant", name: "multiplicationResult")**

This node performs the element-wise multiplication.  It takes two inputs, which have been specified to be from the nodes "inputs" and "constant".  These are the names of the two nodes above this one.

**.targetForModes(["runGraph"])**

This modifies the Multiplication node to make it a target node for the mode we will call "runGraph".  Note we didn't put a mode filter on the PlaceHolder (no parameter defining applicable modes), so it will be available in all modes - including this one.

**}**

End of the graph definition

####  Running a Tensor through this Graph

We will go into more detail on running and encoding the Graph in a later section, but it helps to see how a simple run case would be performed to help understand how the PlaceHolder, modes, and result Tensors fit into the scheme of this Graph.  The following code could be used, assuming you have a TensorFloat32 of shape \[3, 4\] with your data called "inputTensor":

```swift
let inputTensorList: [String: Tensor] = ["input": inputTensor]
let results = try graph.runOne(mode: "runGraph", inputTensors: inputTensorList)
let myResultTensor = results["multiplicationResult"]
```

This code starts out by creating a dictionary of all required inputs (from PlaceHolders valid for the mode being run) with the Tensor keyed by the name of the PlaceHolder.  The only PlaceHolder we added is needed for all modes and is named "input", so we take our data Tensor and assign it into the input feed dictionary

The Graph is then run with the name of the mode and the input dictionary.  The only mode referenced in the Graph is "runGraph".

The returned results is a dictionary of all Tensor outputs from the nodes that are identified as targets for the mode being run.  We specified the output of the Multiplication node as a target for mode "runGraph", which is the one specified on the run, so it should be in the dictionary (although you should always check - we left out that code for simplicity).  The name of the node was "multiplicationResult" so extracting that out of the dictionary gets us our outputs.

#### Automatic Short Paths

It is common for the output of a node to immediately be used by the next node in the graph, and only that node.  This pattern is common enough that MPSGraphDSL allows you to skip the name and the reference when it occurs.  The previous Graph could have been defined as:

```swift
let graph = Graph {
    PlaceHolder(shape: [3, 4], name: "inputs")
    Constant(shape: [3, 4], value: Float32(2.0))
    Multiplication(firstInput: "inputs", name: "multiplicationResult")
        .targetForModes(["runGraph"])
}
```

The Constant node no longer has a name, and the Multiplication node only references one input - the PlaceHolder - which is two nodes back in the list.  The missing second input parameter will cause the Multiplication node to assume that input is from the previous node, which is the Constant node we explicitly referenced before.  If we wanted to multiply by the square-root of two rather than just two (and didn't want to have an irrational constant in the definition) we could use the Graph definition:

```swift
let graph = Graph {
    PlaceHolder(shape: [3, 4], name: "inputs")
    Constant(shape: [3, 4], value: Float32(2.0))
    SquareRoot()
    Multiplication(firstInput: "inputs", name: "multiplicationResult")
        .targetForModes(["runGraph"])
}
```

The Constant flows into the SquareRoot which flows into the Multiplication.  No names or references needed.

### Available Nodes

The Nodes that are available to create a Graph are too many to just list here.  To see all of them, go to the documentation for ``Node`` and look at the 'Inherited By" list, go to each subclass, and look at the inherited subclasses.  It is not expected to have any nodes more than two sub-classes away from Node.  Generally, if you have a math operation, data manipulation, or neural network operation in mind, it can likely be found.  Below are a few of the more common operations:

#### Leaf Nodes

* ``PlaceHolder``
* ``Constant``
* ``Variable`` - explained below

####  Basic Math

* ``Addition``, ``Subtraction``, ``Multiplication``, etc.
* ``SquareRoot``, ``Negative``, ``Absolute``, etc.
* Logarithms with different bases, ``Exponent``
* ``Round``, ``Floor``, ``Ceiling``, etc.

#### Logic

* ``LogicalAND``, ``LogicalOR``, etc.
* ``BitwiseAND``, ``BitwiseOR``, etc.
* ``Equal``, ``GreaterThan``, etc.

#### Trigonometry

* ``Sine``, ``Cosine``, etc.
* ``ArcSine``, ``ArcCosine``, etc.
* ``HyperbolicSine``, ``HyperbolicCosine``, etc.

#### Tensor Modification

* ``Reshape``, ``Slice``, ``Concatenate``, ``Split``, etc.
* ``Sort``, ``Reduction``, ``Transpose``, etc.
*  ``Cast``

#### Neural Network

* ``Sigmoid``, ``ReLU``, ``LeakyReLU``
* ``Padding``
* ``FullyConnectedLayer``, ``Convolution2D``, etc.
* ``SoftMax``, ``SoftMaxCrossEntropy``, ``MeanSquaredErrorLoss``

##  Variables and Learning

In a neural network, and perhaps other Graphs, you have a training phase where you attempt to learn the value of one or more variables.  In a neural network these variables are the weights and biases of the network layers.  For this purpose there is a ``Variable`` node.

A Variable node is (usually) a leaf node that provides an output tensor similar to a Constant node.  The difference is that Variable nodes can be updated by operations in the Graph.  The MPSGraphDSL framework also provides functions to read all the Variable nodes, store previously read values back into, and to reset Variable tensors to their initial or random states.  A modifier for Variable nodes is available that marks them as 'learnable', where a loss function can be calculated and used to update the Variable based on the gradient from the loss back to the Variable node.

An MPSGraphDSL specific node is provided to identify the learning modes and provide learning rates for the back-propogation.  This node is called ``Learning``.

###  Learning Example

The example we will use is quite simple.  Assume you have a set of two numbers, an input and an output.  The output values are an unknown multiple of the input values.  Our example will learn this multiplication constant.  If you would like to actually see this in practice, this is mostly the first test in the 'LearningTests' test suite of the Framework.  Our graph definition will look like the following:

```swift
let multiplicand = TensorFloat32(shape: TensorShape([1]), initialValue: Float32(2.0))
let graph = Graph {
    PlaceHolder(shape: [1], name: "input")
    Variable(values: multiplicand, name: "variable")
        .learnWithRespectTo("loss")
        .targetForModes(["getVar"])
    Multiplication(firstInput: "input", secondInput: "variable", name: "result")
        .targetForModes(["test"])
    PlaceHolder(shape: [1], modes: ["lossCalc", "learn"], name: "expectedValue")
    MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")
        .targetForModes(["lossCalc", "learn"])
    Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])
}
```

Again, we will walk through each line discussing what is happening.

**let multiplicand = TensorFloat32(shape: TensorShape([1]), initialValue: Float32(2.0))**

A Variable node requires an initial value to be specified.  It can come from another node (making the Variable somewhat of a non-leaf node, although the initialization is only done once), from a provided Tensor, from a constant value, or from a random generator.  For this example we are going to provide a Tensor for the initial values (or value, the shape is a single dimension of 1).  This input Tensor is initialized with a value of 2.0.

**let graph = Graph {**

Create the graph for both testing and learning modes

**PlaceHolder(shape: [1], name: "input")**

Our input is a single value.  It will be passed to the graph using a feed dictionary key of "input"

**"Variable(values: multiplicand, name: "variable")**

This defines our Variable.  It gets it's shape and data type from the initialization Tensor, and will be referenced by the name "variable".  Variables are another node that the name is required rather than optional.

**.learnWithRespectTo("loss")**

This modifies the Variable to be updated when in a learning mode.  The update will be a function of the gradient from a future node named "loss", and the learning rate provided by a ``Learning`` node.

**.targetForModes(["getVar"])**

This modifies the Variable to be a target when the Graph is run with mode "getVar".  This allows us to retrieve the value of the Variable for inspection later.

**Multiplication(firstInput: "input", secondInput: "variable", name: "result")**

This node multiplies the input by the variable.

**.targetForModes(["test"])**

This modifies the Multiplication node to be an output when the Graph is run with mode "test".  The predicted output value will be return in the results dictionary with a key of "result".  Nodes identified as targets must be named so the run results can be found amongst any other result targets for the mode.

**PlaceHolder(shape: [1], modes: ["lossCalc", "learn"], name: "expectedValue")**

This Placeholder provides the expected value for the loss function.  It is not needed for "test" mode where we are just getting the results of the multiplication, or "getVar" mode where we are just reading the value of the Variable.  Therefore the PlaceHolder is marked to only be required when the mode is "learn"

**MeanSquaredErrorLoss(actual: "expectedValue", predicted: "result", name: "loss")**

This node calculates a mean squared error between two nodes.  It is often used as the loss function for a calculation, as is being done here.  Note the name of the node is "loss", the node reference by the Variable node as the one it 'learns' relative to.

**.targetForModes(["lossCalc", "learn"])**

The loss value must be a target for learning so that path to the loss node will be calculated.  In addition, we might want to get the loss value without the learning update step, so we will create a mode where that is possible as well.

**Learning(constant: true, learningRate: 0.05, learningModes: ["learn"])**

This is the special ``Learning`` node for MPSGraphDSL.  When it is present the Graph will create gradient nodes, stochastic learning nodes, and Variable assign operations for the specified mode.  All Variables that are to be learned must be modified with a 'learnWithRespectTo' modifier, and the loss term referenced be a target node for the mode.  The learning rate for the stochastic gradient descent is specified in this node as either a constant, or a variable.  If variable it will be set to the initial rate specified in this node but can be changed with any run or encode operation as training procedes.

**}**

End of the Graph definition.

Discussion of how to use this graph is in a following section.

##  Batch Graphs

Often Graphs are created to run multiple data samples through them between Variable updates.  The input Tensors are concatenated together into a 'batch' version that is passed to the Graph.  This speeds up processing as the number of data transfers between the CPU and the GPU is reduced, even though the amount of data remains the same.

To try to make this process easy, Graphs can be built in 'batch' mode.  When a batch Graph is being built, all PlaceHolders that are not marked as exempt automatically have a batch dimension prepended to the shape.  Creating a batch Graph just requires specification of the batch size on the initializer.  The following code creates a batch Graph that processes 16 samples at once:

```swift
graph = Graph(batchSize: 16) {
    PlaceHolder(shape: [28, 28], name: "input")
    .
    .
    .
    SoftMax(name: "result")
        .targetForModes(["infer"])
}
```

The PlaceHolder takes an 28x28 input image.  When the Graph is used a 16x28x28 Tensor will be passed to the Graph, with 16 samples (of shape \[28, 28\]) concatenated together to make the batch.  Any result Tensors may have a batch dimension as well, so check rather than assume.  Testing and Training functions on the Graph will automatically take care of the batch dimension.

Many neural network nodes like FullyConnectedLayer, ConvolutionLayer, etc. deal with batch dimension tensors by running each batch sample through the operations with weight/bias variables that are not expanded for the batch size.  This means you can create a Graph to train/test with the speed inprovement of batch processing, but put the same weights and biases into a future Graph (with the same structure of nodes) that does not require batch inputs for user inference runs.

##  Running Operations with the Graph

To get a result out of the Graph you need to run it or encode it.  Running it starts the Graph with the given input data on a new Metal command buffer and awaits the return Tensors.  Encoding starts the graph on an already existing command buffer, and so doesn't require creating new buffers for every data instance - making it much more efficient for multiple runs.  Encoding can also be set to not wait for results, which speeds up runs where they aren't needed, such as training runs.

### Input (Feed) Dictionary

Whether you are running or encoding you will likely need to provide an input dictionary.  All PlaceHolders required for the mode being run must have an entry in the dictionary.  In our example above we have four modes, "getVar", "test", and "lossCalc", and "learn".  The first PlaceHolder (for the input) is not configured with a mode set, so will be required for all modes even though it really isn't needed for the "getVar" mode.  The second PlaceHolder (for the expected value) is configured to be used when calculating the loss or when learning.  A ``Tensor`` of the desired data type and shape matching the PlaceHolder is put in the feed dictionary with the key being the name of the PlaceHolder node it goes into.  If we have an input Tensor named "inputTensor" and expected value Tensor named "expectedValue" the following code builds the input dictionary:

```swift
var inputTensors : [String : Tensor] = [:]
inputTensors["input"] = inputTensor
inputTensors["expectedValue"] = expectedValue
```

Often these two PlaceHolders are filled from a sample in a ``DataSet`` using sample.inputs and sample.outputs.

If a required PlaceHolder does not have an entry in the dictionary the run or encode call will throw the error ``MPSGraphRunErrors.PlaceHolderInputNotFound``.  Extra entries, like the 'expectedValue' entry would be if running a "test" mode instance, will be ignored.

### Running a Single Instance

To run a single instance of data through the Graph, use the ``Graph.runOne`` or ``Graph.encodeOne`` methods with the mode name and the input dictionary:

```swift
let results = try graph.runOne(mode: "test", inputTensors: inputTensors)
```

Encoding a "learn" operation where we don't need to see the results, but we want to change a variable learning rate (not handled in the above example) could look like:

```swift
let _ = try graph.encodeOne(mode: "test", inputTensors: inputTensors, waitForResults: false, newLearningRate: 0.01)
```

###  Extracting Results

The runOne and encodeOne methods return a dictionary of results.  The dictionary contains Tensors for each target identified for the mode executed, keyed by the name of the target node.  You extract the Tensor from the results using the target nodes' name.  Always assume an error could occur and the result is missing, but without those checks to complicate things the code to get our single value from the output tensor for a "test" mode run looks like:

```swift
let resultTensor = results["result"]
let result = try resultTensor!.getElement(index: 0).asDouble
```

###  Running an Entire DataSet for Classification

Graph has methods to run an entire ``DataSet`` through with the results being handled as a classification or regression problem.  There are testing and training methods, with the testing method returning a fraction correct value for classification or total absolute error for regression, and the training mode doing learning operations.  Our example above isn't a classification problem so the following code assumes a different Graph and a two full DataSets that have the testing and training data for that Graph:

```swift
let result = try classificationGraph.runClassifierTest(mode: "infer", testDataSet: testDataSet, inputTensorName: "inputs", resultTensorName: "inferenceResult")
print("Initial test percentage: \(result.fractionCorrect*100.0)")
```

This method runs all samples in DataSet 'testDataSet' through the Graph 'classificationGraph' in mode "infer", returning a tuple with the number of classifications that succeeded and a fraction value using the number of samples.  The input and result node names must be identified so the method can set up the input dictionary and extract the results appropriately.  Optional parameters on the function allow you to use a subset of the training DataSet as the test inputs.  Optional parameters allow the range of samples (or batches) used for the testing to be specified.  The method 'runRegressionTest' performs a similar operation for regression problems, returning the total absolute difference for the error value.

```swift
_ = try await classificationGraph.runTraining(mode: "learn", trainingDataSet: trainingDataSet, inputTensorName: "inputs", expectedValueTensorName: "expectedValue")
```

This method runs all samples in DataSet 'trainingDataSet' through the graph 'classificationGraph' in mode "learn", performing the learning operations and updating any Variables that are marked to learn versus a loss function.  The input values an expected result nodes must be identified so the method can set up the input dictionary with both PlaceHolder tensors.  Optional parameters on the function allow you to use a subset of the training DataSet as the test inputs and to request the total loss for the run to be returned.

All of these functions can be used with batch Graphs.  In these cases the number of samples for testing must be a multiple of the batch size and the epoch size for training indicates the number of batches run, rather than individual samples.

##  SubGraphs

Sometimes you will find yourself repeating a set of nodes multiple times in a Graph.  Things like a fully-connected neural network layer takes up to 5 nodes for weights, biases, operations, and activation functions (before MPSGraphDSL added one as compound node with ``FullyConnectedLayer``).  Reading a Graph definition with lots of repeating blocks can be difficult.  Therefore MPSGraphDSL supports SubGraph definitions, where you can define a set of nodes that can be repeated as a single node in the main Graph definition.  Mapping data between a main Graph definition and a SubGraph does add a slight complexity, so only use SubGraphs when the repeated node count is high due to the size of the subgraph times the number of times the subgraph is repeated.

###  Defining a SubGraph

To use a SugGraph you first create a ``SubGraphDefinition``.  A SubGraphDefinition is created mostly same as creating a full Graph, with the main difference being a node called ``SubGraphPlaceHolder``.  While a normal ``PlaceHolder`` is used to stand in for inputs coming from outside the Graph when it is being executed, the SubGraphPlaceHolder just marks where Tensors from other nodes within the Graph will come into the SubGraph.

The following SubGraphDefinition  adds 3 to all elements of a Tensor of shape [2] (probably not worth making a SubGraph for, but we wanted a simple example):

```swift
let subGraph = SubGraphDefinition {
    SubGraphPlaceHolder(name: "subInput")
    Constant(shape: [2], value: Float32(3.0), name: "constant")
    Addition(firstInput: "input", secondInput: "constant", name: "result")
        .targetForModes(["runTest"])
}
```

The ``Addition`` node was made a target to get the resulting Tensor for test purposes, and to show that you still set targets and learning modes in SubGraphs.

###  Using a SubGraph

To use a SubGraph you have to first define a mapping of nodes outputs to SubGraphPlaceHolders to allow the SubGraph to get data from other parts of the Graph.  This mapping is defined as a dictionary of strings of the node names being input into the SubGraph keyed by the name of the SubGraphPlaceHolder.  The first line of the code below creates a mapping for the input from the PlaceHolder ("input") to go into the SubGraphPlaceHolder ("subInput").

You can then provide this mapping in a ``SubGraph`` node which takes the definition of created for it and a name.  All SubGraphs must be uniquely named so that repeated nodes can be identified when connections are made.  The following code uses the above SubGraphDefinition:

```swift
let subgraphMap : [String : String?] = ["subInput" : "input"]       //  Map "input" node of graph to "subInput" placeholder of subgraph
let graph = Graph {
    PlaceHolder(shape: [2], name: "input")
    SubGraph(definition: subGraph, name: "subgraph", inputMap: subgraphMap)
    Negative(name: "result")
        .targetForModes(["runTest"])
}
```

The final names of nodes in SubGraphs are a composite name of the SubGraph and the node name.  The ``Constant`` node in the SubGraphDefinition will be added to the Graph as "subgraph_constat" since the SubGraph is named "subgraph" and the Constant is named "constant".  When referenced within the SubGraphDefinition you do not need to know the SubGraph name and just reference it using the name, as seen in the Addition node in the code above.  If referenced outside the SubGraphDefinition you would need to use the full name.  It is recommended that the input mapping is used rather than trying to use the full name.  Note the above definitions both have a "result" node.  As the Addition node would have a final name of "subgraph_result" there is no conflict.  Both nodes are targets, and so both will be in the result Tensor list with their full names.

When a SubGraph is added to a Graph the nodes in the SubGraphDefinition are placed in the Graph just as they would if they were individually added instead.  This means the previous node shortcut will work.  The ``Negative`` node in the Graph definition does not have an input reference and so uses the output of the previous node.  This would be the last node of the SubGraph definition, the Addition node.
