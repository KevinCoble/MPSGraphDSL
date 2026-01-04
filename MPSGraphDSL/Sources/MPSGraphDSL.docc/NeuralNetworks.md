# Neural Networks

While the MPSGraph system has all the pieces to make any neural network you can think of, assembling those pieces can take a good bit of work.  Apple made almost everything configurable so the number of parameters and options and pieces you have to assemble can make it a time-consuming task.  To alleviate this MPSGraphDSL has created composite nodes for neural networks.  They combine the network layer operations with the required variable declarations, tensor manipulation, default option handling, and more - allowing you to define a network in a few lines.

## Overview

The following composite nodes are available:
| Network Layer            | Description |
| ------------------------ | ----------- |
| ``FullyConnectedLayer``  | A standard fully-connected layer with optional bias term and activation function                             |
| ``RNNLayer``             | A standard one-gate recursivve neural newtwork layer.  Weights, biases and activation functions are managed  |
| ``LSTMLayer``            | The long short term memory layer, with all four gates.  Weights, biases and activation functions are managed |

In addition, topics in this article discuss how to use the Graph functions to test and train both classification and regression networks.

### FullyConnectedLayer

The fully-connected layer is a standard neural network layer that takes the input tensor, multiplies it by the weight tensor, optionally adds a bias term to that result, then performs an activation function on the final output.  If needed, tensor reshape operations may be added to coerce tensors to the correct shape.

####  Creating a FullyConnectedLayer

The following initializer is used in a Graph (or SubGraph) definition to add a FullyConnectedLayer:

```swift
FullyConnectedLayer(input: String? = nil, outputShape: TensorShape, activationFunction: ActivationFunction, name: String)
```

The output shape determines the number of 'neurons' in the layer.  Rather than a flat array of them, it is possible to make a multi-dimensional tensor of them.  The result from the node will then be in the shape specified.

The activation function is selected from the enumeration ``ActivationFunction``.  To turn off activation, select the option '.none'.

The input tensor and node name are standard parameters.  See the article on Creating Graphs for more information


####  Modifiers for a FullyConnectedLayer

The following modifiers are available for the FullyConnectedLayer node:

| Modifier            | Description |
| ----------------------------------| ----------- |
|  noBiasTerm()                     |  If added removes the bias Variable and the addition of it to the matrix multiplication  |
|  weightInitialRange(min:, max:)   |  Sets the range for random initialization of the weights Variable                        |
|  biasInitialRange(min:, max:)     |  Sets the range for random initialization of the bias Variable                           |
|  learnWithRespectTo(_ lossNode)   |  Sets the node to have the Variables learn with respect to the specified loss node       |

As with all nodes, the targetForModes modifier is available, but should be added after all other modifiers.  Only some tensors created by the FullyConnectedLayer node will be targetted when this modifier is used.  See the 'Tensors Added' section for more information.


####  Tensors Added by a FullyConnectedLayer

The following tensors are added to the Graph by the node:

| Suffix                | Targetted | Description |
| --------------------- | --------- | ---------------------- |
| "_inputReshape"       |     No     | (if needed) A reshape node to convert the input to shape [1, x], where x is the total input tensor shape volume         |
| "_weights"            |     No     | The weights Variable.  Added to assign/load and learn operations if configured        |
| "_biases"             |     No     | (If configured) The biases Variable.  Added to assign/load and learn operations if configured        |
| "_matrixMult"         | See Desc.  | The multiplication of the input by the weights.  Set as a target if no reshape, bias, or activation tensors needed        |
| "_outputReshape"      | See Desc.  | (if needed) A reshape node to convert the matrix multiplicaiton to the output shape.  Set as target if no bias or activation  |
| "_biasAdded"          | See Desc.  | (If configured) The addition of the bias term.  Set as a target if no activation tensor needed        |
| None.                 |    Yes     | (If configured) The activation function.  Set as a target if node configured as target       |

Note:  Any of the above tensors that can become the target will have their suffix removed (named with just the layer's node name) when they are the target tensor.  This makes the name of the output of the node the same regardless of configuration.

These tensors can be referenced using the name of the node with the suffix added.

### RNNLayer

The recursive neural network layer is a neural network layer that feeds the previous output back into itself as another input.  This allows a sequence in the input data to be learned.  Since the previous output is directly connected to the input, sequences that are long will have a hard time being learned due to the "exploding or vanishing gradient" problem.  LSTM and Gru layers have methods to address this.

####  Creating an RNNLayer

The following initializer is used in a Graph (or SubGraph) definition to add an RNNLayer:

```swift
RNNLayer(input: String? = nil, stateSize: Int, name: String? = nil)
```

The RNN layer has four parameters that define the size of data being handled.  The time period, the state size, the number of features, and the number of inputs.  These parameters go by the initials T, H, N, and I respectively in the literature and Apple documentation.  All of these values are derived from the first two parameters of the initializer.

The state size is explicitly given by the second parameter.  It will be used for weight, bias, and output sizes

The input shape gives the other three configuration values.  The input tensor must be of rank 3, of the shape \[T, N, I\].  It is applied in a 'for' loop to the RNN node across the time (T) variable.  Therefore if your input tensor shape is \[4, N, I\], four \[N, I\] tensors are sequentially sent into the RNN MPSGraph node, with the state values calculated and concatenated into the output tensor.

#### Outputs of an RNNLayer

The outputs of the RNNLayer can be configured with a modifier (and all internal MPSGraph node outputs can be referenced with the use of suffixes on the node name).  The following output tensors are available:

| Suffix                | Shape     | Description |
| --------------------- | --------- | ---------------------- |
| "_state"              |\[T, N, H\]| The rolled version of the state output for each time-step processed        |
| "_lastState"          | \[N, H\]  | The last state output produced        |

The output flags can be set with the setOutput modifier.  The five values to be set are as follows:

| Flag            | Description |
| ---------------------| ----------- |
|  createLastState     |  If true the last state tensor is created and output.  Default is true |
|  targetLoop          |  If true the loop output tensor (state) will be marked as a target if the node is marked as a target.  Default is false |
|  targetLast          |  If true the last time-step tensors (state, if created) will be marked as a target if the node is marked as a target.  Default is true |

####  Modifiers for a RNNLayer

The following modifiers are available for the RNNLayer node:

| Modifier            | Description |
| --------------------------------------------| ----------- |
|  activationFunction(_ activation)           |  Sets the activation function for the layer.  Default is tanh |
|  recurrentWeightInitialRange(min, max)      |  Sets the random value range for initializing the recurrent weights.  Default is -0.5 to 0.5   |
|  inputWeightInitialRange(min, max)          |  Sets the random value range for initializing the input weights.  Default is -0.5 to 0.5   |
|  biasInitialRange(min, max)                 |  Sets the random value range for initializing the bias values.  Default is -0.5 to 0.5   |
|  setOutput(createLastState, targetLoops, targetLasts)  |  Sets the output flags.  See above discussion   |
|  learnWithRespectTo(_ lossNode)             |  Sets the node to have the Variables learn with respect to the specified loss node       |

As with all nodes, the targetForModes modifier is available, but should be added after all other modifiers.  Only some tensors created by the RNNLayer node will be targetted when this modifier is used.  See the 'Tensors Added' section for more information.

####  Tensors Added by an RNNLayer

The following tensors are added to the Graph by the node:

| Suffix                | Targetted | Description |
| --------------------- | --------- | ---------------------- |
| "_recurrentWeights"   |     No     | The recurrent weights for the gate.  Shape \[H, H\]       |
| "_inputWeights"       |     No     | The input weights for the gate.  Shape \[H, I\]       |
| "_bias"               |     No     | The bias values for the gate.  Shape \[H\]       |
| "_state"              |  Can be    | The rolled state output tensor.  Shape \[T, N, H\]       |
| "_z"                  |     No     | If learning configured.  The rolled internal gate states.  Shape \[T, N, H\]       |
| "_lastStateSlice"     |     No     | If createLastState is true.  The result of the slice operation to extract last state from rolled state tensor.  Shape \[1, N, H\]|
| "_lastState"          |   Can be   | If createLastState is true.  The final state output.  Shape \[N, H\]     |

These tensors can be referenced using the name of the node with the suffix added.

### LSTMLayer

The LSTM layer implements the Long Short Term Memory layer.  Apple provides an LSTM tensor, but it requires you to handle all weights and bias variables, requires all configuration options to be set, and only outputs the fully rolled state and cell tensors.  This node takes care of all the variables, offers default configuration, and will extract the last state from the output tensor (which is usually the output wanted).

####  Creating an LSTMLayer

The following initializer is used in a Graph (or SubGraph) definition to add an LSTMLayer:

```swift
LSTMLayer(input: String? = nil, stateSize: Int, name: String? = nil)
```

The input tensor and node name are standard parameters.  See the article on Creating Graphs for more information

The LSTM layer has four parameters that define the size of data being handled.  The time period, the state size, the number of features, and the number of inputs.  These parameters go by the initials T, H, N, and I respectively in the literature and Apple documentation.  All of these values are derived from the first two parameters of the initializer.

The state size is explicitly given by the second parameter.  It will be used for weight, bias, and output sizes

The input shape gives the other three configuration values.  The input tensor must be of rank 3, of the shape \[T, N, I\].  It is applied in a 'for' loop to the LSTM node across the time (T) variable.  Therefore if your input tensor shape is \[4, N, I\], four \[N, I\] tensors are sequentially sent into the LSTM MPSGraph node, with the state and cell values calculated and concatenated into the output tensors.

#### Outputs of an LSTMLayer

The outputs of the LSTMLayer can be configured with a modifier (and all internal MPSGraph node outputs can be referenced with the use of suffixes on the node name).  The following output tensors are available:

| Suffix                | Shape     | Description |
| --------------------- | --------- | ---------------------- |
| "_state"              |\[T, N, H\]| The rolled version of the state output for each time-step processed        |
| "_cell"               |\[T, N, H\]| The rolled version of the cell output for each time-step processed          |
| "_lastState"          | \[N, H\]  | The last state output produced        |
| "_lastCell"           | \[N, H\]  | The last cell output produced        |

The output flags can be set with the setOutput modifier.  The five values to be set are as follows:

| Flag            | Description |
| ---------------------| ----------- |
|  produceCellOutput   |  If true the rolled cell tensor is created and output.  Default is false |
|  createLastState     |  If true the last state tensor is created and output.  Default is true |
|  createLastCell      |  If true the last state tensor is created and output.  If true 'produceCellOutput will also be set true.  Default is false |
|  targetLoops         |  If true the two loop output tensors (state, and cell if created) will be marked as targets if the node is marked as a target.  Default is false |
|  targetLasts         |  If true the two last time-step tensors (state and/or cell, if created) will be marked as targets if the node is marked as a target.  Default is true |

####  Modifiers for a LSTMLayer

The following modifiers are available for the LSTMLayer node:

| Modifier            | Description |
| --------------------------------------------| ----------- |
|  activationFunction(_ activation)           |  Sets the activation function (activation between cell and state).  Default is tanh |
|  cellGateActivationFunction(_ activation)   |  Sets the cell gate activation function.  Default is tanh  |
|  forgetGateActivationFunction(_ activation) |  Sets the forget gate activation function.  Default is sigmoid  |
|  inputGateActivationFunction(_ activation)  |  Sets the input gate activation function.  Default is sigmoid   |
|  outputGateActivationFunction(_ activation) |  Sets the output gate activation function.  Default is sigmoid   |
|  allActivationFunctions(_ activation)       |  Sets all five of the activation functions the same   |
|  recurrentWeightInitialRange(min, max)      |  Sets the random value range for initializing the recurrent weights.  Default is -0.5 to 0.5   |
|  inputWeightInitialRange(min, max)          |  Sets the random value range for initializing the input weights.  Default is -0.5 to 0.5   |
|  biasInitialRange(min, max)                 |  Sets the random value range for initializing the bias values.  Default is -0.5 to 0.5   |
|  setOutput(produceCellOutput, createLastState, createLastCell, targetLoops, targetLasts)  |  Sets the output flags.  See above discussion   |
|  learnWithRespectTo(_ lossNode)             |  Sets the node to have the Variables learn with respect to the specified loss node       |

As with all nodes, the targetForModes modifier is available, but should be added after all other modifiers.  Only some tensors created by the LSTMLayer node will be targetted when this modifier is used.  See the 'Tensors Added' section for more information.

####  Tensors Added by a LSTMLayer

The following tensors are added to the Graph by the node:

| Suffix                | Targetted | Description |
| --------------------- | --------- | ---------------------- |
| "_recurrentWeights"   |     No     | The recurrent weights for all four of the gates.  Shape \[4H, H\]       |
| "_inputWeights"       |     No     | The input weights for all four of the gates.  Shape \[4H, I\]       |
| "_bias"               |     No     | The bias values for all four of the gates.  Shape \[4H\]       |
| "_state"              |  Can be    | The rolled state output tensor.  Shape \[T, N, H\]       |
| "_cell"               |  Can be    | If produceCellOutput is true.  The rolled cell output tensor.  Shape \[T, N, H\]       |
| "_z"                  |     No     | If learning configured.  The rolled internal gate states.  Shape \[T, N, 4H\]       |
| "_lastStateSlice"     |     No     | If createLastState is true.  The result of the slice operation to extract last state from rolled state tensor.  Shape \[1, N, H\]|
| "_lastState"          |   Can be   | If createLastState is true.  The final state output.  Shape \[N, H\]     |
| "_lastCellSlice"      |     No     | If createLastCell is true.  The result of the slice operation to extract last cell from rolled cell tensor.  Shape \[1, N, H\]|
| "_lastCell"           |   Can be   | If createLastCell is true.  The final cell output.  Shape \[N, H\]     |

These tensors can be referenced using the name of the node with the suffix added.
