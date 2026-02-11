# Neural Networks

While the MPSGraph system has all the pieces to make any neural network you can think of, assembling those pieces can take a good bit of work.  Apple made almost everything configurable so the number of parameters and options and pieces you have to assemble can make it a time-consuming task.  To alleviate this MPSGraphDSL has created composite nodes for neural networks.  They combine the network layer operations with the required variable declarations, tensor manipulation, default option handling, and more - allowing you to define a network in a few lines.

## Overview

The following composite nodes are available:
| Network Layer            | Description |
| ------------------------ | ----------- |
| ``MeanSquaredErrorLoss`` | The loss function using a mean-squared error calculation                                                     |
| ``FullyConnectedLayer``  | A standard fully-connected layer with optional bias term and activation function                             |
| ``RNNLayer``             | A standard one-gate recursivve neural newtwork layer.  Weights, biases and activation functions are managed  |
| ``LSTMLayer``            | The long short term memory layer, with all four gates.  Weights, biases and activation functions are managed |
| ``GRULayer`` (alpha!)    | The gated recursive unit layer, with all three gates.  Weights, biases and activation functions are managed |
| ``PoolingLayer``         | A pooling layer, with three possible pooling functions.  Tensor ranks are managed (MPSGraph requires 4D inputs)|
| ``ConvolutionLayer``     | A convolution layer, with two-dimensional or three-dimensional kernels.  Tensor ranks are managed (MPSGraph requires 4D or 5D inputs)|

In addition, topics in this article discuss how to use the Graph functions to test and train both classification and regression networks.

### MeanSquaredErrorLoss

The mean-squared error loss function takes the difference of the actual versus the predicted value, squares it and takes the mean of the values.  It is commonly used as the loss function for a regression problem

####  Creating a MeanSquaredErrorLoss

The following initializer is used in a Graph (or SubGraph) definition to add a MeanSquaredErrorLoss:

```swift
MeanSquaredErrorLoss(actual: String? = nil, predicted: String? = nil, name: String? = nil)
```

The actual and predicted inputs are names of tensors (or if nil it assumes the previous node's tensor) for the actual output (usually a ``PlaceHolder`` with the output tensor of a ``DataSet``) and the networks predicted output

### FullyConnectedLayer

The fully-connected layer is a standard neural network layer that takes the input tensor, multiplies it by the weight tensor, optionally adds a bias term to that result, then performs an activation function on the final output.  If needed, tensor reshape operations may be added to coerce tensors to the correct shape.

The weights for the layer can be initialized in a number of different ways, using uniform random values with a given range, gaussian normal values with given mean a standard deviation, or Xavier/Glorot or He parameters can be used with either of these methods.  The default method is based on the activation function being used.  If a ReLU or related activation function the initialization is done with a normal distribution using the He method parameters.  Any other activation function changes the default initialization method to a normal distribution using the Xavier/Glorot method parameters.  The initialization method and parameters used can set using a modifier on the layer.

Bias initial values, if a bias is to be used, can be initialized to a given value.  The value defaults to zero, but can be set using a modifier.

####  Creating a FullyConnectedLayer

The following initializer is used in a Graph (or SubGraph) definition to add a FullyConnectedLayer:

```swift
FullyConnectedLayer(input: String? = nil, outputShape: TensorShape, activationFunction: ActivationFunction, name: String)
```

The output shape determines the number of 'neurons' in the layer.  Rather than a flat array of them, it is possible to make a multi-dimensional tensor of them.  The result from the node will then be in the shape specified.

The activation function is selected from the enumeration ``ActivationFunction``.  To turn off activation, select the option '.none'.

The input tensor and node name are standard parameters.  See the article on Creating Graphs for more information.


####  Modifiers for a FullyConnectedLayer

The following modifiers are available for the FullyConnectedLayer node:

| Modifier            | Description |
| ----------------------------------| ----------- |
|  noBiasTerm()                     |  If added removes the bias Variable and the addition of it to the matrix multiplication  |
|  weightInitialization(initializerInfo:)   |  Sets the method and parameters for random initialization of the weights Variable       |
|  biasInitialValue(initialValue:)     |  Sets the initial value for initialization of the bias Variable                           |
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

The recursive neural network layer is a neural network layer that feeds the previous output back into itself as another input.  This allows a sequence in the input data to be learned.  Since the previous output is directly connected to the input, sequences that are long will have a hard time being learned due to the "exploding or vanishing gradient" problem.  LSTM and GRU layers have methods to address this.  Orthogonal matrices for the recurrent weights can also help, and are on by default for the RNNLayer.

The RNNLayer accepts time-ordered data as the input sequence.  An option on the layer turns it into two nodes where one takes the sequence in forward order, and the other in reverse order.  This doubles the number of parameters created by the layer.  The two nodes are combined using a sum operation to create the final output of the layer.  Use the 'makeBidirectional' modifier to turn this option on.

####  Creating an RNNLayer

The following initializer is used in a Graph (or SubGraph) definition to add an RNNLayer:

```swift
RNNLayer(input: String? = nil, stateSize: Int, name: String? = nil)
```

The RNN layer has four parameters that define the size of data being handled.  The time period, the state size, the number of features, and the number of inputs.  These parameters go by the initials T, H, N, and I respectively in the literature and Apple documentation.  All of these values are derived from the first two parameters of the initializer.

The state size is explicitly given by the second parameter.  It will be used for weight, bias, and output sizes

The input shape gives the other three configuration values.  The input tensor must be of rank 3, of the shape \[T, N, I\].  It is applied in a 'for' loop to the RNN node across the time (T) variable.  Therefore if your input tensor shape is \[4, N, I\], four \[N, I\] tensors are sequentially sent into the RNN MPSGraph node, with the state values calculated and concatenated into the output tensor.

#### Weight Initialization of an RNNLayer

Both the recurrent and input weights for the layer can be initialized in a number of different ways, using uniform random values with a given range, gaussian normal values with given mean a standard deviation, or Xavier/Glorot or He parameters can be used with either of these methods.  The default method is based on the activation function being used.  If a ReLU or related activation function the initialization for both weight sets is done with a normal distribution using the He method parameters.  Any other activation function changes the default initialization method to a normal distribution using the Xavier/Glorot method parameters.  The initialization method and parameters used can set using a modifier on the layer.

Since the recurrent weights are used repeatedly on the state value, it is especially susceptable to vanishing or exploding gradients.  One way to help avoid this is to have the recurrent weights be an orthonormal matrix.  This option is on by default, but can be set with a modifier.  The recurrent weight matrix is initialized like any other weight matrix, and then a Gram-Schmidt method is used to turn the random matrix to an orthonormal one if the option is on.

Bias initial values, if a bias is to be used, can be initialized to a given value.  The value defaults to zero, but can be set using a modifier.

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
|  recurrentWeightInitialization(initializerInfo:, orthogonalize:)   |  Sets the method and parameters for random initialization of the recurrent weights Variable       |
|  inputWeightInitialization(initializerInfo:)   |  Sets the method and parameters for random initialization of the input weights Variable       |
|  biasInitialValue(initialValue:)     |  Sets the initial value for initialization of the bias Variable                           |
|  makeBidirectional()     |  Makes the layer bidirectional                          |
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

These tensors can be referenced using the name of the node with the suffix added.  Shapes of the weights and bias nodes will be doubled in the first dimension when the node is set to be bidirectional.

### LSTMLayer

The LSTM layer implements the Long Short Term Memory layer.  Apple provides an LSTM tensor, but it requires you to handle all weights and bias variables, requires all configuration options to be set, and only outputs the fully rolled state and cell tensors.  This node takes care of all the variables, offers default configuration, and will extract the last state from the output tensor (which is usually the output wanted).

The LSTMLayer accepts time-ordered data as the input sequence.  An option on the layer turns it into two nodes where one takes the sequence in forward order, and the other in reverse order.  This doubles the number of parameters created by the layer.  The two nodes are combined using a sum operation to create the final output of the layer.  Use the 'makeBidirectional' modifier to turn this option on.

####  Creating an LSTMLayer

The following initializer is used in a Graph (or SubGraph) definition to add an LSTMLayer:

```swift
LSTMLayer(input: String? = nil, stateSize: Int, name: String? = nil)
```

The input tensor and node name are standard parameters.  See the article on Creating Graphs for more information

The LSTM layer has four parameters that define the size of data being handled.  The time period, the state size, the number of features, and the number of inputs.  These parameters go by the initials T, H, N, and I respectively in the literature and Apple documentation.  All of these values are derived from the first two parameters of the initializer.

The state size is explicitly given by the second parameter.  It will be used for weight, bias, and output sizes

The input shape gives the other three configuration values.  The input tensor must be of rank 3, of the shape \[T, N, I\].  It is applied in a 'for' loop to the LSTM node across the time (T) variable.  Therefore if your input tensor shape is \[4, N, I\], four \[N, I\] tensors are sequentially sent into the LSTM MPSGraph node, with the state and cell values calculated and concatenated into the output tensors.

#### Weight Initialization of an LSTMLayer

Both the recurrent and input weights for the layer can be initialized in a number of different ways, using uniform random values with a given range, gaussian normal values with given mean a standard deviation, or Xavier/Glorot or He parameters can be used with either of these methods.  The default method is based on the activation function being used.  If a ReLU or related activation function the initialization for both weight sets is done with a normal distribution using the He method parameters.  Any other activation function changes the default initialization method to a normal distribution using the Xavier/Glorot method parameters.  The initialization method and parameters used can set using a modifier on the layer.

Since the recurrent weights are used repeatedly on the state value, it is especially susceptable to vanishing or exploding gradients.  One way to help avoid this is to have the recurrent weights be an orthonormal matrix.  This option is on by default, but can be set with a modifier.  The recurrent weight matrix is initialized like any other weight matrix, and then a Gram-Schmidt method is used to turn the random matrix to an orthonormal one if the option is on.

Bias initial values, if a bias is to be used, can be initialized to a given value.  The value defaults to zero, but can be set using a modifier.

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
|  recurrentWeightInitialization(initializerInfo:, orthogonalize:)   |  Sets the method and parameters for random initialization of the recurrent weights Variable       |
|  inputWeightInitialization(initializerInfo:)   |  Sets the method and parameters for random initialization of the input weights Variable       |
|  biasInitialValue(initialValue:)     |  Sets the initial value for initialization of the bias Variable                           |
|  makeBidirectional()     |  Makes the layer bidirectional                          |
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

These tensors can be referenced using the name of the node with the suffix added.  Shapes of the weights and bias nodes will be doubled in the first dimension when the node is set to be bidirectional.

### GRULayer

The GRU layer implements the Gated Recurssive Unit layer.  Apple provides an GRU tensor, but it requires you to handle all weights and bias variables, requires all configuration options to be set, and only outputs the fully rolled state tensor.  This node takes care of all the variables, offers default configuration, and will extract the last state from the output tensor (which is usually the output wanted).

####  Creating an GRULayer

The following initializer is used in a Graph (or SubGraph) definition to add an GRULayer:

```swift
GRULayer(input: String? = nil, stateSize: Int, name: String? = nil)
```

The input tensor and node name are standard parameters.  See the article on Creating Graphs for more information

The GRU layer has four parameters that define the size of data being handled.  The time period, the state size, the number of features, and the number of inputs.  These parameters go by the initials T, H, N, and I respectively in the literature and Apple documentation.  All of these values are derived from the first two parameters of the initializer.

The state size is explicitly given by the second parameter.  It will be used for weight, bias, and output sizes

The input shape gives the other three configuration values.  The input tensor must be of rank 3, of the shape \[T, N, I\].  It is applied in a 'for' loop to the GRU node across the time (T) variable.  Therefore if your input tensor shape is \[4, N, I\], four \[N, I\] tensors are sequentially sent into the GRU MPSGraph node, with the state and cell values calculated and concatenated into the output tensors.

#### Weight Initialization of an GRULayer

Both the recurrent and input weights for the layer can be initialized in a number of different ways, using uniform random values with a given range, gaussian normal values with given mean a standard deviation, or Xavier/Glorot or He parameters can be used with either of these methods.  The default method is based on the activation function being used.  If a ReLU or related activation function the initialization for both weight sets is done with a normal distribution using the He method parameters.  Any other activation function changes the default initialization method to a normal distribution using the Xavier/Glorot method parameters.  The initialization method and parameters used can set using a modifier on the layer.

Since the recurrent weights are used repeatedly on the state value, it is especially susceptable to vanishing or exploding gradients.  One way to help avoid this is to have the recurrent weights be an orthonormal matrix.  This option is on by default, but can be set with a modifier.  The recurrent weight matrix is initialized like any other weight matrix, and then a Gram-Schmidt method is used to turn the random matrix to an orthonormal one if the option is on.

Bias initial values, if a bias is to be used, can be initialized to a given value.  The value defaults to zero, but can be set using a modifier.

#### Outputs of an GRULayer

The outputs of the GRULayer can be configured with a modifier (and all internal MPSGraph node outputs can be referenced with the use of suffixes on the node name).  The following output tensors are available:

| Suffix                | Shape     | Description |
| --------------------- | --------- | ---------------------- |
| "_state"              |\[T, N, H\]| The rolled version of the state output for each time-step processed        |
| "_lastState"          | \[N, H\]  | The last state output produced        |

The output flags can be set with the setOutput modifier.  The five values to be set are as follows:

| Flag            | Description |
| ---------------------| ----------- |
|  createLastState     |  If true the last state tensor is created and output.  Default is true |
|  targetLoops         |  If true the two loop output tensors (state, and cell if created) will be marked as targets if the node is marked as a target.  Default is false |
|  targetLasts         |  If true the two last time-step tensors (state and/or cell, if created) will be marked as targets if the node is marked as a target.  Default is true |

####  Modifiers for a GRULayer

The following modifiers are available for the GRULayer node:

| Modifier            | Description |
| --------------------------------------------| ----------- |
|  outputGateActivation(_ activation)         |  Sets the output gate activation function.  Default is tanh   |
|  resetGateActivation(_ activation)          |  Sets the reset gate activation function.  Default is sigmoid  |
|  updateGateActivation(_ activation)         |  Sets the update gate activation function.  Default is sigmoid   |
|  recurrentWeightInitialization(initializerInfo:, orthogonalize:)   |  Sets the method and parameters for random initialization of the recurrent weights Variable       |
|  inputWeightInitialization(initializerInfo:)   |  Sets the method and parameters for random initialization of the input weights Variable       |
|  biasInitialValue(initialValue:)     |  Sets the initial value for initialization of the bias Variable                           |
|  makeBidirectional()     |  Makes the layer bidirectional                          |
|  setOutput(createLastState, targetLoops, targetLasts)  |  Sets the output flags.  See above discussion   |
|  learnWithRespectTo(_ lossNode)             |  Sets the node to have the Variables learn with respect to the specified loss node       |

As with all nodes, the targetForModes modifier is available, but should be added after all other modifiers.  Only some tensors created by the GRULayer node will be targetted when this modifier is used.  See the 'Tensors Added' section for more information.

####  Tensors Added by a GRULayer

The following tensors are added to the Graph by the node:

| Suffix                | Targetted | Description |
| --------------------- | --------- | ---------------------- |
| "_recurrentWeights"   |     No     | The recurrent weights for all four of the gates.  Shape \[3H, H\]       |
| "_inputWeights"       |     No     | The input weights for all four of the gates.  Shape \[3H, I\]       |
| "_bias"               |     No     | The bias values for all four of the gates.  Shape \[3H\]       |
| "_state"              |  Can be    | The rolled state output tensor.  Shape \[T, N, H\]       |
| "_z"                  |     No     | If learning configured.  The rolled internal gate states.  Shape \[T, N, 3H\]       |
| "_lastStateSlice"     |     No     | If createLastState is true.  The result of the slice operation to extract last state from rolled state tensor.  Shape \[1, N, H\]|
| "_lastState"          |   Can be   | If createLastState is true.  The final state output.  Shape \[N, H\]     |

These tensors can be referenced using the name of the node with the suffix added.  Shapes of the weights and bias nodes will be doubled in the first dimension when the node is set to be bidirectional.

### PoolingLayer

The pooling layer implements the pooling operation for tensors between 2 and four dimensions.  The pooling operation can be done with either two or four dimensional gathering using one of three functions: averaging, maximum value, or minimum value.

####  Creating an PoolingLayer

The following initializer is used in a Graph (or SubGraph) definition to add an PoolingLayer:

```swift
PoolingLayer(input: String? = nil, function: PoolingFunction, kernelSize: [Int], strides: [Int]? = nil, name: String? = nil)
```

The input tensor and node name are standard parameters.  See the article on Creating Graphs for more information

The rank of the input tensor and the size of the kernel (and associated optional strides) are limited to the following combinations:

| Input Tensor Rank | # of Kernel sizes | Resulting Operation |
| --------------------- | --------- | ---------------------- |
| 2   |     2     | The tensor is assumed to be rows-columns, with width dimension of the kernel and X stride going along columns dimension       |
| 3   |     2     | The tensor is assumed to be rows-columns-channel.  If a batch graph or extraDimensionIsBatch modifier, it is assumed to be batch-rows-columns.  Width to columns, height to rows.       |
| 4   |     2     | The tensor is assumed to be batch-rows-columns-channel, with width of kernel and X stride going along columns dimension of each channel of each batch       |
| 4   |     4     | The tensor is pooled using the 4-dimensional kernel       |

The pooling function has three options, average, average with zeroes for padding, maximum, and minimum.

####  Output of a PoolingLayer

The output tensor of the pooling layer is the same rank as the input tensor, with pooled dimensions modified based on the settings.  A pooled dimension size will be reduced to '1 + ((length-1)/stride)', with the length of the input dimension and the pooling stride for the dimension being used in that equation.  

output size = floor((length+padding-kernelSize)/stride)+1.  If ceiling mode, replace floor with ceiling operation.  Padding is total of padding on both ends.


####  Modifiers for a PoolingLayer

The following modifiers are available for the PoolingLayer node:

| Modifier            | Description |
| --------------------------------------------| ----------- |
|  extraDimensionIsBatch        |  For rank 3 inputs with 2D kernel, specifies if tensor is NHW or HWC. |
|  HWPadding(bottomPadding, topPadding, leftPadding, rightPadding, paddingStyle)         |  Sets the padding distances for the height (row) and width (column) dimensions, along with the padding style.  Defaults are 0 and TF_SAME   |
|  NCPadding(nLowPadding, nHighPadding, cLowPadding, cHighPadding)         |  Sets the padding distances for the batch (dimension 0) and channel (dimension 3) dimensions.  Defaults are 0.  Only used with 4D kernels   |
|  dilationRates(dilationRateH, dilationRateW, dilationRateN, dilationRateC)         |  Sets the dilation rate (the spacing between kernel entries) for each dimension.  Only the first two dimensions are required, as the last two are only used with 4 dimensional kernels   |
|  setCeilingMode()         |  Turns on ceiling mode, where the output size is rounded up instead of down after dividing the dimension size by the stride   |

### ConvolutionLayer

The ConvolutionLayer passes a 2D or 3D kernel over a tensor (often assumed to be an image), multiplying kernel values by tensor values and summing the results.  A bias term can be applied to the summation and an activation layer can process the final result.  The kernel is then moved along the dimensions it was created for (the 2 or 3 dimensions), where the process is repeated for all positions, resulting in a tensor of the same dimension.  The convolution can do this with multiple kernels (filters) at once, as well as handling batch inputs.

MPSGraph requires a four dimensional input tensor to convoluted with a four dimensional weight matrix to do a two-dimensional convolution, and a five dimensional input tensor to convoluted with a five dimensional weight matrix to do a three-dimensional convolution.  MPSGraphDSLs' ConvolutionLayer tries to manage this mayhem and make inputs and outputs match more common formats like images with height, width, and maybe color channels.

####  Creating a ConvolutionLayer

The following initializer is used in a Graph (or SubGraph) definition to add a ConvolutionLayer with a 2-dimensional kernel:

```swift
ConvolutionLayer(input: String? = nil,
                        kernelHeight: Int, kernelWidth: Int,
                        numFilters: Int = 1, activationFunction: ActivationFunction = .none,
                        heightStride: Int = 1, widthStride: Int = 1,
                        name: String)
```

The following initializer is used in a Graph (or SubGraph) definition to add a ConvolutionLayer with a 3-dimensional kernel:

```swift
ConvolutionLayer(input: String? = nil,
                        kernelHeight: Int, kernelWidth: Int, kernelDepth : Int,
                        numFilters: Int = 1, activationFunction: ActivationFunction = .none,
                        heightStride: Int = 1, widthStride: Int = 1, depthStride: Int = 1,
                        name: String)
```

#### Inputs with a 2-Dimensional Kernel

A two-dimensional input tensor is assumed to be HW (row-column)  It can only be convoluted with a 2D (HW) kernel.  The result (with only one filter) will be a reduced HW (row-column) tensor.

With more than one filter there are multiple convolutions on HW input tensor with different HW kernel values.  Since it is likely you want the different HW results to be easily separable as their own planes, the ConvolutionLayer re-arranges the MPSGraph output to put the filter (O) dimension as the first dimension of the output - giving OHW.  This can be kept closer to the MPSGraph return format HWO with the leaveFilterDimensionLast modifier.

If you have a three dimensional input tensor the extra dimension can be either a batch or channels.  If the graph is built as a batch processing graph it is assumed to be a batch dimension at the beginning (NHW).  Otherwise it is assumed to be channels on the last dimension (HWC).  You can force it to be a batch dimension with the extraDimensionIsBatch modifier

If performing a 2-dimensional convolution on an input with channels (which is now 3-Dimensional), the weight tensor becomes three dimensional (with the third dimension being the size of the channel dimension), and the entire volume is convoluted into a single value on the 2-dimensional output.  No movement of the kernel (the weights) is performed in that third dimension during the operation.

If a batch dimension is present it is assumed to be the first dimension.  A batch dimension is just iterated over, creating a same-sized dimension on the output.

#### Inputs with a 3-Dimensional Kernel

If you enter a three-dimensional kernel (with depth), a three-dimensional input tensor is assumed to be DHW.  If the third dimension is a channel, and you wish to use it as the depth dimension, the useChannelAsDepth will swap the dimensions appropriately.

A four-dimensional input can get the extra dimension from a batch dimension or channels.  If the graph is built as a batch processing graph it is assumed to be a batch dimension at the beginning (NDHW).  Otherwise it is assumed to be channels on the last dimension (DWHC).  You can force it to be a batch dimension with the extraDimensionIsBatch modifier.  The useChannelAsDepth modifier will make the layer assume a four-dimensional input is NHWC, and swap dimensions as necessary to get the depth dimension of the kernel to line up with the channel dimension of the input.

A five-dimensional input is assumed to be NDHWC, with the channel dimension always being summed into the convolution volume.

#### Input/Output Summary
In the following table H and W are the height and width (row and column) dimensions, D is the depth dimension, N is the batch dimension, C is the channel dimension.  H', W', and D' are the output height, width, and depth dimensions, while O is the filter dimension. If the channel dimension is used for depth, the output is marked C'.

| Input            | Kernel  | Filters |  Output  |    Modifiers    |
| :---------------:|:-------:|:-------:|:--------:| :-------------- |
| HW               | 2D      | 1       |  H'W'    |    none    |
| HW               | 2D      | >1      |  OH'W'   |    none    |
| HW               | 2D      | >1      |  H'W'O   |    leaveFilterDimensionLast    |
| HWC              | 2D      | 1       |  H'W'    |    none    |
| HWC              | 2D      | >1      |  OH'W'   |    none    |
| HWC              | 2D      | >1      |  H'W'O   |    leaveFilterDimensionLast    |
| NHW              | 2D      | 1       |  NH'W'   |    extraDimensionIsBatch (or batch graph)    |
| NHW              | 2D      | >1      |  NOH'W'  |    extraDimensionIsBatch (or batch graph)    |
| NHW              | 2D      | >1      |  NH'W'O  |    extraDimensionIsBatch (or batch graph), leaveFilterDimensionLast    |
| NHWC             | 2D      | 1       |  NH'W'   |    none    |
| NHWC             | 2D      | >1      |  NOH'W'  |    none    |
| NHWC             | 2D      | >1      |  NH'W')  |    leaveFilterDimensionLast    |
| DHW              | 3D      | 1       |  D'H'W'  |    none    |
| HWC              | 3D      | 1       |  H'W'C'  |    useChannelAsDepth    |
| DHW              | 3D      | >1      |  OD'H'W' |    none    |
| DHW              | 3D      | >1      |  D'H'W'O |    leaveFilterDimensionLast    |
| HWC              | 3D      | >1      |  OH'W'C' |    useChannelAsDepth    |
| HWC              | 3D      | >1      |  H'W'C'O |    useChannelAsDepth, leaveFilterDimensionLast    |
| NDHW             | 3D      | 1       | ND'H'W'  |    extraDimensionIsBatch (or batch graph)    |
| NHWC             | 3D      | 1       | NH'W'C'  |    extraDimensionIsBatch (or batch graph), useChannelAsDepth    |
| NDHW             | 3D      | >1      | NOD'H'W' |    extraDimensionIsBatch (or batch graph)    |
| NDHW             | 3D      | >1      | ND'H'W'O |    extraDimensionIsBatch (or batch graph), leaveFilterDimensionLast    |
| NHWC             | 3D      | >1      | NOH'W'C' |    extraDimensionIsBatch (or batch graph), useChannelAsDepth    |
| NHWC             | 3D      | >1      | NH'W'C'O |    extraDimensionIsBatch (or batch graph), useChannelAsDepth, leaveFilterDimensionLast    |
| NDHWC            | 3D      | 1       | ND'H'W'  |    extraDimensionIsBatch (or batch graph)    |
| NDHWC            | 3D      | >1      | NOD'H'W' |    extraDimensionIsBatch (or batch graph)    |
| NDHWC            | 3D      | >1      | ND'H'W'O |    extraDimensionIsBatch (or batch graph), leaveFilterDimensionLast    |

####  Modifiers for a ConvolutionLayer

The following modifiers are available for the ConvolutionLayer node:

| Modifier            | Description |
| --------------------------------------------| ----------- |
|  extraDimensionIsBatch        |  For rank 3 inputs with 2D kernel, specifies if tensor is NHW or HWC. For rank 4 inputs with 3D kernel, specifies if tensor is NDHW or DHWC |
|  leaveFilterDimensionLast     | When numFilters>1 output starts with filter dimension last.  If this modifier not used, filter dimension moved to beginning (after batch if present) |
|  useChannelAsDepth     | When using a 3D kernel, this modifier will move the channel dimension to the depth dimension.  i.e.  HWC is changed to DHW (results swap dimension back) |
|  noBiasTerm     | Turns off the bias term.  The bias variable will not be created. |
|  weightInitialRange(min:, max:)   |  Sets the range for random initialization of the weights Variable                        |
|  biasInitialRange(min:, max:)     |  Sets the range for random initialization of the bias Variable                           |
|  padding(bottomPadding:, topPadding:, leftPadding:, rightPadding:, backPadding:, frontPadding:, paddingStyle:)   |  Sets the padding parameters      |
|  dilationRates(dilationRateH:, dilationRateW:, dilationRateD:)   |  Sets the dilation rates for all dimensions of the kernel      |
|  learnWithRespectTo(_ lossNode)   |  Sets the node to have the Variables learn with respect to the specified loss node       |

####  Tensors Added by a ConvolutionLayer

The following tensors may be added to the Graph by the node:

| Suffix (if used)      | When Added | Description |
| --------------------- | ---------- | ---------------------- |
| "_biases"             |     bias not turned off     | The bias variable.  Shape \[numFilters\]       |
| "_inputReshape"       |     When input doesn't match     | The reshape of the input tensor to match the convolution operation      |
| "_inputPermutation"   |     if channelAsDepth     | Move channel dimension to depth dimension      |
| "_weights"            |     Always     | The weight variable for the kernel.  Shape \[H, W, C, numFilters\] or  \[D, H, W, C, numFilters\]     |
| "_convolution"        |     Always     | The actual convolution operation       |
| "_biasAddition"       |     bias not turned off      | Addition of the bias to the convolution result       |
| "_outputReshape"      |     If output needs reshaping     | Reshape the output to smallest rank based on the input|
| "_outputPermutation"  |     If >1 filter and not last or channelAsDepth   | Move dimensions to match standard output formats     |
| "" (none)             |     If activation function not .none  | The activation function tensor.  Always last so never has a suffix    |

These tensors can be referenced using the name of the node with the suffix added.  The last tensor will always be named with just the given (required) node name with no suffix.  This allows the node output to be referenced without regard to options

Only the last tensor will become a target if the node is configured to be a target.
