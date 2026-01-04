import Cocoa
import PlaygroundSupport
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

//: Import the framework
import MPSGraphDSL

//:  This text will be our data source.  Usually this will be in a file, but to avoid path issues in a tutorial we are making it in-line.  It comes from the Iris data set, which gives the sepal and petal sizes for three different Iris species
let textLines = """
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
5.4,3.9,1.7,0.4,setosa
4.6,3.4,1.4,0.3,setosa
5.0,3.4,1.5,0.2,setosa
4.4,2.9,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
4.8,3.4,1.6,0.2,setosa
4.8,3.0,1.4,0.1,setosa
4.3,3.0,1.1,0.1,setosa
5.8,4.0,1.2,0.2,setosa
5.7,4.4,1.5,0.4,setosa
5.4,3.9,1.3,0.4,setosa
5.1,3.5,1.4,0.3,setosa
5.7,3.8,1.7,0.3,setosa
5.1,3.8,1.5,0.3,setosa
5.4,3.4,1.7,0.2,setosa
5.1,3.7,1.5,0.4,setosa
4.6,3.6,1.0,0.2,setosa
5.1,3.3,1.7,0.5,setosa
4.8,3.4,1.9,0.2,setosa
5.0,3.0,1.6,0.2,setosa
5.0,3.4,1.6,0.4,setosa
5.2,3.5,1.5,0.2,setosa
5.2,3.4,1.4,0.2,setosa
4.7,3.2,1.6,0.2,setosa
4.8,3.1,1.6,0.2,setosa
5.4,3.4,1.5,0.4,setosa
5.2,4.1,1.5,0.1,setosa
5.5,4.2,1.4,0.2,setosa
4.9,3.1,1.5,0.1,setosa
5.0,3.2,1.2,0.2,setosa
5.5,3.5,1.3,0.2,setosa
4.9,3.1,1.5,0.1,setosa
4.4,3.0,1.3,0.2,setosa
5.1,3.4,1.5,0.2,setosa
5.0,3.5,1.3,0.3,setosa
4.5,2.3,1.3,0.3,setosa
4.4,3.2,1.3,0.2,setosa
5.0,3.5,1.6,0.6,setosa
5.1,3.8,1.9,0.4,setosa
4.8,3.0,1.4,0.3,setosa
5.1,3.8,1.6,0.2,setosa
4.6,3.2,1.4,0.2,setosa
5.3,3.7,1.5,0.2,setosa
5.0,3.3,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.3,3.3,4.7,1.6,versicolor
4.9,2.4,3.3,1.0,versicolor
6.6,2.9,4.6,1.3,versicolor
5.2,2.7,3.9,1.4,versicolor
5.0,2.0,3.5,1.0,versicolor
5.9,3.0,4.2,1.5,versicolor
6.0,2.2,4.0,1.0,versicolor
6.1,2.9,4.7,1.4,versicolor
5.6,2.9,3.6,1.3,versicolor
6.7,3.1,4.4,1.4,versicolor
5.6,3.0,4.5,1.5,versicolor
5.8,2.7,4.1,1.0,versicolor
6.2,2.2,4.5,1.5,versicolor
5.6,2.5,3.9,1.1,versicolor
5.9,3.2,4.8,1.8,versicolor
6.1,2.8,4.0,1.3,versicolor
6.3,2.5,4.9,1.5,versicolor
6.1,2.8,4.7,1.2,versicolor
6.4,2.9,4.3,1.3,versicolor
6.6,3.0,4.4,1.4,versicolor
6.8,2.8,4.8,1.4,versicolor
6.7,3.0,5.0,1.7,versicolor
6.0,2.9,4.5,1.5,versicolor
5.7,2.6,3.5,1.0,versicolor
5.5,2.4,3.8,1.1,versicolor
5.5,2.4,3.7,1.0,versicolor
5.8,2.7,3.9,1.2,versicolor
6.0,2.7,5.1,1.6,versicolor
5.4,3.0,4.5,1.5,versicolor
6.0,3.4,4.5,1.6,versicolor
6.7,3.1,4.7,1.5,versicolor
6.3,2.3,4.4,1.3,versicolor
5.6,3.0,4.1,1.3,versicolor
5.5,2.5,4.0,1.3,versicolor
5.5,2.6,4.4,1.2,versicolor
6.1,3.0,4.6,1.4,versicolor
5.8,2.6,4.0,1.2,versicolor
5.0,2.3,3.3,1.0,versicolor
5.6,2.7,4.2,1.3,versicolor
5.7,3.0,4.2,1.2,versicolor
5.7,2.9,4.2,1.3,versicolor
6.2,2.9,4.3,1.3,versicolor
5.1,2.5,3.0,1.1,versicolor
5.7,2.8,4.1,1.3,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
7.6,3.0,6.6,2.1,virginica
4.9,2.5,4.5,1.7,virginica
7.3,2.9,6.3,1.8,virginica
6.7,2.5,5.8,1.8,virginica
7.2,3.6,6.1,2.5,virginica
6.5,3.2,5.1,2.0,virginica
6.4,2.7,5.3,1.9,virginica
6.8,3.0,5.5,2.1,virginica
5.7,2.5,5.0,2.0,virginica
5.8,2.8,5.1,2.4,virginica
6.4,3.2,5.3,2.3,virginica
6.5,3.0,5.5,1.8,virginica
7.7,3.8,6.7,2.2,virginica
7.7,2.6,6.9,2.3,virginica
6.0,2.2,5.0,1.5,virginica
6.9,3.2,5.7,2.3,virginica
5.6,2.8,4.9,2.0,virginica
7.7,2.8,6.7,2.0,virginica
6.3,2.7,4.9,1.8,virginica
6.7,3.3,5.7,2.1,virginica
7.2,3.2,6.0,1.8,virginica
6.2,2.8,4.8,1.8,virginica
6.1,3.0,4.9,1.8,virginica
6.4,2.8,5.6,2.1,virginica
7.2,3.0,5.8,1.6,virginica
7.4,2.8,6.1,1.9,virginica
7.9,3.8,6.4,2.0,virginica
6.4,2.8,5.6,2.2,virginica
6.3,2.8,5.1,1.5,virginica
6.1,2.6,5.6,1.4,virginica
7.7,3.0,6.1,2.3,virginica
6.3,3.4,5.6,2.4,virginica
6.4,3.1,5.5,1.8,virginica
6.0,3.0,4.8,1.8,virginica
6.9,3.1,5.4,2.1,virginica
6.7,3.1,5.6,2.4,virginica
6.9,3.1,5.1,2.3,virginica
5.8,2.7,5.1,1.9,virginica
6.8,3.2,5.9,2.3,virginica
6.7,3.3,5.7,2.5,virginica
6.7,3.0,5.2,2.3,virginica
6.3,2.5,5.0,1.9,virginica
6.5,3.0,5.2,2.0,virginica
6.2,3.4,5.4,2.3,virginica
5.9,3.0,5.1,1.8,virginica
"""

//:  Next we create our data set.  It will have the four inputs values and three output values (one for each possible flower classification).  All values will be 32-bit floats
let inputShape = TensorShape([4])
let outputShape = TensorShape([3])
let dataSet = DataSet(inputShape: inputShape, inputType: .float32, outputShape: outputShape, outputType: .float32)

//:  The next step is to define the data structure in a parser.  Since this is comma seperated text we will use a DelineatedTextParser
//:  Both the input and output are in the same source here, but it is perfectly fine to pass the data set to two different parsers, one for the input file and another for a separate output file

//:  Create a delineated text parser (comma separated)
let parser = DelineatedTextParser(lineSeparator: .CommaSeparated) {
    InputFloatString()      //  Sepal length
    InputFloatString()      //  Sepal width
    InputFloatString()      //  Petal length
    InputFloatString()      //  Petal width
    OutputLabelString()     //  The text classification label
}.withCommentIndicators(["s", "#", "//"])

//:  Parse the data into the DataSet
try parser.parseTextLines(dataSet: dataSet, text: textLines)
print("Number of samples = \(dataSet.numSamples)")

//:  At this point, we now have 150 samples in a single DataSet.  We need to split this into two sets, a training set and a testing set.  We will do this randomly, with 30 random samples going into the testing set.  This function also has the effect of shuffling the samples, so all the samples of each Ibis type is not contiguous anymore
let splitSets = try dataSet.splitSetRandomly(secondSetCount: 30)
let trainingDataSet = splitSets.set1
let testDataSet = splitSets.set2

//  Time to create our Graph.  This will be a 2 layer fully-connected Neural Network with 10 nodes in the first layer and 3 nodes in the output layer (to match the classification output size)
let graph = Graph(buildOptions: [.addLoadAssigns, .addResetAssigns]) {
    //:  Placeholders are refernces the graph uses for an input.  When running data through the graph you pass a dictionary of Tensors with the PlaceHolder name as the keys
    PlaceHolder(shape: [4], name: "inputs")     //  This is placeholder for the inputs to the graph
    PlaceHolder(shape: [3], modes: ["learn"], name: "expectedValue")  //  This is a placeholder for the expected results - used when training the graph so only needed in mode "learn"
    FullyConnectedLayer(input: "inputs", outputShape: TensorShape([10]), activationFunction: .sigmoid, name: "hidden")  //  This is a fully connected NN layer with 10 nodes, using a sigmoid activation function.  It gets its data from the node 'inputs' - the placeholder above
        .learnWithRespectTo("loss")     //  This modifies the FullyConnectedLayer to update the weights and bias terms based on the loss function labeled 'loss' -  which is a calcution node near the end of the graph
    FullyConnectedLayer(input: "hidden", outputShape: TensorShape([3]), activationFunction: .sigmoid, name: "result")   //  Another fully connected NN layer, with 3 nodes this time
        .learnWithRespectTo("loss")     //  This NN layer also learns related to the loss function
    SoftMax(name: "inferenceResult")    //  This node takes the output of the previous node (note: no input name indicates you want to use output of the previous node as your input) and runs it through a SoftMax function
        .targetForModes(["infer"], )       //  This marks this node as the output for the Graph when run in mode 'infer'
    SoftMaxCrossEntropy(labels: "expectedValue", reductionType: .sum, name: "loss")     //  This node calculates the loss function using a SoftMax cross entropy function
        .targetForModes(["learn"])//  This marks this node as the output for the Graph when run in mode 'learn'
    Learning(constant: true, learningRate: 0.02, learningModes: ["learn"])  //  This node sets the learning parameters - a constant 0.01 learning rate, and learning is done when in mode 'learn'
}


//:  Get the initial accuracy by running the test data set through the graph.  Note the mode is 'infer', so the loss function is not calculated and learning is not performed
let result = try graph.runClassifierTest(mode: "infer", testDataSet: testDataSet, inputTensorName: "inputs", resultTensorName: "inferenceResult")
print("Initial test percentage: \(result.fractionCorrect*100.0)")

//:  Train the network by running through all the training data 500 times.  Note that the mode is 'learn', so the loss function is calculated and weights and biases updated
for _ in 0..<500 {
    try graph.runTraining(mode: "learn", trainingDataSet: trainingDataSet, inputTensorName: "inputs", expectedValueTensorName: "expectedValue")
}

//:  Get the final accuracy by running the test data set through again
let result2 = try graph.runClassifierTest(mode: "infer", testDataSet: testDataSet, inputTensorName: "inputs", resultTensorName: "inferenceResult")
print("final test percentage: \(result2.fractionCorrect*100.0)")


//:  Let's create an input Tensor for a post-learning inference run
let inputTensor = try TensorFloat32(shape: TensorShape([4]), initialValues: [6.0, 3.4, 4.6, 1.6])
let outputTensor = TensorFloat32(shape: TensorShape([3]), initialValue: 0.0)

//:  And run that input through the graph
var inputTensorList: [String: Tensor] = ["inputs": inputTensor]
inputTensorList["expectedValue"] = outputTensor
let results = try graph.runOne(mode: "infer", inputTensors: inputTensorList)

//:  The results are a dictionary keyed by the names of the nodes that are targets for the mode
let resultTensor = results["inferenceResult"]!

//  Get the classification, and the string label for the result
let resultClass = resultTensor.getClassification()
var classString = "Unknown"
if let string = trainingDataSet.getLabel(labelIndex: resultClass) {
    classString = string
}
print("Resulting Iris class is \(resultClass) - \(classString)")


//:  Get the variable data - this can be saved and reloaded into the graph later
let data = try graph.getVariableData()
print("byte count = \(data.count)")

//:  Reset the variables to their initial state
try graph.resetVariables()

//:  Get the post-reset accuracy by running the test data set through again
let result3 = try graph.runClassifierTest(mode: "infer", testDataSet: testDataSet, inputTensorName: "inputs", resultTensorName: "inferenceResult")
print("After reset percentage: \(result3.fractionCorrect*100.0)")

//:  Load the variables back to the post-learn state.  The data could have been saved to a file on a previous learning session, and read back to load the graph back to that state
try graph.loadVariables(fromData: data)

//:  Get the post-reset accuracy by running the test data set through again
let result4 = try graph.runClassifierTest(mode: "infer", testDataSet: testDataSet, inputTensorName: "inputs", resultTensorName: "inferenceResult")
print("After load percentage: \(result4.fractionCorrect*100.0)")
