//
//  Multinomial.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 2/23/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node for sampling a discrete probability distribution
///     The individual probabilitie values cannot be negative.
///     The probability distribution doesn't need to sum to 1, but must sum to a smaller number (less than half of floating maximum divided by number of logits)
///
public class Multinomial : UnaryNode {
    let seed: Int?

    var suffixes: [String] = []
    var targetIndices: [Int] = []

    /// Create a multinomial node to sample from a probability distribution of logits
    /// - Parameters:
    ///   - probabilities: (Optional).  The name of the tensor providing the probability distributions.  If nil the previous node's output will be used.  The logits are assumed to be on the last dimension, and the size of that dimension being the number of discreet probabilities in the distribution.
    ///   - randomSeed: (Optional).  The seed for the random generator.  If nil a random seed will be generated each time the graph is built.
    ///   - name: The name for this node and its associated tensor.  The sub-nodes will be named &ltname&gt_&ltsuffix&gt
    public init(probabilities: String? = nil, randomSeed: Int? = nil, name: String) {
        self.seed = randomSeed
        super.init(input: probabilities, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        
        //  Get the input tensor
        let probabilities = try graph.getUnaryTensor(name: inputName)
        
        //  If the probabilities are of rank 1, reshape to a [1,x] tensor
        var probabilityShape = probabilities.shape
        if (probabilityShape == nil) { throw MPSGraphDSLErrors.InputShapeError("Unable to get number of logits from probability shape") }
        if (probabilityShape!.count < 0) { throw MPSGraphDSLErrors.InputShapeError("Unable to get logits dimension from probability shape") }
        let shapedProbabilities: MPSGraphTensor
        if (probabilityShape!.count == 1) {
            probabilityShape = [1 as NSNumber, probabilityShape![0]]
            shapedProbabilities = graph.mpsgraph.reshape(probabilities, shape: probabilityShape!, name: graph.getFullName(name)! + "_reshapedProbabilities")
            addedTensors.append(shapedProbabilities)
            suffixes.append("_reshapedProbabilities")
        }
        else {
            shapedProbabilities = probabilities
        }

        //  Get the number of logits and logit dimension from the last dimension of the probability tensor
        let logitsDimension = probabilityShape!.count - 1
        
        //  Get the dimension of the output (probabilities minus logits dimension
        let outputShape = Array(probabilityShape!.dropLast())
         
        //  Get the probability sum (it may not be 1)
        //  If more than 1 dimension, reduction leaves [..., 1] at the end - remove that
        let probabilityTotal = graph.mpsgraph.reductionSum(with: shapedProbabilities, axis: logitsDimension, name: graph.getFullName(name)! + "_probabilityTotal")
        addedTensors.append(probabilityTotal)
        suffixes.append("_probabilityTotal")
        
        //  Get the cumulative sum across the logits dimension
        let cumulativeProbabilities = graph.mpsgraph.cumulativeSum(shapedProbabilities, axis: logitsDimension, name: graph.getFullName(name)! + "_cumulativeProbabilities")
        addedTensors.append(cumulativeProbabilities)
        suffixes.append("_cumulativeProbabilities")

        //  Get the seed for the random generator
        let randomSeed: Int
        if let seed = seed {
            randomSeed = seed
        }
        else {
            randomSeed = Int.random(in: 0...Int.max)
        }
        
        //  Tried the following with the variable initialized inline:
//        let stateTensor = graph.mpsgraph.randomPhiloxStateTensor(withSeed: randomSeed, name: graph.getFullName(name)! + "_initial_state_tensor")
//        let stateVariable = graph.mpsgraph.variableFromTensor(stateTensor, name: graph.getFullName(name)! + "_random_state_variable")
        
        //  Now initializing outside of graph - with a small graph!
        let randomGraph = MPSGraph()
        let randomPhiloxStateTensor = randomGraph.randomPhiloxStateTensor(withSeed: randomSeed, name: "philox_state_tensor")
        let randomStateResults = randomGraph.run(feeds: [:], targetTensors: [randomPhiloxStateTensor], targetOperations: nil)
        let randomStateResult = randomStateResults[randomPhiloxStateTensor]!
        let randomStateTensor = TensorInt32(fromMPSTensorData: randomStateResult)
        
        let stateVariable = graph.mpsgraph.variable(with: randomStateTensor.getData(), shape: [7], dataType: .int32, name: graph.getFullName(name)! + "_random_state_variable")
        addedTensors.append(stateVariable)
        suffixes.append("_random_state_variable")

        //  Get random numbers between zero and the total probability for each distribution
        let randomFrom0To1 = graph.mpsgraph.randomUniformTensor(withShape: outputShape, stateTensor: stateVariable, name: graph.getFullName(name)! + "_randomFrom0To1")
        addedTensors.append(randomFrom0To1[0])
        suffixes.append("_randomFrom0To1_value")
        addedTensors.append(randomFrom0To1[1])
        suffixes.append("_randomFrom0To1_updatedState")
        let outputShapeWithOneAtEnd = outputShape + [NSNumber(value: 1)]
        let randomReshape = graph.mpsgraph.reshape(randomFrom0To1[0], shape: outputShapeWithOneAtEnd, name: graph.getFullName(name)! + "_randomReshape")  //  Bug in randomUniformTensor making output shape [-1]
        addedTensors.append(randomReshape)
        suffixes.append("_randomReshape")
        let random = graph.mpsgraph.multiplication(randomReshape, probabilityTotal, name: graph.getFullName(name)! + "_random")
        addedTensors.append(random)
        suffixes.append("_random")
        
        //  Update the random state
        let randomStateAssign = graph.mpsgraph.assign(stateVariable, tensor: randomFrom0To1[1], name: graph.getFullName(name)! + "_random_state_assign")
        graph.nodeNonLearningOps.append(randomStateAssign)
        graph.nodeLearningOps.append(randomStateAssign)

        //  Subtract the random values from the cumulative probabilities
        let subtraction = graph.mpsgraph.subtraction(cumulativeProbabilities, random, name: graph.getFullName(name)! + "_subtraction")
        addedTensors.append(subtraction)
        suffixes.append("_subtraction")
        
        //  Get the zero tensor
        let zero = graph.getZeroTensor(type: DataType(from: subtraction.dataType))

        //  Find the values that are positive
        let positiveValues = graph.mpsgraph.greaterThan(subtraction, zero, name: graph.getFullName(name)! + "_positiveValues")
        addedTensors.append(positiveValues)
        suffixes.append("_positiveValues")
        
        //  Get a minus-one tensor
        let minusOne = graph.mpsgraph.constant(-1.0, dataType: cumulativeProbabilities.dataType)
        
        //  Select one or zero based on the positive flag (1 if positive)
        let selection = graph.mpsgraph.select(predicate: positiveValues, trueTensor: minusOne, falseTensor: zero, name: graph.getFullName(name)! + "_selection")
        addedTensors.append(selection)
        suffixes.append("_selection")
        
        //  Reduce to the argmin
        let argMin = graph.mpsgraph.reductionArgMinimum(with: selection, axis: logitsDimension, name: graph.getFullName(name)! + "_argMin")
        addedTensors.append(argMin)
        suffixes.append("_argMin")
        
        //  Reshape to the output shape
        let finalReshape = graph.mpsgraph.reshape(argMin, shape: outputShape, name: graph.getFullName(name)!)
        addedTensors.append(finalReshape)
        suffixes.append("")

        targetIndices = [addedTensors.count-1]    //  The final indices are the targetted tensor
        return addedTensors
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }
}
