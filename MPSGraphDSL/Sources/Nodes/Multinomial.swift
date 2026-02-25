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
///     The probability distribution doesn't need to sum to 1, but must sum to a smaller number (less than half of floating maximum divided by number of logits)
///     The node will not work as part of a learning path in the graph
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

        //  Get the number of logits and logit dimension from the last dimension of the probability tensor
        let probabilityShape = probabilities.shape
        if (probabilityShape == nil) { throw MPSGraphDSLErrors.InputShapeError("Unable to get number of logits from probability shape") }
        if (probabilityShape!.count < 0) { throw MPSGraphDSLErrors.InputShapeError("Unable to get logits dimension from probability shape") }
        let logitsDimension = probabilityShape!.count - 1
        let numberOfLogits = Int(truncating: probabilityShape![logitsDimension])
        
        //  Get the dimension of the output (probabilities minus logits dimension
        let outputShape: [NSNumber]
        if (logitsDimension > 0) {
            outputShape = probabilityShape!.dropLast()
        }
        else {
            outputShape = [1 as NSNumber]
        }
        
        //  Get the probability sum (it may not be 1)
        let probabilityTotal: MPSGraphTensor
        if (logitsDimension == 0) {
            probabilityTotal = graph.mpsgraph.reductionSum(with: probabilities, axis: logitsDimension, name: graph.getFullName(name)! + "_probabilityTotal")
            addedTensors.append(probabilityTotal)
            suffixes.append("_probabilityTotal")
        }
        else {
            //  If more than 1 dimension, reduction leaves [..., 1] at the end - remove that
            let probabilityTotalBeforeReshape = graph.mpsgraph.reductionSum(with: probabilities, axis: logitsDimension, name: graph.getFullName(name)! + "_probabilityTotal")
            addedTensors.append(probabilityTotalBeforeReshape)
            suffixes.append("_probabilityTotal")
            probabilityTotal = graph.mpsgraph.reshape(probabilityTotalBeforeReshape, shape: outputShape, name: graph.getFullName(name)! + "_probabilityReshape")
            addedTensors.append(probabilityTotal)
            suffixes.append("_probabilityReshape")
        }

        //  Get the seed for the random generator
        let randomSeed: Int
        if let seed = seed {
            randomSeed = seed
        }
        else {
            randomSeed = Int.random(in: 0...Int.max)
        }
        
        //  Get random numbers between zero and the total probability for each distribution
        let stateTensor = graph.mpsgraph.randomPhiloxStateTensor(withSeed: randomSeed, name: graph.getFullName(name)! + "_initial_state_tensor")
        addedTensors.append(stateTensor)
        suffixes.append("_initial_state_tensor")
        let stateVariable = graph.mpsgraph.variableFromTensor(stateTensor, name: graph.getFullName(name)! + "_random_state_variable")
        addedTensors.append(stateVariable)
        suffixes.append("_random_state_variable")
        let randomFrom0To1 = graph.mpsgraph.randomUniformTensor(withShape: outputShape, stateTensor: stateVariable, name: graph.getFullName(name)! + "_randomFrom0To1")
        addedTensors.append(randomFrom0To1[0])
        suffixes.append("_randomFrom0To1_value")
        addedTensors.append(randomFrom0To1[1])
        suffixes.append("_randomFrom0To1_updatedState")
        let randomReshape = graph.mpsgraph.reshape(randomFrom0To1[0], shape: outputShape, name: graph.getFullName(name)! + "_randomReshape")  //  Bug in randomUniformTensor making output shape [-1]
        addedTensors.append(randomReshape)
        suffixes.append("_randomReshape")
        let random = graph.mpsgraph.multiplication(probabilityTotal, randomReshape, name: graph.getFullName(name)! + "_random")
        addedTensors.append(random)
        suffixes.append("_random")
        
        //  Update the random state
        let randomStateAssign = graph.mpsgraph.assign(stateVariable, tensor: randomFrom0To1[1], name: graph.getFullName(name)! + "_random_state_assign")
        graph.nodeNonLearningOps.append(randomStateAssign)

        //  Get the size tensor for the slice of each discrete probability
        let sliceSize: MPSGraphTensor
        if (logitsDimension == 0) {
            sliceSize = graph.mpsgraph.constant(1.0, dataType: .int32)
            addedTensors.append(sliceSize)
            suffixes.append("_sliceSize")
        }
        else {
            var intSliceSize: [Int32] = probabilityShape!.map{ Int32(truncating: $0) }
            intSliceSize[logitsDimension] = 1
            let numDimensions = probabilityShape!.count
            let sizeData = Data(bytes: intSliceSize, count: numDimensions * MemoryLayout<Int32>.size)
            sliceSize = graph.mpsgraph.constant(sizeData, shape: [numDimensions as NSNumber], dataType: .int32)
            addedTensors.append(sliceSize)
            suffixes.append("_sliceSize")
        }
        
        //  Get a very low value that can be used to stop finding a second matching index (no 'break' capability in mpsgraph for loop)
        let lowBallValue: MPSGraphTensor
        if (probabilities.dataType == .float32) {
            lowBallValue = graph.mpsgraph.constant(Double(-Float32.greatestFiniteMagnitude), shape: [1 as NSNumber], dataType: .float32)
        }
        else {
            lowBallValue = graph.mpsgraph.constant(Double(-Float16.greatestFiniteMagnitude), shape: [1 as NSNumber], dataType: .float16)
        }
        addedTensors.append(lowBallValue)
        suffixes.append("_lowBallValue")

        //  Get the startIndex tensor if needed.  All zero's except the logits dimension a 1 - which will be multiplied by the discrete probability index
        let startIndexExceptLogits: MPSGraphTensor?
        if logitsDimension == 0 {
            startIndexExceptLogits = nil
        }
        else {
            var startInts: [Int32] = Array(repeating: 0, count: logitsDimension)
            startInts.append(1)
            let startData = Data(bytes: startInts, count: startInts.count * MemoryLayout<Int32>.size)
            startIndexExceptLogits = graph.mpsgraph.constant(startData, shape: [startInts.count as NSNumber], dataType: .int32)
            addedTensors.append(startIndexExceptLogits)
            suffixes.append("_startIndexExceptLogits")
        }

        //  For loop to find first index where the accumulated probability crosses the random value
        //  Iteration arguments:  0 - index tensor, 1, accumulated probability
        let numValuesTensor = graph.mpsgraph.constant(Double(numberOfLogits), dataType: .int32)
        addedTensors.append(numValuesTensor)
        suffixes.append("_numValuesTensor")
        let indexTensor = graph.mpsgraph.constant(0.0, shape: outputShape, dataType: .int32)
        addedTensors.append(indexTensor)
        suffixes.append("_indexTensor")
        let accumulatedProbability = graph.mpsgraph.constant(0.0, shape: outputShape, dataType: .float32)
        addedTensors.append(accumulatedProbability)
        suffixes.append("_accumulatedProbability")
        let namePrefix = graph.getFullName(name)!
        let forResult = graph.mpsgraph.for(numberOfIterations: numValuesTensor,
                               initialBodyArguments: [indexTensor, accumulatedProbability],
                                  body: {
            (index: MPSGraphTensor, iterationArguments: [MPSGraphTensor]) -> [MPSGraphTensor] in
                //  Get the start tensor
                let startTensor: MPSGraphTensor
                if (logitsDimension == 0) {
                    startTensor = index
                }
                else {
                    startTensor = graph.mpsgraph.multiplication(startIndexExceptLogits!, index, name: namePrefix + "_start_tensor")
                    addedTensors.append(startTensor)
                    self.suffixes.append("_start_tensor")
                }
                //  Select the probability based on the index  (probabilities[index])
                let probabilitySlice: MPSGraphTensor
                if (logitsDimension == 0) {
                    probabilitySlice = graph.mpsgraph.sliceTensor(probabilities, start: startTensor, sizeTensor: sliceSize, squeezeMask: 0, name: namePrefix + "_probability_slice")
                    addedTensors.append(probabilitySlice)
                    self.suffixes.append("_probability_slice")
                }
                else {
                    let probabilitySliceBeforeReshape = graph.mpsgraph.sliceTensor(probabilities, start: startTensor, sizeTensor: sliceSize, squeezeMask: 0, name: namePrefix + "_probability_slice")
                    addedTensors.append(probabilitySliceBeforeReshape)
                    self.suffixes.append("_probability_slice")
                    probabilitySlice = graph.mpsgraph.reshape(probabilitySliceBeforeReshape, shape: outputShape, name: namePrefix + "_probabilitySliceReshape")
                    addedTensors.append(probabilitySlice)
                    self.suffixes.append("_probabilitySliceReshape")
                }
                let accumulation = graph.mpsgraph.addition(iterationArguments[1], probabilitySlice, name: namePrefix + "_accumulation")
                addedTensors.append(accumulation)
                self.suffixes.append("_accumulation")
                let lessThan = graph.mpsgraph.lessThan(random, accumulation, name: "_less_than")
                addedTensors.append(lessThan)
                self.suffixes.append("_less_than")
                let newIndexes = graph.mpsgraph.select(predicate: lessThan, trueTensor: index, falseTensor: iterationArguments[0], name: namePrefix + "_index_select")
                addedTensors.append(newIndexes)
                self.suffixes.append("_index_select")
                let newAccumulations = graph.mpsgraph.select(predicate: lessThan, trueTensor: lowBallValue, falseTensor: accumulation, name: namePrefix + "_accumulation_select")
                addedTensors.append(newAccumulations)
                self.suffixes.append("_accumulation_select")
                return [newIndexes, newAccumulations]
        }, name: "for_loop")
        targetIndices = [addedTensors.count]    //  The final indices are the targetted tensor
        addedTensors.append(forResult[0])
        self.suffixes.append("")        //  just the name of the node
        addedTensors.append(forResult[1])
        self.suffixes.append("_final_accumulations")

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
