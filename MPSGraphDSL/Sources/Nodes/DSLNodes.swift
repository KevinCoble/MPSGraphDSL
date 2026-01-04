//
//  DSLNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/30/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///  Node that sets up the learning mode parameters.
///  The learning rate can be a constant, or come from a Placeholder (which can be set individually for each run operation)
public class Learning : Node {
    let constant: Bool
    let learningRate: Double
    let learningModes: [String]


    /// Construct a Learning node with the specified parameters
    /// - Parameters:
    ///   - constant: (Optional) If true the learning rate will be constant, set by the parameter of this node.  If variable, it will be initilized to this nodes value, but can be set on each run operation
    ///   - learningRate: (Optional) The learning rate value a constant learning rate, or the initial value for a variable learning rate
    ///   - learningModes: The run modes that should include a learning clause at the end of the graph
    ///   - name: (Optional) The name for this node and its associated tensor.  If not a constant learning rate, the name is required
    public init(constant: Bool = true, learningRate: Double = 0.05, learningModes: [String], name: String? = nil) {
        self.constant = constant
        self.learningRate = learningRate
        self.learningModes = learningModes
        super.init(name: name)
    }
    
    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Make sure we are the only Learning node
        if (graph.seenLearning) {
            throw MPSGraphDSLErrors.MoreThanOneLearningNode
        }
        graph.seenLearning = true
        
        graph.learningRateConstant = constant
        graph.learningRate = learningRate
        graph.learningModes = learningModes
        if constant {
            let constant = graph.mpsgraph.constant(learningRate, shape: [1 as NSNumber], dataType: .float32)
            graph.learningRateTensor = constant
            return [constant]
        }
        else {
            if (name == nil) { throw MPSGraphDSLErrors.VariableLearningNodeMustBeNamed }
            let inputPlaceholder = graph.mpsgraph.placeholder(shape: [1 as NSNumber], name: graph.getFullName(name))
            graph.learningRateTensor = inputPlaceholder
            return [inputPlaceholder]
        }
    }
}


///  Node that calculates the mean squared error for a loss function -> (actual - predicted)²
///    This node assumes the actual and expected tensors are one-dimensional
public class MeanSquaredErrorLoss: BinaryNode {
    /// Constructor for an Mean Squared Error loss calculations
    ///
    /// - Parameters:
    ///   - actual: (Optional) The name of the tensor that will provide the actual (usually from the training data set).  If nil the previous node's output will be used
    ///   - predicted: (Optional) The name of the tensor that will provide the predicted results from the graph.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(actual: String? = nil, predicted: String? = nil, name: String? = nil) {
        super.init(firstInput: actual, secondInput: predicted, name: name)
    }

    
    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input tensors
        let inputTensors = try graph.getBinaryTensors(firstInputName, secondInputName)
        
        //  Add to the graph itself
        let subtraction = graph.mpsgraph.subtraction(inputTensors.firstInputTensor, inputTensors.secondInputTensor, name: nil)
        let squareResult = graph.mpsgraph.square(with: subtraction, name: nil)
        let mean = graph.mpsgraph.mean(of: squareResult, axes: [0 as NSNumber], name: graph.getFullName(name))

        //  Return the created MPSGraphTensor
        return [mean]
    }
}
