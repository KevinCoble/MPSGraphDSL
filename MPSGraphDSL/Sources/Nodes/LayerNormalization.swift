//
//  LayerNormalization.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 3/6/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node for a layer normalization layer, including optional activation function
///
///   LayerNormalization does not calculate and store a running mean and variance of the inputs during training.  Both forward and backward passes use the mean and variance of the current inputs
///
public class LayerNormalization : UnaryNode {
    let normalizedDimensionCount: Int
    let ϵ: Double
    let momentum: Double
    let activationFunction: ActivationFunction
    
    var lossNode: String? = nil
    var gammaLearningOptions: LearningOptions = LearningOptions(clipping: nil, optimizer: .stochasticGradientDescent)
    var betaLearningOptions: LearningOptions = LearningOptions(clipping: nil, optimizer: .stochasticGradientDescent)

    var suffixes: [String] = []
    var targetIndices: [Int] = []
    
    var totalParameterCount: Int = 0

    ///  Constructor for a layer normalization node
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - normalizedDimensionCount: (Optional) The number of trailing dimensions of the input tensor shape that will be normalized.  Defaults to 1
    ///   - ϵ:  (Optional) The small noise parameter used to prevent divide-by-zero errors.  The default is 1.0e-5
    ///   - momentum:  (Optional) The momentum value used to calculate the running mean.  The default is 0.9
    ///   - activationFunction: (Optional) The activation function for the layer.  Defaults to none
    ///   - name: The name for this node and its associated tensor.  The variable and other calculation nodes created will start with this name
    public init(input: String? = nil, normalizedDimensionCount: Int = 1, ϵ: Double = 1.0e-5, momentum: Double = 0.9, activationFunction: ActivationFunction = .none, name: String) {
        self.normalizedDimensionCount = normalizedDimensionCount
        self.ϵ = ϵ
        self.momentum = momentum
        self.activationFunction = activationFunction
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        totalParameterCount = 0
        
        //  See if we have an activation function
        let haveActivationFunction: Bool
        switch (activationFunction) {
        case .none:
            haveActivationFunction = false
        default:
            haveActivationFunction = true
        }
        
        //  Get the input tensor
        let inputTensor = try graph.getUnaryTensor(name: inputName)
        var inputShape: TensorShape
        if let shape = inputTensor.shape {
            inputShape = TensorShape(fromMPS: shape)
        }
        else {
            throw GenericMPSGraphDSLErrors.UnknownShape
        }
        let weightType = DataType(from: inputTensor.dataType)
        
        //  Get the shape and axes for the mean/variance calculation
        if (normalizedDimensionCount > inputShape.numDimensions) { throw GenericMPSGraphDSLErrors.InvalidShape }
        var normalizedShape: [Int] = []
        var meanVarianceAxes: [Int] = []
        let startDimension = inputShape.numDimensions - normalizedDimensionCount
        for i in startDimension..<inputShape.numDimensions {
            normalizedShape.append(inputShape.dimensions[i])
            meanVarianceAxes.append(i)
        }
        let shape = TensorShape(normalizedShape)

        //  Create the gamma variable, initialized to ones
        let gammas = CreateTensor.constantValues(type: weightType, shape: shape, initialValue: 1.0)
        let gammaData = gammas.getData()
        let gammaName = graph.getFullName(name)! + "_gamma"
        let gammaTensor = graph.mpsgraph.variable(with: gammaData, shape: shape.getMPSShape(), dataType: weightType.getMPSDataType(), name: gammaName)
        suffixes.append("_gamma")
        addedTensors.append(gammaTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        var node = Variable(dataType: weightType, shape: shape, initialValue: 1.0, name: gammaName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: gammaTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  If this is a learning layer - add the weights to the list to get assignment operations for
        if let lossNode = lossNode {
            let learningVariable = LearningVariable(variable: node, tensor: gammaTensor, loss: lossNode, learningOptions: gammaLearningOptions)
            graph.learningVariables.append(learningVariable)
            totalParameterCount += shape.totalSize
        }
        
        //  Create the beta variable, initialized to zeros
        let betas = CreateTensor.constantValues(type: weightType, shape: shape, initialValue: 0.0)
        let betaData = betas.getData()
        let betaName = graph.getFullName(name)! + "_beta"
        let betaTensor = graph.mpsgraph.variable(with: betaData, shape: shape.getMPSShape(), dataType: weightType.getMPSDataType(), name: betaName)
        suffixes.append("_beta")
        addedTensors.append(betaTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        node = Variable(dataType: weightType, shape: shape, initialValue: 0.0, name: betaName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: betaTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  If this is a learning layer - add the weights to the list to get assignment operations for
        if let lossNode = lossNode {
            let learningVariable = LearningVariable(variable: node, tensor: betaTensor, loss: lossNode, learningOptions: betaLearningOptions)
            graph.learningVariables.append(learningVariable)
            totalParameterCount += shape.totalSize
        }

        //  Create the ϵ and momentum constants
        let ϵTensor = graph.mpsgraph.constant(ϵ, shape: [1 as NSNumber], dataType: weightType.getMPSDataType())
        suffixes.append("_ϵ")
        addedTensors.append(ϵTensor)
        let momentumTensor = graph.mpsgraph.constant(momentum, shape: [1 as NSNumber], dataType: weightType.getMPSDataType())
        suffixes.append("_momentum")
        addedTensors.append(momentumTensor)
        
        //  Calculate mean a variance from the batch input
        let mean = graph.mpsgraph.mean(of: inputTensor, axes: meanVarianceAxes.map{NSNumber(value: $0)}, name: graph.getFullName(name)! + "_mean")
        suffixes.append("_mean")
        addedTensors.append(mean)
        let variance = graph.mpsgraph.variance(of: inputTensor, mean: mean, axes: meanVarianceAxes.map{NSNumber(value: $0)}, name: graph.getFullName(name)! + "_variance")
        suffixes.append("_variance")
        addedTensors.append(variance)
        
        //  Numerator is inputs minus mean
        let train_numerator = graph.mpsgraph.subtraction(inputTensor, mean, name: graph.getFullName(name)! + "_numerator")
        suffixes.append("_numerator")
        addedTensors.append(train_numerator)
        
        //  Denominator is sqrt of variance + ϵ
        let variance_plus_ϵ = graph.mpsgraph.addition(variance, ϵTensor, name: graph.getFullName(name)! + "_variance_plus_epsilon")
        suffixes.append("_variance_plus_epsilon")
        addedTensors.append(variance_plus_ϵ)
        let train_denominator = graph.mpsgraph.squareRoot(with: variance_plus_ϵ, name: graph.getFullName(name)! + "_denominator")
        suffixes.append("_denominator")
        addedTensors.append(train_denominator)
        
        //  X_hat is numerator over denominator
        let x_hat = graph.mpsgraph.division(train_numerator, train_denominator, name: graph.getFullName(name)! + "_x_hat")
        suffixes.append("_x_hat")
        addedTensors.append(x_hat)

        //  Result is gamma * X_hat + beta
        let xhat_times_gamma = graph.mpsgraph.multiplication(x_hat, gammaTensor, name: graph.getFullName(name)! + "_gamma_times_x_hat")
        suffixes.append("_gamma_times_x_hat")
        addedTensors.append(xhat_times_gamma)
        var name = graph.getFullName(self.name)!
        if (haveActivationFunction) {
            name += "_layerNormalization"
            suffixes.append("_layerNormalization")
        }
        else {
            suffixes.append("")
        }
        let result = graph.mpsgraph.addition(xhat_times_gamma, betaTensor, name: name)
        targetIndices.append(addedTensors.count)
        addedTensors.append(result)
        
        //  Add any activation function
        if (haveActivationFunction) {
            if let activationTensor = try activationFunction.addActivation(graph: graph, inputTensor: result, name: graph.getFullName(name)!) {
                addedTensors.append(activationTensor)
                targetIndices.append(suffixes.count)
                suffixes.append("")
            }
        }
        
        return addedTensors
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return targetIndices
    }

    /// Modifier to configure the layer's variables to learn
    /// - Parameters:
    ///   - mode: lossNode: the name of the loss calculation in the Graph
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String) -> LayerNormalization {
        self.lossNode = lossNode
        return self
    }
    
    /// Modifier to set the optimizer used for learning the gamma variable
    /// - Parameter optimizer: the optimizer method to use for learning the gammas.  Defaults to stochastic gradient descent
    /// - Returns: The modified layer
    public func gammaOptimizer(_ optimizer: LearningOptimizer) -> LayerNormalization {
        gammaLearningOptions = LearningOptions(clipping: gammaLearningOptions.clipping, optimizer: optimizer)
        return self
    }
    
    /// Modifier to set all the learning options for the gamma variable
    /// - Parameter options: The LearningOptions structure with all the learning options
    /// - Returns: The modified layer
    public func gammaLearningOptions(_ options: LearningOptions) -> LayerNormalization {
        gammaLearningOptions = options
        return self
    }
    
    /// Modifier to set the optimizer used for learning the beta variable
    /// - Parameter optimizer: the optimizer method to use for learning the beta.  Defaults to stochastic gradient descent
    /// - Returns: The modified layer
    public func betaOptimizer(_ optimizer: LearningOptimizer) -> LayerNormalization {
        betaLearningOptions = LearningOptions(clipping: betaLearningOptions.clipping, optimizer: optimizer)
        return self
    }
    
    /// Modifier to set all the learning options for the beta variable
    /// - Parameter options: The LearningOptions structure with all the learning options
    /// - Returns: The modified layer
    public func betaLearningOptions(_ options: LearningOptions) -> LayerNormalization {
        betaLearningOptions = options
        return self
    }

    override func getNumberOfParameters() throws -> Int {
        return totalParameterCount
    }
}
