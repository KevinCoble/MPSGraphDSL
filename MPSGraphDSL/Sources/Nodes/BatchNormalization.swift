//
//  BatchNormalization.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 2/12/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node for a batch normalization layer, including optional activation function
///
public class BatchNormalization : UnaryNode {
    let featureDimensions: [Int]?
    let ϵ: Double
    let momentum: Double
    let activationFunction: ActivationFunction
    
    var lossNode: String? = nil
    var learningOptimizer: LearningOptimizer = .stochasticGradientDescent
    var gradientClipping: (min: Double, max: Double)? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []

    ///  Constructor for a batch normalization node
    ///
    ///  If a batch graph the first dimension is assumed to be the batch dimension, and will be normalized across.
    ///  If no featureDimensions are provided, all other non-batch dimensions are assumed to be feature dimensions and are not included in the averaging, etc.
    ///  If featureDimensions are provided, all other dimensions are used in the calculations to get the normalization parameters.
    ///  If the previous layer is a ``FullyConnectedLayer``, you usually want to have all dimensions except the batch dimension as features, so do not provide featureDimensions
    ///  If the previous layer is a ``ConvolutionLayer``, you usually want to not have the row-column dimensions as feature dimensions, so provide the index of the feature dimension
    ///
    /// - Parameters:
    ///   - input: (Optional) The name of the tensor that will provide the input operand.  If nil the previous node's output will be used
    ///   - ϵ:  (Optional) The small noise parameter used to prevent divide-by-zero errors.  The default is 1.0e-5
    ///   - momentum:  (Optional) The momentum value used to calculate the running mean.  The default is 0.9
    ///   - featureDimensions: (Optional) The dimension indices of the input tensor that will be normalized over.  See description for more details.  Defaults to nil
    ///   - activationFunction: (Optional) The activation function for the layer.  Defaults to none
    ///   - name: The name for this node and its associated tensor.  The variable and other calculation nodes created will start with this name
    public init(input: String? = nil, ϵ: Double = 1.0e-5, momentum: Double = 0.9, featureDimensions: [Int]? = nil, activationFunction: ActivationFunction = .none, name: String) {
        self.featureDimensions = featureDimensions
        self.ϵ = ϵ
        self.momentum = momentum
        self.activationFunction = activationFunction
        super.init(input: input, name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        var addedTensors: [MPSGraphTensor?] = []
        
        suffixes = []
        targetIndices = []
        
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

        //  Get the shape of the variables and the axes for the mean/variance calculation
        var normalizedShape = inputShape.dimensions
        var meanVarianceAxes: [Int] = []
        if (graph.batchGraph) {
            //  batch dimension is not features
            normalizedShape[0] = 1
            meanVarianceAxes.append(0)
        }
        if let featureDimensions = featureDimensions {
            for dim in 0..<normalizedShape.count {
                if (!featureDimensions.contains(dim)) {
                    normalizedShape[dim] = 1
                    meanVarianceAxes.append(dim)
                }
            }
        }
        let shape = TensorShape(normalizedShape)
        
        //  Get the training mode tensor
        let trainingModeTensor = graph.getTrainingModeTensor()

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
            let learningVariable = LearningVariable(variable: node, tensor: gammaTensor, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
            graph.learningVariables.append(learningVariable)
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
            let learningVariable = LearningVariable(variable: node, tensor: betaTensor, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
            graph.learningVariables.append(learningVariable)
        }
        
        //  Create the running mean variable, initialized to zeros
        let means = CreateTensor.constantValues(type: weightType, shape: shape, initialValue: 0.0)
        let meanData = means.getData()
        let meanName = graph.getFullName(name)! + "_runningMean"
        let runningMeanTensor = graph.mpsgraph.variable(with: meanData, shape: shape.getMPSShape(), dataType: weightType.getMPSDataType(), name: meanName)
        suffixes.append("_runningMean")
        addedTensors.append(runningMeanTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        node = Variable(dataType: weightType, shape: shape, initialValue: 0.0, name: meanName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: runningMeanTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  Create the running variance variable, initialized to ones
        let variances = CreateTensor.constantValues(type: weightType, shape: shape, initialValue: 1.0)
        let varianceData = variances.getData()
        let varianceName = graph.getFullName(name)! + "_runningVariance"
        let runningVarianceTensor = graph.mpsgraph.variable(with: varianceData, shape: shape.getMPSShape(), dataType: weightType.getMPSDataType(), name: varianceName)
        suffixes.append("_runningVariance")
        addedTensors.append(runningVarianceTensor)
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        node = Variable(dataType: weightType, shape: shape, initialValue: 1.0, name: varianceName)
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: runningVarianceTensor, sourceTensor: nil)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  Create the ϵ and momentum constants
        let ϵTensor = graph.mpsgraph.constant(ϵ, shape: [1 as NSNumber], dataType: weightType.getMPSDataType())
        suffixes.append("_ϵ")
        addedTensors.append(ϵTensor)
        let momentumTensor = graph.mpsgraph.constant(momentum, shape: [1 as NSNumber], dataType: weightType.getMPSDataType())
        suffixes.append("_momentum")
        addedTensors.append(momentumTensor)

    //  Training forward pass
        //  Calculate mean a variance from the batch input
        let mean = graph.mpsgraph.mean(of: inputTensor, axes: meanVarianceAxes.map{NSNumber(value: $0)}, name: graph.getFullName(name)! + "_training_mean")
        suffixes.append("_training_mean")
        addedTensors.append(mean)
        let variance = graph.mpsgraph.variance(of: inputTensor, mean: mean, axes: meanVarianceAxes.map{NSNumber(value: $0)}, name: graph.getFullName(name)! + "_training_variance")
        suffixes.append("_training_variance")
        addedTensors.append(variance)
        
        //  Numerator is inputs minus mean
        let train_numerator = graph.mpsgraph.subtraction(inputTensor, mean, name: graph.getFullName(name)! + "_training_numerator")
        suffixes.append("_training_numerator")
        addedTensors.append(train_numerator)
        
        //  Denominator is sqrt of variance + ϵ
        let variance_plus_ϵ = graph.mpsgraph.addition(variance, ϵTensor, name: graph.getFullName(name)! + "_training_variance_plus_epsilon")
        suffixes.append("_training_variance_plus_epsilon")
        addedTensors.append(variance_plus_ϵ)
        let train_denominator = graph.mpsgraph.squareRoot(with: variance_plus_ϵ, name: graph.getFullName(name)! + "_training_denominator")
        suffixes.append("_training_denominator")
        addedTensors.append(train_denominator)
        
        //  X_hat is numerator over denominator
        let train_x_hat = graph.mpsgraph.division(train_numerator, train_denominator, name: graph.getFullName(name)! + "_training_x_hat")
        suffixes.append("_training_x_hat")
        addedTensors.append(train_x_hat)
        
        //  Update the running mean and variance
        let oneTensor = graph.mpsgraph.constant(1.0, shape: [1 as NSNumber], dataType: weightType.getMPSDataType())
        suffixes.append("_one")
        addedTensors.append(oneTensor)
        let oneMinusMomentumTensor = graph.mpsgraph.subtraction(oneTensor, momentumTensor, name: graph.getFullName(name)! + "_one_minus_momentum")
        suffixes.append("_one_minus_momentum")
        addedTensors.append(oneMinusMomentumTensor)
        let oneMinusMomentumTimesMeanTensor = graph.mpsgraph.multiplication(oneMinusMomentumTensor, mean, name: graph.getFullName(name)! + "_one_minus_momentum_times_mean")
        suffixes.append("_one_minus_momentum_times_mean")
        addedTensors.append(oneMinusMomentumTimesMeanTensor)
        let momentumTimesRunningMeanTensor = graph.mpsgraph.multiplication(momentumTensor, runningMeanTensor, name: graph.getFullName(name)! + "_momentum_times_running_mean")
        suffixes.append("_momentum_times_running_mean")
        addedTensors.append(momentumTimesRunningMeanTensor)
        let newRunningMean = graph.mpsgraph.addition(momentumTimesRunningMeanTensor, oneMinusMomentumTimesMeanTensor, name: graph.getFullName(name)! + "_new_running_mean")
        suffixes.append("_new_running_mean")
        addedTensors.append(newRunningMean)
        let oneMinusMomentumTimesVarianceTensor = graph.mpsgraph.multiplication(oneMinusMomentumTensor, variance, name: graph.getFullName(name)! + "_one_minus_momentum_times_variance")
        suffixes.append("_one_minus_momentum_times_variance")
        addedTensors.append(oneMinusMomentumTimesVarianceTensor)
        let momentumTimesRunningVarianceTensor = graph.mpsgraph.multiplication(momentumTensor, runningVarianceTensor, name: graph.getFullName(name)! + "_momentum_times_running_variance")
        suffixes.append("_momentum_times_running_variance")
        addedTensors.append(momentumTimesRunningVarianceTensor)
        let newRunningVariance = graph.mpsgraph.addition(momentumTimesRunningVarianceTensor, oneMinusMomentumTimesVarianceTensor, name: graph.getFullName(name)! + "_new_running_variance")
        suffixes.append("_new_running_variance")
        addedTensors.append(newRunningVariance)
        
        //  Add assign operations for the running mean and variance (gamma and beta will be set up by the graph learning operations)
        let runningMeanAssignment = graph.mpsgraph.assign(runningMeanTensor, tensor: newRunningMean, name: graph.getFullName(name)! + "_running_mean_assignment")
        graph.nodeLearningOps.append(runningMeanAssignment)
        let runningVarianceAssignment = graph.mpsgraph.assign(runningVarianceTensor, tensor: newRunningVariance, name: graph.getFullName(name)! + "_running_variance_assignment")
        graph.nodeLearningOps.append(runningVarianceAssignment)
        
    //  Testing forward pass
        //  Numerator is inputs minus running mean
        let test_numerator = graph.mpsgraph.subtraction(inputTensor, runningMeanTensor, name: graph.getFullName(name)! + "_testing_numerator")
        suffixes.append("_testing_numerator")
        addedTensors.append(test_numerator)
        
        //  Denominator is sqrt of running variance + ϵ
        let running_variance_plus_ϵ = graph.mpsgraph.addition(runningVarianceTensor, ϵTensor, name: graph.getFullName(name)! + "_testing_variance_plus_epsilon")
        suffixes.append("_testing_variance_plus_epsilon")
        addedTensors.append(variance_plus_ϵ)
        let test_denominator = graph.mpsgraph.squareRoot(with: running_variance_plus_ϵ, name: graph.getFullName(name)! + "_testing_denominator")
        suffixes.append("_testing_denominator")
        addedTensors.append(test_denominator)
        
        //  X_hat is numerator over denominator
        let test_x_hat = graph.mpsgraph.division(test_numerator, test_denominator, name: graph.getFullName(name)! + "_testing_x_hat")
        suffixes.append("_testing_x_hat")
        addedTensors.append(test_x_hat)

        //  Get X_hat depending on the mode
        let x_hat = graph.mpsgraph.if(trainingModeTensor, then: { () -> [MPSGraphTensor] in
            return [train_x_hat]
        }, else: { () -> [MPSGraphTensor] in
            return [test_x_hat]
        }, name: "if")
        suffixes.append("x_hat")
        addedTensors.append(x_hat[0])

        //  Result is gamma * X_hat + beta
        let xhat_times_gamma = graph.mpsgraph.multiplication(x_hat[0], gammaTensor, name: graph.getFullName(name)! + "_gamma_times_x_hat")
        suffixes.append("_gamma_times_x_hat")
        addedTensors.append(xhat_times_gamma)
        var name = graph.getFullName(self.name)!
        if (haveActivationFunction) {
            name += "_batchNormalization"
            suffixes.append("_batchNormalization")
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
        
//        let tempResult = graph.mpsgraph.identity(with: runningVarianceTensor, name: graph.getFullName(name)!)
//        addedTensors.append(tempResult)
//        targetIndices.append(suffixes.count)
//        suffixes.append("")

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
    ///   - using: (Optional) the optimizer method to use for learning.  Defaults to stochastic gradient descent
    ///   - gradientClipping: (Optional) defaults to nil.  A tuple with the minimum and maximum gradient values allowed in the back-propogation for this node.  The gradient is clipped to this range before being used by the optimizer
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String, using: LearningOptimizer = .stochasticGradientDescent, gradientClipping: (min: Double, max: Double)? = nil) -> BatchNormalization {
        self.lossNode = lossNode
        self.learningOptimizer = using
        self.gradientClipping = gradientClipping
        return self
    }
}
