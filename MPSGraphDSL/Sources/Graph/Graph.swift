//
//  Graph.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/18/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

internal struct AddedNode {
    let name : String?
    let node : Node
    let mpstensor : MPSGraphTensor
    let outputShape: TensorShape
}

internal class LoadResetAssignInfo {
    let node: Variable
    let variableTensor: MPSGraphTensor
    let sourceTensor: MPSGraphTensor?
    var placeHolderTensor: MPSGraphTensor?
    
    init(node: Variable, variableTensor: MPSGraphTensor, sourceTensor: MPSGraphTensor?) {
        self.node = node
        self.variableTensor = variableTensor
        self.sourceTensor = sourceTensor
        self.placeHolderTensor = nil
    }
}

internal struct feedTensorInfo {
    let name: String
    let tensor: MPSGraphTensor
    let modes: [String]
    
    func neededForMode(_ mode: String) -> Bool {
        return (modes.isEmpty || modes.contains(mode))
    }
}

///  Options for building a Graph object
public struct BuildOptions: OptionSet, Sendable {
    ///  The complete option set
    public let rawValue: Int

    /// Initializer for a BuildOptions set
    /// - Parameter rawValue: The integer representation of the selected options
    public init(rawValue: Self.RawValue) {
        self.rawValue = rawValue
    }
    
    ///  Option to add assignment operations for all Variables so they can be loaded from saved files
    public static let addLoadAssigns    = BuildOptions(rawValue: 1 << 0)
    ///  Option to add assignment operations for all Variables so they can be reset to their initial (or another set of random) values
    public static let addResetAssigns    = BuildOptions(rawValue: 1 << 1)
    ///  Option to add assignment operations for all Variables so they can be loaded or  reset to their initial (or another set of random) values
    public static let variableAssigns: BuildOptions = [.addLoadAssigns, .addResetAssigns]
}

/// Main MPSGraph definition.  Created using the Graph DSL
public class Graph {
    internal let buildOptions: BuildOptions
    internal var nodes : [Node]
    internal var mpsgraph : MPSGraph!
    internal var device : MTLDevice!
    internal var commandQueue : MTLCommandQueue!
    
    //  Build variables
    internal var allAddedNodes : [AddedNode] = []
    internal var feedTensors : [feedTensorInfo] = []
    internal var targetTensors : [(modes: [String], tensor: MPSGraphTensor)] = []
    internal var lastAddedNode: AddedNode? = nil
    internal var currentSubGraphInputMap: [String : String?] = [:]
    internal var currentNamePrefix: String = ""
    internal var prefixStack: [String] = []
    internal var dataTensorMap: [String : Tensor] = [:]
    internal var loadResetAssignList: [LoadResetAssignInfo] = []
    internal var loadOps: [MPSGraphOperation] = []
    internal var resetOps: [MPSGraphOperation] = []
    internal var learningVariables: [(variable: Variable, tensor: MPSGraphTensor, loss: String)] = []
    internal var seenLearning = false
    
    //  Learning values
    internal var learningRateConstant = true
    internal var learningRate: Double = 0.05
    internal var learningRateTensor: MPSGraphTensor? = nil
    internal var learningModes : [String] = []
    internal var learningOps: [MPSGraphOperation] = []
    
    
    internal init(buildOptions: BuildOptions, nodes: [Node]) {
        self.buildOptions = buildOptions
        self.nodes = nodes
    }
    
    /// Create the MPSGraph object from the nodes in the Graph.  Done automatically from any run or encode operation
    ///
    /// - Throws: `MPSGraphDSLErrors.NamedTensorNotFound` if a referenced tensor name is not found in the Graph's node list
    /// - Throws: `MPSGraphDSLErrors.NodeNotInstantiatedYet` if a referenced tensor name is found in the Graph's node list, but that node has not yet been translated into a tensor
    /// - Throws: `GenericMPSGraphDSLErrors.UnknownShape` The shape of an MPSGraph output tensor was nil, and so could not be verified for correct shape matching
    public func buildGraph() throws {
        
        //  Create the graph
        mpsgraph = MPSGraph()
        
        //  Initialize the build variables
        allAddedNodes = []
        feedTensors = []
        targetTensors = []
        lastAddedNode = nil
        currentSubGraphInputMap = [:]
        currentNamePrefix = ""
        prefixStack = [""]
        dataTensorMap = [:]
        loadResetAssignList = []
        loadOps = []
        resetOps = []
        learningVariables = []
        learningOps = []
        
        //  Process the top-level nodes
        try processNodes(nodes)
        
        //  If no targets, throw
        if (targetTensors.isEmpty) {
            throw MPSGraphDSLErrors.NoTargetsInGraph
        }
        
        //  If learning variables, get the assignment operation
        if (!learningVariables.isEmpty) {
            //  Add the learning rate tensor
            var lambdaTensor: MPSGraphTensor
            if (learningRateConstant) {
                lambdaTensor = mpsgraph.constant(learningRate, shape: [1], dataType: .float32)
            }
            else {
                lambdaTensor = mpsgraph.placeholder(shape: [1], dataType: .float32, name: nil)
            }
            
            //  Get a list of the loss variables
            var lossNames: [String] = []
            for lv in learningVariables {
                if (!lossNames.contains(lv.loss)) {
                    lossNames.append(lv.loss)
                }
            }
            
            //  Process each loss tensor's variables
            for lossName in lossNames {
                if let lossNode = findNamedNode(lossName) {
                    //  Get a list of variables that use this loss node
                    var variableTensors: [MPSGraphTensor] = []
                    for lv in learningVariables {
                        if (lv.loss == lossName) {
                            variableTensors.append(lv.tensor)
                        }
                    }
                    
                    //  Create a gradient tensor for the loss node and all variables that use it
                    let gradTensors = mpsgraph.gradients(of: lossNode.mpstensor, with: variableTensors, name: nil)
                    
                    //  Add a stochastic gradient descent for each variable tensor gradient
                    for (key, value) in gradTensors {
                        let updateTensor = mpsgraph.stochasticGradientDescent(learningRate: lambdaTensor,
                                                                              values: key,
                                                                              gradient: value,
                                                                              name: nil)
                        let assign = mpsgraph.assign(key, tensor: updateTensor, name: nil)
                        learningOps.append(assign)
                    }
                }
                else {
                    throw MPSGraphDSLErrors.NamedTensorNotFound(lossName)
                }
            }
            
            //  If we are adding load or reset assign operations, do that here at the end of the graph
            if (buildOptions.contains(.addLoadAssigns) || buildOptions.contains(.addResetAssigns)) {
                for loadResetAssign in loadResetAssignList {
                    //  Get names for placeholders and assign operations
                    var placeHolderNameString: String
                    var assignNameString: String
                    if let varName = loadResetAssign.node.name {
                        placeHolderNameString = varName + "_loadAssignPlaceHolder"
                        assignNameString = varName + "_loadAssign"
                    }
                    else {
                        placeHolderNameString = "loadAssignPlaceHolder"
                        assignNameString = "loadAssign"
                    }
                    
                    //  Add a placeholder
                    let placeHolder = mpsgraph.placeholder(shape: loadResetAssign.node.shape!.getMPSShape(), dataType: loadResetAssign.node.dataType!.getMPSDataType(), name: placeHolderNameString)
                    loadResetAssign.placeHolderTensor = placeHolder
                    
                    //  Add an assign operation
                    let assign = mpsgraph.assign(loadResetAssign.variableTensor, tensor: placeHolder, name: assignNameString)
                    loadOps.append(assign)
                    
                    switch (loadResetAssign.node.valueSource) {
                    case .inputTensor:
                        //  Add an assign operation
                        let resetAssign = mpsgraph.assign(loadResetAssign.variableTensor, tensor: loadResetAssign.sourceTensor!, name: assignNameString)
                        resetOps.append(resetAssign)
                    default:
                        resetOps.append(assign)
                    }
                }
            }
        }
    }
    
    internal func processNodes(_ nodes: [Node]) throws {
        for node in nodes {
            let addedTensors = try node.addToGraph(graph: self)
            if (node.addToNodeList) {
                let suffixes = node.getNodeSuffixes()
                for i in 0..<addedTensors.count {
                    if let addedTensor = addedTensors[i] {
                        //  Get the output shape
                        let mpsshape = addedTensor.shape
                        guard (mpsshape != nil) else { throw GenericMPSGraphDSLErrors.UnknownShape }
                        let shape = TensorShape(fromMPS: mpsshape!)
                        
                        //  If there is a name, add any required suffix and verify the name is unique
                        var fullName = getFullName(node.name)
                        if let name = fullName {
                            fullName = name + suffixes[i]
                            if (!verifyNameIsUnique(fullName!)) { throw MPSGraphDSLErrors.NameNotUnique(fullName!) }
                        }
                        
                        //  Create the added node struct
                        let addedNode = AddedNode(name: fullName, node: node, mpstensor: addedTensor, outputShape: shape)
                        
                        //  Put added node into the list
                        allAddedNodes.append(addedNode)
                        lastAddedNode = addedNode
                    }
                }
            }
            
            //  If there is a build error on the node, throw it now
            if (node.buildError != nil) { throw node.buildError! }
            
            //  If a target node, add it to the list
            if !node.targetModes.isEmpty {
                if let targetIndices = node.getTargetIndices() {
                    //  We have a list of tensors to target, add them
                    for index in targetIndices {
                        if let addedTensor = addedTensors[index] {
                            targetTensors.append((modes: node.targetModes, tensor: addedTensor))
                        }
                    }
                }
                else {
                    //  Nil for target indices list - add them all as a target
                    for addedTensor in addedTensors {
                        if let addedTensor = addedTensor {
                            targetTensors.append((modes: node.targetModes, tensor: addedTensor))
                        }
                    }
                }
            }
        }
    }
    
    internal func setNewCurrentPrefix(_ newPrefix : String) {
        currentNamePrefix = newPrefix
        prefixStack.append(newPrefix)
    }
    
    internal func popLastFromPrefixStack() {
        prefixStack.removeLast()
        currentNamePrefix = prefixStack.last!
    }
    
    internal func getFullName(_ name: String?) -> String? {
        if (name == nil) {return nil}
        return currentNamePrefix + name!
    }
    
    internal func verifyNameIsUnique(_ name: String) -> Bool {
        for addedNode in allAddedNodes {
            if (addedNode.name == name) {
                return false
            }
        }
        return true
    }
    
    internal func findNamedNode(_ name : String) -> AddedNode? {
        
        //  Look for the node starting with the current prefix, then working all the way down to initial empty prefix
        for prefix in prefixStack.reversed() {
            let fullName = prefix + name
            let addedNode = allAddedNodes.first(where: { $0.name == fullName })
            if let addedNode = addedNode {
                return addedNode
            }
        }
        
        return nil
    }
    
    internal func findMPSGraphTensor(_ tensor : MPSGraphTensor) -> AddedNode? {
        for addedNode in allAddedNodes {
            if (addedNode.mpstensor === tensor) {
                return addedNode
            }
        }
        return nil
    }
    
    internal func getUnaryTensor(name: String? = nil) throws -> MPSGraphTensor {
        //  If the name is not nil, find the tensor in the graph
        if let name = name {
            if let addedNode = findNamedNode(name) {
                return addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(name)
            }
        }
        
        //  If the name is nil, return the last output tensor
        if (lastAddedNode == nil) {throw MPSGraphDSLErrors.NoPreviousNode }
        return lastAddedNode!.mpstensor
    }
    
    internal func getBinaryTensors(_ name1: String? = nil, _ name2: String? = nil) throws -> (firstInputTensor: MPSGraphTensor, secondInputTensor: MPSGraphTensor) {
        let firstTensor = try getOptionalTensor(name1)
         
        let secondTensor = try getOptionalTensor(name2)
        
        return (firstInputTensor: firstTensor, secondInputTensor: secondTensor)
    }
    
    internal func getTernaryTensors(_ name1: String? = nil, _ name2: String? = nil, _ name3: String? = nil) throws -> (firstInputTensor: MPSGraphTensor, secondInputTensor: MPSGraphTensor, thirdInputTensor: MPSGraphTensor) {
        let firstTensor = try getOptionalTensor(name1)
        
       let secondTensor = try getOptionalTensor(name2)
        
       let thirdTensor = try getOptionalTensor(name3)

        return (firstInputTensor: firstTensor, secondInputTensor: secondTensor, thirdInputTensor: thirdTensor)
    }

    //  Get the named tensor, or previous node if name is nil
    internal func getOptionalTensor(_ name: String?) throws -> MPSGraphTensor {
        if let tensorName = name {
            if let addedNode = findNamedNode(tensorName) {
                return addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(tensorName)
            }
        }
        else {
            //  nil state tensor node name - use last node
            if (lastAddedNode == nil) {throw MPSGraphDSLErrors.NoPreviousNode }
            return lastAddedNode!.mpstensor
        }
    }
    
    //  Get the metal command buffer
    internal func getCommandBuffer() {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
    }
    
    /// Run a single input tensor (or single set if multiple inputs are required) through the graph, returning the output tensors
    ///       If the graph has not been built yet, it is built at the start of this function - see BuildGraph for errors that occur from the build process
    ///       Only use this function if you are only running a few cases.  More that a few hundred can cause memory errors in Metal.  Use 'encodeOne' instead for larger runs
    /// - Parameters:
    ///   - mode: The mode that specifies what output tensors are to be computed
    ///   - inputTensors: an dictionary of input tensors to feed to the graph, with the key being the ``PlaceHolder`` name
    ///   - newLearningRate: (Optional) if a non-constant learning rate is specified (see ``Learning``, the value can be changed for subsequent runs with this parameter
    /// - Returns: an array of output tensors that are the result of the graph run
    public func runOne(mode: String, inputTensors: [String : Tensor], newLearningRate: Double? = nil) throws -> [String : Tensor] {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandBuffer() }
        
        //  If a new learning rate entered - set it
        if let newlearningRate = newLearningRate {
            learningRate = newlearningRate
        }
        
        //  Convert the tensors to MPS version and create the feed dictionary
        var feedDict : [MPSGraphTensor: MPSGraphTensorData] = [:]
        for feedTensor in feedTensors {
            //  Find the input tensor that matches
            if let inputTensor = inputTensors.first(where: { $0.key == feedTensor.name }) {
                feedDict[feedTensor.tensor] = inputTensor.value.getMPSGraphTensorData(forGraph: self)
            }
            else {
                if (feedTensor.neededForMode(mode)) {
                    throw MPSGraphRunErrors.PlaceHolderInputNotFound(feedTensor.name)
                }
            }
        }
        if (!learningRateConstant) {
            let learningRateValueTensor = TensorFloat32(shape : TensorShape([1]), initialValue : Float32(learningRate))
            feedDict[learningRateTensor!] = learningRateValueTensor.getMPSGraphTensorData(forGraph: self)
        }
        
        //  Get the targetTensors
        var targets : [MPSGraphTensor] = []
        for targetTensor in targetTensors {
            if targetTensor.modes.contains(mode) {
                targets.append(targetTensor.tensor)
            }
        }
        if (targets.isEmpty) { throw MPSGraphDSLErrors.NoTargetsInGraph }
        
        //  Set up any learning operations
        var targetOperations : [MPSGraphOperation]? = nil
        if (learningModes.contains(mode)) {
            targetOperations = learningOps
        }
        
        //  Run the graph
        let results = mpsgraph.run(feeds: feedDict,
                                   targetTensors: targets,
                                   targetOperations: targetOperations)
        
        var resultTensors : [String: Tensor] = [:]
        for (tensor, data) in results {
            //  Find the name of the tensor
            let addedNode = findMPSGraphTensor(tensor)
            if let addedNode = addedNode {
                resultTensors[addedNode.name!] = CreateTensor.fromMPSTensorData(data)
            }
        }
        
        return resultTensors
    }
    
    /// Run a single input tensor (or single set if multiple inputs are required) through the graph, returning the output tensors
    ///       If the graph has not been built yet, it is built at the start of this function - see BuildGraph for errors that occur from the build process
    ///       Only use this function if you are only running a few cases.  More that a few hundred can cause memory errors in Metal.  Use 'encodeOne' instead for larger runs
    /// - Parameters:
    ///   - mode: The mode that specifies what output tensors are to be computed
    ///   - inputTensors: an dictionary of input tensors to feed to the graph, with the key being the ``PlaceHolder`` name
    ///   - waitForResults: (Optional) defaults to true.  If true the operation waits for the results back from Metal.  If not needed (e.g. during training operations), set to false
    ///   - newLearningRate: (Optional) if a non-constant learning rate is specified (see ``Learning``, the value can be changed for subsequent runs with this parameter
    /// - Returns: an array of output tensors that are the result of the graph run
    public func encodeOne(mode: String, inputTensors: [String : Tensor], waitForResults: Bool = true, newLearningRate: Double? = nil) throws -> [String : Tensor] {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandBuffer() }
        
        let commandBuffer = MPSCommandBuffer(commandBuffer: commandQueue.makeCommandBuffer()!)
        
        //  If a new learning rate entered - set it
        if let newlearningRate = newLearningRate {
            learningRate = newlearningRate
        }
        
        //  Convert the tensors to MPS version and create the feed dictionary
        var feedDict : [MPSGraphTensor: MPSGraphTensorData] = [:]
        for feedTensor in feedTensors {
            //  Find the input tensor that matches
            if let inputTensor = inputTensors.first(where: { $0.key == feedTensor.name }) {
                feedDict[feedTensor.tensor] = inputTensor.value.getMPSGraphTensorData(forGraph: self)
            }
            else {
                if (feedTensor.neededForMode(mode)) {
                    throw MPSGraphRunErrors.PlaceHolderInputNotFound(feedTensor.name)
                }
            }
        }
        if (!learningRateConstant) {
            let learningRateValueTensor = TensorFloat32(shape : TensorShape([1]), initialValue : Float32(learningRate))
            feedDict[learningRateTensor!] = learningRateValueTensor.getMPSGraphTensorData(forGraph: self)
        }
        
        //  Get the targetTensors
        var targets : [MPSGraphTensor] = []
        for targetTensor in targetTensors {
            if targetTensor.modes.contains(mode) {
                targets.append(targetTensor.tensor)
            }
        }
        if (targets.isEmpty) { throw MPSGraphDSLErrors.NoTargetsInGraph }
        
        //  Set up any learning operations
        var targetOperations : [MPSGraphOperation]? = nil
        if (learningModes.contains(mode)) {
            targetOperations = learningOps
        }
        
        //  Run the graph
        let results = mpsgraph.encode(to: commandBuffer,
                                      feeds: feedDict,
                                      targetTensors: targets,
                                      targetOperations: targetOperations,
                                      executionDescriptor: nil)
        
        commandBuffer.commit()
        if (waitForResults) { commandBuffer.waitUntilCompleted() }
        
        var resultTensors : [String: Tensor] = [:]
        for (tensor, data) in results {
            //  Find the name of the tensor
            let addedNode = findMPSGraphTensor(tensor)
            if let addedNode = addedNode {
                resultTensors[addedNode.name!] = CreateTensor.fromMPSTensorData(data)
            }
        }
        
        return resultTensors
    }
    
    /// Run all samples of a testing dataset through the graph and return the number/percentage correct
    /// - Parameters:
    ///   - mode: The name of the inference mode of the graph
    ///   - testDataSet: The testing DataSet
    ///   - inputTensorName: The name of the input tensor for the graph (PlaceHolder)
    ///   - resultTensorName: The name of the result tensor for the graph (must be a targe for the mode specified)
    /// - Returns: A tuple with the fraction correct and the total number of samples that classified correctly
    public func runClassifierTest(mode: String, testDataSet: DataSet, inputTensorName: String, resultTensorName: String) throws -> (fractionCorrect: Double, totalCorrect : Int) {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandBuffer() }
        
        var numCorrect: Int = 0
        for sampleIndex in 0..<testDataSet.numSamples {
            //  Get the input tensor
            let sample = try testDataSet.getSample(sampleIndex: sampleIndex)
            
            //  Run the sample
            var inputTensors : [String : Tensor] = [:]
            inputTensors[inputTensorName] = sample.inputs
            let results = try encodeOne(mode: mode, inputTensors: inputTensors)
            let result = results[resultTensorName]
            
            if let result = result {
                let actual = sample.outputClass
                let predictedClass = result.getClassification()
                
                if (predictedClass == actual) {
                    numCorrect += 1
                }
            }
            else {
                throw MPSGraphRunErrors.ResultTensorNotFound
            }
        }
        
        let fractionCorrect: Double = Double(numCorrect) / Double(testDataSet.numSamples)
        return (fractionCorrect: fractionCorrect, totalCorrect : numCorrect)
    }
    
    /// Run all samples of a testing dataset through a graph set up for batch running and return the number/percentage correct
    /// - Parameters:
    ///   - mode: The name of the inference mode of the graph
    ///   - testDataSet: The testing DataSet
    ///   - inputTensorName: The name of the input tensor for the graph (PlaceHolder)
    ///   - resultTensorName: The name of the result tensor for the graph (must be a targe for the mode specified)
    ///   - batchSize: The number of samples in a batch
    /// - Returns: A tuple with the fraction correct and the total number of samples that classified correctly
    public func runBatchedClassifierTest(mode: String, testDataSet: DataSet, inputTensorName: String, resultTensorName: String, batchSize: Int = 1) throws -> (fractionCorrect: Double, totalCorrect : Int) {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandBuffer() }
        
        let numBatches = testDataSet.numSamples / batchSize
        var sampleIndex: Int = 0
        var numCorrect: Int = 0
        for _ in 0..<numBatches {
            //  Get the batch sample indices
            var sampleIndices:  [Int] = []
            for _ in 0..<batchSize {
                sampleIndices.append(sampleIndex)
                sampleIndex += 1
            }
            
            //  Get the batch input tensor
            let tensors = try testDataSet.getBatch(sampleIndices: sampleIndices)
            
            //  Run the batch
            var inputTensors : [String : Tensor] = [:]
            inputTensors[inputTensorName] = tensors.inputTensor
            let results = try encodeOne(mode: mode, inputTensors: inputTensors)
            let result = results[resultTensorName]
            
            if let result = result {
                for i in 0..<batchSize {
                    let sample = try testDataSet.getSample(sampleIndex: i)
                    let actual = sample.outputClass
                    let predictedClass = try result.getClassificationForBatch(i)
                    
                    if (predictedClass == actual) {
                        numCorrect += 1
                    }
                }
            }
            else {
                throw MPSGraphRunErrors.ResultTensorNotFound
            }
        }
        
        let fractionCorrect: Double = Double(numCorrect) / Double(batchSize * numBatches)
        return (fractionCorrect: fractionCorrect, totalCorrect : numCorrect)
    }
    
    
    /// Run all samples of a training DataSet through the graph, with the variable learning operations enaboed
    /// - Parameters:
    ///   - mode: The name of the learning mode of the graph
    ///   - trainingDataSet: The training DataSet
    ///   - inputTensorName: The name of the input tensor for the graph (PlaceHolder)
    ///   - expectedValueTensorName: The name of the input tensor for the expected value (PlaceHolder)
    public func runTraining(mode: String, trainingDataSet: DataSet, inputTensorName: String, expectedValueTensorName : String) throws {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandBuffer() }
        
        var inputTensors : [String : Tensor] = [:]
        for sample in trainingDataSet.samples {
            inputTensors[inputTensorName] = sample.inputs
            inputTensors[expectedValueTensorName] = sample.outputs
            _ = try encodeOne(mode: mode, inputTensors: inputTensors, waitForResults: false)
        }
    }
    
    /// Run all samples of a training DataSet through the graph set up for batch running ,  with the variable learning operations enaboed
    /// - Parameters:
    ///   - mode: The name of the learning mode of the graph
    ///   - trainingDataSet: The training DataSet
    ///   - inputTensorName: The name of the input tensor for the graph (PlaceHolder)
    ///   - expectedValueTensorName: The name of the input tensor for the expected value (PlaceHolder)
    ///   - numBatches: The number of batches to be run
    ///   - random: If true the samples are shuffled before being added to the batches.  Otherwise the samples are put into batches in the order they are in the DataSet
    ///   - batchSize: The number of samples in a batch
    public func runBatchTraining(mode: String, trainingDataSet: DataSet, inputTensorName: String, expectedValueTensorName : String, numBatches: Int, random: Bool = true, batchSize: Int = 1) throws {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandBuffer() }
        
        //  Set up for batch indexing
        let numSamples = trainingDataSet.numSamples
        var sequentialBatchIndex = 0
        var batchIndices: [Int] = []
        
        
        //  Do for all the batches
        for _ in 0..<numBatches {
            //  Get the batch data
            batchIndices = []
            if (random) {
                for _ in 0..<batchSize {
                    var index = Int.random(in: 0..<numSamples)
                    while (batchIndices.contains(index)) { index = Int.random(in: 0..<numSamples)}
                    batchIndices.append(index)
                }
            }
            else {
                for _ in 0..<batchSize {
                    batchIndices.append(sequentialBatchIndex)
                    sequentialBatchIndex += 1
                    if (sequentialBatchIndex >= numSamples) { sequentialBatchIndex = 0 }
                }
            }
            let tensors = try trainingDataSet.getBatch(sampleIndices: batchIndices)
            
            //  Run the batch
            var inputTensors : [String : Tensor] = [:]
            inputTensors[inputTensorName] = tensors.inputTensor
            inputTensors[expectedValueTensorName] = tensors.outputTensor
            _ = try encodeOne(mode: mode, inputTensors: inputTensors, waitForResults: false)
        }
    }
    
    /// Print each added node's shape, in order, with node names if available.  This is convenient for debugging a Graph
    public func printShapeList() throws {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        
        for addedNode in allAddedNodes {
            var nameString = "* Unnamed *"
            if let name = addedNode.name {
                nameString = name
            }
            
            print ("\(addedNode.outputShape.dimensions) - \(nameString)")
        }
    }
    
    /// Write an MPSPackage file from the Graph
    /// - Parameters:
    ///   - url: The URL to write the package to
    ///   - mode: The mode for the Graph.  Different modes can take different paths, so this dictates what goes into the package
    ///   - inputTensors: A dictionary of Tensors keyed by Placeholder names for inputs to the Graph
    public func makeMPSPackage(url: URL, mode: String, inputTensors: [String : Tensor]) throws {
        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandBuffer() }
        
        //  Get the targetTensors
        var targets : [MPSGraphTensor] = []
        for targetTensor in targetTensors {
            if targetTensor.modes.contains(mode) {
                targets.append(targetTensor.tensor)
            }
        }
        if (targets.isEmpty) { throw MPSGraphDSLErrors.NoTargetsInGraph }
        
        //  Convert the tensors to MPS version and create the feed dictionary
        var feedDict : [MPSGraphTensor: MPSGraphShapedType] = [:]
        for feedTensor in feedTensors {
            //  Find the input tensor that matches
            if let inputTensor = inputTensors.first(where: { $0.key == feedTensor.name }) {
                feedDict[feedTensor.tensor] = MPSGraphShapedType(shape: inputTensor.value.shape.getMPSShape(), dataType: inputTensor.value.type.getMPSDataType())
            }
            else {
                if (feedTensor.neededForMode(mode)) {
                    throw MPSGraphRunErrors.PlaceHolderInputNotFound(feedTensor.name)
                }
            }
        }
        if (!learningRateConstant) {
            feedDict[learningRateTensor!] = MPSGraphShapedType(shape: [1 as NSNumber], dataType: .float32)
        }
        
        //  Set up any learning operations
        var targetOperations : [MPSGraphOperation]? = nil
        if (learningModes.contains(mode)) {
            targetOperations = learningOps
        }
        
        //  Compile into an executable
        let compilationDescriptor = MPSGraphCompilationDescriptor()
        let executable = mpsgraph.compile(with: MPSGraphDevice(mtlDevice: device), feeds: feedDict, targetTensors: targets, targetOperations: targetOperations, compilationDescriptor: compilationDescriptor)
        
        //  Serialize into a package
        let serializationDescriptor = MPSGraphExecutableSerializationDescriptor()
        executable.serialize(package: url, descriptor: serializationDescriptor)
    }
    
    /// Get the values of all Variable objects and encode it into a Data object
    /// - Returns: A Data object with the Variable data encoded within
    public func getVariableData() throws -> Data {
        if (learningVariables.isEmpty) {
            throw MPSGraphDSLErrors.NoLearningVariablesInGraph
        }
        
        //  Set up a list of the variable tensors as targets for a run
        var variableTensors: [MPSGraphTensor] = []
        for learningVariable in learningVariables {
            variableTensors.append(learningVariable.tensor)
        }
        
        //  run the graph with the variables as the targets
        let results = mpsgraph.run(feeds: [:],
                                   targetTensors: variableTensors,
                                   targetOperations: nil)
        
        //  Extract the variable numbers into a Data object
        var data = Data()
        data.appendValue(results.count)
        for result in results {
            //  Find the learning variable the result is for
            if let learningVariable = learningVariables.first(where: { $0.tensor === result.key }) {
                let name = learningVariable.variable.name!
                data.appendString(name)
                
                let resultTensor = CreateTensor.fromMPSTensorData(result.value)
                let tensorData = resultTensor.getData()
                let tensorSize = tensorData.count
                data.appendValue(tensorSize)
                data.append(tensorData)
            }
            else {
                throw PersistanceErrors.ErrorFindingLearningVariableInResults
            }
        }
        
        return data
    }
    
    /// Store saved Variable data back into the nodes.  This requires the Graph to have been built with the 'addLoadAssigns' option
    /// - Parameter fromData: The Data object to load the Variables from
    public func loadVariables(fromData: Data) throws {
        if (!buildOptions.contains(.addLoadAssigns)) { throw MPSGraphRunErrors.GraphNotBuiltForOperation("Load Variables") }
        
        //  Get the number of variables
        var offset: Int = 0
        let count: Int? = fromData.extractValue(offset: &offset)
        if let count = count {
            if (count != learningVariables.count) { throw PersistanceErrors.SavedCountMismatchWithLearnVariableCount }
            if (count != loadOps.count) { throw PersistanceErrors.SavedCountMismatchWithLearnVariableCount }
            var feedDict : [MPSGraphTensor: MPSGraphTensorData] = [:]
            for _ in 0..<count {
                let name = fromData.extractString(offset: &offset)
                
                //  Find the name in the load assign list
                if let loadAssign = loadResetAssignList.first(where: { $0.node.name == name }) {
                    //  Get the size of the data
                    if let dataSize: Int = fromData.extractValue(offset: &offset) {
                        //  extract the data
                        let range = offset..<(offset+dataSize)
                        let tensorData = fromData[range]
                        offset += dataSize
                        
                        //  Convert the data to MPSGraphTensorData
                        let descriptor = MPSNDArrayDescriptor(dataType: loadAssign.node.dataType!.getMPSDataType(), shape: loadAssign.node.shape!.getMPSShape())
                        let NDArray = MPSNDArray(device: device, descriptor: descriptor)
                        var byteArray: [UInt8] = [UInt8](tensorData)
                        NDArray.writeBytes(&byteArray, strideBytes: nil)
                        let graphData = MPSGraphTensorData(NDArray)
                        
                        //  Add the data to the feed
                        feedDict[loadAssign.placeHolderTensor!] = graphData
                    }
                    else {
                        throw PersistanceErrors.UnexpectedDataReadError
                    }
                }
                else {
                    throw PersistanceErrors.SavedVariableNotFoundInLoadList(name!)
                }
            }
            
            //  Run the assignment operations
            _ = mpsgraph.run(feeds: feedDict, targetTensors: [], targetOperations: loadOps)
        }
        else {
            throw PersistanceErrors.UnexpectedDataReadError
        }
    }
    
    /// Rest all Variables to their initial values or, if from a random generator, to a new set of random values.  This function requires the Graph to have been built with the 'addResetAssigns' option
    public func resetVariables() throws {
        if (!buildOptions.contains(.addResetAssigns)) { throw MPSGraphRunErrors.GraphNotBuiltForOperation("Variable Reset") }
        
        //  Create a feed dictionary
        var feedDict : [MPSGraphTensor: MPSGraphTensorData] = [:]
        for assign in loadResetAssignList {
            let data = assign.node.getResetData(forGraph: self)
            if let data = data {
                if let placeHolderTensor = assign.placeHolderTensor {
                    feedDict[placeHolderTensor] = data
                }
            }
        }
        
        //  Run the assignment operations
        _ = mpsgraph.run(feeds: feedDict, targetTensors: [], targetOperations: resetOps)
    }
}
