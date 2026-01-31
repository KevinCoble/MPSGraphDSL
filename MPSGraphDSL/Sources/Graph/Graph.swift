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
    let operation: String
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
    let batchExemption: Bool
    
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
    
    ///  Flag for if the graph is built to accomodate batch processing
    public private(set) var batchGraph = false
    public private(set) var batchSize = 1

    //  Learning values
    internal var learningRateConstant = true
    internal var learningRate: Double = 0.05
    internal var learningRateTensor: MPSGraphTensor? = nil
    internal var learningModes : [String] = []
    internal var learningOps: [MPSGraphOperation] = []
    
    
    internal init(batchSize: Int, buildOptions: BuildOptions, nodes: [Node]) {
        self.batchSize = batchSize
        if (batchSize > 1) { batchGraph = true }
        self.buildOptions = buildOptions
        self.nodes = nodes
    }
    
    internal func clearReferencedFlags() {
        for node in nodes {
            node.clearReferencedFlag()
        }
    }

    /// Create the MPSGraph object from the nodes in the Graph.  Done automatically from any run or encode operation
    ///
    /// - Throws: `MPSGraphDSLErrors.NamedTensorNotFound` if a referenced tensor name is not found in the Graph's node list
    /// - Throws: `MPSGraphDSLErrors.NodeNotInstantiatedYet` if a referenced tensor name is found in the Graph's node list, but that node has not yet been translated into a tensor
    /// - Throws: `GenericMPSGraphDSLErrors.UnknownShape` The shape of an MPSGraph output tensor was nil, and so could not be verified for correct shape matching
    public func buildGraph() throws {
        //  Clear the reference list
        clearReferencedFlags()
        
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
        
        //  Check if all the nodes are referenced by another node or a target destination
        for node in nodes {
            try node.isReferenced()
        }
        
        //  If learning variables, get the assignment operation
        if (!learningVariables.isEmpty) {
            //  Verify we have a learning rate tensor
            if (learningRateTensor == nil) {
                throw MPSGraphDSLErrors.NoLearningNode
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
                        let updateTensor = mpsgraph.stochasticGradientDescent(learningRate: learningRateTensor!,
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
                        
                        var opName = String(describing: node)
                        if (opName.starts(with: "MPSGraphDSL.")) { opName = String(opName.trimmingPrefix("MPSGraphDSL.")) }
                        
                        //  Create the added node struct
                        let addedNode = AddedNode(name: fullName, operation: opName, node: node, mpstensor: addedTensor, outputShape: shape)
                        
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
        let tensor = try getOptionalTensor(name)
        return tensor
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
                addedNode.node.referencedByAnotherNode = true
                return addedNode.mpstensor
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(tensorName)
            }
        }
        else {
            //  nil state tensor node name - use last node
            if (lastAddedNode == nil) {throw MPSGraphDSLErrors.NoPreviousNode }
            lastAddedNode!.node.referencedByAnotherNode = true
            return lastAddedNode!.mpstensor
        }
    }
    
    //  Get the metal command buffer
    internal func getCommandQueue() {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
    }
    
    /// Run a single input tensor (or single set if multiple inputs are required) through the graph, returning the output tensors
    ///       If the graph has not been built yet, it is built at the start of this function - see BuildGraph for errors that occur from the build process
    ///       If the graph is a batch graph, all the input tensors (that are not batch exempt) must be sized with the batch dimension or none of them.  If none, the function will fill a batch tensor with the (assumed) single inputs and extract the single results from the returned batch tensors
    ///       Only use this function if you are only running a few cases.  More that a few hundred can cause memory errors in Metal.  Use 'encodeOne' instead for larger runs
    /// - Parameters:
    ///   - mode: The mode that specifies what output tensors are to be computed
    ///   - inputTensors: an dictionary of input tensors to feed to the graph, with the key being the ``PlaceHolder`` name
    ///   - newLearningRate: (Optional) if a non-constant learning rate is specified (see ``Learning``, the value can be changed for subsequent runs with this parameter
    /// - Returns: an array of output tensors that are the result of the graph run
    public func runOne(mode: String, inputTensors: [String : Tensor], newLearningRate: Double? = nil) throws -> [String : Tensor] {
        //  Verify the input tensors are usable
        var allAreBatchTensors: Bool = true
        for (key, value) in inputTensors {
            if (!value.type.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
            if (batchGraph) {
                //  Check if we need a batch dimension on the tensor
                if let feedInfo = feedTensors.first(where: { $0.name == key }) {
                    if (!feedInfo.batchExemption) {
                        if (!value.shape.firstDimensionIsBatchSize(batchSize)) { allAreBatchTensors = false }
                    }
                }
                else {
                    throw MPSGraphRunErrors.PlaceHolderInputNotFound(key)
                }
            }
        }

        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandQueue() }
        
        //  If a new learning rate entered - set it
        if let newlearningRate = newLearningRate {
            learningRate = newlearningRate
        }
        
        //  Convert the tensors to MPS version and create the feed dictionary
        var feedDict : [MPSGraphTensor: MPSGraphTensorData] = [:]
        for feedTensor in feedTensors {
            //  Find the input tensor that matches
            if let inputTensor = inputTensors.first(where: { $0.key == feedTensor.name }) {
                if (!allAreBatchTensors && !feedTensor.batchExemption) {
                    //  Repeat the tensor batchSize times to make a batch tensor out of it
                    let batchTensorShape = inputTensor.value.shape.shapeWithAddedBatchDimension(batchSize)
                    var batchTensor = CreateTensor.constantValues(type: inputTensor.value.type, shape: batchTensorShape, initialValue: 0.0)
                    for i in 0..<batchSize {
                        try batchTensor.setBatchSample(tensor: inputTensor.value, batchIndex: i)
                    }
                    feedDict[feedTensor.tensor] = try batchTensor.getMPSGraphTensorData(forGraph: self)
                }
                else {
                    feedDict[feedTensor.tensor] = try inputTensor.value.getMPSGraphTensorData(forGraph: self)
                }
            }
            else {
                if (feedTensor.neededForMode(mode)) {
                    throw MPSGraphRunErrors.PlaceHolderInputNotFound(feedTensor.name)
                }
            }
        }
        if (!learningRateConstant) {
            let learningRateValueTensor = TensorFloat32(shape : TensorShape([1]), initialValue : Float32(learningRate))
            feedDict[learningRateTensor!] = try learningRateValueTensor.getMPSGraphTensorData(forGraph: self)
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
        
        //  Convert batch tensors to single output ones where needed
        if (!allAreBatchTensors) {
            for (key, value) in resultTensors {
                if (value.shape.firstDimensionIsBatchSize(batchSize)) {
                    let singleTensor = try value.getTensorForBatch(0)
                    resultTensors[key] = singleTensor
                }
            }
        }

        return resultTensors
    }
    
    /// Run a single input tensor (or single set if multiple inputs are required) through the graph, returning the output tensors
    ///       If the graph has not been built yet, it is built at the start of this function - see BuildGraph for errors that occur from the build process
    ///       If the graph is a batch graph, all the input tensors (that are not batch exempt) must be sized with the batch dimension or none of them.  If none, the function will fill a batch tensor with the (assumed) single inputs and extract the single results from the returned batch tensors
    ///       Only use this function if you are only running a few cases.  More that a few hundred can cause memory errors in Metal.  Use 'encodeOne' instead for larger runs
    /// - Parameters:
    ///   - mode: The mode that specifies what output tensors are to be computed
    ///   - inputTensors: an dictionary of input tensors to feed to the graph, with the key being the ``PlaceHolder`` name
    ///   - waitForResults: (Optional) defaults to true.  If true the operation waits for the results back from Metal.  If not needed (e.g. during training operations), set to false
    ///   - newLearningRate: (Optional) if a non-constant learning rate is specified (see ``Learning``, the value can be changed for subsequent runs with this parameter
    /// - Returns: an array of output tensors that are the result of the graph run
    public func encodeOne(mode: String, inputTensors: [String : Tensor], waitForResults: Bool = true, newLearningRate: Double? = nil) throws -> [String : Tensor] {
        //  Verify the input tensors are usable
        var allAreBatchTensors: Bool = true
        for (key, value) in inputTensors {
            if (!value.type.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
            if (batchGraph) {
                //  Check if we need a batch dimension on the tensor
                if let feedInfo = feedTensors.first(where: { $0.name == key }) {
                    if (!feedInfo.batchExemption) {
                        if (!value.shape.firstDimensionIsBatchSize(batchSize)) { allAreBatchTensors = false }
                    }
                }
                else {
                    throw MPSGraphRunErrors.PlaceHolderInputNotFound(key)
                }
            }
        }

        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandQueue() }
        
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
                if (!allAreBatchTensors && !feedTensor.batchExemption) {
                    //  Repeat the tensor batchSize times to make a batch tensor out of it
                    let batchTensorShape = inputTensor.value.shape.shapeWithAddedBatchDimension(batchSize)
                    var batchTensor = CreateTensor.constantValues(type: inputTensor.value.type, shape: batchTensorShape, initialValue: 0.0)
                    for i in 0..<batchSize {
                        try batchTensor.setBatchSample(tensor: inputTensor.value, batchIndex: i)
                    }
                    feedDict[feedTensor.tensor] = try batchTensor.getMPSGraphTensorData(forGraph: self)
                }
                else {
                    feedDict[feedTensor.tensor] = try inputTensor.value.getMPSGraphTensorData(forGraph: self)
                }
            }
            else {
                if (feedTensor.neededForMode(mode)) {
                    throw MPSGraphRunErrors.PlaceHolderInputNotFound(feedTensor.name)
                }
            }
        }
        if (!learningRateConstant) {
            let learningRateValueTensor = TensorFloat32(shape : TensorShape([1]), initialValue : Float32(learningRate))
            feedDict[learningRateTensor!] = try learningRateValueTensor.getMPSGraphTensorData(forGraph: self)
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
        
        //  Convert batch tensors to single output ones where needed
        if (!allAreBatchTensors) {
            for (key, value) in resultTensors {
                if (value.shape.firstDimensionIsBatchSize(batchSize)) {
                    let singleTensor = try value.getTensorForBatch(0)
                    resultTensors[key] = singleTensor
                }
            }
        }

        return resultTensors
    }
    
    /// Run all, or a specified set of,  samples of a testing dataset through the graph and return the number/percentage correct
    /// - Parameters:
    ///   - mode: The name of the inference mode of the graph
    ///   - testDataSet: The testing DataSet
    ///   - inputTensorName: The name of the input tensor for the graph (PlaceHolder)
    ///   - resultTensorName: The name of the result tensor for the graph (must be a targe for the mode specified)
    ///   - sampleRange: (Optional) The range of samples to put into the test.  If nil all samples are used for the test.  Must be divisible by batch size if Graph batch processing in use
    /// - Returns: A tuple with the fraction correct and the total number of samples that classified correctly
    public func runClassifierTest(mode: String, testDataSet: DataSet, inputTensorName: String, resultTensorName: String, sampleRange: ClosedRange<Int>? = nil) async throws -> (fractionCorrect: Double, totalCorrect : Int) {
        //  Verify the input tensors are usable
        if (!testDataSet.inputType.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
        if (!testDataSet.outputType.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
        
        //  Verify the number of samples is divisible by the batch size if needed
        if (batchGraph && sampleRange != nil) {
            let usedSamples = sampleRange!.upperBound - sampleRange!.lowerBound + 1
            if ((usedSamples % batchSize) != 0) { throw MPSGraphRunErrors.SampleSizeNotDivisibleByBatchSize }
        }

        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandQueue() }
        
        //  Get the targetTensors
        var targets : [MPSGraphTensor] = []
        for targetTensor in targetTensors {
            if targetTensor.modes.contains(mode) {
                targets.append(targetTensor.tensor)
            }
        }
        if (targets.isEmpty) { throw MPSGraphDSLErrors.NoTargetsInGraph }
        
        //  Get the input feed tensor
        let inputFeedTensorInfo = feedTensors.first(where: { $0.name == inputTensorName })
        if (inputFeedTensorInfo == nil) { throw MPSGraphRunErrors.PlaceHolderInputNotFound(inputTensorName) }
        
        //  Get the output result tensor
        let outputResultTensor: MPSGraphTensor
        let addedNode = findNamedNode(resultTensorName)
        if let addedNode = addedNode {
            outputResultTensor = addedNode.mpstensor
        }
        else {
            throw MPSGraphRunErrors.PlaceHolderInputNotFound(resultTensorName)
        }
        
        //  Get the double-buffer handler
        let doubleBufferHandler = DoubleBufferActor()
        
        //  Lock the dataset
        try await testDataSet.lockForMultiSampleUse()
        
        //  Get the sample range
        let numSamples = await testDataSet.numSamples
        let startSample: Int
        let endSample: Int
        if let sampleRange = sampleRange {
            startSample = sampleRange.lowerBound
            endSample = sampleRange.upperBound
        }
        else {
            startSample = 0
            endSample = numSamples-1
        }
        
        //  Get the number of runs required
        var numberOfRuns: Int = endSample - startSample + 1
        if (batchGraph) {
            numberOfRuns /= batchSize
        }
        
        //  Create the batch tensor
        var batchInputTensor: Tensor?
        if (batchGraph) {
            let batchInputTensorShape = testDataSet.inputShape.shapeWithAddedBatchDimension(batchSize)
            batchInputTensor = CreateTensor.constantValues(type: testDataSet.inputType, shape: batchInputTensorShape, initialValue: 0.0)
        }
        
        do {
            var sampleIndex: Int = startSample
            for _ in 0..<numberOfRuns {
                //  Wait for our turn
                await doubleBufferHandler.getAllowance()

                //  Get the command buffer
                let commandBuffer = MPSCommandBuffer(commandBuffer: commandQueue.makeCommandBuffer()!)
                
                //  Get the input Tensor
                var inputTensor: Tensor
                var actualClassifications = [Int](repeating: 0, count: batchSize)
                if (batchGraph) {
                    for i in 0..<batchSize {
                        let sample = try await testDataSet.getSample(sampleIndex: sampleIndex)
                        try batchInputTensor!.setBatchSample(tensor: sample.inputs, batchIndex: i)
                        actualClassifications[i] = sample.outputClass
                        sampleIndex += 1
                    }
                    inputTensor = batchInputTensor!
                }
                else {
                    let sample = try await testDataSet.getSample(sampleIndex: sampleIndex)
                    inputTensor = sample.inputs
                    actualClassifications[0] = sample.outputClass
                    sampleIndex += 1
                }
                
                //  Convert the input tensor to MPS version and create the feed dictionary
                let feedDict : [MPSGraphTensor: MPSGraphTensorData] = [inputFeedTensorInfo!.tensor : try inputTensor.getMPSGraphTensorData(forGraph: self)]
                
                //  Create the callback
                let executionDesc = MPSGraphExecutionDescriptor()
                let bSize = batchSize
                executionDesc.completionHandler = { (resultsDictionary, nil) in
                    let result: MPSGraphTensorData = resultsDictionary[outputResultTensor]!
                    if (bSize == 1) {
                        let actual = actualClassifications[0]
                        let resultTensor = CreateTensor.fromMPSTensorData(result)
                        let predictedClass = resultTensor.getClassification()
                        
                        Task {
                            await doubleBufferHandler.operationComplete(correct: predictedClass == actual)
                        }
                    }
                    else {
                        var numCorrect: Int = 0
                        let resultTensor = CreateTensor.fromMPSTensorData(result)
                        for i in 0..<bSize {
                            let actual = actualClassifications[i]
                            do {
                                let predictedClass = try resultTensor.getClassificationForBatch(i)
                                if (predictedClass == actual) { numCorrect += 1 }
                            }
                            catch {
                                fatalError("Batch index failure in runClassifierTest")
                            }
                        }
                        
                        Task {
                            await doubleBufferHandler.operationComplete(numCorrect: numCorrect)
                        }
                   }
                }

                //  Run the graph
                let _ = mpsgraph.encode(to: commandBuffer,
                                              feeds: feedDict,
                                              targetTensors: targets,
                                              targetOperations: nil,
                                              executionDescriptor: executionDesc)

                commandBuffer.commit()
            }
        }
        catch {
            //  Make sure we release the dataset.  Cannot use defer with actor isolated code
            try await testDataSet.releaseLock()
            throw error
        }
        try await testDataSet.releaseLock()

        await doubleBufferHandler.waitTillComplete()
        let numCorrect = await doubleBufferHandler.numCorrect
        let fractionCorrect: Double = Double(numCorrect) / Double(endSample - startSample + 1)
        return (fractionCorrect: fractionCorrect, totalCorrect : numCorrect)
    }
    
    /// Run all, or a specified set of,  samples of a testing dataset through the graph and return the total error value
    /// - Parameters:
    ///   - mode: The name of the inference mode of the graph
    ///   - testDataSet: The testing DataSet
    ///   - inputTensorName: The name of the input tensor for the graph (PlaceHolder)
    ///   - resultTensorName: The name of the result tensor for the graph (must be a targe for the mode specified)
    ///   - sampleRange: (Optional) The range of samples to put into the test.  If nil all samples are used for the test.  Must be divisible by batch size if Graph batch processing in use
    /// - Returns: The total error - the sum of the absolute value of the difference between the predicted and actual values of all elements of the result tensors of all samples specified
    public func runRegressionTest(mode: String, testDataSet: DataSet, inputTensorName: String, resultTensorName: String, sampleRange: ClosedRange<Int>? = nil) async throws -> Double {
        //  Verify the input tensors are usable
        if (!testDataSet.inputType.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
        if (!testDataSet.outputType.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
        
        //  Verify the number of samples is divisible by the batch size if needed
        if (batchGraph && sampleRange != nil) {
            let usedSamples = sampleRange!.upperBound - sampleRange!.lowerBound + 1
            if ((usedSamples % batchSize) != 0) { throw MPSGraphRunErrors.SampleSizeNotDivisibleByBatchSize }
        }

        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandQueue() }
        
        //  Get the targetTensors
        var targets : [MPSGraphTensor] = []
        for targetTensor in targetTensors {
            if targetTensor.modes.contains(mode) {
                targets.append(targetTensor.tensor)
            }
        }
        if (targets.isEmpty) { throw MPSGraphDSLErrors.NoTargetsInGraph }
        
        //  Get the input feed tensor
        let inputFeedTensorInfo = feedTensors.first(where: { $0.name == inputTensorName })
        if (inputFeedTensorInfo == nil) { throw MPSGraphRunErrors.PlaceHolderInputNotFound(inputTensorName) }
        
        //  Get the output result tensor
        let outputResultTensor: MPSGraphTensor
        let addedNode = findNamedNode(resultTensorName)
        if let addedNode = addedNode {
            outputResultTensor = addedNode.mpstensor
        }
        else {
            throw MPSGraphRunErrors.PlaceHolderInputNotFound(resultTensorName)
        }
        
        //  Get the double-buffer handler
        let doubleBufferHandler = DoubleBufferActor()
        
        //  Lock the dataset
        try await testDataSet.lockForMultiSampleUse()
        
        //  Get the sample range
        let numSamples = await testDataSet.numSamples
        let startSample: Int
        let endSample: Int
        if let sampleRange = sampleRange {
            startSample = sampleRange.lowerBound
            endSample = sampleRange.upperBound
        }
        else {
            startSample = 0
            endSample = numSamples-1
        }
        
        //  Get the number of runs required
        var numberOfRuns: Int = endSample - startSample + 1
        if (batchGraph) {
            numberOfRuns /= batchSize
        }
        
        //  Create the batch tensor
        var batchInputTensor: Tensor?
        var batchOutputTensor: Tensor?
        if (batchGraph) {
            let batchInputTensorShape = testDataSet.inputShape.shapeWithAddedBatchDimension(batchSize)
            batchInputTensor = CreateTensor.constantValues(type: testDataSet.inputType, shape: batchInputTensorShape, initialValue: 0.0)
            let batchOutputTensorShape = testDataSet.outputShape.shapeWithAddedBatchDimension(batchSize)
            batchOutputTensor = CreateTensor.constantValues(type: testDataSet.outputType, shape: batchOutputTensorShape, initialValue: 0.0)
        }
        
        do {
            var sampleIndex: Int = startSample
            for _ in 0..<numberOfRuns {
                //  Wait for our turn
                await doubleBufferHandler.getAllowance()

                //  Get the command buffer
                let commandBuffer = MPSCommandBuffer(commandBuffer: commandQueue.makeCommandBuffer()!)
                
                //  Get the input and output Tensors
                var inputTensor: Tensor
                var outputTensor: Tensor
                if (batchGraph) {
                    for i in 0..<batchSize {
                        let sample = try await testDataSet.getSample(sampleIndex: sampleIndex)
                        try batchInputTensor!.setBatchSample(tensor: sample.inputs, batchIndex: i)
                        try batchOutputTensor!.setBatchSample(tensor: sample.outputs, batchIndex: i)
                        sampleIndex += 1
                    }
                    inputTensor = batchInputTensor!
                    outputTensor = batchOutputTensor!
                }
                else {
                    let sample = try await testDataSet.getSample(sampleIndex: sampleIndex)
                    inputTensor = sample.inputs
                    outputTensor = sample.outputs
                    sampleIndex += 1
                }
                
                //  Convert the input tensor to MPS version and create the feed dictionary
                let feedDict : [MPSGraphTensor: MPSGraphTensorData] = [inputFeedTensorInfo!.tensor : try inputTensor.getMPSGraphTensorData(forGraph: self)]
                
                //  Create the callback
                let executionDesc = MPSGraphExecutionDescriptor()
                executionDesc.completionHandler = { (resultsDictionary, nil) in
                    let result: MPSGraphTensorData = resultsDictionary[outputResultTensor]!
                    let resultTensor = CreateTensor.fromMPSTensorData(result)
                    let runError = outputTensor.totalDifference(with: resultTensor)
                    
                    Task {
                        await doubleBufferHandler.operationComplete(error: runError)
                    }
                 }

                //  Run the graph
                let _ = mpsgraph.encode(to: commandBuffer,
                                              feeds: feedDict,
                                              targetTensors: targets,
                                              targetOperations: nil,
                                              executionDescriptor: executionDesc)

                commandBuffer.commit()
            }
        }
        catch {
            //  Make sure we release the dataset.  Cannot use defer with actor isolated code
            try await testDataSet.releaseLock()
            throw error
        }
        try await testDataSet.releaseLock()

        await doubleBufferHandler.waitTillComplete()
        let totalError = await doubleBufferHandler.totalError
        return totalError
    }

    /// Run all samples of a training DataSet through the graph, with the variable learning operations enaboed
    /// - Parameters:
    ///   - mode: The name of the learning mode of the graph
    ///   - trainingDataSet: The training DataSet
    ///   - inputTensorName: The name of the input tensor for the graph (PlaceHolder)
    ///   - expectedValueTensorName: The name of the input tensor for the expected value (PlaceHolder)
    ///   - lossTensorName: (Optional) The name of the output tensor for the loss calculation.  This node must be a target for the mode passed in
    ///   - epochSize: (Optional) The number of random samples, or random batches if a batch graph, the graph is trained on.  If nil all samples are processed sequentially.  Default is nil
    /// - Returns: The average loss value for the run - or nil if no loss tensor name was provided
    public func runTraining(mode: String, trainingDataSet: DataSet, inputTensorName: String, expectedValueTensorName : String, lossTensorName: String? = nil, epochSize: Int? = nil) async throws -> Double? {
        //  Verify the input tensors are usable
        if (!trainingDataSet.inputType.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
        if (!trainingDataSet.outputType.usableInGraph()) { throw GenericMPSGraphDSLErrors.TypeNotUsableByGraph }
        
        //  Verify the number of samples is divisible by the batch size if needed
        let numSamples = await trainingDataSet.numSamples
        if (batchGraph && epochSize == nil) {
            if ((numSamples % batchSize) != 0) { throw MPSGraphRunErrors.SampleSizeNotDivisibleByBatchSize }
        }

        //  Get the graph ready
        if (mpsgraph == nil) { try buildGraph() }
        if (device == nil) { getCommandQueue() }
        
        //  Get the targetTensors
        var targets : [MPSGraphTensor] = []
        for targetTensor in targetTensors {
            if targetTensor.modes.contains(mode) {
                targets.append(targetTensor.tensor)
            }
        }
        if (targets.isEmpty) { throw MPSGraphDSLErrors.NoTargetsInGraph }
        
        //  Get the input feed tensor
        let inputFeedTensorInfo = feedTensors.first(where: { $0.name == inputTensorName })
        if (inputFeedTensorInfo == nil) { throw MPSGraphRunErrors.PlaceHolderInputNotFound(inputTensorName) }
        
        //  Get the expected value feed tensor
        let expectedValueFeedTensorInfo = feedTensors.first(where: { $0.name == expectedValueTensorName })
        if (expectedValueFeedTensorInfo == nil) { throw MPSGraphRunErrors.PlaceHolderInputNotFound(expectedValueTensorName) }
        
        //  Find the loss tensor, if a name was provided
        var lossTensor: MPSGraphTensor? = nil
        if let lossTensorName = lossTensorName {
            if let addedNode = findNamedNode(lossTensorName) {
                lossTensor = addedNode.mpstensor
                if (!targets.contains(lossTensor!)) { throw MPSGraphRunErrors.LossNodeNotATargetForMode(lossTensorName)}
            }
            else {
                throw MPSGraphRunErrors.LossNodeNotFound(lossTensorName)
            }
        }

        //  Get the double-buffer handler
        let doubleBufferHandler = DoubleBufferActor()
        
        //  Create the batch tensors
        var batchInputTensor: Tensor?
        var batchOutputTensor: Tensor?
        if (batchGraph) {
            let batchInputTensorShape = trainingDataSet.inputShape.shapeWithAddedBatchDimension(batchSize)
            batchInputTensor = CreateTensor.constantValues(type: trainingDataSet.inputType, shape: batchInputTensorShape, initialValue: 0.0)
            let batchOutputTensorShape = trainingDataSet.outputShape.shapeWithAddedBatchDimension(batchSize)
            batchOutputTensor = CreateTensor.constantValues(type: trainingDataSet.inputType, shape: batchOutputTensorShape, initialValue: 0.0)
        }

        //  Lock the dataset
        try await trainingDataSet.lockForMultiSampleUse()

        do {
            var epochCount = numSamples
            if (batchGraph) { epochCount = numSamples / batchSize }
            if let epochSize = epochSize { epochCount = epochSize }
            var sampleIndex = 0
            for _ in 0..<epochCount {
                //  Wait for our turn
                await doubleBufferHandler.getAllowance()

                //  Get the command buffer
                let commandBuffer = MPSCommandBuffer(commandBuffer: commandQueue.makeCommandBuffer()!)

                //  Get the input and output Tensors
                var inputTensor: Tensor
                var outputTensor: Tensor
                if (batchGraph) {
                    for i in 0..<batchSize {
                        let sample: DataSample
                        if (epochSize == nil) {
                            sample = try await trainingDataSet.getSample(sampleIndex: sampleIndex)
                            sampleIndex += 1
                        }
                        else {
                            sampleIndex = Int.random(in: 0..<numSamples)
                            sample = try await trainingDataSet.getSample(sampleIndex: sampleIndex)
                        }
                        try batchInputTensor!.setBatchSample(tensor: sample.inputs, batchIndex: i)
                        try batchOutputTensor!.setBatchSample(tensor: sample.outputs, batchIndex: i)
                    }
                    inputTensor = batchInputTensor!
                    outputTensor = batchOutputTensor!
                }
                else {
                    let sample: DataSample
                    if (epochSize == nil) {
                        sample = try await trainingDataSet.getSample(sampleIndex: sampleIndex)
                        sampleIndex += 1
                    }
                    else {
                        sampleIndex = Int.random(in: 0..<numSamples)
                        sample = try await trainingDataSet.getSample(sampleIndex: sampleIndex)
                    }
                    inputTensor = sample.inputs
                    outputTensor = sample.outputs
                }

                //  Convert the input and output tensors to MPS versions and create the feed dictionary
                var feedDict : [MPSGraphTensor: MPSGraphTensorData] = [:]
                feedDict[inputFeedTensorInfo!.tensor] = try inputTensor.getMPSGraphTensorData(forGraph: self)
                feedDict[expectedValueFeedTensorInfo!.tensor] = try outputTensor.getMPSGraphTensorData(forGraph: self)

                //  Create the callback
                let executionDesc = MPSGraphExecutionDescriptor()
                executionDesc.completionHandler = { (resultsDictionary, nil) in
                    //  If a loss tensor name was provided, accumulate the loss
                    var lossValue: Double = 0.0
                    if let lossTensor = lossTensor {
                        let result = resultsDictionary[lossTensor]!
                        let resultTensor = CreateTensor.fromMPSTensorData(result)
                        let elements = resultTensor.getElements()
                        for element in elements {
                            lossValue += element
                        }
                    }
                    Task {
                        await doubleBufferHandler.operationComplete(error: lossValue)
                    }
                }

                //  Run the graph
                let _ = mpsgraph.encode(to: commandBuffer,
                                              feeds: feedDict,
                                              targetTensors: targets,
                                              targetOperations: learningOps,
                                              executionDescriptor: executionDesc)
                commandBuffer.commit()
            }
        }
        catch {
            //  Make sure we release the dataset.  Cannot use defer with actor isolated code
            try await trainingDataSet.releaseLock()
            throw error
        }
        try await trainingDataSet.releaseLock()
        await doubleBufferHandler.waitTillComplete()
        
        if (lossTensorName == nil) {
            return nil
        }
        else {
            return await doubleBufferHandler.totalError
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
            
            print ("\(addedNode.operation) produces Tensor of shape: \(addedNode.outputShape.dimensions) - named: \(nameString)")
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
        if (device == nil) { getCommandQueue() }
        
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
            let data = try assign.node.getResetData(forGraph: self)
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

internal actor DoubleBufferActor {
    var numOutstandingOperations: Int
    var numCorrect: Int     //  Used for classification testing
    var totalError: Double     //  Used for regression testing and loss accumulation

    init() {
        numOutstandingOperations = 0
        numCorrect = 0
        totalError = 0.0
    }
    
    func getAllowance() async {
        while (numOutstandingOperations >= 2) {
            await Task.yield()
        }
        
        numOutstandingOperations += 1
    }
    
    func operationComplete() {
        numOutstandingOperations -= 1
    }
    
    func operationComplete(correct: Bool) {
        if (correct) { numCorrect += 1 }
        numOutstandingOperations -= 1
    }
    
    func operationComplete(numCorrect: Int) {
        self.numCorrect += numCorrect
        numOutstandingOperations -= 1
    }
    
    func operationComplete(error: Double) {
        self.totalError += error
        numOutstandingOperations -= 1
    }

    func waitTillComplete() async {
        while (numOutstandingOperations > 0) {
            await Task.yield()
        }
    }
}
