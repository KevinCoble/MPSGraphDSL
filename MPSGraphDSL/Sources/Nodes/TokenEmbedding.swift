//
//  TokenEmbedding.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 2/20/26.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

///   Node for a Token Embedding layer, converting a token index value to an embedding vector
///
public class TokenEmbedding : UnaryNode {
    let vocabularySize: Int
    let embeddingLength: Int
    let embeddingType: DataType
    let table: Tensor?      //  If not nil, set translation table to constant values from this tensor
    
    var tableInitialization: WeightInitialization = .normal(mean: 0.0, standardDeviation: 1.0)
    var lossNode: String? = nil
    var learningOptimizer: LearningOptimizer = .stochasticGradientDescent
    var gradientClipping: (min: Double, max: Double)? = nil

    var suffixes: [String] = []
    var targetIndices: [Int] = []

    /// Create a learnable (translation table is an initialized variable node) token embedding node
    /// - Parameters:
    ///   - tokens: The tensor, generally of type Int32, containing the indices of the tokens to be converted into an embedded token tensor
    ///   - vocabularySize: The number of tokens that can be embedded
    ///   - embeddingLength: The length of each embedding vector
    ///   - embeddingType: the data type for the output embedding tensor
    ///   - name: The name for this node and its associated tensor.  The variable node will be named &ltname&gt_table'
    public init(tokens: String? = nil, vocabularySize: Int, embeddingLength: Int, embeddingType: DataType, name: String) {
        self.vocabularySize = vocabularySize
        self.embeddingLength = embeddingLength
        self.embeddingType = embeddingType
        table = nil
        super.init(input: tokens, name: name)
    }
    
    /// Create a pre-determined (translation table is a constant matrix) token embedding node
    /// - Parameters:
    ///   - tokens: The tensor, generally of type Int32, containing the indices of the tokens to be converted into an embedded token tensor
    ///   - translationTable: The tensor to containing the matrix of tokens (rows) to embedding vectors (columns).  The data type of this tensor dictates the output tensor type
    ///   - name: The name for this node and its associated tensor.  The variable node will be named &ltname&gt_table'
    public init(tokens: String? = nil, translationTable: Tensor, name: String) {
        let shape = translationTable.shape.dimensions
        self.vocabularySize = shape[0]
        self.embeddingLength = shape[1]
        self.embeddingType = translationTable.type
        self.table = translationTable
        super.init(input: tokens, name: name)

        if (shape.count != 2) {
            buildError = GenericMPSGraphDSLErrors.InvalidShape
        }
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the input token tensor
        let tokenTensor = try graph.getUnaryTensor(name: inputName)
        
        //  Add the table.  How depends on the configuration
        let tableTensor: MPSGraphTensor
        let tableName = graph.getFullName(name)! + "_table"
        if let table = table {
            //  Constant table
            let data = table.getData()
            tableTensor = graph.mpsgraph.constant(data, shape: [vocabularySize as NSNumber, embeddingLength as NSNumber], dataType: .float32)
        }
        else {
            //  Learnable variable table
            let tableShape = TensorShape([vocabularySize, embeddingLength])
            let table = try CreateTensor.createWeightInitializationTensor(type: embeddingType, shape: tableShape, initializationInfo: tableInitialization, numInputs: vocabularySize, numOutput: embeddingLength)
            let tableData = table.getData()
            tableTensor = graph.mpsgraph.variable(with: tableData, shape: tableShape.getMPSShape(), dataType: embeddingType.getMPSDataType(), name: tableName)
            
            //  If we are adding load or reset assignments, put this variable on the list for load assignments
            let node = try Variable.createWeightInitializationVariable(type: embeddingType, shape: tableShape, initializationInfo: tableInitialization, numInputs: vocabularySize, numOutput: embeddingLength, name: tableName)
            if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
                let loadResetAssignInfo = LoadResetAssignInfo(node: node, variableTensor: tableTensor, sourceTensor: nil)
                graph.loadResetAssignList.append(loadResetAssignInfo)
            }

            //  If this is a learning layer - add the weights to the list to get assignment operations for
            if let lossNode = lossNode {
                let learningVariable = LearningVariable(variable: node, tensor: tableTensor, loss: lossNode, clipping: gradientClipping, optimizer: learningOptimizer)
                graph.learningVariables.append(learningVariable)
            }
        }
        suffixes.append("_table")

        let gather = graph.mpsgraph.gather(withUpdatesTensor: tableTensor, indicesTensor: tokenTensor, axis: 0, batchDimensions: 0, name: name)
        suffixes.append("")

        return [tableTensor, gather]
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
    
    //  Get the indices for added tensors of target nodes that should be added to the target tensor list - if nil returned all tensors get targetted
    override internal func getTargetIndices() -> [Int]? {
        return [1]  //  The gather tensor is the target
    }
    
    ///  Modifier to set the initialization info for the random initialization of the table
    public func weightInitialization(initializerInfo: WeightInitialization) -> TokenEmbedding {
        tableInitialization = initializerInfo
        return self
    }
    
    /// Modifier to configure the layer's variables to learn
    /// - Parameters:
    ///   - mode: lossNode: the name of the loss calculation in the Graph
    ///   - using: (Optional) the optimizer method to use for learning.  Defaults to stochastic gradient descent
    ///   - gradientClipping: (Optional) defaults to nil.  A tuple with the minimum and maximum gradient values allowed in the back-propogation for this node.  The gradient is clipped to this range before being used by the optimizer
    /// - Returns: The modified layer
    public func learnWithRespectTo(_ lossNode: String, using: LearningOptimizer = .stochasticGradientDescent, gradientClipping: (min: Double, max: Double)? = nil) -> TokenEmbedding {
        self.lossNode = lossNode
        self.learningOptimizer = using
        self.gradientClipping = gradientClipping
        return self
    }
}
