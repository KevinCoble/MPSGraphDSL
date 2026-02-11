//
//  LeafNodes.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/18/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph


///  Leaf Node that represents an input to be supplied later
///  Not for SubGraphs - use SubGraphPlaceholder instead
public class PlaceHolder : Node {
    let shape : TensorShape
    let modes: [String]
    var batchExempt: Bool = false   //  If true the placeholder doesn't get a batch dimension added to it at build time
    
    /// Constructor for an input tensor placeholder
    /// 
    /// - Parameters:
    ///   - shape: The dimensional shape of the output tensor, specified as an array of Int.  Do not include batch dimension if using Graph batch
    ///   - modes: (Optional) An array of strings for the modes this placeholder needs to be filled with data.  If an empty array, all modes are assumed.  Default is the empty array
    ///   - name:  The name for this node and its associated tensor.  Required for mapping input tensors into the graph
    public init(shape: [Int], modes: [String] = [], name: String) {
        self.shape = TensorShape(shape)
        self.modes = modes
        super.init(name: name)
    }
    
    ///  Constructor for an input tensor placeholder
    ///
    /// - Parameters:
    ///   - shape: The dimensional shape of the output tensor, specified as a TensorShape struct.  Do not include batch dimension if using Graph batch
    ///   - modes: (Optional) An array of strings for the modes this placeholder needs to be filled with data.  If an empty array, all modes are assumed.  Default is the empty array
    ///   - name:  The name for this node and its associated tensor.  Required for mapping input tensors into the graph
    public init(shape: TensorShape, modes: [String] = [], name: String) {
        self.shape = shape
        self.modes = modes
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Get the shape
        var mpsShape = shape.getMPSShape()
        if (graph.batchGraph && !batchExempt) {
            mpsShape.insert(NSNumber(value: graph.batchSize), at: 0)
        }
        
        //  Add to the graph itself
        let inputPlaceholder = graph.mpsgraph.placeholder(shape: mpsShape, name: graph.getFullName(name))
        
        //  Add the placeholder to the feeds list
        let feedTensorInfo = feedTensorInfo(name: graph.getFullName(name)!, tensor: inputPlaceholder, modes: modes, batchExemption: batchExempt)
        graph.feedTensors.append(feedTensorInfo)
        
        //  Return the created MPSGraphTensor
        return [inputPlaceholder]
    }
    
    ///  Modifier to turn on batch exemption for the PlaceHolder.  Default is false
    public func isBatchExempt() -> PlaceHolder {
        self.batchExempt = true
        return self
    }
}


///  Leaf Node that represents an input into a SubGraph
///  Not for Graphs - use Placeholder instead
public class SubGraphPlaceHolder : Node {
    
    /// Constructor for an input tensor placeholder
    ///
    /// - Parameters:
    ///   - name: The name for this node - used by the SubGraph node to map inputs
    public init(name: String) {
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Use the current input map to find the name of input to be used
        if let inputName = graph.currentSubGraphInputMap[self.name!] {
            //  Find the node with the input name and set its output tensor to our output tensor
            let referenceInputTensor = try graph.getOptionalTensor(inputName)
            
            return [referenceInputTensor]
        }
        else {
            throw MPSGraphDSLErrors.SubGraphPlaceHolderNotInInputMap(self.name!)
        }
    }
    
    override internal func isReferenced() throws {
        //  Don't throw
    }
}


///  Leaf Node that provides a constant value tensor in the specified shape
public class Constant : Node {
    var shape : TensorShape
    let value: DataElement?
    let values: Tensor?
    let tensorReference: String?

    /// Construct a Constant node with the specified shape and repeated value
    /// - Parameters:
    ///   - shape: The dimensional shape of the output tensor, specified as an array of Int
    ///   - value: The value that should be repeated into all elements of the constant tensor.  The value supplied should be a DataElement of the type wanted for the tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(shape: [Int], value: DataElement, name: String? = nil) {
        self.shape = TensorShape(shape)
        self.value = value
        values = nil
        tensorReference = nil
        super.init(name: name)
    }
    /// Construct a Constant node with the specified shape and repeated value
    /// - Parameters:
    ///   - shape: The dimensional shape of the output tensor, specified as a TensorShape struct
    ///   - value: The value that should be repeated into all elements of the constant tensor.  The value supplied should be a DataElement of the type wanted for the tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(shape: TensorShape, value: DataElement, name: String? = nil) {
        self.shape = shape
        self.value = value
        values = nil
        tensorReference = nil
        super.init(name: name)
    }
    
    
    /// Construct a Constant node with the specified shape and given initial values
    /// - Parameters:
    ///   - values: The tensor that will be used to initialize the constant MPSGraphTensor.  This also provides the shape and data type of the MPSGraphTensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(values: Tensor, name: String? = nil) {
        self.shape = values.shape
        self.value = nil
        self.values = values
        tensorReference = nil
        super.init(name: name)
    }

    /// Construct a Constant node with the specified shape and given initial values from a ''Tensor'' referenced by string from a map on an enclosing ''SubGraph'' node
    /// - Parameters:
    ///   - tensorReference: The reference string for a ''Tensor'' provided by an enclosing ''SubGraph'' node
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(tensorReference: String, name: String? = nil) {
        self.shape = TensorShape([1])       //  To be overwritten by referenced tensor
        self.value = nil
        self.values = nil
        self.tensorReference = tensorReference
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        //  Add to the graph itself
        if let value = value {
            let constant = graph.mpsgraph.constant(value.asDouble, shape: shape.getMPSShape(), dataType: value.dataType.getMPSDataType())
            
            //  Return the created MPSGraphTensor
            return [constant]
        }
        else {
            //  Get the data tensor
            let tensor: Tensor
            //  If a reference version - get the Tensor from the current SubGraph map
            if let tensorReference = tensorReference {
                if let referenceTensor = graph.dataTensorMap[tensorReference] {
                    tensor = referenceTensor
                    shape = tensor.shape
                }
                else {
                    throw DataParsingErrors.ReferencedDataTensorNotFound(tensorReference)
                }
            }
            else {
                tensor = values!
            }

            //  Add to the graph itself
            let data = tensor.getData()
            let constant = graph.mpsgraph.constant(data, shape: shape.getMPSShape(), dataType: tensor.type.getMPSDataType())
            
            //  Return the created MPSGraphTensor
            return [constant]
        }
    }
}


///  Leaf Node that provides a variable value tensor in the specified shape, initialized with given values from a tensor
///
///  It can be created using the initialization ''Tensor'', or with a string reference to a ''Tensor'' to be provided by a mapping in a ''SubGraph''
public class Variable : Node {
    internal enum VariableSource {
        case tensor(Tensor)
        case tensorReference(String)
        case randomUniformValues(range: ParameterRange, orthogonal: Bool)
        case randomNormalValues(mean: Double, standardDeviation: Double, orthogonal: Bool)
        case constant(Double)
        case inputTensor(String)
    }
    let valueSource: VariableSource
    var dataType: DataType?
    var shape: TensorShape?
    var lossNode: String? = nil
    var referenceTensor: Tensor? = nil

    /// Construct a Variable node with shape, type,  and initial values from a ``Tensor``
    /// - Parameters:
    ///   - values: The tensor that will be used to initialize the constant MPSGraphTensor.  This also provides the shape and data type of the MPSGraphTensor
    ///   - name: The name for this node and its associated tensor
    public init(values: Tensor, name: String) {
        self.valueSource = .tensor(values)
        self.dataType = values.type
        self.shape = values.shape
        super.init(name: name)
    }

    /// Construct a Variable node with the  shape, type,  and initial values from a ''Tensor'' referenced by string from a map on an enclosing ''SubGraph'' node
    /// - Parameters:
    ///   - tensorReference: The reference string for a ''Tensor'' provided by an enclosing ''SubGraph'' node
    ///   - name: The name for this node and its associated tensor
    public init(tensorReference: String, name: String) {
        self.valueSource = .tensorReference(tensorReference)
        self.dataType = nil
        self.shape = nil
        super.init(name: name)
    }
    
    /// Contruct a Variable node with the specified type, shape, and uniform random values
    /// - Parameters:
    ///   - dataType: the type of data stored by the variable
    ///   - shape: the shape of the data stored by the variable
    ///   - randomValueRange: the range of random values used to initialize the variable
    ///   - name: The name for this node and its associated tensor
    public init(dataType: DataType, shape : TensorShape, randomValueRange: ParameterRange, orthogonal: Bool = false, name: String) {
        self.valueSource = .randomUniformValues(range: randomValueRange, orthogonal: orthogonal)
        self.dataType = dataType
        self.shape = shape
        super.init(name: name)
    }
    
    /// Contruct a Variable node with the specified type, shape, and gaussian random values
    /// - Parameters:
    ///   - dataType: the type of data stored by the variable
    ///   - shape: the shape of the data stored by the variable
    ///   - randomValueRange: the range of random values used to initialize the variable
    ///   - name: The name for this node and its associated tensor
    public init(dataType: DataType, shape : TensorShape, randomMean: Double, randomStdDev: Double, orthogonal: Bool = false, name: String) {
        self.valueSource = .randomNormalValues(mean: randomMean, standardDeviation: randomStdDev, orthogonal: orthogonal)
        self.dataType = dataType
        self.shape = shape
        super.init(name: name)
    }

    /// Construct a Variable node with the specified type and shape, and constant values for all entries
    /// - Parameters:
    ///   - dataType: the type of data stored by the variable
    ///   - shape: the shape of the data stored by the variable
    ///   - initialValue: The initial value set (after casting) to all entries of the variable
    ///   - name: The name for this node and its associated tensor
    public init(dataType: DataType, shape : TensorShape, initialValue: Double, name: String) {
        self.valueSource = .constant(initialValue)
        self.dataType = dataType
        self.shape = shape
        super.init(name: name)
    }
    
    /// Construct a Variable node from another graph node
    /// - Parameters:
    ///   - inputTensor: The name of the Node that provides the initial shape, type and data for the Variable
    ///   - name: The name for this node and its associated tensor
    public init(inputTensor: String, name: String) {
        self.valueSource = .inputTensor(inputTensor)
        self.dataType = nil
        self.shape = nil
        super.init(name: name)
    }

    /// Modifier to configure the Variable to learn
    /// - Parameter lossNode: the name of the loss calculation in the Graph
    /// - Returns: The modified Variable
    public func learnWithRespectTo(_ lossNode: String) -> Variable {
        self.lossNode = lossNode
        return self
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        let tensor: Tensor?
        
        //  Get the data tensor
        switch (valueSource) {
        case .tensor(let sourceTensor):
            tensor = sourceTensor
        case .tensorReference(let tensorReference):
            if let refTensor = graph.dataTensorMap[tensorReference] {
                tensor = refTensor
                referenceTensor = refTensor
            }
            else {
                throw DataParsingErrors.ReferencedDataTensorNotFound(tensorReference)
            }
        case .randomUniformValues(let range, let orthogonal):
            if (orthogonal) {
                //  Assume square for each gate
                let numGates = shape!.dimensions[0] / shape!.dimensions[1]
                let squareShape = TensorShape([shape!.dimensions[1], shape!.dimensions[1]])
                let initializationInfo = WeightInitialization.uniform(min: range.min.asDouble, max: range.max.asDouble)
                tensor = try CreateTensor.createOrthogonalWeightInitializationTensor(type: dataType!, shape: squareShape, initializationInfo: initializationInfo, numGates: numGates)
            }
            else {
                tensor = CreateTensor.randomUniformValues(type: dataType!, shape: shape!, range: range)
            }
        case .randomNormalValues(let mean, let standardDeviation, let orthogonal):
            if (orthogonal) {
                //  Assume square for each gate
                let numGates = shape!.dimensions[0] / shape!.dimensions[1]
                let squareShape = TensorShape([shape!.dimensions[1], shape!.dimensions[1]])
                let initializationInfo = WeightInitialization.normal(mean: mean, standardDeviation: standardDeviation)
                tensor = try CreateTensor.createOrthogonalWeightInitializationTensor(type: dataType!, shape: squareShape, initializationInfo: initializationInfo, numGates: numGates)
            }
            else {
                tensor = CreateTensor.randomNormalValues(type: dataType!, shape: shape!, mean: mean, standardDeviation: standardDeviation)
            }
        case .constant(let value):
            tensor = CreateTensor.constantValues(type: dataType!, shape: shape!, initialValue: value)
        case .inputTensor:
            tensor = nil
        }

        //  Add to the graph itself
        let variable: MPSGraphTensor
        var sourceTensor : MPSGraphTensor? = nil
        if case let .inputTensor(inputName) = valueSource {
            if let addedNode = graph.findNamedNode(inputName) {
                sourceTensor = addedNode.mpstensor
                variable = graph.mpsgraph.variableFromTensor(sourceTensor!, name: graph.getFullName(name))
                dataType = DataType(from: sourceTensor!.dataType)
                if let sourceShape = sourceTensor!.shape {
                    shape = TensorShape(fromMPS: sourceShape)
                }
            }
            else {
                throw MPSGraphDSLErrors.NamedTensorNotFound(inputName)
            }
        }
        else {
            let data = tensor!.getData()
            variable = graph.mpsgraph.variable(with: data, shape: shape!.getMPSShape(), dataType: dataType!.getMPSDataType(), name: graph.getFullName(name))

        }
        
        //  If we are adding load or reset assignments, put this variable on the list for load assignments
        if (graph.buildOptions.contains(.addLoadAssigns) || graph.buildOptions.contains(.addResetAssigns)) {
            let loadResetAssignInfo = LoadResetAssignInfo(node: self, variableTensor: variable, sourceTensor: sourceTensor)
            graph.loadResetAssignList.append(loadResetAssignInfo)
        }

        //  If this is a learning variable - add to the list to get assignment operations for
        if let lossNode = lossNode {
            graph.learningVariables.append((variable: self, tensor: variable, loss: lossNode))
        }
        
        //  Return the created MPSGraphTensor
        return [variable]
    }
    
    public static func createWeightInitializationVariable(type: DataType, shape: TensorShape, initializationInfo: WeightInitialization, numInputs: Int, numOutput: Int, orthogonal: Bool = false, name: String) throws -> Variable {
        switch initializationInfo {
        case .uniform(let min, let max):
            let range = try ParameterRange(minimum: min, maximum: max)
            return Variable(dataType: type, shape : shape, randomValueRange: range, orthogonal: orthogonal, name: name)
        case .normal(let mean, let standardDeviation):
            return Variable(dataType: type, shape : shape, randomMean: mean, randomStdDev: standardDeviation, orthogonal: orthogonal, name: name)
        case .XavierGlorotUniform:
            let max = sqrt(6.0 / Double(numInputs + numOutput))
            let min = -max
            let range = try ParameterRange(minimum: min, maximum: max)
            return Variable(dataType: type, shape : shape, randomValueRange: range, orthogonal: orthogonal, name: name)
        case .HeUniform:
            let max = sqrt(6.0 / Double(numInputs))
            let min = -max
            let range = try ParameterRange(minimum: min, maximum: max)
            return Variable(dataType: type, shape : shape, randomValueRange: range, orthogonal: orthogonal, name: name)
        case .XavierGlorotNormal:
            let standardDeviation = sqrt(2.0 / Double(numInputs + numOutput))
            return Variable(dataType: type, shape : shape, randomMean: 0.0, randomStdDev: standardDeviation, orthogonal: orthogonal, name: name)
        case .HeNormal:
            let standardDeviation = sqrt(2.0 / Double(numInputs))
            return Variable(dataType: type, shape : shape, randomMean: 0.0, randomStdDev: standardDeviation, orthogonal: orthogonal, name: name)
        }
    }
    
    internal func getResetData(forGraph: Graph) throws -> MPSGraphTensorData? {
        switch (valueSource) {
        case .tensor(let sourceTensor):
            return try sourceTensor.getMPSGraphTensorData(forGraph: forGraph)
        case .tensorReference:
            if let referenceTensor = referenceTensor {
                return try referenceTensor.getMPSGraphTensorData(forGraph: forGraph)
            }
            else {
                return nil
            }
        case .randomUniformValues(let range, let orthogonal):
            if (orthogonal) {
                //  Assume square for each gate
                let numGates = shape!.dimensions[0] / shape!.dimensions[1]
                let squareShape = TensorShape([shape!.dimensions[1], shape!.dimensions[1]])
                let initializationInfo = WeightInitialization.uniform(min: range.min.asDouble, max: range.max.asDouble)
                let tensor = try CreateTensor.createOrthogonalWeightInitializationTensor(type: dataType!, shape: squareShape, initializationInfo: initializationInfo, numGates: numGates)
                return try tensor.getMPSGraphTensorData(forGraph: forGraph)
            }
            else {
                let tensor = CreateTensor.randomUniformValues(type: dataType!, shape: shape!, range: range)
                return try tensor.getMPSGraphTensorData(forGraph: forGraph)
            }
        case .randomNormalValues(let mean, let standardDeviation, let orthogonal):
            if (orthogonal) {
                //  Assume square for each gate
                let numGates = shape!.dimensions[0] / shape!.dimensions[1]
                let squareShape = TensorShape([shape!.dimensions[1], shape!.dimensions[1]])
                let initializationInfo = WeightInitialization.normal(mean: mean, standardDeviation: standardDeviation)
                let tensor = try CreateTensor.createOrthogonalWeightInitializationTensor(type: dataType!, shape: squareShape, initializationInfo: initializationInfo, numGates: numGates)
                return try tensor.getMPSGraphTensorData(forGraph: forGraph)
            }
            else {
                let tensor = CreateTensor.randomNormalValues(type: dataType!, shape: shape!, mean: mean, standardDeviation: standardDeviation)
                return try tensor.getMPSGraphTensorData(forGraph: forGraph)
            }
        case .constant(let value):
            let tensor = CreateTensor.constantValues(type: dataType!, shape: shape!, initialValue: value)
            return try tensor.getMPSGraphTensorData(forGraph: forGraph)
        case .inputTensor:
            return nil
        }
    }
}


///  Leaf Node that provides a random uniform value tensor in the specified shape, initialized with given random data
public class RandomUniformTensor : Node {
    let shape: [Int]
    let shapeTensor: String?
    let useShapeTensor: Bool
    let seed: Int?
    let stateTensor: String?
    let useStateTensor: Bool
    
    var suffixes: [String] = []

    /// Create a uniform random Float32 tensor of the given shape.  Range \[0,1)
    /// - Parameters:
    ///   - shape: The shape of the output tensor
    ///   - seed: (Optional) The seed for the random number generator.  If nil a random value is used for the seed.  Defaults to nil
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(shape: TensorShape, seed: Int? = nil, name: String? = nil) {
        self.shape = shape.dimensions
        self.shapeTensor = nil
        useShapeTensor = false
        self.seed = seed
        self.stateTensor = nil
        useStateTensor = false
        super.init(name: name)
    }
    
    /// Create a uniform random Float32 tensor of the given shape.  Range \[0,1)
    /// - Parameters:
    ///   - shape: The shape of the output tensor
    ///   - seed: (Optional) The seed for the random number generator.  If nil a random value is used for the seed.  Defaults to nil
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(shape: [Int], seed: Int? = nil, name: String? = nil) {
        self.shape = shape
        self.shapeTensor = nil
        useShapeTensor = false
        self.seed = seed
        self.stateTensor = nil
        useStateTensor = false
        super.init(name: name)
    }
    
    /// Create a uniform random Float32 tensor of a shape given by a tensor.  Range \[0,1)
    /// - Parameters:
    ///   - shapeTensor: (Optional) The name of a tensor that provedes the shape of the output tensor.  If nil the previous node's output will be used
    ///   - seed: (Optional) The seed for the random number generator.  If nil a random value is used for the seed.  Defaults to nil
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(shapeTensor: String? = nil, seed: Int? = nil, name: String? = nil) {
        self.shape = []
        self.shapeTensor = shapeTensor
        useShapeTensor = true
        self.seed = seed
        self.stateTensor = nil
        useStateTensor = false
        super.init(name: name)
    }

    /// Create a uniform random Float32 tensor of the given shape using a state tensor to control the random values.  Range \[0,1)
    ///   Output will be two tensors with suffixes "_random" and "_state"
    /// - Parameters:
    ///   - shape: The shape of the output tensor
    ///   - stateTensor: (Optional) The name of the tensor for the state of the random generator.  If nil a the previous node's output will be used  Defaults to nil
    ///   - name:The name for this node and its associated tensor
    public init(shape: TensorShape, stateTensor: String? = nil, name: String) {
        self.shape = shape.dimensions
        self.shapeTensor = nil
        useShapeTensor = false
        self.seed = 0
        self.stateTensor = stateTensor
        useStateTensor = true
        super.init(name: name)
    }
    
    /// Create a uniform random Float32 tensor of the given shape using a state tensor to control the random values..  Range \[0,1)
    /// - Parameters:
    ///   - shape: The shape of the output tensor
    ///   - stateTensor: (Optional) The name of the tensor for the state of the random generator.  If nil a the previous node's output will be used  Defaults to nil
    ///   - name:The name for this node and its associated tensor
    public init(shape: [Int], stateTensor: String? = nil, name: String? = nil) {
        self.shape = shape
        self.shapeTensor = nil
        useShapeTensor = false
        self.seed = 0
        self.stateTensor = stateTensor
        useStateTensor = true
        super.init(name: name)
    }
    
    /// Create a uniform random Float32 tensor of a shape given by a tensor using a state tensor to control the random values..  Range \[0,1)
    /// - Parameters:
    ///   - shapeTensor: (Optional) The name of a tensor that provedes the shape of the output tensor.  If nil the previous node's output will be used
    ///   - stateTensor: (Optional) The name of the tensor for the state of the random generator.  If nil a the previous node's output will be used  Defaults to nil
    ///   - name:The name for this node and its associated tensor
    public init(shapeTensor: String? = nil, stateTensor: String? = nil, name: String? = nil) {
        self.shape = []
        self.shapeTensor = shapeTensor
        useShapeTensor = true
        self.seed = 0
        self.stateTensor = stateTensor
        useStateTensor = true
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        if (useStateTensor) {
            let results: [MPSGraphTensor]

            //  Get the state tensor
            let stateMPSTensor = try graph.getOptionalTensor(stateTensor)

            if (useShapeTensor) {
                let shapeMPSTensor = try graph.getOptionalTensor(shapeTensor)
                
                results = graph.mpsgraph.randomUniformTensor(withShapeTensor: shapeMPSTensor, stateTensor: stateMPSTensor, name: graph.getFullName(name))
            }
            else {
                results = graph.mpsgraph.randomUniformTensor(withShape: shape.map { NSNumber(value: $0)}, stateTensor: stateMPSTensor, name: graph.getFullName(name))
             }
            suffixes = ["_random", "_state"]
            
            return results
        }
        else {
            let result: MPSGraphTensor
            
            if (useShapeTensor) {
                let shapeMPSTensor = try graph.getOptionalTensor(shapeTensor)
                
                if let seed = seed {
                    result = graph.mpsgraph.randomUniformTensor(withShapeTensor: shapeMPSTensor, seed: seed, name: graph.getFullName(name))
                }
                else {
                    result = graph.mpsgraph.randomUniformTensor(withShapeTensor: shapeMPSTensor, name: graph.getFullName(name))
                }
            }
            else {
                if let seed = seed {
                    result = graph.mpsgraph.randomUniformTensor(withShape: shape.map { NSNumber(value: $0)}, seed: seed, name: graph.getFullName(name))
                }
                else {
                    result = graph.mpsgraph.randomUniformTensor(withShape: shape.map { NSNumber(value: $0)}, name: graph.getFullName(name))
                }
            }
            
            suffixes = [""]
            
            return [result]
        }
    }
    
    override internal func getNodeSuffixes() -> [String] {
        return suffixes
    }
}



///  Leaf Node that provides a tensor of a specified shape with coordinate values along the given axis
public class CoordinateTensor : Node {
    let axis: Int
    let axisTensor: String?
    let useAxisTensor: Bool
    let shape: [Int]
    let shapeTensor: String?
    let useShapeTensor: Bool

    /// Create a coordinate tensor of the given shape with the coordinate index along the specified axis filled into the values
    /// - Parameters:
    ///   - alongAxis: the axis along which the coordinate values are used to fill in the tensor
    ///   - withShape: the shape of the output tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(alongAxis: Int, withShape: [Int], name: String? = nil) {
        self.axis = alongAxis
        self.axisTensor = nil
        useAxisTensor = false
        self.shape = withShape
        self.shapeTensor = nil
        useShapeTensor = false
        super.init(name: name)
    }

    /// Create a coordinate tensor of the given shape with the coordinate index along the specified axis filled into the values
    /// - Parameters:
    ///   - alongAxis: the axis along which the coordinate values are used to fill in the tensor
    ///   - withShape: the shape of the output tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(alongAxis: Int, withShape: TensorShape, name: String? = nil) {
        self.axis = alongAxis
        self.axisTensor = nil
        useAxisTensor = false
        self.shape = withShape.dimensions
        self.shapeTensor = nil
        useShapeTensor = false
        super.init(name: name)
    }

    /// Create a coordinate tensor of the given shape with the coordinate index along the specified axis filled into the values
    /// - Parameters:
    ///   - alongAxis: the axis along which the coordinate values are used to fill in the tensor
    ///   - withShapeTensor: (Optional) The name of the tensor that will provide the shape of the output tensor.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(alongAxis: Int, withShapeTensor: String? = nil, name: String? = nil) {
        self.axis = alongAxis
        self.axisTensor = nil
        useAxisTensor = false
        self.shape = []
        self.shapeTensor = withShapeTensor
        useShapeTensor = true
        super.init(name: name)
    }

    /// Create a coordinate tensor of the given shape with the coordinate index along the specified axis filled into the values
    /// - Parameters:
    ///   - alongAxisTensor: (Optional) The name of the tensor that will provide the axis along which the coordinate values are used to fill in the tensor.  If nil the previous node's output will be used
    ///   - withShape: the shape of the output tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(alongAxisTensor: String? = nil, withShape: [Int], name: String? = nil) {
        self.axis = 0
        self.axisTensor = alongAxisTensor
        useAxisTensor = true
        self.shape = withShape
        self.shapeTensor = nil
        useShapeTensor = false
        super.init(name: name)
    }

    /// Create a coordinate tensor of the given shape with the coordinate index along the specified axis filled into the values
    /// - Parameters:
    ///   - alongAxisTensor: (Optional) The name of the tensor that will provide the axis along which the coordinate values are used to fill in the tensor.  If nil the previous node's output will be used
    ///   - withShape: the shape of the output tensor
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(alongAxisTensor: String? = nil, withShape: TensorShape, name: String? = nil) {
        self.axis = 0
        self.axisTensor = alongAxisTensor
        useAxisTensor = true
        self.shape = withShape.dimensions
        self.shapeTensor = nil
        useShapeTensor = false
        super.init(name: name)
    }

    /// Create a coordinate tensor of the given shape with the coordinate index along the specified axis filled into the values
    /// - Parameters:
    ///   - alongAxisTensor: (Optional) The name of the tensor that will provide the axis along which the coordinate values are used to fill in the tensor.  If nil the previous node's output will be used
    ///   - withShapeTensor: (Optional) The name of the tensor that will provide the shape of the output tensor.  If nil the previous node's output will be used
    ///   - name: (Optional) The name for this node and its associated tensor
    public init(alongAxisTensor: String? = nil, withShapeTensor: String? = nil, name: String? = nil) {
        self.axis = 0
        self.axisTensor = alongAxisTensor
        useAxisTensor = true
        self.shape = []
        self.shapeTensor = withShapeTensor
        useShapeTensor = true
        super.init(name: name)
    }

    override internal func addToGraph(graph: Graph) throws -> [MPSGraphTensor?] {
        let result: MPSGraphTensor
        if (useAxisTensor) {
            let axisMPSTensor = try graph.getOptionalTensor(axisTensor)
            if (useShapeTensor) {
                let shapeMPSTensor = try graph.getOptionalTensor(shapeTensor)
                result = graph.mpsgraph.coordinate(alongAxisTensor: axisMPSTensor, withShapeTensor: shapeMPSTensor, name: graph.getFullName(name))
            }
            else {
                result = graph.mpsgraph.coordinate(alongAxisTensor: axisMPSTensor, withShape: shape.map { NSNumber(value: $0)}, name: graph.getFullName(name))
            }
        }
        else {
            if (useShapeTensor) {
                let shapeMPSTensor = try graph.getOptionalTensor(shapeTensor)
                result = graph.mpsgraph.coordinate(alongAxis: axis, withShapeTensor: shapeMPSTensor, name: graph.getFullName(name))
            }
            else {
                result = graph.mpsgraph.coordinate(alongAxis: axis, withShape: shape.map { NSNumber(value: $0)}, name: graph.getFullName(name))
            }
        }
        
        //  Return the created MPSGraphTensor
        return [result]
    }
}
