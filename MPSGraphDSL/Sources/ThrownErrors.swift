//
//  ThrownErrors.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 11/17/25.
//

import Foundation

///  Errors that can be thrown by any subsystem
public enum GenericMPSGraphDSLErrors : Error {
    ///  Invalid parameter value
    case InvalidValue
    ///  Invalid type of value passed to a function
    case InvalidType
    ///  The shape of a tensor (or a location within a tensor) did not match the shape of the receiving tensor
    case InvalidShape
    ///  Dimension passed to method was outside that of defining tensor
    case InvalidDimension
    ///  The index passed to a function is outside the range of the data
    case InvalidIndex
    ///  The shape of the referenced tensor was not determinable
    case UnknownShape
    ///  Classification value outside of range
    case ClassificationValueOutOfRange
    ///  The number of values passed was not enough for the operation
    case NotEnoughValues
    ///  Internal error - should not be seen outside of public calls
    case InternalError
}


/// Errors that can be thrown by a DataSet
public enum DataSetErrors : Error {
    /// A passed in sample has an input shape that does not match the DataSet
    case SampleInputShapeMismatch
    /// A passed in sample has an output shape that does not match the DataSet
    case SampleOutputShapeMismatch
    /// A passed in sample has an input type that does not match the DataSet
    case SampleInputTypeMismatch
    /// A passed in sample has an output type that does not match the DataSet
    case SampleOutputTypeMismatch
}

/// Errors that can be thrown by the data parsing system
public enum DataParsingErrors : Error {
    /// An error occurred reading binary data from a file
    case ErrorReadingBinaryData
    /// The parse location for a DataChunk is outside of the DataSet input Tensor shape
    case InputLocationOutOfRange
    /// The parse location for a DataChunk is outside of the DataSet output Tensor shape
    case OutputLocationOutOfRange
    /// The specified Color channel is outside range of dimension 3 of the Tensor"
    case ColorChannelOutOfRange
    /// Not enough components on a text line being parsed to match format specified
    case NotEnoughComponentsOnLine
    /// String value being parsed was invalid - often occurs when the string is not a numeric value when expected
    case InvalidStringValue(String)
    ///  Unsupported text format used
    case UnsupportedTextFormat
    ///  Error reading the specified number of 'skip' lines in a text file
    case CannotReadEnoughSkipLines
    ///  Referenced data Tensor not found in current data tensor map
    case ReferencedDataTensorNotFound(String)
    ///  The number of unique labels exceed the output dimension
    case MoreUniqueLabelsThanOutputDimension
}

///  Errors that can be thrown by the MPSGraph building system
public enum MPSGraphDSLErrors : Error {
    ///  The named tensor was not found
    case NamedTensorNotFound(String)
    ///  The shapes of the two inputs to a standard binary node (add, multiply, etc.) did not match
    case NoPreviousNode
    ///   The shapes of a binary operation do not match
    case BinaryShapesDontMatch(String, String)
    ///   The shapes of a ternary operation do not match
    case TernaryShapesDontMatch(String, String, String)
    ///  The name for a node is already used in the Graph or SubGraph
    case NameNotUnique(String)
    ///  The name for a SubGraphPlaceHolder was not found in the input map for the SubGraph
    case SubGraphPlaceHolderNotInInputMap(String)
    ///  An error occurred matching the input tensor shapes to the operation.  See associated string for more information
    case InputShapeError(String)
    ///  All nodes designated as targets must have a name
    case TargetNodesMustBeNamed
    ///  The Learning node set as variable (not constant) must have a name
    case VariableLearningNodeMustBeNamed
    ///  The node designated as a target cannot be a target (usually occurs with subgraph being designated as a target)
    case NodeCannotBeTarget
    ///  The Graph must contain at least one node that is a target.  Naming the last node will automatically make it a target
    case NoTargetsInGraph
    ///  Only one Learning node is allowed per graph
    case MoreThanOneLearningNode
    ///  The tensor does not contain the dimension chosen for an operation
    case DoesNotContainDimension
    ///  The tensor specified is not one-dimensional, which is required by the operation
    case TensorNot1Dimensional(String)
    ///  The number of partitions requested is above the size of the split dimension
    case MorePartitionsThanDimensions
    ///  A modifier was added to a node that does not support it
    case ModifierNotAvailableOnNode(String)
    ///  There were no Variable nodes set to learn in the graph
    case NoLearningVariablesInGraph
    ///  A Transpose operation on a one-dimensional Tensor (vector) is undefined
    case TransposeOnVectorNotSupported
    ///  A Transpose operation on more than a two-dimensional Tensor requires dimension specification
    case TransposeNeedsDimensionSpecification
    ///  A Transpose permutation array was used that did not match the rank of the tensor, or have all axis specified
    case PermutationArrayMustHaveAllDimensions
    ///  A slice operation specifies parameters that fall outside the source tensors dimensions
    case SliceExceedsTensorDimensions
    ///  An array entry required values for each dimension of the input tensor and did not receive it (or received extra)
    case EntryRequiredForEachDimensions
    ///  A NaN propogation request was made on an operation type that doesn't support it (reduction - min and max only)
    case NaNPropogationNotSupported
    ///  The node in question does not support the operation among multiple dimensions.
    case MultipleDimensionsNotSupported
}

///  Errors that can be thrown by the MPSGraph building system when dealing with LSTM and GRU layers
public enum MPSGraphLSTMGRUErrors : Error {
    ///  Input tensors to LSTMs must have rank 3 \[T, N, I\]
    case InputTensorNot3D
    ///  The node is configured to not produce any targettable tensors although it has been configured as a target
    case NoConfiguredTargetTensorsForTargettedNode
}

///  Errors that can be thrown by the MPSGraph run system
public enum MPSGraphRunErrors : Error {
    ///  The tensor having a sample added to it does not have enough dimensions to be a batch tensor
    case NotABatchTensor
    ///  The tensor being added to a batch tensor does not have the correct shape
    case SampleDoesntMatchBatchShape
    ///  The named result tensor was not found in the graph result list
    case ResultTensorNotFound
    ///  A ``PlaceHolder`` did not have an input tensor assigned to it
    case PlaceHolderInputNotFound(String)
    ///  The graph was not built with the required option for the operation
    case GraphNotBuiltForOperation(String)
}

///  Errors that can be thrown by the persistance subsystem
public enum PersistanceErrors : Error {
    ///  Version read for a file was above that known for the object type (file written by newer version of code)
    case VersionAboveKnown
    ///  Data read did not match expected data
    case UnexpectedDataReadError
    ///  Unable to open the specified file
    case UnableToOpenFile(String)
    ///  An error occured on the input stream
    case InputStreamError(String)
    ///  The number of variables in the loading data stream does not match the number of learning variables in the graph
    case SavedCountMismatchWithLearnVariableCount
    ///  The name of a Variable was not found in the load list (created when the graph is built from all Variables)
    case SavedVariableNotFoundInLoadList(String)
    ///  The learning variable did not appear in the results of a variable read run
    case ErrorFindingLearningVariableInResults
}
