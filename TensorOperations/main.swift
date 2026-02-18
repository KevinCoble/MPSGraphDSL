//
//  main.swift
//  TensorOperations
//
//  Created by Kevin Coble on 12/5/25.
//

import Foundation
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import MPSGraphDSL

//  Set the flags for the operations to be performed
let matrixMultiplication = false
let concatenate = false
let splitting = false
let reshaping = false
let reversing = false
let transposing = false
let slicing = false
let sorting = false
let findNonZeroes = false
let reduction = false
let cumulate = false
let tiling = false
let padding = false
let squeezing = false
let topAndBottom = false
let expand = false
let banding = false
let argSort = false
let broadcast = false
let flatten = false
let onehot = false
let dropout = false
let quantize = false
let pool = false
let shapeOf = false
let depthToSpace = false
let gather = false
let sliceUpdate = true

if matrixMultiplication {
    let MMvectorTensor = try TensorFloat32(shape: TensorShape([2]), initialValues: [7.0, 8.0])
    try MMvectorTensor.print1D(elementWidth: 6, precision: 1)
    
    let MMmatrixTensor = try TensorFloat32(shape: TensorShape([2, 3]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    //                                                                              [1.0, 2.0, 3.0
    //                                                                               4.0, 5.0, 6.0]
    try MMmatrixTensor.print2D(elementWidth: 6, precision: 1)
    
    let MMgraph = Graph {
        Constant(values: MMvectorTensor, name: "vector")
        Constant(values: MMmatrixTensor, name: "matrix")
        MatrixMultiplication(primary: "vector", secondary: "matrix", name: "product")
            .targetForModes(["mmTest"])
    }
    
    let MMresults = try MMgraph.runOne(mode: "mmTest", inputTensors: [:])
    let MMresult = MMresults["product"]!
    try MMresult.print1D(elementWidth: 6, precision: 1)
    
    let element : Double = try MMmatrixTensor.getElement(location: [0, 1])     //  Remember, location is zero based!
    print("element [0, 1] = \(element)")
    let element3 : Double = try MMmatrixTensor.getElement(location: [1, 0])     //  Remember, location is zero based!
    print("element [1, 0] = \(element3)")
    let element2 : Double = try MMmatrixTensor.getElement(location: [1, 1])     //  Remember, location is zero based!
    print("element [1, 1] = \(element2)")
    let element4 : Double = try MMmatrixTensor.getElement(location: [1, 2])     //  Remember, location is zero based!
    print("element [1, 2] = \(element4)")
}

if (concatenate) {
    let matrix1Tensor = try TensorFloat32(shape: TensorShape([2, 3]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    //                                                                              [1.0, 2.0, 3.0
    //                                                                               4.0, 5.0, 6.0]
    print("tensor 1")
    try matrix1Tensor.print2D(elementWidth: 6, precision: 1)
    let matrix2Tensor = try TensorFloat32(shape: TensorShape([2, 3]), initialValues: [11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    //                                                                              [11.0, 12.0, 13.0
    //                                                                               14.0, 15.0, 16.0]
    print("tensor 2")
    try matrix2Tensor.print2D(elementWidth: 6, precision: 1)

    
    let concatGraph = Graph {
        Constant(values: matrix1Tensor, name: "matrix1")
        Constant(values: matrix2Tensor, name: "matrix2")
        Concatenate(["matrix1", "matrix2"], dimension: 0, name: "result")
            .targetForModes(["concatTest"])
    }
    
    print("Concatenate along dimension 0")
    let concatResults = try concatGraph.runOne(mode: "concatTest", inputTensors: [:])
    let concatResult = concatResults["result"]!
    try concatResult.print2D(elementWidth: 6, precision: 1)
    
    let concatGraph2 = Graph {
        Constant(values: matrix1Tensor, name: "matrix1")
        Constant(values: matrix2Tensor, name: "matrix2")
        Concatenate(["matrix1", "matrix2"], dimension: 1, name: "result")
            .targetForModes(["concatTest"])
    }
    
    print("Concatenate along dimension 1")
    let concatResults2 = try concatGraph2.runOne(mode: "concatTest", inputTensors: [:])
    let concatResult2 = concatResults2["result"]!
    try concatResult2.print2D(elementWidth: 6, precision: 1)
}

if (splitting) {
    let splitMatrixTensor = try TensorFloat32(shape: TensorShape([2, 8]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
    //                                                                        [1.0,  2.0,  3.0  4.0,  5.0,  6.0,  7.0,  8.0]
    //                                                                        11.0, 12.0, 13.0 14.0, 15.0, 16.0, 17.0, 18.0]
    print("starting tensor")
    try splitMatrixTensor.print2D(elementWidth: 6, precision: 1)
    
    let splitGraph = Graph {
        Constant(values: splitMatrixTensor, name: "matrix")
        Split(input: "matrix", axis: 1, numberOfSplits: 4, name: "result")
            .targetForModes(["splitTest"])
    }

    print("Split evenly 4 times in dimension 1 (results may not be in a particular order)")
    let splitResults = try splitGraph.runOne(mode: "splitTest", inputTensors: [:])
    for (key, value) in splitResults {
        print("\(key)")
        try value.print2D(elementWidth: 6, precision: 1)
    }
    
    let splitGraph2 = Graph {
        Constant(values: splitMatrixTensor, name: "matrix")
        Split(input: "matrix", axis: 1, splitSizes: [2, 4, 2], name: "result")
            .targetForModes(["splitTest"])
    }

    print("Split into [2, 4, 2] partitions in dimension 1 (results may not be in a particular order)")
    let splitResults2 = try splitGraph2.runOne(mode: "splitTest", inputTensors: [:])
    for (key, value) in splitResults2 {
        print("\(key)")
        try value.print2D(elementWidth: 6, precision: 1)
    }
}

if (reshaping) {
    let reshapeMatrixTensor = try TensorFloat32(shape: TensorShape([2, 3]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    //                                                                              [1.0, 2.0, 3.0
    //                                                                               4.0, 5.0, 6.0]
    print("Original tensor")
    try reshapeMatrixTensor.print2D(elementWidth: 6, precision: 1)
    
    let reshapeGraph = Graph {
        Constant(values: reshapeMatrixTensor, name: "matrix")
        Reshape(shape: TensorShape([3, 2]), name: "result")
            .targetForModes(["reshapeTest"])
    }

    print("Reshape to [3,2]")
    let reshapeResults = try reshapeGraph.runOne(mode: "reshapeTest", inputTensors: [:])
    let reshapeResult = reshapeResults["result"]!
    try reshapeResult.print2D(elementWidth: 6, precision: 1)
}

if (reversing) {
    let reverseVectorTensor = try TensorFloat32(shape: TensorShape([4]), initialValues: [1.0, 2.0, 3.0, 4.0])
    print("Original vector tensor")
    try reverseVectorTensor.print1D(elementWidth: 6, precision: 1)

    let reverseGraph = Graph {
        Constant(values: reverseVectorTensor, name: "vector")
        Reverse(name: "result")
            .targetForModes(["reverseTest"])
    }

    print("Reverse")
    var reverseResults = try reverseGraph.runOne(mode: "reverseTest", inputTensors: [:])
    var reverseResult = reverseResults["result"]!
    try reverseResult.print1D(elementWidth: 6, precision: 1)
    
    let matrixTensor = try TensorFloat32(shape: TensorShape([2, 3]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    //                                                                              [1.0, 2.0, 3.0
    //                                                                               4.0, 5.0, 6.0]
    print("")
    print("Original matrix tensor")
    try matrixTensor.print2D(elementWidth: 6, precision: 1)
    
    let reverseGraph2 = Graph {
        Constant(values: matrixTensor, name: "matrix")
        Reverse(name: "result")
            .targetForModes(["reverseTest"])
    }

    print("Reverse")
    reverseResults = try reverseGraph2.runOne(mode: "reverseTest", inputTensors: [:])
    reverseResult = reverseResults["result"]!
    try reverseResult.print2D(elementWidth: 6, precision: 1)

    let reverseGraph3 = Graph {
        Constant(values: matrixTensor, name: "matrix")
        Reverse(axes: [1], name: "result")
            .targetForModes(["reverseTest"])
    }

    print("Reverse along axis 1")
    reverseResults = try reverseGraph3.runOne(mode: "reverseTest", inputTensors: [:])
    reverseResult = reverseResults["result"]!
    try reverseResult.print2D(elementWidth: 6, precision: 1)
}

if (transposing) {
    let transposeMatrixTensor = try TensorFloat32(shape: TensorShape([3, 4]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    //                                                                              [1.0,  2.0,  3.0   4.0
    //                                                                               5.0,  6.0,  7.0,  8.0
    //                                                                               9.0, 10.0, 11.0, 12.0]
    print("Original tensor")
    try transposeMatrixTensor.print2D(elementWidth: 6, precision: 1)

    var transposeGraph = Graph {
        Constant(values: transposeMatrixTensor, name: "matrix")
        Transpose(name: "transpose")
            .targetForModes(["transposeTest"])
    }

    print("Standard transpose of Matrix")
    var transposeResults = try transposeGraph.runOne(mode: "transposeTest", inputTensors: [:])
    var transposeResult = transposeResults["transpose"]!
    try transposeResult.print2D(elementWidth: 6, precision: 1)

    transposeGraph = Graph {
        Constant(values: transposeMatrixTensor, name: "matrix")
        Transpose(permutation: [1, 0], name: "transpose")
            .targetForModes(["transposeTest"])
    }

    print("Permutation transpose of Matrix")
    transposeResults = try transposeGraph.runOne(mode: "transposeTest", inputTensors: [:])
    transposeResult = transposeResults["transpose"]!
    try transposeResult.print2D(elementWidth: 6, precision: 1)
}

if (slicing) {
    var initialValues : [Float32] = []
    for i in 1...50 { initialValues.append(Float32(i)) }
    let sliceMatrixTensor = try TensorFloat32(shape: TensorShape([5, 10]), initialValues: initialValues)
    //                                                            [ 1.0,  2.0,  3.0   4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0
    //                                                             11.0, 12.0, 13.0  14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
    //                                                             21.0, 22.0, 23.0  24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0
    //                                                             31.0, 32.0, 33.0  34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0
    //                                                             41.0, 42.0, 43.0  44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0]
    print("Original tensor")
    try sliceMatrixTensor.print2D(elementWidth: 6, precision: 1)

    var sliceGraph = Graph {
        Constant(values: sliceMatrixTensor, name: "matrix")
        Slice(dimension: 1, start: 3, length: 2, name: "slice")
            .targetForModes(["sliceTest"])
    }

    print("Slice of Tensor along dimension 1 (columns), starting at 3 and going for 2 elements")
    var sliceResults = try sliceGraph.runOne(mode: "sliceTest", inputTensors: [:])
    var sliceResult = sliceResults["slice"]!
    try sliceResult.print2D(elementWidth: 6, precision: 1)

    sliceGraph = Graph {
        Constant(values: sliceMatrixTensor, name: "matrix")
        Slice(starts: [1, 2], ends: [4, 9], strides: [1, 2], name: "slice")
            .targetForModes(["sliceTest"])
    }

    print("Slice of Tensor starting at [1, 2], ending at [4,9], using strides [1, 2]")
    sliceResults = try sliceGraph.runOne(mode: "sliceTest", inputTensors: [:])
    sliceResult = sliceResults["slice"]!
    try sliceResult.print2D(elementWidth: 6, precision: 1)

    sliceGraph = Graph {
        Constant(values: sliceMatrixTensor, name: "matrix")
        Slice(starts: [1, 2], ends: [4, 9], strides: [1, 2], ignoreStartsOnDimension: [1], removeDimensions: [0], name: "slice")
            .targetForModes(["sliceTest"])
    }

    print("Slice of Tensor starting at [1, 2], ending at [4,9], using strides [1, 2], ignoring starts on dimension 1, and removing dimension 0")
    sliceResults = try sliceGraph.runOne(mode: "sliceTest", inputTensors: [:])
    sliceResult = sliceResults["slice"]!
    try sliceResult.print1D(elementWidth: 6, precision: 1)
}

if (sorting) {
    let sortMatrixTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Original tensor")
    try sortMatrixTensor.print2D(elementWidth: 6, precision: 1)

    var sortGraph = Graph {
        Constant(values: sortMatrixTensor, name: "matrix")
        Sort(axis: 0, name: "sort")
            .targetForModes(["sortTest"])
    }

    print("Sort of Tensor along axis 0 (rows)")
    var sortResults = try sortGraph.runOne(mode: "sortTest", inputTensors: [:])
    var sortResult = sortResults["sort"]!
    try sortResult.print2D(elementWidth: 6, precision: 1)

    sortGraph = Graph {
        Constant(values: sortMatrixTensor, name: "matrix")
        Sort(axis: 1, descending: true, name: "sort")
            .targetForModes(["sortTest"])
    }

    print("Sort of Tensor along axis 1 (columns) - descending")
    sortResults = try sortGraph.runOne(mode: "sortTest", inputTensors: [:])
    sortResult = sortResults["sort"]!
    try sortResult.print2D(elementWidth: 6, precision: 1)
}

if (findNonZeroes) {
    let someZeroesTensor = try TensorFloat32(shape: TensorShape([3, 4]), initialValues: [1.0, 0.0, 3.0, 4.0, 0.0, 6.0, 0.0, 8.0, 9.0, 0.0, 11.0, 0.0])
    //                                                                              [1.0, 0.0, 3.0, 4.0
    //                                                                               0.0, 6.0, 0.0, 8.0
    //                                                                               9.0, 0.0,11.0, 0.0]
    print("tensor with some non-zeroes")
    try someZeroesTensor.print2D(elementWidth: 6, precision: 1)
    
    let findZeroesGraph = Graph {
        Constant(values: someZeroesTensor, name: "someZeroesTensor")
        NonZeroIndices(name: "result")
            .targetForModes(["findNonZeroesTest"])
    }
    
    print("Find non-zero elements")
    let findResults = try findZeroesGraph.runOne(mode: "findNonZeroesTest", inputTensors: [:])
    let findResult = findResults["result"]!
    try findResult.print2D(elementWidth: 6, precision: 0)
}

if (reduction) {
    let reductionTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor before reduction")
    try reductionTensor.print2D(elementWidth: 6, precision: 1)

    var reductionGraph = Graph {
        Constant(values: reductionTensor, name: "inputTensor")
        Reduction(op: .sum, axis: 0, name: "result")
            .targetForModes(["reductionTest"])
    }
    
    print("Reduce with sum along axis 0 (rows)")
    var reductionResults = try reductionGraph.runOne(mode: "reductionTest", inputTensors: [:])
    var reductionResult = reductionResults["result"]!
    try reductionResult.print2D(elementWidth: 6, precision: 1)

    reductionGraph = Graph {
        Constant(values: reductionTensor, name: "inputTensor")
        Reduction(op: .max, axes: [0, 1], name: "result")
            .targetForModes(["reductionTest"])
    }
    
    print("Reduce with maximum along both axes")
    reductionResults = try reductionGraph.runOne(mode: "reductionTest", inputTensors: [:])
    reductionResult = reductionResults["result"]!
    try reductionResult.print2D(elementWidth: 6, precision: 1)
}

if (cumulate) {
    let cumulationTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor before cumulation")
    try cumulationTensor.print2D(elementWidth: 6, precision: 1)
    
    var cumulationGraph = Graph {
        Constant(values: cumulationTensor, name: "inputTensor")
        Cumulate(op: .sum, axis: 0, name: "result")
            .targetForModes(["cumulationTest"])
    }
    
    print("Cumulate sum along axis 0")
    var cumulationResults = try cumulationGraph.runOne(mode: "cumulationTest", inputTensors: [:])
    var cumulationResult = cumulationResults["result"]!
    try cumulationResult.print2D(elementWidth: 6, precision: 1)
    
    cumulationGraph = Graph {
        Constant(values: cumulationTensor, name: "inputTensor")
        Cumulate(op: .max, axis: 0, exclusive: true, reverse: true, name: "result")
            .targetForModes(["cumulationTest"])
    }
    
    print("Cumulate max along axis 0, exclusive and reversed")
    cumulationResults = try cumulationGraph.runOne(mode: "cumulationTest", inputTensors: [:])
    cumulationResult = cumulationResults["result"]!
    try cumulationResult.print2D(elementWidth: 6, precision: 1)
}

if (tiling) {
    let initialTensor = try TensorFloat32(shape: TensorShape([2, 3]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    //                                                                              [1.0, 2.0, 3.0
    //                                                                               4.0, 5.0, 6.0]
    print("")
    print("Original tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let tileGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        TileTensor("startTensor", multipliers: [3, 2], name: "result")
            .targetForModes(["tileTest"])
    }

    print("Tiled [3, 2]")
    let tileResults = try tileGraph.runOne(mode: "tileTest", inputTensors: [:])
    let tileResult = tileResults["result"]!
    try tileResult.print2D(elementWidth: 6, precision: 1)
}

if padding {
    let paddingTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor before padding")
    try paddingTensor.print2D(elementWidth: 6, precision: 1)
    
    var padGraph = Graph {
        Constant(values: paddingTensor, name: "startTensor")
        Padding("startTensor", padMode: .clampToEdge, leftPadding: [3, 2], rightPadding: [1, 2], name: "result")
            .targetForModes(["padTest"])
    }

    print("padded with clamp-to-edge, left [3, 2], right: [1,2]")
    var padResults = try padGraph.runOne(mode: "padTest", inputTensors: [:])
    var padResult = padResults["result"]!
    try padResult.print2D(elementWidth: 6, precision: 1)
    
    padGraph = Graph {
        Constant(values: paddingTensor, name: "startTensor")
        Padding("startTensor", padMode: .symmetric, leftPadding: [2, 2], rightPadding: [2, 2], name: "result")
            .targetForModes(["padTest"])
    }

    print("padded with symetric, left [2, 2], right: [2,2]")
    padResults = try padGraph.runOne(mode: "padTest", inputTensors: [:])
    padResult = padResults["result"]!
    try padResult.print2D(elementWidth: 6, precision: 1)
}

if (squeezing) {
    let initialTensor = try TensorFloat32(shape: TensorShape([6, 1]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    print("Initial Tensor before squeezing")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let squeezeGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        Squeeze("startTensor", name: "result")
            .targetForModes(["squeezeTest"])
    }

    print("Squeezed on all dimensions of size 1")
    let squeezeResults = try squeezeGraph.runOne(mode: "squeezeTest", inputTensors: [:])
    let squeezeResult = squeezeResults["result"]!
    try squeezeResult.print1D(elementWidth: 6, precision: 1)
}

if (topAndBottom) {
    let initialTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let bottomKGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        BottomK("startTensor", axis: 0, k: 2, name: "result")
            .targetForModes(["bottomTest"])
        Cast(input: "result_indices", newType: .float32, name: "indices")
            .targetForModes(["bottomTest"])
    }

    print("Bottom K on dimension 0 (rows) of size 2")
    print("Values:")
    var results = try bottomKGraph.runOne(mode: "bottomTest", inputTensors: [:])
    var result = results["result_values"]!
    try result.print2D(elementWidth: 6, precision: 1)
    print("Indices:")
    result = results["indices"]!
    try result.print2D(elementWidth: 6, precision: 0)
    
    let topKGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        TopK("startTensor", axis: 1, k: 3, name: "result")
            .targetForModes(["topTest"])
        Cast(input: "result_indices", newType: .float32, name: "indices")
            .targetForModes(["topTest"])
    }

    print("Top K on dimension 1 (columns) of size 3")
    print("Values:")
    results = try topKGraph.runOne(mode: "topTest", inputTensors: [:])
    result = results["result_values"]!
    try result.print2D(elementWidth: 6, precision: 1)
    print("Indices:")
    result = results["indices"]!
    try result.print2D(elementWidth: 6, precision: 0)
}

if expand {
    let initialTensor = try TensorFloat32(shape: TensorShape([2, 3]), initialValues: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    //                                                                              [1.0, 2.0, 3.0
    //                                                                               4.0, 5.0, 6.0]
    print("")
    print("Original tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    print("shape of initial tensor is \(initialTensor.shape.dimensions)")
    
    let expandGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        ExpandDimension("startTensor", axis: 1, name: "result")
            .targetForModes(["expandTest"])
    }

    print("expanded on axis 1 (columns)")
    let expandResults = try expandGraph.runOne(mode: "expandTest", inputTensors: [:])
    let expandResult = expandResults["result"]!
    print("shape of expanded tensor is \(expandResult.shape.dimensions)")
}

if (banding) {
    let initialTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7, 8.2, 11.1, 5.9, 3.3, 2.4])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7
    //                                                                               8.2, 11.1,  5.9,   3.3, 2.4]
    print("")
    print("Original tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    print("shape of initial tensor is \(initialTensor.shape.dimensions)")

    let bandGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        BandPart("startTensor", numLower: 1, numUpper: 2, name: "result")
            .targetForModes(["bandTest"])
    }

    print("banded with lower 1, upper 2")
    let bandResults = try bandGraph.runOne(mode: "bandTest", inputTensors: [:])
    let bandResult = bandResults["result"]!
    try bandResult.print2D(elementWidth: 6, precision: 1)
}

if (argSort) {
    let initialTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let argSortGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        ArgSort("startTensor", axis: 1, name: "result")
            .targetForModes(["argSortTest"])
    }

    print("ArgSort  on dimension 1 (columns)")
    let results = try argSortGraph.runOne(mode: "argSortTest", inputTensors: [:])
    let result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 0)
}

if (broadcast) {
    let initialTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let broadCastGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        BroadCast("startTensor", shape: [2, 10], name: "result")
            .targetForModes(["broadcastTest"])
    }

    print("broadcast to shape [2, 10]")
    let results = try broadCastGraph.runOne(mode: "broadcastTest", inputTensors: [:])
    let result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 1)

}

if (flatten) {
    let initialTensor = try TensorFloat32(shape: TensorShape([2, 3, 3, 2]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7, 8.2, 11.1, 5.9, 3.3, 2.4, 44.0, 12.3, 36.0,  17.2,  50.3, 66.7, 81.2, 61.1, 95.9, 43.3, 52.4])
    print("Initial Tensor shape [2, 3, 3, 2]")
    
    let flattenGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        Flatten2D("startTensor", axis: 1, name: "result")
            .targetForModes(["flattenTest"])
    }

    print("flatten along axis 1")
    var results = try flattenGraph.runOne(mode: "flattenTest", inputTensors: [:])
    var result = results["result"]!
    print("result tensor shape \(result.shape.dimensions)")
    try result.print2D(elementWidth: 6, precision: 1)
    
    let flattenGraph2 = Graph {
        Constant(values: initialTensor, name: "startTensor")
        Flatten2D("startTensor", axis: 2, name: "result")
            .targetForModes(["flattenTest"])
    }

    print("flatten along axis 2")
    results = try flattenGraph2.runOne(mode: "flattenTest", inputTensors: [:])
    result = results["result"]!
    print("result tensor shape \(result.shape.dimensions)")
    try result.print2D(elementWidth: 6, precision: 1)
}

if (onehot) {
    let initialTensor = try TensorFloat32(shape: TensorShape([6]), initialValues: [0.0, 4.0, 3.0, 1.0, 5.0, 2.0])
    print("")
    print("Original tensor")
    try initialTensor.print1D(elementWidth: 6, precision: 1)
    print("shape of initial tensor is \(initialTensor.shape.dimensions)")
    
    let oneHotGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        OneHot(depth: 6, axis: 0, name: "result")
            .targetForModes(["oneHotTest"])
    }

    print("one-hot along axis 0")
    let results = try oneHotGraph.runOne(mode: "oneHotTest", inputTensors: [:])
    let result = results["result"]!
    print("result tensor shape \(result.shape.dimensions)")
    try result.print2D(elementWidth: 6, precision: 1)
}

if (dropout) {
    let initialTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let dropOutGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        Dropout(rate: 0.3, name: "result")
            .targetForModes(["dropOutTest"])
    }

    print("Drop-out with rate of 0.3")
    let results = try dropOutGraph.runOne(mode: "dropOutTest", inputTensors: [:])
    let result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 1)
}

if (quantize) {
    let initialTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,  0.85,  0.95])
    //                                                                              [0.0,  0.1,  0.2,   0.3,  0.4
    //                                                                               0.5,  0.6,  0.7,   0.8,  0.9
    //                                                                               1.0,  0.15, 0.25,  0.35, 0.45
    //                                                                               0.55, 0.65, 0.75,  0.85, 0.95]
    print("Initial Tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let quantizeGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        Quantize(scale: 1.0 / 255.0, zeroPoint: 0.0, signed: false, name: "result")
            .targetForModes(["quantizeTest"])
    }

    print("Quantize with scale = 1/255, zeropoint = 0")
    let results = try quantizeGraph.runOne(mode: "quantizeTest", inputTensors: [:])
    let result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 0)
    
//    let quantizeGraph2 = Graph {
//        Constant(values: initialTensor, name: "startTensor")
//        Constant(shape: [5], value: Float32(1.0 / 255.0), name: "scaleConstant")
//        Quantize("startTensor", scaleTensor: "scaleConstant", zeroPoint: 3.0, signed: false, axis: 0, name: "result")
//            .targetForModes(["quantizeTest"])
//    }
//
//    print("Quantize with tensor scale = 1/255, zeropoint = 3, axis = 0")
//    let results2 = try quantizeGraph2.runOne(mode: "quantizeTest", inputTensors: [:])
//    let result2 = results2["result"]!
//    try result2.print1D(elementWidth: 6, precision: 1)
    
    let dequantizeGraph = Graph {
        Constant(values: result, name: "quantizedTensor")
        Dequantize("quantizedTensor", scale: 1.0 / 255.0, zeroPoint: 0.0, dataType: .float32, name: "result")
            .targetForModes(["dequantizeTest"])
    }

    print("De-Quantize above quantized tensor")
    let results3 = try dequantizeGraph.runOne(mode: "dequantizeTest", inputTensors: [:])
    let result3 = results3["result"]!
    try result3.print2D(elementWidth: 6, precision: 1)

}

if (pool) {
    let initialTensor = try TensorFloat32(shape: TensorShape([4, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7]
    print("Initial Tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    
    let poolingGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        PoolingLayer(function: .max, kernelSize: [2, 2], strides: [2, 3], name: "result")
            .targetForModes(["poolingTest"])
    }

    print("Max Pooling with kernal [2, 2] and strides [2, 3]")
    var results = try poolingGraph.runOne(mode: "poolingTest", inputTensors: [:])
    var result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 1)
    
    let threeDTensor = try TensorFloat32(shape: TensorShape([3, 3, 3]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7, 8.2, 11.1, 5.9, 3.3, 2.4, 44.0, 12.3])
    //                                                                              [[   7.0,   9.0,   3.0]   [[   2.0,   5.0,   8.0]   [[   1.0,   6.0,   1.2]
    //                                                                               [  -2.0,   6.3,   2.3]    [   9.0,   5.8,  16.0]    [  17.0,  14.0,   7.2]
    //                                                                               [  10.3,  11.1,   2.4]]   [   6.7,   5.9,  44.0]]   [   8.2,   3.3,  12.3]]
    print("Three-dimensional Tensor split on last dimension")
    try threeDTensor.print3D(elementWidth: 6, precision: 1)
    
    let poolingGraph3D = Graph {
        Constant(values: threeDTensor, name: "startTensor")
        PoolingLayer(function: .max, kernelSize: [3, 3], strides: [3, 3], name: "result")
            .targetForModes(["poolingTest"])
    }

    print("Max Pooling with kernal [3, 3] and strides [3, 3]")
    results = try poolingGraph3D.runOne(mode: "poolingTest", inputTensors: [:])
    result = results["result"]!
    try result.print3D(elementWidth: 6, precision: 1)
    
    let wideTensor = try TensorFloat32(shape: TensorShape([2, 10]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7])
    print("")
    print("")
    print("Initial Wide Tensor")
    try wideTensor.print2D(elementWidth: 6, precision: 1)

    let poolingGraph2 = Graph {
        Constant(values: wideTensor, name: "startTensor")
        PoolingLayer(function: .max, kernelSize: [2, 2], strides: [1, 3], name: "result")
            .setCeilingMode()
            .dilationRates(dilationRateH: 2, dilationRateW: 2)
            .HWPadding(bottomPadding: 0, topPadding: 0, leftPadding: 5, rightPadding: 5)
            .targetForModes(["poolingTest"])
    }

    print("Max Pooling with kernal [2, 2] and strides [1, 2]")
    results = try poolingGraph2.runOne(mode: "poolingTest", inputTensors: [:])
    result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 1)
}

if (shapeOf) {
    let initialTensor = try TensorFloat32(shape: TensorShape([2, 1, 3, 4]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7, 1.0, 2.0, 3.0, 4.0])
    print("Initial Tensor shape : [2, 1, 3, 4]")
    
    let shapeOfGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        ShapeOfTensor(name: "result")
            .targetForModes(["shapeOfTest"])
    }

    print("Shape of operation")
    let results = try shapeOfGraph.runOne(mode: "shapeOfTest", inputTensors: [:])
    let result = results["result"]!
    try result.print1D(elementWidth: 3, precision: 0)
}

if (depthToSpace) {
    var initValues = [Float32](repeating: 0.0, count: 64)
    for i in 0..<64 { initValues[i] = Float32(i) }
    let initialTensor = try TensorFloat32(shape: TensorShape([4, 4, 4]), initialValues: initValues)
    try initialTensor.print3D(elementWidth: 6, precision: 1)
    
    let depthToSpaceGraph = Graph {
        Constant(values: initialTensor, name: "startTensor")
        DepthToSpace2D(widthAxis: 1, heightAxis: 0, depthAxis: 2, blockSize: 2, name: "result")
            .targetForModes(["depthToSpaceTest"])
    }
    
    print("DepthToSpace operation - width dimension 1, height dimension 0, depth dimension 2, block size 2")
    var results = try depthToSpaceGraph.runOne(mode: "depthToSpaceTest", inputTensors: [:])
    var result = results["result"]!
    try result.print3D(elementWidth: 6, precision: 1)
    
    let spaceToDepthGraph = Graph {
        Constant(values: result, name: "startTensor")
        SpaceToDepth2D(widthAxis: 1, heightAxis: 0, depthAxis: 2, blockSize: 2, name: "result")
            .targetForModes(["depthToSpaceTest"])
    }
    
    print("SpaceToDepth operation - width dimension 1, height dimension 0, depth dimension 2, block size 2")
    results = try spaceToDepthGraph.runOne(mode: "depthToSpaceTest", inputTensors: [:])
    result = results["result"]!
    try result.print3D(elementWidth: 6, precision: 1)
}

if (gather) {
    let updateTensor = try TensorFloat32(shape: TensorShape([5, 5]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2, -2.0, 9.0, 17.0, 6.3, 5.8, 14.0, 2.3, 16.0,  7.2,  10.3, 6.7, 8.2, 11.1, 5.9, 3.3, 2.4])
    //                                                                              [7.0,  2.0,  1.0,  9.0,  5.0
    //                                                                               6.0,  3.0,  8.0,  1.2, -2.0
    //                                                                               9.0, 17.0,  6.3,  5.8, 14.0
    //                                                                               2.3, 16.0,  7.2,  10.3, 6.7
    //                                                                               8.2, 11.1,  5.9,   3.3, 2.4]
    print("")
    print("Update tensor")
    try updateTensor.print2D(elementWidth: 6, precision: 1)
    
    let indicesTensor = try TensorInt32(shape: TensorShape([2, 5]), initialValues: [0, 1, 2, 3, 4, 3, 2, 1, 0, 4])
    //                                                                              [0,  1,  2,  3,  4
    //                                                                               3,  2,  1,  0,  4]
    print("")
    print("Indices tensor")
    try indicesTensor.print2D(elementWidth: 6, precision: 0)
    
    let gatherGraph = Graph {
        Constant(values: updateTensor, name: "updateTensor")
        Constant(values: indicesTensor, name: "indicesTensor")
        Gather(updateTensor: "updateTensor", indicesTensor: "indicesTensor", axis: 0, name: "result")
            .targetForModes(["gatherTest"])
    }
    
    print("")
    print("Gather operation along axis 0 (rows)")
    let results = try gatherGraph.runOne(mode: "gatherTest", inputTensors: [:])
    let result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 1)
}

if (sliceUpdate) {
    let initialTensor = TensorFloat32(shape: TensorShape([5, 5]), initialValue: 0.0)
    print("Initial tensor")
    try initialTensor.print2D(elementWidth: 6, precision: 1)
    let updateTensor = try TensorFloat32(shape: TensorShape([3, 3]), initialValues: [7.0, 2.0, 1.0, 9.0, 5.0, 6.0, 3.0, 8.0, 1.2])
    //                                                                              [[   7.0,   2.0,   1.0]
    //                                                                               [   9.0,   5.0,   6.0]
    //                                                                               [   3.0,   8.0,   1.2]]
    print("Update tensor")
    try updateTensor.print2D(elementWidth: 6, precision: 1)
    
    let sliceUdateGraph = Graph {
        Constant(values: initialTensor, name: "initialTensor")
        Constant(values: updateTensor, name: "updateTensor")
        SliceUpdateDataTensor(tensorToUpdate: "initialTensor", sourceTensor: "updateTensor", starts: [0, 0], ends: [5, 5], strides: [2, 2], name: "result")
            .targetForModes(["sliceUpdateTest"])
    }
    
    print("")
    print("SliceUpdate operation starting at [0,0] going to [5, 5], with strides [2, 2]")
    var results = try sliceUdateGraph.runOne(mode: "sliceUpdateTest", inputTensors: [:])
    var result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 1)
    
    let sliceUdateGraph2 = Graph {
        Constant(values: initialTensor, name: "initialTensor")
        Constant(values: updateTensor, name: "updateTensor")
        SliceUpdateDataTensor(tensorToUpdate: "initialTensor", sourceTensor: "updateTensor", starts: [1, 1], ends: [4, 4], strides: [1, 1], name: "result")
            .targetForModes(["sliceUpdateTest"])
    }

    print("")
    print("SliceUpdate operation starting at [1,1] going to [4, 4], with strides [1, 1]")
    results = try sliceUdateGraph2.runOne(mode: "sliceUpdateTest", inputTensors: [:])
    result = results["result"]!
    try result.print2D(elementWidth: 6, precision: 1)
}
