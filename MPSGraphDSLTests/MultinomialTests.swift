//
//  MultiNomialt.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 2/23/26.
//

import Testing
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSGraphDSL

struct MultinomialTest {

    @Test(.disabled("Passes, but get Insufficient Memory errors when ran in parallel"))
    func SingleDistribution() async throws {
        //  Create a single probability distribution of four logits
        let probabilityTensor = try TensorFloat32(shape: TensorShape([4]), initialValues: [0.2, 0.1, 0.4, 0.3])
        
        //  Create the graph
        let multinomialGraph = Graph {
            Constant(values: probabilityTensor, name: "probabilities")
            Multinomial(name: "multinomial")
                .targetForModes(["multinomialTest"])
        }

        //  Run it 1000 times to try get the distribution back
        var hits: [Int] = Array(repeating: 0, count: 4)
        for _ in 0..<1000 {
            let results = try multinomialGraph.runOne(mode: "multinomialTest", inputTensors: [:])
            let result = results["multinomial"]! as! TensorInt32
            let index: Int32 = try result.getElement(index: 0)
            hits[Int(index)] += 1
        }
        
        #expect((hits[0]) > 150 && (hits[0] < 250))
        #expect((hits[1]) >  50 && (hits[1] < 150))
        #expect((hits[2]) > 350 && (hits[2] < 450))
        #expect((hits[3]) > 250 && (hits[3] < 350))
    }

    @Test(.disabled("Passes, but get Insufficient Memory errors when ran in parallel"))
    func MultipleDistribution() async throws {
        //  Create a two probability distributions of four logits
        let probabilityTensor = try TensorFloat32(shape: TensorShape([2, 4]), initialValues: [0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1])
        
        //  Create the graph
        let multinomialGraph = Graph {
            Constant(values: probabilityTensor, name: "probabilities")
            Multinomial(name: "multinomial")
                .targetForModes(["multinomialTest"])
        }

        //  Run it 1000 times to try get the distributions back
        var hits: [[Int]] = Array(repeating: Array(repeating: 0, count: 4), count: 2)
        for _ in 0..<1000 {
            let results = try multinomialGraph.runOne(mode: "multinomialTest", inputTensors: [:])
            let result = results["multinomial"]! as! TensorInt32
            let index: Int32 = try result.getElement(index: 0)
            let index2: Int32 = try result.getElement(index: 1)
            hits[0][Int(index)] += 1
            hits[1][Int(index2)] += 1
        }
        
        #expect((hits[0][0]) > 150 && (hits[0][0] < 250))
        #expect((hits[0][1]) >  50 && (hits[0][1] < 150))
        #expect((hits[0][2]) > 350 && (hits[0][2] < 450))
        #expect((hits[0][3]) > 250 && (hits[0][3] < 350))
        #expect((hits[1][0]) >  50 && (hits[1][0] < 150))
        #expect((hits[1][1]) >  50 && (hits[1][1] < 150))
        #expect((hits[1][2]) > 650 && (hits[1][2] < 750))
        #expect((hits[1][3]) >  50 && (hits[1][3] < 150))
    }
    
    @Test(.disabled("Passes, but get Insufficient Memory errors when ran in parallel"))
    func MultiDimensionalDistribution() async throws {
        //  Create a two-dimensional matrix of probability distributions of four logits
        let probabilityTensor = try TensorFloat32(shape: TensorShape([2, 2, 4]), initialValues: [0.2, 0.1, 0.4, 0.3, 0.1, 0.1, 0.7, 0.1, 0.3, 0.2, 0.2, 0.4, 0.4, 0.3, 0.1, 0.2])
        
        //  Create the graph
        let multinomialGraph = Graph {
            Constant(values: probabilityTensor, name: "probabilities")
            Multinomial(name: "multinomial")
                .targetForModes(["multinomialTest"])
        }

        //  Run it 1000 times to try get the distributions back
        var hits: [[[Int]]] = Array(repeating:Array(repeating: Array(repeating: 0, count: 4), count: 2), count: 2)
        var resultShape: TensorShape! = nil
        for i in 0..<1000 {
            let results = try multinomialGraph.runOne(mode: "multinomialTest", inputTensors: [:])
            let result = results["multinomial"]! as! TensorInt32
            if (i == 0) { resultShape = result.shape}
            let index: Int32 = try result.getElement(index: 0)
            let index2: Int32 = try result.getElement(index: 1)
            let index3: Int32 = try result.getElement(index: 2)
            let index4: Int32 = try result.getElement(index: 3)
            hits[0][0][Int(index)] += 1
            hits[0][1][Int(index2)] += 1
            hits[1][0][Int(index3)] += 1
            hits[1][1][Int(index4)] += 1
        }
        
        //  Make sure the shape is [2, 2]
        let validShape = try #require(resultShape, "Shape should not be nil")
        #expect(validShape.numDimensions == 2)
        #expect(validShape.dimensions[0] == 2)
        #expect(validShape.dimensions[1] == 2)

        //  Check the distributions
        #expect((hits[0][0][0]) > 150 && (hits[0][0][0] < 250))
        #expect((hits[0][0][1]) >  50 && (hits[0][0][1] < 150))
        #expect((hits[0][0][2]) > 350 && (hits[0][0][2] < 450))
        #expect((hits[0][0][3]) > 250 && (hits[0][0][3] < 350))
        #expect((hits[0][1][0]) >  50 && (hits[0][1][0] < 150))
        #expect((hits[0][1][1]) >  50 && (hits[0][1][1] < 150))
        #expect((hits[0][1][2]) > 650 && (hits[0][1][2] < 750))
        #expect((hits[0][1][3]) >  50 && (hits[0][1][3] < 150))
        
        #expect((hits[1][0][0]) > 250 && (hits[1][0][0] < 350))
        #expect((hits[1][0][1]) > 150 && (hits[1][0][1] < 250))
        #expect((hits[1][0][2]) > 150 && (hits[1][0][2] < 250))
        #expect((hits[1][0][3]) > 350 && (hits[1][0][3] < 450))
        #expect((hits[1][1][0]) > 350 && (hits[1][1][0] < 450))
        #expect((hits[1][1][1]) > 250 && (hits[1][1][1] < 350))
        #expect((hits[1][1][2]) >  50 && (hits[1][1][2] < 150))
        #expect((hits[1][1][3]) > 150 && (hits[1][1][3] < 250))
    }
}
