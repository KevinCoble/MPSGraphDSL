//
//  TensorTest.swift
//  MPSGraphDSLTests
//
//  Created by Kevin Coble on 3/14/26.
//

import Testing
@testable import MPSGraphDSL

struct TensorTest {

    @Test func TestMapping() async throws {
        //  Create a test tensor
        let testTensorDoubleData: [Double] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let initialTensorDouble = try TensorDouble(shape: TensorShape([2, 3]), initialValues: testTensorDoubleData)
        
        //  Double the values with a map
        let newTensorDouble = try initialTensorDouble.map { (value, _, _) in
            value * 2.0
        }

        var resultValue = try newTensorDouble.getElement(index: 0)
        #expect(resultValue == 2.0)
        resultValue = try newTensorDouble.getElement(index: 1)
        #expect(resultValue == 4.0)
        resultValue = try newTensorDouble.getElement(index: 2)
        #expect(resultValue == 6.0)
        resultValue = try newTensorDouble.getElement(index: 3)
        #expect(resultValue == 8.0)
        resultValue = try newTensorDouble.getElement(index: 4)
        #expect(resultValue == 10.0)
        resultValue = try newTensorDouble.getElement(index: 5)
        #expect(resultValue == 12.0)
    }

}
