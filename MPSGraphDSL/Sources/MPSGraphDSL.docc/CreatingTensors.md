# Creating and Using Tensors

This article describes some of the methods used to create Tensors

## Overview

There are two ways to create a ``Tensor`` - based on whether the data type (specified using a ``DataType`` enumeration) is known when writing your code, or if the type can be dynamic.  Both methods require a shape and a source for initial values.

### For a Fixed DataType

If at coding time you know what data type is needed, it is usually best to just create the Tensor using one of the classes that conform to the ``Tensor`` protocol that have fixed type.  These classes include:
* ``TensorFloat32``
* ``TensorFloat16``
* ``TensorInt32``
* ``TensorUInt8``
* ``TensorDouble``

Note: Double tensors are not usable by the MPSGraph system, but the data parsers will create them if you want.

The Tensor classes have some additional initializers, such as one that accepts an array of specific data type as initial values for the Tensor

### For a Dynamic DataType

If the data type is not known when writing the code (i.e. it is read from a file or user input), then the ``CreateTensor`` class can be used.  This class consists of static functions that create a Tensor based on a passed in data type, along with the other parameters needed to create the Tensor.  All of the initializers are available on the conforming Tensor classes, without the data type parameter as that is implicit in the class name.

## Tensor Shape

Besides the data type, each Tensor requires a shape - specified using a ``TensorShape`` structure.  The shape defines the number of dimensions, and the size of each dimension

### Dimensions

A Tensor shape must have at least one dimension and can have as many as 16.  A one-dimensional Tensor can be considered a vector, while a two-dimensional one can be thought of as a matrix.  Often you will need more dimensions then is obvious.  The dimensions and sizes are specified as an array of integers.

For example, the CIFAR-10 data set is a collection of images.  The images are RGB pixels in a 32 pixel square.  A shape that would encapsulate a single image could be shaped with the following: TensorShape(\[32, 32, 3\]).  This is a shape for three dimensional Tensor with 32 rows, 32 columns, and 3 data items each (one for each value of the pixel, red, green, and blue).

### Total Size

The total size of a tensor (number of elements) can be calculated by multiplying the dimenstion sizes together.  The ``TensorShape`` function totalSize() will do this for you

### Element Storage

Tensors store their element in a row-major fashion.  That means the left-most dimension (the 'row' in a matrix) changes the storage index the most.  An example is a two-dimensional shape \[R, C\].  The storage location for a particular element \[r,c\] would be r\*C+c.  This means that the items indexed by the right-most dimension are next to each other in the storage.  In the CIFAR-10 example above, the RGB values for each pixel would be right next to each other.

## Initial Values

Once you have a data type and a shape, the only other thing you need to create a tensor is what the initial value of each element should be.  There are four options:
| Source               | Description |
| -------------------- | ----------- |
| Constant             | Each element has the same value that is passed to the initializer                                      |
| Array                | Each element is assigned the value from an array that is passed to the initializer (see Storage above) |
| Random Uniform       | Each element is assigned a uniform random value from a range that is passed to the initializer         |
| Random Normal        | Each element is assigned a normal distribution random value from a mean and standard deviation that is passed to the initializer |
| MPSGraphTensorData   | The elements are transfered from an MPSGraph result (usually done internally)                          |

## Creation Examples

Once you have the data type, the shape, and the initial values, you can create a Tensor.  The following examples show a tensor created for a 5x3 matrix:

```swift
let shape = TensorShape([5, 3])

//  Create Float32 tensor with all values initialized to 3
let type = DataType.float32
let tensor = CreateTensor.constantValues(type: type, shape: shape, initialValue: 3.0)

//  Create UInt8 tensor with all values initialized to random values in the range of 0-127
let range = try ParameterRange(minimum: UInt8(0), maximum: UInt8(127))
let tensor = TensorUInt8(shape: shape, randomValueRange: range)
```

## Accessing Values

Much of the time you might not need to directly access values in Tensors, the data parsers and Graph functions will do it for you.  But you can access the values directly if you need

###  Referencing Values

There are two ways to reference a position within the Tensor, an index or a location.  An index is a direct offset into the element array, which is a single-dimension array that spans the entire Tensor.  A location is an array of Int values, one for each dimension of the Tensor, with the value of the location between 0 and the size of that dimension minus 1.  It can be turned into an index using the TensorShape function 'storageIndex(location: \[Int\])'

###  Getting Values

The following examples show accessing the 1, 3 (remember these are zero based!) position of a 5,4 matrix Tensor

```swift
let shape = TensorShape([5, 4])

let value = try tensor.getElement(location: [1, 3])     //  Always returns a Double

let index = try shape.storageIndex(location: [1, 3])        //  Would return 8  (5 * 1 + 3)
let value = try tensor.getElement(index : index)
```
It is also possible to get the entire element set as a Double array

```swift
let values = tensor.getElements()
```

###  Setting Values

The following examples show setting the 2, 1 element of the 5x4 matrix Tensor

```swift
let shape = TensorShape([5, 4])

try tensor.setElement(location: [2, 1], newValue)     //  newValue must be a Double when referencing by location

let index = try shape.storageIndex(location: [2, 1])        //  Would return 11  (5 * 2 + 1)
try tensor.setElement(index : index, newValue)     //  newValue can be a Double or the type of the Tensor when referencing by index
```
