# MPSGraphDSL
A framework/swift package that makes it much easier parse data files, manage data sets, and to use Apple's Metal Performance Shaders Graph (MPSGraph) library.

Domain Specific Languages (DSL) are available for parsing source data and building calculation graphs using a declarative approach similar to SwiftUI.

Classes are available for managing multi-dimensional tensors and sets of data for feeding into and out of the calculation graphs.

When building a calculation graph, most of the MPSGraph operators are present. However, and more importantly, additional graph nodes are made available to handle all the details of neural network code such as variable creation, persistent storage of learned parameters, loss node back-propogation, and the tensor shape changes needed for use of the MPSGraph neural network operations provided by Apple.

##  Documentation

The framework has in-line documentation for all items (classes, enums, structs, methods) and can be used with XCode.  In addition there are five articles in the documentation for working with tensors, data sets, data parsers, calculation graphs, and neural networks

##  Examples

Included with the XCode project (not the swift package) is a playground tutorial for a simple neural network.  The network runs agains a testing data set, trains, and then retries the test set to determine the improvements.  Then the learned parameters are extracted from the network, weights and biases are reset, and the test data set ran again to show the learning was lost.  Lastly, the extracted parameters are reloaded into the network and the test set run one last time to show how to load trained parameters into a network.

Included with the XCode project (not the swift package) is a command-line app for seeing the affects of various tensor modification nodes that are available when graph building.  These include concatenation, reshaping, casting, slicing, tiling, banding, squeezing, and many more.

Additionally, there are swift testing files that can be perused for how to use many of the main objects like data parsers, calculation graphs, and neural networks.

##  License

The code is licensed under the  [GNU GPL v3.0](LICENSE) restrictions.  

##  Use

This top level of the repository gets you the XCode project with the library, documentation, swift tests, tutorial playground, and tensor modification example app.

One level down, into MPSGraphDSL, you can reference the swift package if that is all you need.

##  Use Notes
### Known Issues
The GRU node cannot be trained (ran through back-propagation) at this time.  This is due to a reported error in the MPSGraph framework, and it is unknown if/when it will be fixed by Apple
