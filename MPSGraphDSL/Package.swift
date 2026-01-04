// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MPSGraphDSL",
    platforms: [
            .macOS(.v15),    // Supports macOS 15 and newer
            .iOS(.v14),      // Supports iOS 14 and newer
            .tvOS(.v14),     // Supports tvOS 14 and newer
            .visionOS(.v1),  // Supports visionOS 1 and newer
         ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MPSGraphDSL",
            targets: ["MPSGraphDSL"]
        ),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MPSGraphDSL"
        ),
        .testTarget(
            name: "MPSGraphDSLTests",
            dependencies: ["MPSGraphDSL"]
        ),
    ]
)
