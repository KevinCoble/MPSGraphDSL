//
//  ImageParser.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 2/15/26.
//

import Foundation
import CoreGraphics
import ImageIO

@resultBuilder
///  The Result Builder for building image data parsers
public enum ImageParserBuilder {
    public static func buildBlock(_ imageChunks: [ImageChunk]...) -> [ImageChunk] {
        return imageChunks.flatMap({$0})
    }
    public static func buildExpression(_ expression: ImageChunk) -> [ImageChunk] {
        return [expression]
    }
    public static func buildOptional(_ component: [ImageChunk]?) -> [ImageChunk] {
        return component ?? []
    }
    public static func buildEither(first component: [ImageChunk]) -> [ImageChunk] {
        return component
    }
    public static func buildEither(second component: [ImageChunk]) -> [ImageChunk] {
        return component
    }
}


///  Image parser class
public class ImageParser {
    var chunks : [ImageChunk]

    ///  Initializer to create an ImageParser with a trailing closure of a list of ImageChunks
    public convenience init(@ImageParserBuilder _ buildChunks: () -> [ImageChunk]) {
        self.init()
        chunks = buildChunks()
    }

    ///  Initializer for creating an image parser
    public init() {
        chunks = []
    }
    
    /// Parse image files that are underneath a specified top-level directory URL
    /// - Parameters:
    ///   - intoDataSet: the DataSet to parse the images into
    ///   - topLevelDirectory: the directory that all sub-directories from the definition nodes are relative to.
    ///   - maxConcurrency: (Optional) the maximum number of concurrent processing tasks.  Set to 1 or below to turn off concurrency.  Default is 4
    public func parse(intoDataSet: DataSet, topLevelDirectory: URL, maxConcurrency : Int = 4) async throws
    {
        //  Process each chunk
        for chunk in chunks {
            try await chunk.process(into: intoDataSet, tld: topLevelDirectory, maxConcurrent: maxConcurrency)
        }
    }
}

internal enum ImageChunkType {
    case file
    case labeledDirectory
    case topLevelDirectory  //  Assumes labeled sub-directories
}

public class ImageChunk {
    let type: ImageChunkType
    let fileName: String
    let range: ParameterRange? = nil
    let classificationIndex: Int?
    let classificationLabel: String?
    
    init (type: ImageChunkType, fileName: String, index: Int?, label: String?) {
        self.type = type
        self.fileName = fileName
        self.classificationIndex = index
        self.classificationLabel = label
    }
    
    internal func process(into: DataSet, tld: URL, maxConcurrent: Int) async throws {
        //  Get the url with the chunk addition
        let chunkURL = tld.appendingPathComponent(fileName)
        
        //  Get the range for the values if not passed in
        let inputRange: ParameterRange
        if let range = range {
            inputRange = range
        }
        else {
            //  Set a default range based on the data type
            switch (into.inputType) {
            case .uInt8, .int32:
                inputRange = try ParameterRange(minimum: 0.0, maximum: 255.0)
            case .float16, .float32, .double:
                inputRange = try ParameterRange(minimum: 0.0, maximum: 1.0)
            }
        }
        
        switch (type) {
        case .file:
            let image = try ImageChunk.loadImage(fileURL: chunkURL)
            let inputTensor = try ImageChunk.getInputTensorFromImage(image, type: into.inputType, range: inputRange, shape: into.inputShape)
            let outputTensor = try await ImageChunk.getOutputTensorForDataset(into, classificationIndex: classificationIndex, classificationLabel: classificationLabel)
            let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
            try await into.appendSample(sample)
        case .labeledDirectory:
            if (try !ImageChunk.urlIsDirectory(chunkURL)) { throw DataParsingErrors.NotDirectory}
            let fileManager = FileManager.default
            let directoryContentsURLs = try fileManager.contentsOfDirectory(at: chunkURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            var numSubmitted = 0
            let index = classificationIndex
            let label = classificationLabel
            for fileURL in directoryContentsURLs {
                if (maxConcurrent <= 1) {
                    //  Make sure it is not a directory
                    if (try !ImageChunk.urlIsDirectory(fileURL)) {
                        //  Process the file
                        let image = try ImageChunk.loadImage(fileURL: fileURL)
                        let inputTensor = try ImageChunk.getInputTensorFromImage(image, type: into.inputType, range: inputRange, shape: into.inputShape)
                        let outputTensor = try await ImageChunk.getOutputTensorForDataset(into, classificationIndex: index, classificationLabel: label)
                        let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                        try await into.appendSample(sample)
                    }
                }
                else {
                    try await withThrowingTaskGroup(of: Void.self) { taskGroup in
                        
                        //  If less than max concurrency submitted, add it immediately.  Otherwise wait for one to finish
                        if (numSubmitted >= maxConcurrent) {
                            try await taskGroup.next()
                        }
                        
                        let added = taskGroup.addTaskUnlessCancelled {
                            //  Make sure it is not a directory
                            if (try !ImageChunk.urlIsDirectory(fileURL)) {
                                //  Process the file
                                let image = try ImageChunk.loadImage(fileURL: fileURL)
                                let inputTensor = try ImageChunk.getInputTensorFromImage(image, type: into.inputType, range: inputRange, shape: into.inputShape)
                                let outputTensor = try await ImageChunk.getOutputTensorForDataset(into, classificationIndex: index, classificationLabel: label)
                                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                                try await into.appendSample(sample)
                            }
                        }
                        if (added) { numSubmitted += 1 }
                    }
                }
            }
        case .topLevelDirectory:
            if (try !ImageChunk.urlIsDirectory(chunkURL)) { throw DataParsingErrors.NotDirectory }
            let fileManager = FileManager.default
            let directoryContentsURLs = try fileManager.contentsOfDirectory(at: chunkURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
            for directoryURL in directoryContentsURLs {
                var numSubmitted = 0
                //  Make sure it is a directory
                if (try ImageChunk.urlIsDirectory(directoryURL)) {
                    //  Get the label from the directory name
                    let label = directoryURL.lastPathComponent
                    //  Process each file in the directory
                    let subDirectoryContentsURLs = try fileManager.contentsOfDirectory(at: directoryURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
                    for fileURL in subDirectoryContentsURLs {
                        if (maxConcurrent <= 1) {
                            //  Make sure it is not a directory
                            if (try !ImageChunk.urlIsDirectory(fileURL)) {
                                //  Process the file
                                let image = try ImageChunk.loadImage(fileURL: fileURL)
                                let inputTensor = try ImageChunk.getInputTensorFromImage(image, type: into.inputType, range: inputRange, shape: into.inputShape)
                                let outputTensor = try await ImageChunk.getOutputTensorForLabel(label, dataset: into)
                                let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                                try await into.appendSample(sample)
                            }
                        }
                        else {
                            try await withThrowingTaskGroup(of: Void.self) { taskGroup in
                                
                                //  If less than max concurrency submitted, add it immediately.  Otherwise wait for one to finish
                                if (numSubmitted >= maxConcurrent) {
                                    try await taskGroup.next()
                                }
                                
                                let added = taskGroup.addTaskUnlessCancelled {
                                    //  Make sure it is not a directory
                                    if (try !ImageChunk.urlIsDirectory(fileURL)) {
                                        //  Process the file
                                        let image = try ImageChunk.loadImage(fileURL: fileURL)
                                        let inputTensor = try ImageChunk.getInputTensorFromImage(image, type: into.inputType, range: inputRange, shape: into.inputShape)
                                        let outputTensor = try await ImageChunk.getOutputTensorForLabel(label, dataset: into)
                                        let sample = DataSample(inputs: inputTensor, outputs: outputTensor)
                                        try await into.appendSample(sample)
                                    }
                                }
                                if (added) { numSubmitted += 1 }
                            }
                        }
                    }
                }
            }
         }
    }
    
    internal static func loadImage(fileURL: URL) throws -> CGImage {
        guard let imageSource = CGImageSourceCreateWithURL(fileURL as CFURL, nil) else {
            throw DataParsingErrors.ImageSourceNotCreated
        }
        let options: [NSString: AnyObject] = [kCGImageSourceShouldCacheImmediately: true as AnyObject]
        guard let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, options as CFDictionary) else {
            throw DataParsingErrors.ImageNotLoaded
        }
        
        return cgImage
    }
    
    internal static func getInputTensorFromImage(_ image: CGImage, type: DataType, range: ParameterRange, shape: TensorShape) throws -> Tensor {
        if (shape.numDimensions == 3) {
            return try ImageToTensor.rgbImageToTensor(image: image, type: type, range: range, shape: shape)
        }
        else {
            return try ImageToTensor.greyscaleImageToTensor(image: image, type: type, range: range, shape: shape)
        }
    }
    
    internal static func getOutputTensorForDataset(_ dataset: DataSet, classificationIndex: Int?, classificationLabel: String?) async throws -> Tensor {
        //  Make sure we have a label or index, but not both
        if (classificationIndex == nil && classificationLabel == nil) { throw DataParsingErrors.NeedClassificationIndexOrLabel }
        if (classificationIndex != nil && classificationLabel != nil) { throw DataParsingErrors.ConflictingClassificationMethods }

        //  Get the classification index
        let index: Int
        if let label = classificationLabel {
            index = await dataset.getLabelIndex(label: label)
        }
        else {
            index = classificationIndex!
        }
        if (index < 0 || index >= dataset.outputShape.totalSize) { throw DataParsingErrors.MoreUniqueLabelsThanOutputDimension }
        
        // Create the tensor
        var outputTensor = CreateTensor.constantValues(type: dataset.outputType, shape: dataset.outputShape, initialValue: 0.0)
        try outputTensor.setElement(index: index, value: 1.0)
        
        return outputTensor
    }
    
    internal static func urlIsDirectory(_ url: URL) throws -> Bool {
        return ((try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true)
    }
    
    internal static func getOutputTensorForLabel(_ label: String, dataset: DataSet) async throws -> Tensor {
        let index = await dataset.getLabelIndex(label: label)
        if (index < 0 || index >= dataset.outputShape.totalSize) { throw DataParsingErrors.MoreUniqueLabelsThanOutputDimension }
        
        // Create the tensor
        var outputTensor = CreateTensor.constantValues(type: dataset.outputType, shape: dataset.outputShape, initialValue: 0.0)
        try outputTensor.setElement(index: index, value: 1.0)
        
        return outputTensor
    }
}


///  An image chunk where the specified image file is read and made into a tensor
public class ImageFile : ImageChunk  {
    /// Create the ImageFile chunk for an Image parser with a classification index
    /// - Parameters:
    ///   - fileName: The name of the file to load (with possible prefix of  subdirectories from parse top-level directory)
    ///   - classificationIndex: The classification index for the image
    public init(_ fileName: String, classificationIndex: Int) {
        super.init(type: .file, fileName: fileName, index: classificationIndex, label: nil)
    }
    
    /// Create the ImageFile chunk for an Image parser with a classification label
    /// - Parameters:
    ///   - fileName: The name of the file to load (with possible prefix of  subdirectories from parse top-level directory)
    ///   - classificationLabel: The classification label for the image
    public init(_ fileName: String, classificationLabel: String) {
        super.init(type: .file, fileName: fileName, index: nil, label: classificationLabel)
    }
}

///  An image chunk where the specified image directory is read and all images are made into a tensor with the given label
public class ImageDirectory : ImageChunk  {
    /// Create the ImageFile chunk for an Image parser with a classification index
    /// - Parameters:
    ///   - directoryName: The name of the directory to load images from (with possible prefix of  subdirectories from parse top-level directory)
    ///   - classificationIndex: The classification index for the image
    public init(_ directoryName: String, classificationIndex: Int) {
        super.init(type: .labeledDirectory, fileName: directoryName, index: classificationIndex, label: nil)
    }
    
    /// Create the ImageFile chunk for an Image parser with a classification label
    /// - Parameters:
    ///   - directoryName: The name of the directory to load images from (with possible prefix of  subdirectories from parse top-level directory)
    ///   - classificationLabel: The classification label for the image
    public init(_ directoryName: String, classificationLabel: String) {
        super.init(type: .labeledDirectory, fileName: directoryName, index: nil, label: classificationLabel)
    }
}

///  An image chunk where the specified image directory is read and all subdirectories are parsed for images that are made into a tensor with the subdirectory name as the label
public class ImageDirectoryWithSubdirectories : ImageChunk  {
    /// Create the ImageFile chunk for an Image parser with a classification index
    /// - Parameters:
    ///   - directoryName: The name of the directory to load images from (with possible prefix of  subdirectories from parse top-level directory)
    public init(_ directoryName: String) {
        super.init(type: .topLevelDirectory, fileName: directoryName, index: nil, label: nil)
    }
}
