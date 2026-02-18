//
//  ImageTensors.swift
//  MPSGraphDSL
//
//  Created by Kevin Coble on 2/14/26.
//

import Foundation
import CoreGraphics

///  Class with static methods to create a ``Tensor`` from a CGIImage
public class ImageToTensor {
    /// Get a tensor of a specified type and shape (must be a 2D shape) from a greyscale image
    /// - Parameters:
    ///   - image: The image to convert into a tensor
    ///   - type: The data type of the returned tensor
    ///   - range: The range of data to map the unsigned bytes into (unused for UInt8 tensors).  0 value bytes get assigned min value, 255 value bytes max value.  Others are interpolated
    ///   - shape: The shape of the returned tensor.  Must be a 2-dimensional tensor
    /// - Returns: A Tensor of the shape and type specified
    public static func greyscaleImageToTensor(image: CGImage, type: DataType, range: ParameterRange, shape: TensorShape) throws -> Tensor {
        //  Shape must be 2D for grayscale images
        if (shape.numDimensions != 2) { throw GenericMPSGraphDSLErrors.InvalidShape }
        
        //  Resize the image if needed
        let height = shape.dimensions[0]
        let width  = shape.dimensions[1]
        let scaledImage: CGImage
        if ((image.width != width) || (image.height != height)) {
            scaledImage = ImageToTensor.resizeImage(image, to: CGSize(width: CGFloat(width), height: CGFloat(height)))
        }
        else {
            scaledImage = image
        }
        
        //  Get the data bytes from the image
        let result = ImageToTensor.getGreyscalePixelValues(fromCGImage: scaledImage)
        
        switch (type) {
        case .uInt8:
            let tensor = try TensorUInt8(shape: shape, initialValues: result.pixelValues)
            return tensor
        case .int32:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Int32(Double($0) * scale + min + 0.5)}
            let tensor = try TensorInt32(shape: shape, initialValues: elements)
            return tensor
        case .float16:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Float16(Double($0) * scale + min)}
            let tensor = try TensorFloat16(shape: shape, initialValues: elements)
            return tensor
        case .float32:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Float32(Double($0) * scale + min)}
            let tensor = try TensorFloat32(shape: shape, initialValues: elements)
            return tensor
        case .double:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Double($0) * scale + min}
            let tensor = try TensorDouble(shape: shape, initialValues: elements)
            return tensor
        }
    }
    
    /// Get a tensor of a specified type and shape (must be a 3D shape) from an RGB image
    /// - Parameters:
    ///   - image: The image to convert into a tensor
    ///   - type: The data type of the returned tensor
    ///   - range: The range of data to map the unsigned bytes into (unused for UInt8 tensors).  0 value bytes get assigned min value, 255 value bytes max value.  Others are interpolated
    ///   - shape: The shape of the returned tensor.  Must be a 3-dimensional tensor with the last dimension a 3 (for the three color channels)
    /// - Returns: A Tensor of the shape and type specified
    public static func rgbImageToTensor(image: CGImage, type: DataType, range: ParameterRange, shape: TensorShape) throws -> Tensor {
        //  Shape must be 3D for RGB images, with the last dimension 3
        if (shape.numDimensions != 3) { throw GenericMPSGraphDSLErrors.InvalidShape }
        if (shape.dimensions[2] != 3) { throw GenericMPSGraphDSLErrors.InvalidShape }
        
        //  Resize the image if needed
        let height = shape.dimensions[0]
        let width  = shape.dimensions[1]
        let scaledImage: CGImage
        if ((image.width != width) || (image.height != height)) {
            scaledImage = ImageToTensor.resizeImage(image, to: CGSize(width: CGFloat(width), height: CGFloat(height)))
        }
        else {
            scaledImage = image
        }
        
        //  Get the data bytes from the image
        let result = ImageToTensor.getRGBPixelValues(fromCGImage: scaledImage)
        
        switch (type) {
        case .uInt8:
            let tensor = try TensorUInt8(shape: shape, initialValues: result.pixelValues)
            return tensor
        case .int32:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Int32(Double($0) * scale + min + 0.5)}
            let tensor = try TensorInt32(shape: shape, initialValues: elements)
            return tensor
        case .float16:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Float16(Double($0) * scale + min)}
            let tensor = try TensorFloat16(shape: shape, initialValues: elements)
            return tensor
        case .float32:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Float32(Double($0) * scale + min)}
            let tensor = try TensorFloat32(shape: shape, initialValues: elements)
            return tensor
        case .double:
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 1.0 / (Double(UInt8.max) * (max - min))
            let elements = result.pixelValues.map{Double($0) * scale + min}
            let tensor = try TensorDouble(shape: shape, initialValues: elements)
            return tensor
        }
    }
    

    internal static func resizeImage(_ image: CGImage, to targetSize: CGSize) -> CGImage {
        let bitsPerComponent = image.bitsPerComponent
        let colorSpace = image.colorSpace
        let bitmapInfo = image.bitmapInfo
        
        let context = CGContext(
            data: nil,
            width: Int(targetSize.width),
            height: Int(targetSize.height),
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: 0,             //       <--  lets system pick, not width (100)
            space: colorSpace,
            bitmapInfo: bitmapInfo
        )

        // Set the interpolation quality for better results (e.g., .high, .medium, .low)
        context?.interpolationQuality = .high

        // Draw the image into the new context
        context?.draw(image, in: CGRect(origin: .zero, size: targetSize))

        let rescaledImage = context?.makeImage()!
        return rescaledImage!
    }
    
    internal static func getGreyscalePixelValues(fromCGImage imageRef: CGImage) -> (pixelValues: [UInt8], width: Int, height: Int)
    {
        var pixelValues: [UInt8]
        let width = imageRef.width
        let height = imageRef.height
        let bitsPerComponent = imageRef.bitsPerComponent
        let bytesPerRow = imageRef.bytesPerRow
        let totalBytes = height * bytesPerRow

        let colorSpace = CGColorSpaceCreateDeviceGray()
        var intensities = [UInt8](repeating: 0, count: totalBytes)
        
        let bitmapInfo = CGBitmapInfo(rawValue: 0)

        let contextRef = CGContext(data: &intensities, width: width, height: height, bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo)
        contextRef?.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(width), height: CGFloat(height)))

        //  Remove any extra allocation on each row
        if (bytesPerRow != width) {
            var offset = 0
            pixelValues = []
            for _ in 0..<height {
                pixelValues += Array(intensities[offset..<(offset + width)])
                offset += bytesPerRow
            }
        }
        else {
            pixelValues = intensities
        }

        return (pixelValues, width, height)
    }
    
    internal static func getRGBPixelValues(fromCGImage imageRef: CGImage) -> (pixelValues: [UInt8], width: Int, height: Int)
    {
        let rgbWidth = imageRef.width
        let rgbheight = imageRef.height
        let rgbBitsPerComponent = imageRef.bitsPerComponent
        let rgbBytesPerRow = imageRef.bytesPerRow
        let rgbTotalBytes = rgbheight * rgbBytesPerRow

        let rgbColorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        var rgbPixelValues = [UInt8](repeating: 0, count: rgbTotalBytes)
        
        let rgbBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue).union(.byteOrder32Big)

        let rgbContext = CGContext(
                    data: &rgbPixelValues,
                    width: rgbWidth,
                    height: rgbheight,
                    bitsPerComponent: rgbBitsPerComponent,
                    bytesPerRow: rgbBytesPerRow,
                    space: rgbColorSpace,
                    bitmapInfo: rgbBitmapInfo
                )
        rgbContext!.draw(imageRef, in: CGRect(x: 0.0, y: 0.0, width: CGFloat(rgbWidth), height: CGFloat(rgbheight)))

        //  Remove the extra allocation in bytesPerRow and the alpha channel
        var rgbElements: [UInt8] = []
        var offset = 1  //  Skip the alpha channel at the beginning (assume ARGB)
        for _ in 0..<rgbheight {
            var pixelOffset = offset
            for _ in 0..<rgbWidth {
                rgbElements.append(rgbPixelValues[pixelOffset])
                rgbElements.append(rgbPixelValues[pixelOffset + 1])
                rgbElements.append(rgbPixelValues[pixelOffset + 2])
                pixelOffset += 4
            }
            offset += rgbBytesPerRow
        }
        return (rgbElements, rgbWidth, rgbheight)
    }
}

///  Class with static methods to create a CGImage from a ``Tensor``
public class TensorToImage {
    /// Convert a tensor into a greyscale (if tensor is 2-dimensional) or RGB (if tensor 3-dimensional with last dimension sized to 3) image
    /// - Parameters:
    ///   - tensor: The tensor to convert
    ///   - range: The range of values of the tensor that maps to the 0-255 range of the UInt8 data used to make the image (ignored for UInt8 tensors)
    /// - Returns: The CGImage created from the tensor
    public static func tensorToImage(_ tensor: Tensor, range: ParameterRange) throws -> CGImage? {
        //  Tensor must be two-dimensional or three-dimensional with last dimension 3
        var rgb: Bool = false
        if (tensor.shape.numDimensions == 2) {
            rgb = false
        }
        else if (tensor.shape.numDimensions == 3) {
            if (tensor.shape.dimensions[2] != 3) { throw GenericMPSGraphDSLErrors.InvalidShape }
            rgb = true
        }
        else {
            throw GenericMPSGraphDSLErrors.InvalidShape
        }

        //  Convert the tensor data to a UInt8 array
        if (range.max.asDouble == range.min.asDouble) { throw GenericMPSGraphDSLErrors.InvalidValue }
        var elements: [UInt8]
        switch tensor.type {
        case .uInt8:
            let castTensor = tensor as! TensorUInt8
            elements = castTensor.elements
        case .int32:
            let castTensor = tensor as! TensorInt32
            let tensorElements = castTensor.elements
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 255.0 / (max - min)
            elements = []
            for element in tensorElements {
                let value = (Double(element) - min) * scale
                if (value < 0) { elements.append(0) }
                else if (value > 255)  { elements.append(255) }
                else { elements.append(UInt8(value + 0.5)) }
            }
        case .float16:
            let castTensor = tensor as! TensorFloat16
            let tensorElements = castTensor.elements
            let min = Float16(range.min.asDouble)
            let max = Float16(range.max.asDouble)
            let scale: Float16 = 255.0 / (max - min)
            elements = []
            for element in tensorElements {
                let value = (element - min) * scale
                if (value < 0) { elements.append(0) }
                else if (value > 255)  { elements.append(255) }
                else { elements.append(UInt8(value + 0.5)) }
            }
        case .float32:
            let castTensor = tensor as! TensorFloat32
            let tensorElements = castTensor.elements
            let min = Float32(range.min.asDouble)
            let max = Float32(range.max.asDouble)
            let scale: Float32 = 255.0 / (max - min)
            elements = []
            for element in tensorElements {
                let value = (element - min) * scale
                if (value < 0) { elements.append(0) }
                else if (value > 255)  { elements.append(255) }
                else { elements.append(UInt8(value + 0.5)) }
            }
        case .double:
            let castTensor = tensor as! TensorDouble
            let tensorElements = castTensor.elements
            let min = range.min.asDouble
            let max = range.max.asDouble
            let scale: Double = 255.0 / (max - min)
            elements = []
            for element in tensorElements {
                let value = (element - min) * scale
                if (value < 0) { elements.append(0) }
                else if (value > 255)  { elements.append(255) }
                else { elements.append(UInt8(value + 0.5)) }
            }
        }
        
        //  Convert the UInt8 data to a CGImage
        let width = tensor.shape.dimensions[0]
        let height = tensor.shape.dimensions[1]
        let image: CGImage?
        if (rgb) {
            image = imageFromRGBElements(elements: elements, width: width, height: height)
        }
        else {
            image = imageFromGrayscaleElements(elements: elements, width: width, height: height)
        }
        
        return image
    }
    
    internal static func imageFromGrayscaleElements(elements: [UInt8], width: Int, height: Int) -> CGImage? {
        //  Define image properties
        let bitsPerComponent = 8
        let bitsPerPixel = 8 // 1 sample per pixel (grayscale)
        let bytesPerRow = width // No extra padding needed for grayscale (kvImageNoAllocate ensures this in Accelerate)
        let colorSpace = CGColorSpaceCreateDeviceGray() // Use device gray color space
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        //  Create the CGImage from the raw data
        let pixelData: Data = Data(elements)
        guard let providerRef = CGDataProvider(data: pixelData as CFData) else { return nil }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: providerRef,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        ) else { return nil }
        
        return cgImage
    }
    
    internal static func imageFromRGBElements(elements: [UInt8], width: Int, height: Int) -> CGImage? {
        //  Add an alpha channel to the data
        var rgbPixels: [UInt8] = []
        var pixelIndex = 0
        for _ in 0..<height {
            for _ in 0..<width {
                rgbPixels.append(0xff)
                rgbPixels.append(elements[pixelIndex])
                pixelIndex += 1
                rgbPixels.append(elements[pixelIndex])
                pixelIndex += 1
                rgbPixels.append(elements[pixelIndex])
                pixelIndex += 1
            }
        }
        //  Convert to an RGB image
        let rgbBitsPerComponent = 8
        let rgbBitsPerPixel = 32 // 4 samples per pixel (ARGB)
        let rgbBytesPerRow = height * 4 // 32 bits per pixel
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let rgbBitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue)

        let rgbPixelData: Data = Data(rgbPixels)
        let rgbProviderRef = CGDataProvider(data: rgbPixelData as CFData)

        let rgbCgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: rgbBitsPerComponent,
            bitsPerPixel: rgbBitsPerPixel,
            bytesPerRow: rgbBytesPerRow,
            space: rgbColorSpace,
            bitmapInfo: rgbBitmapInfo,
            provider: rgbProviderRef!,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
        
        return rgbCgImage
    }
}
