import argparse
import chainer
import numpy
import cupy
import cv2
import time
import os
import pickle
import re
from PIL import Image
from skimage.measure import compare_mse, compare_ssim

import models


# Returns the generator file name for a given training level
def generatorFileName(trainingLevel):
    return ("generator_" + str(trainingLevel) + "_examples.npz")

# Returns the original file name for a given image file name
def originalImageName(fileNameWithoutExtension):
    return (fileNameWithoutExtension + "_original.png")

# Returns the input file name for a given image file name
def inputImageName(fileNameWithoutExtension, jpegQualityPercent):
    return (fileNameWithoutExtension + "_input_" + str(jpegQualityPercent) + ".jpg")

# Returns the output file name for a given image file name
def outputImageName(fileNameWithoutExtension, modelTrainingLevel):
    return (fileNameWithoutExtension + "_output_" + str(modelTrainingLevel) + ".png")


# Prints a progress update to console at a specified interval
def printProgressUpdateAtInterval(nComplete, nTotal, interval):
    if (nComplete % interval) == 0:
        print("  {} of {} complete ({}%)".format(nComplete, nTotal, str(round(100.0 * nComplete / nTotal, 2))))


# Loads an image and ensures it has 3 colour channels
def loadImageAndEnsure3Channels(filePath):
    # Load image from file
    inputImage = numpy.array(Image.open(filePath))

    # If monochrome, duplicate single channel 3 times to get RGB
    if len(inputImage.shape) == 2:
        inputImage = numpy.dstack((inputImage, inputImage, inputImage))

    # If alpha channel present, trim it out
    if inputImage.shape[2] == 4:
        inputImage = inputImage[:, :, :3]

    return inputImage


# Prepares a given test image for testing, creating the original and input files
def prepareTestImage(testImageFilename, inputDirectoryPath, outputDirectoryPath, jpegQualityPercent):
    # Copy original file to output directory, converting it to PNG
    fileNameWithoutExtension = os.path.splitext(testImageFilename)[0]
    testFilePath = os.path.join(inputDirectoryPath, testImageFilename)
    originalFilePath = os.path.join(outputDirectoryPath, originalImageName(fileNameWithoutExtension))
    os.system("convert " + testFilePath + " " + originalFilePath + " > /dev/null 2>&1")

    # Compress original file to desired JPEG quality, creating input file
    inputFilePath = os.path.join(outputDirectoryPath, inputImageName(fileNameWithoutExtension, jpegQualityPercent))
    os.system("convert " + originalFilePath + " -quality " + str(jpegQualityPercent) + " " + inputFilePath + " > /dev/null 2>&1")


# Runs a given input image through the model and saves the output image
def generateOutputForInputImage(inputFilePath, outputFilePath, generatorModel):
    # Load image from file
    inputImage = loadImageAndEnsure3Channels(inputFilePath)

    # Convert image data to a chainer variable
    inputData = chainer.Variable(xp.array([inputImage.transpose(2, 0, 1)], dtype=xp.float32))

    # Run the image through the model and get the output
    with chainer.using_config("test", True):
        outputData = generatorModel(inputData)

    # Convert the chainer output variable back into image data
    # TODO: slowest step in pipeline
    outputImage = xp.clip(outputData.data[0, :, :, :], 0, 255)
    outputImage = outputImage.transpose(1, 2, 0)
    if useGpu >= 0:
        outputImage = chainer.cuda.to_cpu(outputImage)

    # Save output image
    cv2.imwrite(outputFilePath, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))
    
    # Convert values as necessary and return the two images
    outputImage = outputImage.astype(numpy.uint8)


# Evaluates a given image against a given baseline image
def evaluateImage(imageToEvaluate, baselineImage):
    mse = compare_mse(imageToEvaluate, baselineImage)
    ssim = compare_ssim(imageToEvaluate, baselineImage, multichannel=True)
    return mse, ssim


# Parse string argument to boolean
def stringToBoolean(str):
    return str.lower() in ("yes", "true", "t", "1")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--models_directory_path", required=True)
parser.add_argument("--input_directory_path", required=True)
parser.add_argument("--output_directory_path", required=True)
parser.add_argument("--jpeg_quality_percent", type=int, default=10)
parser.add_argument("--use_gpu", type=int, default=-1)
parser.add_argument("--max_test_images", type=int, default=-1)
parser.add_argument("--do_preprocessing", type=stringToBoolean, default=True)
parser.add_argument("--do_output_generation", type=stringToBoolean, default=True)
parser.add_argument("--do_performance_evaluation", type=stringToBoolean, default=True)
args = parser.parse_args()

# Determine computation engine (CPU or GPU)
useGpu = args.use_gpu
if useGpu >= 0:
    print("Using GPU with ID {} for compute".format(useGpu))
    print("  cuda support enabled: {}".format(chainer.cuda.available))
    print("  cudnn support enabled: {}".format(chainer.cuda.cudnn_enabled))
    chainer.cuda.get_device(useGpu).use()
    xp = chainer.cuda.cupy
else:
    print("Using CPU for compute")
    xp = numpy

# Parse directories and other parameters
inputDirectoryPath = args.input_directory_path
outputDirectoryPath = args.output_directory_path
jpegQualityPercent = args.jpeg_quality_percent
modelDirectoryPath = args.models_directory_path
maxTestImages = args.max_test_images
doPreprocessing = args.do_preprocessing
doOutputGeneration = args.do_output_generation
doPerformanceEvaluation = args.do_performance_evaluation


# Get list of input images
testFiles = os.listdir(inputDirectoryPath)
if len(testFiles) <= 0:
    print("Unable to find any input images in directory specified")
    exit(-1)
print("Found {} test images".format(len(testFiles)))

# Get all test image file names without extension
fileNamesWithoutExtension = []
for testFileIndex, testFile in enumerate(testFiles):
    # Enforce max number of test images
    if testFileIndex == maxTestImages:
        break

    # Trim out file name without extension
    fileNameWithoutExtension = os.path.splitext(testFile)[0]
    fileNamesWithoutExtension.append(fileNameWithoutExtension)


if doPreprocessing == False:
    print("Skipping preprocessing (flag set)")
else:
    # Iterate over all test images and prepare original and input images
    print("Copying and transforming all original images to {}% JPEG quality inputs".format(jpegQualityPercent))
    startTime = time.time()
    for testFileIndex, testFile in enumerate(testFiles):
        # Enforce max number of test images
        if testFileIndex == maxTestImages:
            break

        # Make a PNG copy of the original image and preprocess a compressed input image
        prepareTestImage(testFile, inputDirectoryPath, outputDirectoryPath, jpegQualityPercent)
        
        # Show progress updates at regular intervals
        printProgressUpdateAtInterval(testFileIndex + 1, len(testFiles), 100)

    # Compute timing statistics
    currentTime = time.time()
    elapsedTime = currentTime - startTime
    elapsedTimeString = time.strftime("%H:%M:%S", time.gmtime(elapsedTime))
    print("  Time elapsed: {}".format(elapsedTimeString))


# Get list of trained models
modelFiles = os.listdir(modelDirectoryPath)
if len(modelFiles) <= 0:
    print("Unable to find any model files in directory specified")
    exit(-1)

# Determine what models exist to be tested
pattern = re.compile("^generator_([0-9]+)_examples\.npz$")
modelTrainingLevels = []
for modelFile in modelFiles:
    # Strip out the training level from the filename
    match = pattern.search(modelFile)
    if match == None:
        continue
    trainingLevel = match.group(1)
    modelTrainingLevels.append(int(trainingLevel))

# Sort by training level
modelTrainingLevels.sort()
if len(modelTrainingLevels) <= 0:
    print("Unable to find any valid trained models in directory specified")
    exit(-1)

# DEBUG: customize list of models to process
#modelTrainingLevels = modelTrainingLevels[8:]

print("Found {} trained models".format(len(modelTrainingLevels)))
print("  Training levels: {}".format(modelTrainingLevels))

if doOutputGeneration == False:
    print("Skipping output generation (flag set)")
else:
    # Generate output images for all models and input images
    print("Beginning output generation with all models")
    for modelIndex, modelTrainingLevel in enumerate(modelTrainingLevels):
        # Load pre-trained generator model
        print("Generating outputs for model {} of {}, with {}K training".format(modelIndex + 1, len(modelTrainingLevels), (modelTrainingLevel / 1000)))
        generatorModel = models.Generator()
        chainer.serializers.load_npz(os.path.join(modelDirectoryPath, generatorFileName(modelTrainingLevel)), generatorModel)
        if useGpu >= 0:
            generatorModel.to_gpu()
        
        # Iterate over all input images and gather statistics
        startTime = time.time()
        for imageIndex, fileNameWithoutExtension in enumerate(fileNamesWithoutExtension):
            # Construct file paths for this image
            originalFilePath = os.path.join(outputDirectoryPath, originalImageName(fileNameWithoutExtension))
            inputFilePath = os.path.join(outputDirectoryPath, inputImageName(fileNameWithoutExtension, jpegQualityPercent))
            outputFilePath = os.path.join(outputDirectoryPath, outputImageName(fileNameWithoutExtension, modelTrainingLevel))
            
            # Run the input image through the model and save output
            generateOutputForInputImage(inputFilePath, outputFilePath, generatorModel)

            # Show progress updates at regular intervals
            printProgressUpdateAtInterval(imageIndex + 1, len(fileNamesWithoutExtension), 100)

        # Compute timing statistics and print data
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        elapsedTimeString = time.strftime("%H:%M:%S", time.gmtime(elapsedTime))
        print("  Time elapsed for this model: {}".format(elapsedTimeString))


if doPerformanceEvaluation == False:
    print("Skipping performance evaluation (flag set)")
else:
    # Set up variables to collect statistics data
    originalFileSizes = numpy.zeros((len(fileNamesWithoutExtension)))
    inputFileSizes = numpy.zeros((len(fileNamesWithoutExtension)))

    outputFileSizes = numpy.zeros((len(fileNamesWithoutExtension), len(modelTrainingLevels)))

    mseOriginalInputs = numpy.zeros((len(fileNamesWithoutExtension)))
    ssimOriginalInputs = numpy.zeros((len(fileNamesWithoutExtension)))

    mseOriginalOutputs = numpy.zeros((len(fileNamesWithoutExtension), len(modelTrainingLevels)))
    ssimOriginalOutputs = numpy.zeros((len(fileNamesWithoutExtension), len(modelTrainingLevels)))

    # Determine statistics for original and input images
    print("Computing statistics between original and input images")
    startTime = time.time()
    for imageIndex, fileNameWithoutExtension in enumerate(fileNamesWithoutExtension):
        # Construct file paths
        originalFilePath = os.path.join(outputDirectoryPath, originalImageName(fileNameWithoutExtension))
        inputFilePath = os.path.join(outputDirectoryPath, inputImageName(fileNameWithoutExtension, jpegQualityPercent))
        
        # Compute file sizes
        originalFileSizes[imageIndex] = os.path.getsize(originalFilePath)
        inputFileSizes[imageIndex] = os.path.getsize(inputFilePath)
        
        # Load images
        originalImage = loadImageAndEnsure3Channels(originalFilePath)
        inputImage = loadImageAndEnsure3Channels(inputFilePath)
        
        # Compute and save statistics
        mse, ssim = evaluateImage(inputImage, originalImage)
        mseOriginalInputs[imageIndex] = mse
        ssimOriginalInputs[imageIndex] = ssim

        # Show progress updates at regular intervals
        printProgressUpdateAtInterval(imageIndex + 1, len(fileNamesWithoutExtension), 100)

    # Compute timing statistics
    currentTime = time.time()
    elapsedTime = currentTime - startTime
    elapsedTimeString = time.strftime("%H:%M:%S", time.gmtime(elapsedTime))
    print("  Time elapsed: {}".format(elapsedTimeString))

    # Compute performance statistics for all models and generated outputs
    print("Beginning performance calculations with all models")
    for modelIndex, modelTrainingLevel in enumerate(modelTrainingLevels):
        # Iterate over all images and gather statistics
        print("Evaluating performance of model {} of {}, with {}K training".format(modelIndex + 1, len(modelTrainingLevels), (modelTrainingLevel / 1000)))
        startTime = time.time()
        for imageIndex, fileNameWithoutExtension in enumerate(fileNamesWithoutExtension):
            # Construct file paths for this image
            originalFilePath = os.path.join(outputDirectoryPath, originalImageName(fileNameWithoutExtension))
            outputFilePath = os.path.join(outputDirectoryPath, outputImageName(fileNameWithoutExtension, modelTrainingLevel))
            
            # Load original and output images
            originalImage = loadImageAndEnsure3Channels(originalFilePath)
            outputImage = loadImageAndEnsure3Channels(outputFilePath)
            
            # Evaluate output against original
            mse, ssim = evaluateImage(outputImage, originalImage)
            mseOriginalOutputs[imageIndex, modelIndex] = mse
            ssimOriginalOutputs[imageIndex, modelIndex] = ssim
            
            # Evaluate file size statistics
            outputFileSize = os.path.getsize(outputFilePath)
            outputFileSizes[imageIndex, modelIndex] = outputFileSize

            # Show progress updates at regular intervals
            printProgressUpdateAtInterval(imageIndex + 1, len(fileNamesWithoutExtension), 100)

        # Compute timing statistics
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        elapsedTimeString = time.strftime("%H:%M:%S", time.gmtime(elapsedTime))
        print("  Time elapsed for this model: {}".format(elapsedTimeString))

    # Save statistics data to file
    statsFilePath = os.path.join(outputDirectoryPath, "statistics.pkl")
    print("Saving statistics to file (in pickled format): {}".format(statsFilePath))
    with open(statsFilePath, "wb") as statsFile:
        pickle.dump([fileNamesWithoutExtension, modelTrainingLevels, originalFileSizes, inputFileSizes, outputFileSizes,
            mseOriginalInputs, ssimOriginalInputs, mseOriginalOutputs, ssimOriginalOutputs], statsFile)
