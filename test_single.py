import logging
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

# Returns the output file name for a given image file name
def outputImageName(modelTrainingLevel):
    return ("output_" + str(modelTrainingLevel) + ".png")


# Loads an image and ensures it has 3 colour channels
def loadImageAndEnsure3Channels(filePath):
    # Load image from file
    with Image.open(filePath) as inputFile:
        inputImage = numpy.array(inputFile, dtype=numpy.uint8)

    # If monochrome, duplicate single channel 3 times to get RGB
    if len(inputImage.shape) == 2:
        inputImage = numpy.dstack((inputImage, inputImage, inputImage))

    # If alpha channel present, trim it out
    if inputImage.shape[2] == 4:
        inputImage = inputImage[:, :, :3]

    return inputImage


# Runs a given input image through the model and saves the output image
def generateOutputForInputImage(inputFilePath, outputFilePath, generatorModel, modelTrainingLevel):
    # Load image from file
    inputImage = loadImageAndEnsure3Channels(inputFilePath)

    # Convert image data to a chainer variable
    inputData = chainer.Variable(xp.array([inputImage.transpose(2, 0, 1)], dtype=xp.float32))

    # Run the image through the model and get the output
    with chainer.using_config("test", True):
        outputData = generatorModel(inputData)

    # Convert the chainer output variable back into image data
    outputImage = xp.clip(outputData.data[0, :, :, :], 0, 255)
    outputImage = outputImage.transpose(1, 2, 0)
    if useGpu >= 0:
        outputImage = chainer.cuda.to_cpu(outputImage)

    # Write training level to image if applicable
    cvImage = cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR)
    if printTrainingLevels:
        cv2.putText(cvImage, "Training: " + str(modelTrainingLevel / 1000) + "K", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (128, 128, 128), 2)

    # Save output image
    cv2.imwrite(outputFilePath, cvImage)
    
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
parser.add_argument("--input_file_path", required=True)
parser.add_argument("--output_directory_path", required=True)
parser.add_argument("--jpeg_quality_percent", type=int, default=10)
parser.add_argument("--use_gpu", type=int, default=-1)
parser.add_argument("--generate_animated", type=stringToBoolean, default=True)
parser.add_argument("--print_training_levels", type=stringToBoolean, default=True)
args = parser.parse_args()

# Print configuration
print("Using configuration:")
print(args)

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
originalFilePath = args.input_file_path
outputDirectoryPath = args.output_directory_path
jpegQualityPercent = args.jpeg_quality_percent
modelDirectoryPath = args.models_directory_path
generateAnimated = args.generate_animated
printTrainingLevels = args.print_training_levels


# Create input image from original image
print("Transforming original image to {}% JPEG quality input".format(jpegQualityPercent))
inputFilePath = os.path.join(outputDirectoryPath, "input.jpg")
os.system("convert " + originalFilePath + " -quality " + str(jpegQualityPercent) + " " + inputFilePath + " > /dev/null 2>&1")


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
modelTrainingLevels = modelTrainingLevels[int(16/16)-1:int(640/16)]

print("Found {} trained models".format(len(modelTrainingLevels)))
print("  Training levels: {}".format(modelTrainingLevels))


# Generate output images for all models
print("Beginning output generation with all models")
for modelIndex, modelTrainingLevel in enumerate(modelTrainingLevels):
    # Load pre-trained generator model
    print("Generating output for model {} of {}, with {}K training".format(modelIndex + 1, len(modelTrainingLevels), (modelTrainingLevel / 1000)))
    generatorModel = models.Generator()
    chainer.serializers.load_npz(os.path.join(modelDirectoryPath, generatorFileName(modelTrainingLevel)), generatorModel)
    if useGpu >= 0:
        generatorModel.to_gpu()
    
    # Run the input image through the model and save output
    outputFilePath = os.path.join(outputDirectoryPath, outputImageName(modelTrainingLevel))
    generateOutputForInputImage(inputFilePath, outputFilePath, generatorModel, modelTrainingLevel)


# Generate animated version if applicable
if generateAnimated:
    print("Generating animation of output sequence with generated output images")
    cmd = "cd " + outputDirectoryPath + " && convert -delay 10 -loop 0 "
    for modelTrainingLevel in modelTrainingLevels:
        cmd += outputImageName(modelTrainingLevel) + " "
    cmd += "animated.mp4 > /dev/null 2>&1"
    os.system(cmd)

# Generate animated version of beginning portion only if applicable
if generateAnimated:
    print("Generating animation of first portion of output sequence with generated output images")
    cmd = "cd " + outputDirectoryPath + " && convert -delay 30 -loop 0 "
    for modelTrainingLevel in modelTrainingLevels[:10]:
        cmd += outputImageName(modelTrainingLevel) + " "
    cmd += "animated_beginning.mp4 > /dev/null 2>&1"
    os.system(cmd)


print("Finished; exiting")

