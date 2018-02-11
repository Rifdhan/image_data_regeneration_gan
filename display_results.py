import argparse
import numpy
import pickle
import matplotlib.pyplot as plt


# Print averages to console
def printAverages():
    # Compute averages
    avgMseOriginalInput = numpy.mean(mseOriginalInputs, axis=0)
    avgMseOriginalOutput = numpy.mean(mseOriginalOutputs, axis=0)
    avgSsimOriginalInput = numpy.mean(ssimOriginalInputs, axis=0)
    avgSsimOriginalOutput = numpy.mean(ssimOriginalOutputs, axis=0)
    avgOriginalFileSize = numpy.mean(originalFileSizes, axis=0)
    avgInputFileSize = numpy.mean(inputFileSizes, axis=0)
    avgOutputFileSize = numpy.mean(outputFileSizes, axis=0)
    
    # Print statistics
    print("  Average change in MSE: {}".format(avgMseOriginalOutput - avgMseOriginalInput))
    print("  Average change in SSIM: {}".format(avgSsimOriginalOutput - avgSsimOriginalInput))
    print("  Average original file size: {} KB".format(avgOriginalFileSize / 1000))
    print("  Average input file size: {} KB".format(avgInputFileSize / 1000))
    print("  Average output file size: {} KB".format(avgOutputFileSize / 1000))

# Show chart with metric vs training level, supporting multiple graphs on the same axes
def graphMetricVsTraining(xValues, yValues, xTicks=None, legendLabels=None, xLabel=None, yLabel=None, title=None):
    # Plot single or multiple plots on same axes
    if yValues.ndim == 1:
        # Single plot
        plt.plot(xValues, yValues)
    elif legendLabels is not None:
        # Multiple plots with legend
        for yValuesRow, legendLabel in zip(yValues, legendLabels):
            plt.plot(xValues, yValuesRow, label=legendLabel)
        plt.legend(loc="upper right")
    else:
        # Multiple plots without legend
        for yValuesRow in yValues:
            plt.plot(xValues, yValuesRow)
    
    # Add labels/decorations and display graph
    if xTicks is None:
        plt.xticks(xValues)
    else:
        plt.xticks(numpy.arange(len(xTicks)), xTicks)
    
    plt.grid()
    
    if xLabel is not None:
        plt.xlabel(xLabel)
    if yLabel is not None:
        plt.ylabel(yLabel)
    if title is not None:
        plt.title(title)
    
    plt.show()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--statistics_file_path", required=True)
args = parser.parse_args()

# Parse directories and parameters
statsFilePath = args.statistics_file_path

# Read statistics file data
print("Reading data from pickled format file")
with open(statsFilePath, "rb") as statsFile:
    # Unpickle data
    fileNamesWithoutExtension, modelTrainingLevels, originalFileSizes, inputFileSizes, outputFileSizes, \
        mseOriginalInputs, ssimOriginalInputs, mseOriginalOutputs, ssimOriginalOutputs = pickle.load(statsFile)
    
    # Convert to numpy arrays instead of lists
    fileNamesWithoutExtension = numpy.array(fileNamesWithoutExtension)
    modelTrainingLevels = numpy.array(modelTrainingLevels)
    
    originalFileSizes = numpy.array(originalFileSizes)
    inputFileSizes = numpy.array(inputFileSizes)
    
    outputFileSizes = numpy.array(outputFileSizes)
    
    mseOriginalInputs = numpy.array(mseOriginalInputs)
    ssimOriginalInputs = numpy.array(ssimOriginalInputs)
    
    mseOriginalOutputs = numpy.array(mseOriginalOutputs)
    ssimOriginalOutputs = numpy.array(ssimOriginalOutputs)

# Compute global statistics across all images
print("Statistics for test dataset:")

# Create version with input images treated as "0" training examples
modelTrainingLevelsWithZero = numpy.concatenate(([0], modelTrainingLevels))
mseOriginalOutputsWithZero = numpy.hstack((mseOriginalInputs.reshape((-1, 1)), mseOriginalOutputs))
ssimOriginalOutputsWithZero = numpy.hstack((ssimOriginalInputs.reshape((-1, 1)), ssimOriginalOutputs))

# Show selected graphs

# All individual test images
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevelsWithZero / 1000),
        yValues=mseOriginalOutputsWithZero,
        xLabel="Training Examples (Thousands)",
        yLabel="MSE",
        title="MSE vs # Training Examples for Every Test Image")
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevelsWithZero / 1000),
        yValues=ssimOriginalOutputsWithZero,
        xLabel="Training Examples (Thousands)",
        yLabel="SSIM Value",
        title="SSIM Value vs # Training Examples for Every Test Image")

# Average, min, and max of test images
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevelsWithZero / 1000),
        yValues=numpy.vstack((numpy.mean(mseOriginalOutputsWithZero, axis=0),
            numpy.min(mseOriginalOutputsWithZero, axis=0),
            numpy.max(mseOriginalOutputsWithZero, axis=0))),
        legendLabels=["Average", "Min", "Max"],
        xLabel="Training Examples (Thousands)",
        yLabel="MSE",
        title="MSE vs # Training Examples")
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevelsWithZero / 1000),
        yValues=numpy.vstack((numpy.mean(ssimOriginalOutputsWithZero, axis=0),
            numpy.min(ssimOriginalOutputsWithZero, axis=0),
            numpy.max(ssimOriginalOutputsWithZero, axis=0))),
        legendLabels=["Average", "Min", "Max"],
        xLabel="Training Examples (Thousands)",
        yLabel="SSIM Value",
        title="SSIM Value vs # Training Examples")

# All individual test images, SSIM score
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevels / 1000),
        yValues=(ssimOriginalOutputs - ssimOriginalInputs.reshape((-1, 1))),
        xLabel="Training Examples (Thousands)",
        yLabel="SSIM Score",
        title="SSIM Score vs # Training Examples for Every Test Image")

# Average, min, and max SSIM score
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevels / 1000),
        yValues=numpy.vstack((numpy.mean(ssimOriginalOutputs - ssimOriginalInputs.reshape((-1, 1)), axis=0),
            numpy.min(ssimOriginalOutputs - ssimOriginalInputs.reshape((-1, 1)), axis=0),
            numpy.max(ssimOriginalOutputs - ssimOriginalInputs.reshape((-1, 1)), axis=0))),
        legendLabels=["Average", "Min", "Max"],
        xLabel="Training Examples (Thousands)",
        yLabel="SSIM Score",
        title="SSIM Score vs # Training Examples")

# All individual test images, output file size
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevels / 1000),
        yValues=(outputFileSizes / 1000),
        xLabel="Training Examples (Thousands)",
        yLabel="Output File Size (KB)",
        title="Output File Size vs # Training Examples for Every Test Image")

# Average, min, and max output file sizes
if False:
    graphMetricVsTraining(
        xValues=(modelTrainingLevels / 1000),
        yValues=numpy.vstack((numpy.mean(outputFileSizes / 1000, axis=0),
            numpy.min(outputFileSizes / 1000, axis=0),
            numpy.max(outputFileSizes / 1000, axis=0))),
        legendLabels=["Average", "Min", "Max"],
        xLabel="Training Examples (Thousands)",
        yLabel="Output File Size (KB)",
        title="Output File Size vs # Training Examples")

# All individual test images, original, input, and output file sizes
if True:
    graphMetricVsTraining(
        xValues=numpy.array([0, 1, 2]),
        yValues=numpy.array([originalFileSizes / 1000, inputFileSizes / 1000, outputFileSizes[:, 0] / 1000]).transpose(),
        xTicks=["Original", "Input", "Output"],
        yLabel="File Size (KB)",
        title="Original, Input, and Output File Size for Every Test Image")

