import argparse
import numpy
import pickle
import matplotlib
import matplotlib.pyplot as plt


# Font size parameters for graphs
TITLE_FONT_SIZE = 28
LABEL_FONT_SIZE = 22
BODY_FONT_SIZE = 18

matplotlib.rcParams.update({"font.size": BODY_FONT_SIZE})

# Ensure graph windows are maximized
manager = plt.get_current_fig_manager()
manager.resize(*manager.window.maxsize())


# Print averages to console
def printAverages():
    print("Statistics for test dataset:")
    
    # Compute averages
    avgMseOriginalInput = numpy.mean(mseOriginalInputs, axis=0)
    avgMseOriginalOutput = numpy.mean(mseOriginalOutputs, axis=0)
    avgSsimOriginalInput = numpy.mean(ssimOriginalInputs, axis=0)
    avgSsimOriginalOutput = numpy.mean(ssimOriginalOutputs, axis=0)
    avgOriginalFileSize = numpy.mean(originalFileSizes, axis=0)
    avgInputFileSize = numpy.mean(inputFileSizes, axis=0)
    avgOutputFileSize = numpy.mean(outputFileSizes, axis=0)
    
    # Computer other statistics
    print(ssimScores[:, -1])
    nNegativeSsimScore = sum(ssimScores[:, -1] < 0)
    
    indexOfHighestSsimScore = numpy.argmax(ssimScores[:, -1])
    indexOfLowestSsimScore = numpy.argmin(ssimScores[:, -1])
    
    # Print statistics
    print("  Average change in MSE: {}".format(avgMseOriginalOutput - avgMseOriginalInput))
    print("  Average change in SSIM: {}".format(avgSsimOriginalOutput - avgSsimOriginalInput))
    print("  Average original file size: {} KB".format(avgOriginalFileSize / 1000))
    print("  Average input file size: {} KB".format(avgInputFileSize / 1000))
    print("  Average output file size: {} KB".format(avgOutputFileSize / 1000))
    print("  Output images with negative SSIM Scores after {}K training: {}/{} ({}%)".format(modelTrainingLevels[-1] / 1000,
        nNegativeSsimScore, len(fileNamesWithoutExtension), nNegativeSsimScore / len(fileNamesWithoutExtension) * 100))
    print("  Output image with highest SSIM Score: {} (Score of {})".format(fileNamesWithoutExtension[indexOfHighestSsimScore], ssimScores[indexOfHighestSsimScore, -1]))
    print("  Output image with lowest SSIM Score: {} (Score of {})".format(fileNamesWithoutExtension[indexOfLowestSsimScore], ssimScores[indexOfLowestSsimScore, -1]))

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
        plt.legend(loc="lower right")
    else:
        # Multiple plots without legend
        for yValuesRow in yValues:
            plt.plot(xValues, yValuesRow)
    
    # Add labels/decorations and display graph
    if xTicks is None:
        plt.xticks(xValues, rotation=90)
    else:
        plt.xticks(numpy.arange(len(xTicks)), xTicks)
    
    plt.grid()
    plt.margins(0.02)
    
    if xLabel is not None:
        plt.xlabel(xLabel, fontsize=LABEL_FONT_SIZE)
    if yLabel is not None:
        plt.ylabel(yLabel, fontsize=LABEL_FONT_SIZE)
    if title is not None:
        plt.title(title, fontsize=TITLE_FONT_SIZE)
    
    plt.subplots_adjust(left=0.085, right=0.985, top=0.935, bottom=0.125)
    plt.show()

# Show scatter plot with metric vs metric
def graphMetricVsMetric(xValues, yValues, legendLabels=None, xLabel=None, yLabel=None, title=None):
    # Plot with or without legend
    if legendLabels is not None:
        # Multiple points with legend
        for xValue, yValue, legendLabel in zip(xValues, yValues, legendLabels):
            plt.scatter(xValue, yValue, label=legendLabel, s=4)
        plt.legend(loc="lower right")
    else:
        # Multiple points without legend
        plt.scatter(xValues, yValues, s=4)
    
    # Add labels/decorations and display graph
    plt.grid()
    plt.margins(0.02)
    
    if xLabel is not None:
        plt.xlabel(xLabel, fontsize=LABEL_FONT_SIZE)
    if yLabel is not None:
        plt.ylabel(yLabel, fontsize=LABEL_FONT_SIZE)
    if title is not None:
        plt.title(title, fontsize=TITLE_FONT_SIZE)
    
    plt.subplots_adjust(left=0.085, right=0.985, top=0.935, bottom=0.125)
    plt.show()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--statistics_file_path", required=True)
args = parser.parse_args()

# Parse directories and parameters
statsFilePath = args.statistics_file_path

# Read statistics file data
print("Reading data from statistics file")
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

# Generate SSIM score statistics
ssimScores = ssimOriginalOutputs - ssimOriginalInputs.reshape((-1, 1))

# Create version with input images treated as "0" training examples
modelTrainingLevelsWithZero = numpy.concatenate(([0], modelTrainingLevels))
mseOriginalOutputsWithZero = numpy.hstack((mseOriginalInputs.reshape((-1, 1)), mseOriginalOutputs))
ssimOriginalOutputsWithZero = numpy.hstack((ssimOriginalInputs.reshape((-1, 1)), ssimOriginalOutputs))

# Show selected graphs
print("Beginning statistics visualizations")
showAll = True

if True or showAll:
    printAverages()

# MSE/SSIM vs training examples

# All individual test images
if False or showAll:
    graphMetricVsTraining(
        xValues=(modelTrainingLevelsWithZero / 1000),
        yValues=mseOriginalOutputsWithZero,
        xLabel="Training Examples (Thousands)",
        yLabel="MSE",
        title="MSE vs # Training Examples for {} Test Images".format(len(fileNamesWithoutExtension)))
if False or showAll:
    graphMetricVsTraining(
        xValues=(modelTrainingLevelsWithZero / 1000),
        yValues=ssimOriginalOutputsWithZero,
        xLabel="Training Examples (Thousands)",
        yLabel="SSIM Value",
        title="SSIM Value vs # Training Examples for {} Test Images".format(len(fileNamesWithoutExtension)))

# Average, min, and max of test images
if False or showAll:
    graphMetricVsTraining(
        xValues=(modelTrainingLevelsWithZero / 1000),
        yValues=numpy.vstack((numpy.mean(mseOriginalOutputsWithZero, axis=0),
            numpy.min(mseOriginalOutputsWithZero, axis=0),
            numpy.max(mseOriginalOutputsWithZero, axis=0))),
        legendLabels=["Average", "Min", "Max"],
        xLabel="Training Examples (Thousands)",
        yLabel="MSE",
        title="MSE vs # Training Examples")
if False or showAll:
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
if False or showAll:
    graphMetricVsTraining(
        xValues=(modelTrainingLevels / 1000),
        yValues=ssimScores,
        xLabel="Training Examples (Thousands)",
        yLabel="SSIM Score",
        title="SSIM Score vs # Training Examples for {} Test Images".format(len(fileNamesWithoutExtension)))

# Average, min, and max SSIM score
if False or showAll:
    graphMetricVsTraining(
        xValues=(modelTrainingLevels / 1000),
        yValues=numpy.vstack((numpy.mean(ssimScores, axis=0),
            numpy.min(ssimScores, axis=0),
            numpy.max(ssimScores, axis=0))),
        legendLabels=["Average", "Min", "Max"],
        xLabel="Training Examples (Thousands)",
        yLabel="SSIM Score",
        title="SSIM Score vs # Training Examples")

# File sizes vs training examples

# All individual test images, output file size
if False or showAll:
    graphMetricVsTraining(
        xValues=(modelTrainingLevels / 1000),
        yValues=(outputFileSizes / 1000),
        xLabel="Training Examples (Thousands)",
        yLabel="Output File Size (KB)",
        title="Output File Size vs # Training Examples for {} Test Images".format(len(fileNamesWithoutExtension)))

# Average, min, and max output file sizes
if False or showAll:
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
if False or showAll:
    graphMetricVsTraining(
        xValues=numpy.array([0, 1, 2]),
        yValues=numpy.array([originalFileSizes / 1000, inputFileSizes / 1000, outputFileSizes[:, 0] / 1000]).transpose(),
        xTicks=["Original", "Input", "Output"],
        yLabel="File Size (KB) ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="Original, Input, and Output File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# MSE/SSIM vs file sizes

# All individual test images, input file size vs SSIM
if False or showAll:
    graphMetricVsMetric(
        xValues=(inputFileSizes / 1000),
        yValues=(ssimOriginalOutputs[:, -1]),
        xLabel="Input File Size (KB)",
        yLabel="SSIM Value ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Value vs Input File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# All individual test images, input file size vs SSIM
if False or showAll:
    graphMetricVsMetric(
        xValues=(inputFileSizes / 1000),
        yValues=(ssimScores[:, -1]),
        xLabel="Input File Size (KB)",
        yLabel="SSIM Score ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Score vs Input File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# All individual test images, output file size vs SSIM
if False or showAll:
    graphMetricVsMetric(
        xValues=(outputFileSizes[:, -1] / 1000),
        yValues=(ssimOriginalOutputs[:, -1]),
        xLabel="Output File Size (KB) ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="SSIM Value ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Value vs Output File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# All individual test images, output file size vs SSIM
if False or showAll:
    graphMetricVsMetric(
        xValues=(outputFileSizes[:, -1] / 1000),
        yValues=(ssimScores[:, -1]),
        xLabel="Output File Size (KB) ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="SSIM Score ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Score vs Output File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# All individual test images, output file size vs MSE
if False or showAll:
    graphMetricVsMetric(
        xValues=(outputFileSizes[:, -1] / 1000),
        yValues=(mseOriginalOutputs[:, -1]),
        xLabel="Output File Size (KB) ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="MSE ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="MSE vs Output File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# MSE/SSIM vs change in file sizes

# All individual test images, change in file size from original to output image vs SSIM
if False or showAll:
    graphMetricVsMetric(
        xValues=((outputFileSizes[:, -1] - originalFileSizes) / 1000),
        yValues=(ssimOriginalOutputs[:, -1]),
        xLabel="Change in File Size from Original to Output Image (KB) ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="SSIM Value ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Value vs Change in File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# All individual test images, change in file size from original to output image vs SSIM
if False or showAll:
    graphMetricVsMetric(
        xValues=((outputFileSizes[:, -1] - originalFileSizes) / 1000),
        yValues=(ssimScores[:, -1]),
        xLabel="Change in File Size from Original to Output Image (KB) ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="SSIM Score ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Score vs Change in File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# All individual test images, change in file size from original to output image vs MSE
if False or showAll:
    graphMetricVsMetric(
        xValues=((outputFileSizes[:, -1] - originalFileSizes) / 1000),
        yValues=(mseOriginalOutputs[:, -1]),
        xLabel="Change in File Size from Original to Output Image (KB) ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="MSE ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="MSE vs Change in File Size for {} Test Images".format(len(fileNamesWithoutExtension)))

# MSE vs SSIM

# All individual test images, SSIM value vs MSE
if False or showAll:
    graphMetricVsMetric(
        xValues=(mseOriginalOutputs[:, -1] / 1000),
        yValues=(ssimOriginalOutputs[:, -1]),
        xLabel="MSE ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="SSIM Value ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Value vs MSE for {} Test Images".format(len(fileNamesWithoutExtension)))

# All individual test images, SSIM score vs MSE
if False or showAll:
    graphMetricVsMetric(
        xValues=(mseOriginalOutputs[:, -1] / 1000),
        yValues=(ssimScores[:, -1]),
        xLabel="MSE ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        yLabel="SSIM Score ({}K Training)".format(modelTrainingLevels[-1] / 1000),
        title="SSIM Score vs MSE for {} Test Images".format(len(fileNamesWithoutExtension)))

