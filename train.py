import logging
import os
import argparse
import chainer
import numpy
import cupy
import glob
import time

import dataset
import models


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_paths", required=True)
parser.add_argument("--model_output_path", required=True)
parser.add_argument("--use_gpu", type=int, default=-1)
parser.add_argument("--jpeg_quality", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--loss_adversarial_coefficient", type=float, default=0.00001)
parser.add_argument("--loss_mse_coefficient", type=float, default=0.0001)
args = parser.parse_args()

# Create output directory
outputDirectory = args.model_output_path
os.makedirs(outputDirectory)

# Start logging
logging.basicConfig(filename=os.path.join(outputDirectory, "log.txt"), level=logging.DEBUG)
logging.getLogger("").addHandler(logging.StreamHandler())
logging.info("Using configuration:")
logging.info(args)

# Determine computation engine (CPU or GPU)
useGpu = args.use_gpu
if useGpu >= 0:
    logging.info("Using GPU with ID {} for compute".format(useGpu))
    logging.info("  cuda support enabled: {}".format(chainer.cuda.available))
    logging.info("  cudnn support enabled: {}".format(chainer.cuda.cudnn_enabled))
    chainer.cuda.get_device(useGpu).use()
    xp = chainer.cuda.cupy
else:
    logging.info("Using CPU for compute")
    xp = numpy

# Determine logging and model save intervals (more frequent if training on CPU since it's slower)
if useGpu >= 0:
    nExamplesBetweenLogs = args.batch_size * 10
    nExamplesBetweenModelSaves = args.batch_size * 1000
else:
    nExamplesBetweenLogs = args.batch_size * 5
    nExamplesBetweenModelSaves = args.batch_size * 500

# Determine JPEG compression model to use
jpegQuality = args.jpeg_quality
if jpegQuality < 1 or jpegQuality > 100:
    logging.info("Invalid JPEG quality argument: {}; please enter a value between 1 and 100 (inclusively)".format(jpegQuality))
    exit(-1)

# Parameters for preprocessing dataset
# All input images will be resized to (resizeDimension, resizeDimension), being stretched as required to do so
# During training, a random (cropDimension, cropDimension) crop is taken of a randomly selected resized input image, and used as the training example
# This procedure allows us to generate many "different" training examples for any given input image
resizeDimension = 300
cropDimension = 96

# Load training dataset
trainDataPaths = glob.glob(args.train_data_paths)
trainDataset = dataset.PreprocessedImageDataset(xp=xp, useGpu=useGpu, jpegQuality=jpegQuality, dataPaths=trainDataPaths,
    targetSize=cropDimension, resizeTo=(resizeDimension, resizeDimension))

# Create iterator to step through training data
iterator = chainer.iterators.SerialIterator(trainDataset, batch_size=args.batch_size, repeat=True, shuffle=True)

# Create generator network
generator = models.Generator()
generatorOptimizer = chainer.optimizers.Adam()
generatorOptimizer.setup(generator)
if useGpu >= 0:
    generator.to_gpu()

# Create discriminator network
discriminator = models.Discriminator()
discriminatorOptimizer = chainer.optimizers.Adam()
discriminatorOptimizer.setup(discriminator)
if useGpu >= 0:
    discriminator.to_gpu()

# Determine coefficients for measuring loss
lossMseCoefficient = args.loss_mse_coefficient
lossAdversarialCoefficient = args.loss_adversarial_coefficient

# Will keep track of loss values for the last few examples to average out when logging statistics
totalLossGenerator, totalLossGeneratorAdversarial, totalLossGeneratorContent, totalLossDiscriminator = 0, 0, 0, 0

# Iterate through training data
logging.info("Beginning training")
startTime = time.time()
lastIntervalEndTime = startTime
nExamplesSeen = 0
for example in iterator:
    # example is of shape [batch_size, 2, channels, x, y]
    # And of types [list, tuple, xp.array, xp.array, xp.array)
    # We want to split on the second axis to get two arrays of shape [batch_size, channels, x, y]
    trainInput = xp.stack([item[0] for item in example], axis=3).transpose(3, 0, 1, 2)
    trainTarget = xp.stack([item[1] for item in example], axis=3).transpose(3, 0, 1, 2)
    trainInput = chainer.Variable(trainInput)
    trainTarget = chainer.Variable(trainTarget)
    
    # Get generator output for this example
    trainOutput = generator(trainInput)
    
    # Test the output with the discriminator
    discriminatedFromTrainOutput = discriminator(trainOutput)
    discriminatedFromTrainTarget = discriminator(trainTarget)
    
    # Compute losses for generator
    lossGeneratorAdversarial = chainer.functions.softmax_cross_entropy(
        discriminatedFromTrainOutput,
        chainer.Variable(xp.zeros(discriminatedFromTrainOutput.data.shape[0], dtype=xp.int32))
    )
    lossGeneratorContent = chainer.functions.mean_squared_error(
        trainOutput,
        trainTarget
    )
    lossGenerator = lossMseCoefficient * lossGeneratorContent + lossAdversarialCoefficient * lossGeneratorAdversarial
    
    # Accumulate total losses of each type for discriminator
    totalLossGeneratorAdversarial += chainer.cuda.to_cpu(lossGeneratorAdversarial.data)
    totalLossGeneratorContent += chainer.cuda.to_cpu(lossGeneratorContent.data)
    totalLossGenerator += chainer.cuda.to_cpu(lossGenerator.data)

    # Compute losses for discriminator
    lossDiscriminatorTrainOutput = chainer.functions.softmax_cross_entropy(
        discriminatedFromTrainOutput,
        chainer.Variable(xp.ones(discriminatedFromTrainOutput.data.shape[0], dtype=xp.int32))
    )
    lossDiscriminatorTrainTarget = chainer.functions.softmax_cross_entropy(
        discriminatedFromTrainTarget,
        chainer.Variable(xp.zeros(discriminatedFromTrainTarget.data.shape[0], dtype=xp.int32))
    )
    lossDiscriminator = lossDiscriminatorTrainOutput + lossDiscriminatorTrainTarget
    
    # Accumulate total losses for generator
    totalLossDiscriminator += chainer.cuda.to_cpu(lossDiscriminator.data)

    # Back-propagate losses and update generator network
    generator.zerograds()
    lossGenerator.backward()
    generatorOptimizer.update()

    # Back-propagate losses and update discriminator network
    discriminator.zerograds()
    lossDiscriminator.backward()
    discriminatorOptimizer.update()
    
    # Update example count
    nExamplesSeen += len(trainOutput.data)

    # Determine if we should log some statistics
    if nExamplesSeen % nExamplesBetweenLogs == 0:
        # Update timing parameters
        currentTime = time.time()
        elapsedTimeTotal = currentTime - startTime
        elapsedTimeTotalString = time.strftime("%H:%M:%S", time.gmtime(elapsedTimeTotal))
        elapsedTimeForInterval = currentTime - lastIntervalEndTime
        elapsedTimeForIntervalString = time.strftime("%H:%M:%S", time.gmtime(elapsedTimeForInterval))
        lastIntervalEndTime = currentTime
        
        # Log average statistics for the last bunch of examples
        logging.info("examples seen: {}".format(nExamplesSeen))
        logging.info("  loss generator (total): {}".format(totalLossGenerator / nExamplesSeen))
        logging.info("    loss generator (adversarial): {}".format(totalLossGeneratorAdversarial / nExamplesSeen))
        logging.info("    loss generator (mean squared error): {}".format(totalLossGeneratorContent / nExamplesSeen))
        logging.info("  loss discriminator: {}".format(totalLossDiscriminator / nExamplesSeen))
        logging.info("  total time elapsed: {} ({} this interval)".format(elapsedTimeTotalString, elapsedTimeForIntervalString))
        
        # Reset accumulators for the next bunch
        totalLossGenerator, totalLossGeneratorAdversarial, totalLossGeneratorContent, totalLossDiscriminator = 0, 0, 0, 0
    
    # Determine if we should save the current model data
    if nExamplesSeen % nExamplesBetweenModelSaves == 0:
        # Save generator model data
        savePath = os.path.join(outputDirectory, "generator_{}_examples.npz".format(nExamplesSeen))
        logging.info("saving trained model: {}".format(savePath))
        chainer.serializers.save_npz(savePath, generator)

