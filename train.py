import logging
import os
import argparse
import chainer
import numpy
import glob

import dataset
import models


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_data_paths", required=True)
parser.add_argument("--model_output_path", required=True)
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

# Determine logging and nodel save intervals
nExamplesBetweenLogs = args.batch_size * 5
nExamplesBetweenModelSaves = args.batch_size * 500

# Determine computation engine
# TODO add GPU support
xp = numpy

# Load training dataset
trainDataPaths = glob.glob(args.train_data_paths)
trainDataset = dataset.PreprocessedImageDataset(dataPaths=trainDataPaths, targetSize=96, resizeTo=(300, 300))

# Create iterator to step through training data
iterator = chainer.iterators.MultiprocessIterator(trainDataset, batch_size=args.batch_size, repeat=True, shuffle=True)

# Create generator network
generator = models.Generator()
generatorOptimizer = chainer.optimizers.Adam()
generatorOptimizer.setup(generator)

# Create discriminator network
discriminator = models.Discriminator()
discriminatorOptimizer = chainer.optimizers.Adam()
discriminatorOptimizer.setup(discriminator)

# Determine coefficients for measuring loss
lossMseCoefficient = args.loss_mse_coefficient
lossAdversarialCoefficient = args.loss_adversarial_coefficient

# Will keep track of loss values for the last few examples to average out when logging statistics
totalLossGenerator, totalLossGeneratorAdversarial, totalLossGeneratorContent, totalLossDiscriminator = 0, 0, 0, 0

# Iterate through training data
logging.info("Beginning training")
nExamplesSeen = 0
for example in iterator:
    # Extract training data from example
    trainInput = chainer.Variable(xp.array([item[0] for item in example]))
    trainTarget = chainer.Variable(xp.array([item[1] for item in example]))
    
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
        # Log average statistics for the last bunch of examples
        logging.info("examples seen: {}".format(nExamplesSeen))
        logging.info("  loss generator (total): {}".format(totalLossGenerator / nExamplesSeen))
        logging.info("    loss generator (adversarial): {}".format(totalLossGeneratorAdversarial / nExamplesSeen))
        logging.info("    loss generator (mean squared error): {}".format(totalLossGeneratorContent / nExamplesSeen))
        logging.info("  loss discriminator: {}".format(totalLossDiscriminator / nExamplesSeen))
        
        # Reset accumulators for the next bunch
        totalLossGenerator, totalLossGeneratorAdversarial, totalLossGeneratorContent, totalLossDiscriminator = 0, 0, 0, 0
    
    # Determine if we should save the current model data
    if nExamplesSeen % nExamplesBetweenModelSaves == 0:
        # Save generator model data
        chainer.serializers.save_npz(os.path.join(outputDirectory, "generator_{}_examples.npz".format(nExamplesSeen)), generator)

