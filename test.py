import argparse
import chainer
import numpy
import cupy
import cv2
import time
from PIL import Image

import models


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--input_path", required=True)
parser.add_argument("--output_path")
parser.add_argument("--use_gpu", type=int, default=-1)
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

# Load pre-trained generator model
generator = models.Generator()
chainer.serializers.load_npz(args.model_path, generator)
if useGpu >= 0:
    generator.to_gpu()

# Load image from file
inputImage = xp.array(Image.open(args.input_path))

# If monochrome, duplicate single channel 3 times to get RGB
if len(inputImage.shape) == 2:
    inputImage = xp.dstack((inputImage, inputImage, inputImage))

# If alpha channel present, trim it out
if inputImage.shape[2] == 4:
    inputImage = inputImage[:, :, :3]

# Convert image data to a chainer variable
inputData = chainer.Variable(xp.array([inputImage.transpose(2, 0, 1)], dtype=xp.float32))

# Run the image through the model and get the output
startTime = time.time()
with chainer.using_config("test", True):
    outputData = generator(inputData)

# Compute timing statistics
currentTime = time.time()
elapsedTime = currentTime - startTime
elapsedTimeString = time.strftime("%H:%M:%S", time.gmtime(elapsedTime))
print("Time elapsed: {}".format(elapsedTimeString))

# Convert the chainer output variable back into image data
outputImage = xp.clip(outputData.data[0, :, :, :], 0, 255)
outputImage = outputImage.transpose(1, 2, 0)
if useGpu >= 0:
    outputImage = chainer.cuda.to_cpu(outputImage)

# Save the image to disk or display in a window as appropriate
if args.output_path is None:
    # Display in windows
    print("Displaying output image on screen (no --output_path specified)")
    cv2.imshow("Input", cv2.cvtColor(inputImage, cv2.COLOR_RGB2BGR))
    cv2.imshow("Output", cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
else:
    # Save to disk
    print("Saving output image: {}".format(args.output_path))
    cv2.imwrite(args.output_path, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))

