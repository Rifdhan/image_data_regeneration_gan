import argparse
import chainer
import numpy
import cv2
from PIL import Image

import models


def clipImagePixelValues(x):
    return numpy.uint8(0 if x < 0 else (255 if x > 255 else x))


# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True)
parser.add_argument("--input_path", required=True)
parser.add_argument("--output_path")
args = parser.parse_args()

# Load pre-trained generator model
generator = models.Generator()
chainer.serializers.load_npz(args.model_path, generator)

# Load image from file
inputImage = numpy.array(Image.open(args.input_path))

# If monochrome, duplicate single channel 3 times to get RGB
if len(inputImage.shape) == 2:
    inputImage = numpy.dstack((inputImage, inputImage, inputImage))

# If alpha channel present, trim it out
if inputImage.shape[2] == 4:
    inputImage = inputImage[:, :, :3]

# Convert image data to a chainer variable
inputData = chainer.Variable(numpy.array([inputImage.transpose(2, 0, 1)], dtype=numpy.float32))

# Run the image through the model and get the output
with chainer.using_config("test", True):
    outputData = generator(inputData)

# Convert the chainer output variable back into image data
outputImage = (numpy.vectorize(clipImagePixelValues)(outputData.data[0, :, :, :])).transpose(1, 2, 0)

# Save the image to disk or display in a window as appropriate
if args.output_path is None:
    # Display in windows
    cv2.imshow("Input", cv2.cvtColor(inputImage, cv2.COLOR_RGB2BGR))
    cv2.imshow("Output", cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
else:
    # Save to disk
    cv2.imwrite(args.output_path, cv2.cvtColor(outputImage, cv2.COLOR_RGB2BGR))

