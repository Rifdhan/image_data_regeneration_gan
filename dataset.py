import os
import six
import cv2
import random
import chainer
from PIL import Image
from chainer.dataset import dataset_mixin


class ImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataPaths, pathRoot, resizeTo):
        if isinstance(dataPaths, six.string_types):
            with open(dataPaths) as paths:
                dataPaths = [path.strip() for path in paths]
        self._dataPaths = dataPaths
        self._pathRoot = pathRoot
        self._resizeTo = resizeTo

    def __len__(self):
        return len(self._dataPaths)

    def get_example(self, i):
        # Open image at specified index
        path = os.path.join(self._pathRoot, self._dataPaths[i])
        image = Image.open(path)
        
        # Resize if applicable and return
        if not self._resizeTo is None:
            return image.resize(self._resizeTo)
        else:
            return image


class ResizedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, xp, dataPaths, pathRoot, resizeTo):
        self._xp = xp
        self.images = ImageDataset(dataPaths=dataPaths, pathRoot=pathRoot, resizeTo=resizeTo)

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        # Get example at specified index
        image = self._xp.array(self.images[i])
        
        # If monochrome, duplicate single channel 3 times to get RGB
        if len(image.shape) == 2:
            image = self._xp.dstack((image, image, image))
        
        # Transform data from (x, y, channels) to (channels, x, y)
        image_data = image.transpose(2, 0, 1)
        
        # If alpha channel present, trim it out
        if image_data.shape[0] == 4:
            image_data = image_data[:3]

        return image_data


class PreprocessedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, xp, useGpu, dataPaths, pathRoot=".", targetSize=96, resizeTo=None):
        self._xp = xp
        self._useGpu = useGpu
        self.resizedImages = ResizedImageDataset(xp=xp, dataPaths=dataPaths, pathRoot=pathRoot, resizeTo=resizeTo)
        self._dtype = xp.float32
        self.targetSize = targetSize

    def __len__(self):
        return len(self.resizedImages)

    def get_example(self, i):
        # Get indicated image
        originalImage = self.resizedImages[i]
        
        # Determine random bounds to crop image at
        cropStartX = random.randint(0, originalImage.shape[1] - self.targetSize)
        cropEndX = cropStartX + self.targetSize
        cropStartY = random.randint(0, originalImage.shape[2] - self.targetSize)
        cropEndY = cropStartY + self.targetSize
        
        # Crop image
        croppedOriginal = originalImage[:, cropStartX:cropEndX, cropStartY:cropEndY]
        
        # Compress image to JPEG
        if self._useGpu >= 0:
            croppedOriginalCpu = chainer.cuda.to_cpu(croppedOriginal)
            result, encodedImage = cv2.imencode('.jpg', croppedOriginalCpu.transpose(1, 2, 0), [int(cv2.IMWRITE_JPEG_QUALITY), 10])
            decodedImage = cv2.imdecode(encodedImage, 1)
            croppedCompressedCpu = decodedImage.transpose(2, 0, 1)
            croppedCompressed = chainer.cuda.to_gpu(croppedCompressedCpu, device=0)
        else:
            result, encodedImage = cv2.imencode('.jpg', croppedOriginal.transpose(1, 2, 0), [int(cv2.IMWRITE_JPEG_QUALITY), 10])
            decodedImage = cv2.imdecode(encodedImage, 1)
            croppedCompressed = decodedImage.transpose(2, 0, 1)
        
        # Convert to desired data type and return the example pairing
        croppedOriginal = self._xp.asarray(croppedOriginal, dtype=self._dtype)
        croppedCompressed = self._xp.asarray(croppedCompressed, dtype=self._dtype)
        
        return croppedCompressed, croppedOriginal

