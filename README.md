# Dejpeg GAN

Chainer implementation of a GAN for anti-JPEG data recovery.
Adapted from superresolution model: https://github.com/Hi-king/superresolution_gan

## Training Without GPU

```
python train.py --train_data_paths "/path/to/images/*.jpg" --model_output_path "/path/to/save/trained/models"
```
Note: the output directory must not exist, it will be created during runtime.

## Training With GPU

```
python train.py --train_data_paths "/path/to/images/*.jpg" --model_output_path "/path/to/save/trained/models" --use_gpu=0
```
Note: cupy must be installed and set up. You will need to identify your GPU device ID. In this example, it is 0. You may simply try numbers from 0 upwards until one works.

### Testing

```
python test.py --model_path "/path/to/model.npz" --input_path "/path/to/input/image.jpg" --output_path "/path/to/output/image.png"
```
Note: you may omit the output_path argument and the results will be displayed in a window instead.

### Image Manipulation

To convert images between formats:
```
convert input.jpg output.png
```

To change image resolution:
```
mogrify -resize 50% image.jpg
```

To compress to specified JPEG compression level:
```
convert input.png -quality 10 output.jpg
```

