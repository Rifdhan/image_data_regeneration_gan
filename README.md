# Dejpeg GAN

Chainer implementation of a GAN for anti-JPEG data recovery.
Adapted from superresolution model: https://github.com/Hi-king/superresolution_gan

## Training Without GPU

```
python train.py --train_data_paths "/path/to/images/*.jpg" --model_output_path "/path/to/trained/model/output"
```
Note: the output directory must not exist, it will be created during runtime.

## Training With GPU

GPU support coming soon!

```
python train.py --train_data_paths "/path/to/images/*.jpg" --model_output_path "/path/to/model/output" --use_gpu=0
```
Note: cupy and cuDNN must be installed and set up. You will need to identify your GPU device ID (usually guess and check works fine). In this example it is `0`.

### Testing

```
python test.py --model_path "/path/to/model.npz" --input_path "/path/to/input/image.jpg" --output_path "/path/to/output/image.png"
```
Note: you may omit the output_path argument and the results will be displayed in a window instead.

