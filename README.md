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


### Testing Individual Images with Individual Models

```
python test.py --model_path "/path/to/model.npz" --input_path "/path/to/input/image.jpg" --output_path "/path/to/output/image.png"
```
Note: you may omit the output_path argument and the results will be displayed in a window instead.


### Testing Batches of Test Images with Multiple Models

```
python test_batch.py --models_directory_path "/path/to/models/directory" --input_directory_path "/path/to/test/images" --output_directory_path "/path/to/output/generated/images"
```
Note: consists of 3 steps: preprocessing original images to input images, then generating all model output images, then computing statistics across all images. Additional flags are available to skip any step(s) in the process: --do_preprocessing=true, --do_output_generation=true, --do_performance_evaluation=true

```
python display_results.py --statistics_file_path "/path/to/statistics/file.pkl"
```
Note: the .pkl file will be generated after the final step of the previous script above this one. Graphs displayed can be customized in this script itself.


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

