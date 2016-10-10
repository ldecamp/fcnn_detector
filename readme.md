Fully Convolutional Binary Detector
===================================

# General
Contains the necessary code to train a fully convolutional Logo detector using tensorflow for one single class.
The code is inspired from [Fully Convolutional Networks for Semantic Segmentation][1] but does not contain any upsampling layer. 

The network downsample the grayscale input to create a final mask. 
A bouding box is then extracted from the mask and the coordinated are then mapped back to obtain a bounding box related to the input image.

## Assumptions - Dataset

The training data is split in 2 folders, named:

1. negative: contains a list of negative images
2. positive: contains a list of images containing the logo to detect.

Each positive sample file name must be defined as follows: "{some unique name}_{logo center y}_{logo center x}"
where {logo center y} is the height pixel coordinate of the logo center and {logo center x} is the width pixel coordinate of the logo center.

During training the dataset generates a square mask of 50 pixels around the logo center. 
No data augmentation is used.

## Processing workflow

1. Grayscale and downsample input image (CPU)
2. Process image with FCNN (GPU)
3. extract bouding box from mask (CPU)

## FCNN Model

- Input layer
- Convolution Layer: (3, 3) Kernel, RELu activation, SAME Padding, 32 Filters
- Convolution Layer: (3, 3) Kernel, RELu activation, SAME Padding, 32 Filters
- Max Pooling: (2, 2) Kernel
- Convolution Layer: (3, 3) Kernel, RELu activation, SAME Padding, 64 Filters
- Convolution Layer: (3, 3) Kernel, RELu activation, SAME Padding, 64 Filters
- Max Pooling: (2, 2) Kernel
- Convolution Layer: (3, 3) Kernel, RELu activation, SAME Padding, 128 Filters
- Convolution Layer: (3, 3) Kernel, RELu activation, SAME Padding, 128 Filters
- Convolution Layer: (1, 1) Kernel, RELu activation, SAME Padding, 256 Filters
- Convolution Layer: (1, 1) Kernel, no activation, SAME Padding, 2 Filters

## Results

Updated when training completed.


[1][https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf]