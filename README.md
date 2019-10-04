# MICNN - Multiple Instance Convolutional Neural Network

Multiple instance (MI) learning with a convolutional neural network enables end-to-end training in the presence of weak image-level labels. We propose a new method for aggregating predictions from smaller regions of the image into an image-level classification by using the quantile function. The quantile function provides a more complete description of the heterogeneity within each image, improving image-level classification. We also adapt image augmentation to the MI framework by randomly selecting cropped regions on which to apply MI aggregation during each epoch of training. This provides a mechanism to study the importance of MI learning.

The methods implemented here include those discussed in the following two publications:

H. D. Couture, J. S. Marron, C. M. Perou, M. A. Troester, M. Niethammer, Multiple Instance Learning for Heterogeneous Images: Training a CNN for Histopathology, Proc. MICCAI, 2018.

H. D. Couture, Discriminative Representations for Heterogeneous Images and Multimodal Data. PhD thesis, Department of Computer Science, University of North Carolina at Chapel Hill, Chapel Hill, NC, 2019.

This code is not the original used in these publications but an upgraded version to work with the latest version of Keras/TensorFlow and other libraries.  Tested with Python 3.7.4, TensorFlow 2.0Keras 2.2.4, sklearn 0.21.3, and skimage 0.15.0.

## Setup

Basic installation requires a number of python packages tat are most easily installed wit conda and pip:
```
conda install numpy scipy scikit-image scikit-learn cudatoolkit cudnn
pip install tensorflow-gpu
```

If using learning rate range test or cyclic learning rates, this repo is needed: https://github.com/psklight/keras_one_cycle_clr.  Setup the submole using the follow:
```
git submodule init
git submodule update
```

## Data Setup

The above referenced publications used data from the [Carolina Breast Cancer Study](http://cbcs.web.unc.edu/for-researchers/).  You may apply for access to this data set.

Running this code requires two files: labels.csv and sample_images.csv.  If masks (e.g., tissue vs. background) are available, then an additional file sample_masks.csv is needed, matching the format of sample_images.csv.

labels.csv should use the following format for up to N samples and K classification tasks:
```
sample,class1,class2,...,classK
sample1,label11,label12,...,label1K
...
sampleN,labelN1,labelN2,...,labelNK
```

Each class can be binary or multi-class.  Any string or number can be used to identify the classes.

sample_images.csv allows one or more image files to be specified for each sample:
```
sample1,image11,image12,...,image1M
...
sampleN,imageN1,imageN2,...,imageNM
```

Each sample may have a different number of associated images.

If a specific train/val/test split is needed, a file or files may be provided in the following format:
```
sample1,train
sample2,train
sample3,test
...
sampleN,val
```

## Example Usage for CBCS

```

```

## Citation

If you use this code, please cite:

```
@inproceedings{Couture2018_MICCAI,
 author = {Couture, Heather D. and Marron, J. S. and Perou, Charles M. and Troester, Melissa A. and Niethammer, Marc},
 booktitle = {Proc. MICCAI},
 title = {Multiple Instance Learning for Heterogeneous Images: Training a CNN for Histopathology},
 year = {2018}
}
