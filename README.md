# GMM_MRI

Code repository for a Gaussian Mixture Model to segment MRI data. This repository is for Statistical Learning III at Portland State University.

Data is omitted. Download available data into the data folder. Expected files are in data.txt and gt.txt.

## Prerequisites
Python 3.0
TensorFlow
Scipy
Numpy
Matplotlib

## Directory structure
### data
Data folder that contains all avaliable images with list of filenames.
### latex
Latex source file for derivation.pdf

## Source Files:
### data.py
Files for reading in images and parsing out ground truth into numpy arrays
### kmeans.py
Applies kmeans to data using TensorFlow. Outputs an .npy of cluster means into data folder.
### semi_gmm.py
Applies semi-supervised GMM using EM, with test images as missing data.
### sup_gmm.py
Applies supervised GMM using EM with only training images, and applies E step on test images
### unsup_gmm.py
Applies unsupervised GMM using EM with all available data

## Documents:
### derivation.pdf
Derivation of update rules for semi-supervised GMM using EM
### slides.pdf
Slides for output images comparing kmeans, unsupervised, supervised, and semi-supervised GMM on test image

