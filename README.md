# Numpy-neural-network-two-layers-classifier
<h1 align="center"> Neural Network Two Layers Classifier</h1>

<div align="center"> Author: 22110980014 Xu Shi</div>

## Contents

- [Target](#target)
  * [Train](#train)
  * [Parameter Seek](#parameter-seek)
  * [Test](#test)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Instructions](#instructions)


## Target
Constructing a two-layer neural network classifier, including at least the following three code files/sections:

### Train
Activation function
Calculation of back propagation, loss, and gradient
Learning rate reduction strategy
L2 regularization
Optimizer SGD
Save Model

### Perparameter Seek 
Learning rate 
Hidden layer size 
Regularization intensity

### Test
Import the model, test it with the model after parameter search, and output the classification accuracy

## Datasets
MNIST: https://academictorrents.com/details/323a0048d87ca79b68f12a6350a57776b6a3b7fb 

**Can not** use pytorch, tensorflow or any python package, you can use numpy. 

Upload the code to your own public github repo, and edit the training and testing steps in the repo's readme file. The trained model is uploaded to online disks such as Baidu Cloud/google drive.

## Requirements
1. Numpy
2. Tqdm
3. matplotlib
4. pickle
5. gzip
6. time

## Instructions 

* **neural_network.py**

This is where we implement the Network class. One can customize arbitrary DNN models with it. 
 
 * **mnist_train.py**

It illustrates an example of training the model and plotting the loss functions. With the given random seed anyone is able to reach a 99.994% accuracy on training data and 98.46% accuracy on testing data. It also involves instructions on loading and saving the model.

* **plotter.py**

An auxiliary function for plotting the training history. It requires Matplotlib.

* **mnist_train.ipynb**

In the Jupyter notebook includes hyperparameter searching and visualizations besides simply training, saving and loading models. The last step, visualizing the weights by PCA, requires the python package Sklearn.
