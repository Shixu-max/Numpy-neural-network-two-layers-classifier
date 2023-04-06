<h1 align="center"> Neural Network Two Layers Classifier</h1>

<div align="center"> Author: 22110980014 Xu Shi</div>

GitHub: https://github.com/Shixu-max/Numpy-neural-network-two-layers-classifier

Model in Baidu Netdisk: https://pan.baidu.com/s/1hzkMCHe8OqT3ZuXBRyiuug

where Baidu Netdisk's fetch code: t5jp

## Contents

- [Target](#target)
  * [Train](#train)
  * [Parameter Seek](#parameter-seek)
  * [Test](#test)
- [Datasets](#datasets)
- [Packages](#packages)
- [Codes](#codes)


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

Can't use pytorch, tensorflow or any python package, you can use numpy. 

Upload the code to your own public github repo, and edit the training and testing steps in the repo's readme file. The trained model is uploaded to online disks such as Baidu Cloud/google drive.

## Packages
1. Numpy
2. Tqdm
3. matplotlib
4. pickle
5. gzip
6. time

## Codes 

* **twolayerclassifier.py**

This code file is which we tarin model, contains activation function; calculation of back propagation, loss, and gradient; learning rate reduction strategy; L2 regularization; optimizer SGD and save Model. 
 
* **figureplot.py**

This code is which we plot the process of training process. 

* **main.ipynb**

The main Jupyter notebook contains train model, seek optimal model, test model and visualization. If you want to get the results in PDF, you should experiment in order. When you get the optimal parameters, you should change the parameters of 'Train' in the first part, then you can get the results of visualization. 