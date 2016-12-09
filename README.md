## ml-assisted-hyperparameter-search

Overview
---------------------
Using a random forest to optimise convolutional neural network (CNN) hyperparameters on the famous CIFAR-10 image dataset. 

Documentation
---------------------

## Dependencies

On Windows follow Phil Ferriere's guide to building a deep learning setup - 
https://github.com/philferriere/dlwin.

## Installation

1.  Clone this repo with `git clone https://github.com/ayeright/ml-assisted-hyperparameter-search.git`.
2.  Switch to the repo folder with `cd ml-assisted-hyperparameter-search`.
    
## Usage

1. You can run `python cnn_randomsearch_cifar10.py' to train and validate CNNs on the CIFAR-10 dataset, with hyperparameter values drawn randomly from pre-defined distributions. You must pass four input arguments in the following order:
  * number of random search experiments 
  * number of CNNs to train in each experiment
  * maximum number of epochs for which to train each CNN
  * path to the directory where hyperparameter values and the corresponding validation results will be saved.

2. You can run `python cnn_mlsearch_cifar10.py' to train and validate CNNs on the CIFAR-10 dataset, with hyperparameter values chosen using a random forest. The random forest is trained using results from random search trials to approximate the relationship between the hyperparameter values and the validation loss. The trained random forest is asked to predict the validation loss for 1,000,000 candidate hyperparameter set, drawn randomly from the same distributions as in random search. Those predicted to achieve the smallest loss are trained for real. You must pass four input arguments in the following order:
  * number of ML-assisted search experiments 
  * number of CNNs to train in each experiment
  * maximum number of epochs for which to train each CNN
  * path to the directory containing random search results, also where ML-assisted search results will be saved.
