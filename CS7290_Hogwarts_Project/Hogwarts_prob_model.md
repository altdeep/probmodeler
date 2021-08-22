# Probabilistic Modeling of Hogwarts Student House Classification Using Stochastic Variational Inference in Pyro

This is the final project for CS7290: Applied Bayesian Machine Learning Summer 2021. 

# Author

Georgian Tutuianu

Robert Osazuwa Ness

# Introduction and Problem Statement

Given a set of features: 
- student personality type: Type A, B, C or D
- student herbology strength: Strong or Weak
- student quidditch rank score: real number > 0

Model the process generating the feature data with probabilistic parameters.

Predict which Hogwarts House the student most likely comes from. 

The Hogwarts Houses:
- Gryffindor 
- Hufflepuff 
- Slytherin
- Ravenclaw 

Using the Transmitter-Receiver framework and the aid of Dr. Ness, 
I built a probabilistic model capable of performing Stochastic Variational Inference (SVI) using simulated data.

I looked at the model's loss and accuracy sensitivity due to data sampling.

I have prior and posterior predictive checks as well.

Finally, I built a Multi-Layer Perceptron (MLP) neural net (NN) with the intent of implementing amortized variational inference (AVI).

However, due to time constraints AVI is not implemented.

# Getting Started

This project is built in a series of Google Colab Notebooks. 

I recommend forking the repo, adding the data files to your Google Drive and updating the filepath links.
Then the notebooks should work out of the box.

It should also be possible to run the code by copying it into python files and then using a virtual environment to install all of the dependencies below.

This repo contains the following files:

Google Colab Notebooks:
- Hogwarts_Model_Inference.ipynb
- Prior_vs_Posterior.ipynb
- Model_Sensitivity_With_Data_Size_Best_Case.ipynb 
- Make_Test_Set.ipynb
- Make_Training_Set.ipynb
- MLP.ipynb 

Data files:
- data100k.csv
- data100k_ground_truth.csv
- hogwarts_supervised_ground_truth.csv
- hogwarts_unsupervised_ground_truth.csv
- test_set.csv

# Current Project Version
- **Version 1.0.0**

## Module Version
- Python version 3.7.11
- Pytorch version 1.9.0
- Pyro version 1.7.0
- Numpy version 1.19.5
- Pandas version 1.1.5
- Matplotlib Version 3.2.2
- Scipy version 1.4.1

# Build Status
- Pre-production

# Code style
- Python auto-pep8

# Additional Project Info

## Future Considerations and To Dos
Hogwarts_Model_Inference
- Figure out how to accelerate pyro with GPU
- Figure out how to better constrain the priors or how to choose better priors since sometimes probabilities collapse
- Figure out how to better compare probability distributions vs parameters
- Add the true ground truth probabilities when comparing prior vs posterior probabilities
- Try Beta distribution for modeling the personality and house probabilities since Dirichlet is unstable unless very constrained
- Add amortized variational inference using the MLP trained on data simulated from the prior distribution

Prior_vs_Posterior
- Figure out why quidditch priors on sigma are outrageously large and figure out what more judicious priors look like
- Consider merging this notebook with the one above

Model_Sensitivity_With_Data_Size_Best_Case
- Instead of looking at the best case scenario where I calculate accuracy look at the how sensitive the learned parameters are to data

MLP
- Confirm feature standardization by removing the mean and scaling to unit variance is a bad idea
- Tune hyperparameters
- Consider using pytorch lightning or fastai to wrap the NN model so that I can iterate more quickly

Design Modifications
- Add Unit Tests
- Complete Transmitter-Receiver framework