# SciNet (PyTorch)

This is a PyTorch implementation of the SciNet architecture, introduced in the paper  [*Discovering physical concepts
with neural networks*](https://arxiv.org/abs/1807.10300) by Raban Iten, Tony Metger, Henrik Wilming, Lidia del Rio and Renato Renner. It uses artificial neural networks with a modified variational autoencoder approach to discover fundamental physical properties from a given set of observations.

## Requirements
* [Python 3.6](https://www.python.org/downloads/release/python-366/)
* [Pytorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/) 
* [Jupyter](http://jupyter.org/)

## Usage
The SciNet architecture is defined in models.py. The exemplary use case below can be found in various IPython notebooks. The Generate_Trainingdata.ipynb notebook can be used to generate pendulum training data as outlined in the paper. The Training.ipynb and Analysis.ipynb notebooks can be used to see how the model is trained and how to analyse it when training is complete.

## Example use case
The provided notebooks aim to recreate the damped oscillator example from the paper. Although the sequence of latent neurons is different from what is shown in the paper, the network has still learned to encode a pendulum's trajectory using its physical properties. This can be seen by looking at the activation of the neurons in the latent layer of the variational autoencoder. One neuron is practically linearly dependent on the spring constant *k* of the input, the other neuron is linearly dependent on the damping factor *b*. The system's time evolution can be fully described using these two parameters, so the third neuron is superficial and consequently shows no activation.

![Latent layer activation](https://github.com/fd17/SciNet_PyTorch/blob/master/latent_layer.png "Logo Title Text 1")


