# SciNet (PyTorch)

This is a PyTorch implementation of the SciNet architecture, introduced in the paper Discovering physical concepts
with neural networks ([arxiv.org/abs/1807.10300](https://arxiv.org/abs/1807.10300)) by Raban Iten, Tony Metger, Henrik Wilming, Lidia del Rio and Renato Renner. It uses a modified variational autoencoder approach to discover fundamental physical properties from a given set of observations.

## Requirements
* [Python 3.6](https://www.python.org/downloads/release/python-366/)
* [Pytorch](https://pytorch.org/)
* [Matplotlib](https://matplotlib.org/) (only for plotting training scores)
* [Jupyter](http://jupyter.org/) (only for the notebook implementation)

## Example use case
The network is tested on the damped oscillator example described in the paper. Although the sequence of latent neurons is different from what is shown in the paper, the network has still learned to encode a pendulum's trajectory using its fundamental properties. This can be seen by looking at the activation of the neurons in the latent layer of the variational autoencoder. One neuron is practically linearly dependent on the spring constant *k* of the input, the other neuron is linearly dependent on the damping factor *b*. The system's motion can be fully predicted using two parameters, so the third neuron shows no activation as it is superficial.

![Latent layer activation](https://github.com/fd17/SciNet_PyTorch/blob/master/latent_layer.png "Logo Title Text 1")
