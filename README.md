### LSNN: Efficient spiking recurrent neural networks

This repository provides a tensorflow library and a tutorial train a recurrent spiking neural network (ours is called LSNN).
For more details about LSNN see [1]. This model uses a method of network rewiring to keep a sparse connectivity during training, this method is called DEEP R and is described in [2].

In the tutorial `tutorial_sequential_mnist_with_LSNN.py`, you can classify the MNIST digits when the pixels are provided one after the other.
Note that for the purpose of this tutorial, we simplified the task used from [1], the inputs are given in grey level without analog to spike conversion, and the network output is based directly on the membrane potentials of the readout neurons at the last time step instead of averaging it over tens of milliseconds.

The code was written by Guillaume Bellec and Darjan Salaj at the IGI institute of TU Graz between 2017 and 2018.

[1] Long short-term memory and Learning-to-learn in networks of spiking neurons  
Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, Wolfgang Maass  
Arxiv 1803.09574, https://arxiv.org/abs/1803.09574

[2] Deep Rewiring: Training very sparse deep networks  
Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein  
ICLR 2018, (https://arxiv.org/abs/1711.05136)


### Installation

From the main folder run:  
`` pip3 install --user .``  
You can now import the tensorflow cell called ALIF (for adaptive leakey integrate and fire) as well as the rewiring wrapper to update connectivity matrices after each call to the optimizer.

## Troubleshooting

If the scripts fail with the following error:
`` Illegal instruction (core dumped) ``

It is most probably due to the lack of AVX instructions on the machine you are using.
A known workaround is to reinstall the LSNN package with older tensorflow version (1.5).
Change requirements.txt to contain:

`` tensorflow==1.5 ``