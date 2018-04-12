# LSNN

This is a minimal-code to train a recurrent spiking neural network (ours is called LSNN) to classify the MNIST digits when the pixels are provided one after the other.
For more details about LSNN see [1]. This model uses a method of network rewiring to keep a sparse connectivity during training, this method is called DEEP R and is described in [2].
Note that for the purpose of this tutorial, we simplified the task used from [1], the inputs are given in grey level without analog to spike conversion, and the network output is based directly on the membrane potentials of the readout neurons at the last time step instead of averaging it over tens of time milliseconds.

The code was written by Guillaume Bellec and Darjan Salaj at the IGI institute of TU Graz between 2017 and 2018.

[1] Long short-term memory and Learning-to-learn in networks of spiking neurons  
Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, Wolfgang Maass  
Arxiv 1803.09574, https://arxiv.org/abs/1803.09574


[2] Deep Rewiring: Training very sparse deep networks  
Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein  
Published at ICLR 2018, (https://arxiv.org/abs/1711.05136)

