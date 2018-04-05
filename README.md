# LSNN

This is a minimal-code to train a recurrent spiking neural network (ours is called LSNN) to classify the MNIST digits when the pixels are provided one after the other.
For more details about LSNN see [1]. This model uses a method a network rewiring to keep a sparse connectivity during training, this method is called DEEP R and is described in [2].
Note that for the purpose of this tutorial, we simplified the task used in [1], the inputs are given in grey level within analog to spike conversion and the output is based on the level of the membrane potentials of the readouts at the last time step.

The code was written by Guillaume Bellec and Darjan Salaj at the IGI institute of TU Graz between 2017 and 2018.

[1] Long short-term memory and Learning-to-learn in networks of spiking neurons  
Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, Wolfgang Maass  
Arxiv 1803.09574, https://arxiv.org/abs/1803.09574



[2] Deep Rewiring: Training very sparse deep networks  
Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein  
Published at ICLR 2018, (https://arxiv.org/abs/1711.05136)

