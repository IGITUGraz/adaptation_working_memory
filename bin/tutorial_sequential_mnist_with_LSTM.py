""" Recurrent Neural Network.

A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import json
import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import init_ops, math_ops, array_ops, nn_ops

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class IRNNCell(rnn.BasicRNNCell):
    """Le, Quoc V., Navdeep Jaitly, and Geoffrey E. Hinton.
    “A Simple Way to Initialize Recurrent Networks of Rectified Linear Units.”
    ArXiv:1504.00941 [Cs], April 3, 2015. http://arxiv.org/abs/1504.00941.
    """

    def __init__(self, num_units, reuse=None, name=None):
        super(IRNNCell, self).__init__(num_units, activation=tf.nn.relu, reuse=reuse, name=name)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)

        input_depth = inputs_shape[1].value
        rec_init = np.identity(self._num_units) + rd.normal(loc=0.0, scale=0.001,
                                                            size=[self._num_units, self._num_units])
        # init_range = np.sqrt(6.0 / (input_depth + self._num_units))
        # in_init = rd.uniform(-init_range, init_range, size=[input_depth, self._num_units])
        in_init = rd.normal(loc=0.0, scale=0.001, size=[input_depth, self._num_units])
        init = tf.constant_initializer(np.vstack((in_init, rec_init)))
        self._kernel = self.add_variable(_WEIGHTS_VARIABLE_NAME, initializer=init,
                                         shape=[input_depth + self._num_units, self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True


class RINCell(rnn.BasicRNNCell):
    """Hu, Yuhuang, Adrian Huber, Jithendar Anumula, and Shih-Chii Liu.
    “Overcoming the Vanishing Gradient Problem in Plain Recurrent Networks.”
    ArXiv:1801.06105 [Cs], January 18, 2018. http://arxiv.org/abs/1801.06105.
    """

    def __init__(self, num_units, reuse=None, name=None):
        super(RINCell, self).__init__(num_units, activation=tf.nn.relu, reuse=reuse, name=name)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        init_range = np.sqrt(6.0 / (input_depth + self._num_units))
        rec_init = rd.uniform(-init_range, init_range, size=[self._num_units, self._num_units])
        in_init = rd.normal(loc=0.0, scale=0.001, size=[input_depth, self._num_units])
        init = tf.constant_initializer(np.vstack((in_init, rec_init)))
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self._num_units, self._num_units],
            initializer=init)
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))
        self._surrogate_memory = tf.Variable(
            np.vstack((np.zeros((input_depth, self._num_units)),
                       np.identity(self._num_units))),
            dtype=self.dtype, name="SurrogateMemory", trainable=False)
        self._kernel = self._kernel + self._surrogate_memory
        self.built = True

    def call(self, inputs, state):
        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)
        output = self._activation(gate_inputs)
        return output, output

'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('ext_time', 1, 'Repeat factor to extend time of mnist task')
tf.app.flags.DEFINE_float('test_at', -1, 'Early stoppying accuracy criterion')
tf.app.flags.DEFINE_float('lr', 2e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('lr_decay', 0.95, 'Decay factor for learning rate')
tf.app.flags.DEFINE_float('gc', -1, 'gradient clipping')
tf.app.flags.DEFINE_integer('decay_lr_steps', 3000, 'Learning rate decay interval')
tf.app.flags.DEFINE_integer('training_steps', 37000, 'Number of training steps')
tf.app.flags.DEFINE_integer('batch_size', 256, 'Training batch size')
tf.app.flags.DEFINE_integer('print_every', 400, 'Print after every steps')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'Number of hidden units in LSTM')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'adam or rms or gd')
tf.app.flags.DEFINE_string('model', 'LSTM', 'LSTM or RNN or IRNN or RIN')
tf.app.flags.DEFINE_bool('prm', False, 'Fixed permutation of pixels')

# Save parameters and training log
try:
    flag_dict = FLAGS.flag_values_dict()
except:
    print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
    flag_dict = FLAGS.__flags

# Training Parameters
learning_rate = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False)
decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)
training_steps = FLAGS.training_steps
batch_size = FLAGS.batch_size
display_step = FLAGS.print_every
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

# Network Parameters
num_input = 1  # MNIST data input (img shape: 28*28)
timesteps = 28 * 28  # timesteps
num_hidden = FLAGS.num_hidden  # hidden layer num of features
num_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps*FLAGS.ext_time, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps*FLAGS.ext_time, 1)

    # Define a lstm cell with tensorflow
    if FLAGS.model == 'LSTM':
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    elif FLAGS.model == 'IRNN':
        lstm_cell = IRNNCell(num_hidden)
    elif FLAGS.model == 'RIN':
        lstm_cell = RINCell(num_hidden)
    elif FLAGS.model == 'RNN':
        lstm_cell = rnn.BasicRNNCell(num_hidden)
    else:
        raise NotImplementedError("Unknown model: " + FLAGS.model)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
if FLAGS.optimizer == "adam":
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
elif FLAGS.optimizer == "rms":
    opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
elif FLAGS.optimizer == "gd":
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
else:
    raise NotImplementedError("Unknown optimizer selected")

if FLAGS.gc > 0:
    gradients, variables = zip(*opt.compute_gradients(loss_op))
    gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.gc)
    grads_vars = [(g, v) for g, v in zip(gradients, variables)]
    train_op = opt.apply_gradients(grads_vars, global_step=global_step)
else:
    train_op = opt.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

if FLAGS.prm:
    permutation = np.random.permutation(np.arange(28*28))


def get_data(batch_size, test=False, shuffle=True):
    if test:
        batch_x, batch_y = mnist.test.next_batch(batch_size, shuffle=shuffle)
    else:
        batch_x, batch_y = mnist.train.next_batch(batch_size, shuffle=shuffle)
    if FLAGS.prm:
        batch_x[:] = batch_x[:, permutation]
    return batch_x, batch_y


print(json.dumps(flag_dict, indent=4))
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print("TOTAL PARAMS", total_parameters)

    for step in range(1, training_steps+1):
        batch_x, batch_y = get_data(batch_size)
        if FLAGS.ext_time > 1:
            batch_x = np.repeat(batch_x, FLAGS.ext_time, axis=1)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps*FLAGS.ext_time, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % FLAGS.decay_lr_steps == 0:
            old_lr = sess.run(learning_rate)
            new_lr = sess.run(decay_learning_rate_op)
            print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr,new_lr))
        if step % display_step == 0 or step == 1:
            batch_x, batch_y = get_data(batch_size * 4, test=True)
            if FLAGS.ext_time > 1:
                batch_x = np.repeat(batch_x, FLAGS.ext_time, axis=1)
            batch_x = batch_x.reshape((batch_size * 4, timesteps*FLAGS.ext_time, num_input))
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Test Accuracy= " + \
                  "{:.3f}".format(acc))
        if acc > FLAGS.test_at > 0:
            print("Early stopping!")
            break

    print("Optimization Finished!")

    # Calculate accuracy for all mnist test images
    test_len = 256
    n_test_batches = (mnist.test.num_examples//test_len) + 1
    test_accuracy = []
    for i in range(n_test_batches):  # cover the whole test set
        test_data, test_label = get_data(batch_size, test=True, shuffle=False)
        test_data = test_data.reshape((-1, timesteps, num_input))
        if FLAGS.ext_time > 1:
            test_data = np.repeat(test_data, FLAGS.ext_time, axis=1)

        # test_label = mnist.test.labels[:test_len]
        test_accuracy.append(sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    print('''Statistics on the test set average accuracy {:.4g} +- {:.4g} (averaged over {} test batches of size {})'''
          .format(np.mean(test_accuracy), np.std(test_accuracy), n_test_batches, test_len))