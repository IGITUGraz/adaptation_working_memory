import datetime
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import numpy.random as rd
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.variables import Variable

from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler
from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bi_ijk_to_bjk
from lsnn.guillaume_toolbox.tensorflow_utils import tf_roll

Cell = tf.contrib.rnn.BasicRNNCell


def placeholder_container_for_rnn_state(cell_state_size, dtype, batch_size, name='TupleStateHolder'):
    with tf.name_scope(name):
        default_dict = cell_state_size._asdict()
        placeholder_dict = OrderedDict({})
        for k, v in default_dict.items():
            if np.shape(v) == ():
                v = [v]
            shape = np.concatenate([[batch_size], v])
            placeholder_dict[k] = tf.placeholder(shape=shape, dtype=dtype, name=k)

        placeholder_tuple = cell_state_size.__class__(**placeholder_dict)
        return placeholder_tuple


def feed_dict_with_placeholder_container(dict_to_update, state_holder, state_value, batch_selection=None):
    if state_value is None:
        return dict_to_update

    assert state_holder.__class__ == state_value.__class__, 'Should have the same class, got {} and {}'.format(
        state_holder.__class__, state_value.__class__)

    for k, v in state_value._asdict().items():
        if batch_selection is None:
            dict_to_update.update({state_holder._asdict()[k]: v})
        else:
            dict_to_update.update({state_holder._asdict()[k]: v[batch_selection]})

    return dict_to_update


#################################
# Rewirite the Spike function without hack
#################################

@function.Defun()
def SpikeFunctionGrad(v_scaled, dampening_factor, grad):
    dE_dz = grad
    dz_dv_scaled = tf.maximum(1 - tf.abs(v_scaled), 0)
    dz_dv_scaled *= dampening_factor

    dE_dv_scaled = dE_dz * dz_dv_scaled

    return [dE_dv_scaled,
            tf.zeros_like(dampening_factor)]


@function.Defun(grad_func=SpikeFunctionGrad)
def SpikeFunction(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)
    return tf.identity(z_, name="SpikeFunction")

def weight_matrix_with_delay_dimension(w, d, n_delay):
    """
    Generate the tensor of shape n_in x n_out x n_delay that represents the synaptic weights with the right delays.

    :param w: synaptic weight value, float tensor of shape (n_in x n_out)
    :param d: delay number, int tensor of shape (n_in x n_out)
    :param n_delay: number of possible delays
    :return:
    """
    with tf.name_scope('WeightDelayer'):
        w_d_list = []
        for kd in range(n_delay):
            mask = d == kd
            w_d = tf.where(condition=mask, x=w, y=tf.zeros_like(w))
            w_d_list.append(w_d)

        delay_axis = len(d.shape)
        WD = tf.stack(w_d_list, axis=delay_axis)

    return WD


# PSP on output layer
def exp_convolve(tensor, decay):  # tensor shape (trial, time, neuron)
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16, tf.float32, tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1 - decay) * x, tensor_time_major, initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor, perm=[1, 0, 2])
    return filtered_tensor


LIFStateTuple = namedtuple('LIFStateTuple', ('v', 'z', 'i_future_buffer', 'z_buffer'))


def tf_cell_to_savable_dict(cell, sess, supplement={}):
    """
    Usefull function to return a python/numpy object from of of the tensorflow cell object defined here.
    The idea is simply that varaibles and Tensors given as attributes of the object with be replaced by there numpy value evaluated on the current tensorflow session.

    :param cell: tensorflow cell object
    :param sess: tensorflow session
    :param supplement: some possible
    :return:
    """

    dict_to_save = {}
    dict_to_save['cell_type'] = str(cell.__class__)
    time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    dict_to_save['time_stamp'] = time_stamp

    dict_to_save.update(supplement)

    for k, v in cell.__dict__.items():
        if k == 'self':
            pass
        elif type(v) in [Variable, Tensor]:
            dict_to_save[k] = sess.run(v)
        elif type(v) in [bool, int, float, np.ndarray]:
            dict_to_save[k] = v
        else:
            print('WARNING: attribute {} is of type {}, not recording it.'.format(k, v))
            dict_to_save[k] = str(type(v))

    return dict_to_save

class LIF(Cell):
    def __init__(self, n_in, n_rec, tau=20., thr=0.03,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1, rewiring_connectivity=-1,
                 in_neuron_sign=None, rec_neuron_sign=None,
                 dampening_factor=0.3,
                 injected_noise_current=0.):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.

        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        :param reset: method of resetting membrane potential after spike thr-> by fixed threshold amount, zero-> to zero
        """

        if np.isscalar(tau): tau = tf.ones(n_rec, dtype=dtype) * np.mean(tau)
        if np.isscalar(thr): thr = tf.ones(n_rec, dtype=dtype) * np.mean(thr)

        self.dampening_factor = dampening_factor

        # Parameters
        self.n_delay = n_delay
        self.n_refractory = n_refractory

        self.dt = dt
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = tf.exp(-dt / tau)
        self.thr = thr

        self.injected_noise_current = injected_noise_current

        with tf.name_scope('WeightDefinition'):

            # Input weights
            if 0 < rewiring_connectivity < 1:
                self.w_in_val, self.w_in_sign, self.w_in_var, _ = weight_sampler(n_in, n_rec, rewiring_connectivity, neuron_sign=in_neuron_sign)
            else:
                self.w_in_var = tf.Variable(rd.randn(n_in, n_rec) / np.sqrt(n_in), dtype=dtype, name="InputWeight")
                self.w_in_val = self.w_in_var

            self.w_in_delay = rd.randint(self.n_delay, size=n_in * n_rec).reshape(n_in, n_rec)
            self.W_in = weight_matrix_with_delay_dimension(self.w_in_val, self.w_in_delay, self.n_delay)

            if 0 < rewiring_connectivity < 1:
                self.w_rec_val, self.w_rec_sign, self.w_rec_var, _ = weight_sampler(n_rec, n_rec, rewiring_connectivity, neuron_sign=rec_neuron_sign)
            else:
                if not(rec_neuron_sign is None and in_neuron_sign is None):
                    raise NotImplementedError('Neuron sign requested but this is only implemented with rewiring')
                self.w_rec_var = tf.Variable(rd.randn(n_rec, n_rec) / np.sqrt(n_rec), dtype=dtype,
                                             name='RecurrentWeight')
                self.w_rec_val = self.w_rec_var

            recurrent_disconnect_mask = np.diag(np.ones(n_rec, dtype=bool))

            self.w_rec_val = tf.where(recurrent_disconnect_mask, tf.zeros_like(self.w_rec_val),
                                      self.w_rec_val)  # Disconnect autotapse
            self.w_rec_delay = rd.randint(self.n_delay, size=n_rec * n_rec).reshape(n_rec, n_rec)
            self.W_rec = weight_matrix_with_delay_dimension(self.w_rec_val, self.w_rec_delay, self.n_delay)

    @property
    def state_size(self):
        return LIFStateTuple(v=self.n_rec,
                             z=self.n_rec,
                             i_future_buffer=(self.n_rec, self.n_delay),
                             z_buffer=(self.n_rec, self.n_refractory))

    @property
    def output_size(self):
        return self.n_rec

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return LIFStateTuple(
            v=v0,
            z=z0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
            state.z, self.W_rec)

        add_current = 0.
        if self.injected_noise_current > 0:
            add_current = tf.random_normal(shape=state.z.shape, stddev=self.injected_noise_current)

        new_v, new_z = self.LIF_dynamic(
            v=state.v,
            z=state.z,
            z_buffer=state.z_buffer,
            i_future_buffer=i_future_buffer,
            add_current=add_current)

        new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
        new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

        new_state = LIFStateTuple(v=new_v,
                                  z=new_z,
                                  i_future_buffer=new_i_future_buffer,
                                  z_buffer=new_z_buffer)
        return new_z, new_state

    def LIF_dynamic(self, v, z, z_buffer, i_future_buffer, thr=None, decay=None, n_refractory=None, add_current=0.):
        """
        Function that generate the next spike and voltage tensor for given cell state.
.expand_dims(thr,axis=1)
            v_buffer_scaled = (v_buffer - thr_expanded) / thr_expanded
            new_z = different
        :param v
        :param z
        :param z_buffer:
        :param i_future_buffer:
        :param thr:
        :param decay:
        :param n_refractory:
        :param add_current:
        :return:
        """

        with tf.name_scope('LIFdynamic'):
            if thr is None: thr = self.thr
            if decay is None: decay = self._decay
            if n_refractory is None: n_refractory = self.n_refractory

            i_t = i_future_buffer[:, :, 0] + add_current

            I_reset = z * thr

            new_v = decay * v + (1 - decay) * i_t - I_reset

            # Spike generation
            v_scaled = (v - thr) / thr

            # new_z = differentiable_spikes(v_scaled=v_scaled)
            new_z = SpikeFunction(v_scaled, self.dampening_factor)
            new_z.set_shape([None, v.get_shape()[1]])

            if n_refractory > 0:
                is_ref = tf.greater(tf.reduce_max(z_buffer[:, :, -n_refractory:], axis=2), .5)
                new_z = tf.where(is_ref, tf.zeros_like(new_z), new_z)

            return new_v, new_z

ALIFStateTuple = namedtuple('ALIFState', (
    'z',
    'v',
    'b',

    'i_future_buffer',
    'z_buffer'))


class ALIF(LIF):
    def __init__(self, n_in, n_rec, tau=20, thr=0.01,
                 dt=1., n_refractory=0, dtype=tf.float32, n_delay=1,
                 tau_adaptation=200., beta=1.6,
                 rewiring_connectivity=-1, dampening_factor=0.3,
                 in_neuron_sign=None, rec_neuron_sign=None, injected_noise_current=0.):
        """
        Tensorflow cell object that simulates a LIF neuron with an approximation of the spike derivatives.

        :param n_in: number of input neurons
        :param n_rec: number of recurrent neurons
        :param tau: membrane time constant
        :param thr: threshold voltage
        :param dt: time step of the simulation
        :param n_refractory: number of refractory time steps
        :param dtype: data type of the cell tensors
        :param n_delay: number of synaptic delay, the delay range goes from 1 to n_delay time steps
        """

        super(ALIF, self).__init__(n_in=n_in, n_rec=n_rec, tau=tau, thr=thr, dt=dt, n_refractory=n_refractory,
                                   dtype=dtype, n_delay=n_delay,
                                   rewiring_connectivity=rewiring_connectivity,
                                   dampening_factor=dampening_factor, in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                                   injected_noise_current=injected_noise_current)

        if tau_adaptation is None: raise ValueError("alpha parameter for adaptive bias must be set")
        if beta is None: raise ValueError("beta parameter for adaptive bias must be set")

        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = np.exp(-dt / tau_adaptation)

    @property
    def output_size(self):
        return [self.n_rec,self.n_rec]

    @property
    def state_size(self):
        return ALIFStateTuple(v=self.n_rec,
                              z=self.n_rec,
                              b=self.n_rec,
                              i_future_buffer=(self.n_rec, self.n_delay),
                              z_buffer=(self.n_rec, self.n_refractory))

    def zero_state(self, batch_size, dtype, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        v0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)
        b0 = tf.zeros(shape=(batch_size, n_rec), dtype=dtype)

        i_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_delay), dtype=dtype)
        z_buff0 = tf.zeros(shape=(batch_size, n_rec, self.n_refractory), dtype=dtype)

        return ALIFStateTuple(
            v=v0,
            z=z0,
            b=b0,
            i_future_buffer=i_buff0,
            z_buffer=z_buff0
        )

    def __call__(self, inputs, state, scope=None, dtype=tf.float32):

        i_future_buffer = state.i_future_buffer + einsum_bi_ijk_to_bjk(inputs, self.W_in) + einsum_bi_ijk_to_bjk(
            state.z, self.W_rec)

        new_b = self.decay_b * state.b + (np.ones(self.n_rec) - self.decay_b) * state.z
        thr = self.thr + new_b * self.beta

        new_v, new_z = self.LIF_dynamic(
            v=state.v,
            z=state.z,
            z_buffer=state.z_buffer,
            i_future_buffer=i_future_buffer,
            decay=self._decay,
            thr=thr)

        new_z_buffer = tf_roll(state.z_buffer, new_z, axis=2)
        new_i_future_buffer = tf_roll(i_future_buffer, axis=2)

        new_state = ALIFStateTuple(v=new_v,
                                   z=new_z,
                                   b=new_b,
                                   i_future_buffer=new_i_future_buffer,
                                   z_buffer=new_z_buffer)
        return [new_z,new_b], new_state