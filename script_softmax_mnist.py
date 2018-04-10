import matplotlib

from guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik

import datetime
import os
import socket
from time import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from guillaume_toolbox.file_saver_dumper_no_h5py import save_file
from guillaume_toolbox.matplotlib_extension import strip_right_top_axis, raster_plot

from spiking_models import tf_cell_to_savable_dict, exp_convolve, ALIF
from guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper
import json
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.app.flags.FLAGS

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
##
tf.app.flags.DEFINE_integer('n_batch', 256, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('n_in', 1, 'number of input units to convert gray level input spikes.')
tf.app.flags.DEFINE_integer('n_regular', 140, 'number of regular spiking units in the recurrent layer.')
tf.app.flags.DEFINE_integer('n_adaptive', 100, 'number of adaptive spiking units in the recurrent layer')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target firing rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 100000, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 10, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 5, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('n_output_average_steps', 28 * 2, 'steps using for averaged the output logits')
tf.app.flags.DEFINE_integer('lr_decay_every', 2500, 'Decay learning rate every n steps')
tf.app.flags.DEFINE_integer('print_every', 100, '')
##
tf.app.flags.DEFINE_float('beta', 1.8, 'Scaling constant of the adaptive threshold')
# to solve safely set tau_a == expected recall delay
tf.app.flags.DEFINE_float('tau_a', 600, 'Adaptation time constant')
tf.app.flags.DEFINE_float('tau_v', 20, 'Membrane time constant of output readouts')
tf.app.flags.DEFINE_float('thr', 0.01, 'Baseline threshold voltage')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 1e-3, 'regularization coefficient to target a specific firing rate')
tf.app.flags.DEFINE_float('rewiring_temperature', 0., 'regularization coefficient')
tf.app.flags.DEFINE_float('proportion_excitatory', 0.75, 'proportion of excitatory neurons')
##
tf.app.flags.DEFINE_bool('interactive_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('verbose', True, 'Print many info during training')
tf.app.flags.DEFINE_bool('neuron_sign', True, 'If rewiring is active, this will fix the sign of input and recurrent neurons')

tf.app.flags.DEFINE_float('rewiring_connectivity', 0.2, 'possible usage of rewiring with ALIF and LIF (0.2 and 0.5 have been tested)')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring (irrelevant without rewiring)')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, 'Parameter necessary to approximate the spike derivative')

# Define the flag object as dictionnary for saving purposes
try: # TENSORFLOW 1.4
    print('MODEL', FLAGS.reg)  # should print at least one element to display the correct flag
    flag_dict = FLAGS.flag_values_dict()
except: # TENSORFLOW 1.6>=
    print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform lag to dict')
    flag_dict = FLAGS.__flags
print(json.dumps(flag_dict, indent=4))


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Fix the random seed if given as an argument
dt = 1. # Time step is by default 1 ms
n_output_symbols = 10

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(FLAGS.proportion_excitatory * FLAGS.n_in)
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    in_neuron_sign = np.random.shuffle(in_neuron_sign)

    n_excitatory = int(FLAGS.proportion_excitatory * (FLAGS.n_regular + FLAGS.n_adaptive))
    n_inhibitory = FLAGS.n_regular + FLAGS.n_adaptive - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not (FLAGS.neuron_sign == False): print(
        'WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Define the cell
beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=FLAGS.tau_v, n_delay=FLAGS.n_delay,
                n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=FLAGS.tau_a, beta=beta, thr=FLAGS.thr,
                rewiring_connectivity=FLAGS.rewiring_connectivity,
                in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                dampening_factor=FLAGS.dampening_factor)


# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(FLAGS.n_batch, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
targets = tf.placeholder(dtype=tf.int64, shape=(FLAGS.n_batch,),
                         name='Targets')  # Lists of target characters of the recall task

# Build a batch
def get_data_dict(batch_size, test=False):
    '''
    Generate the dictionary to be fed when running a tensorflow op.

    :param batch_size:
    :param test:
    :return:
    '''
    if test:
        input_px, target_oh = mnist.test.next_batch(batch_size)
    else:
        input_px, target_oh = mnist.train.next_batch(batch_size)
    target_num = np.argmax(target_oh, axis=1)

    # transform target one hot from batch x classes to batch x time x classes
    data_dict = {input_spikes: input_px[:,:,None], targets: target_num}
    return data_dict, input_px

outputs, final_state = tf.nn.dynamic_rnn(cell, input_spikes, dtype=tf.float32)
z,b = outputs
z_regular = z[:,:,:FLAGS.n_regular]
z_adaptive = z[:,:,FLAGS.n_regular:]

with tf.name_scope('ClassificationLoss'):
    psp_decay = np.exp(-dt / FLAGS.tau_v)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
    psp = exp_convolve(z, decay=psp_decay)
    n_neurons = z.get_shape()[2]

    # Define the readout weights
    if 0 < FLAGS.rewiring_connectivity:
        w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, n_output_symbols,
                                                         FLAGS.rewiring_connectivity,
                                                         neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])
    b_out = tf.get_variable(name='out_bias', shape=[n_output_symbols], initializer=tf.zeros_initializer())

    # Define the loss function
    out = einsum_bij_jk_to_bik(psp, w_out) + b_out
    Y_predict = out[:, -1, :]  # shape batch x classes, 32 x 10

    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=Y_predict))

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)

    # Define the accuracy
    Y_predict_num = tf.argmax(Y_predict, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(targets, Y_predict_num), dtype=tf.float32))

# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z, axis=(0, 1)) / dt
    regularization_f0 = FLAGS.reg_rate / 1000
    loss_regularization = tf.reduce_sum(tf.square(av - regularization_f0)) * FLAGS.reg

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay) # Op to decay learning rate

    loss = loss_regularization + loss_recall

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if 0 < FLAGS.rewiring_connectivity:

        train_step = rewiring_optimizer_wrapper(optimizer, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                                FLAGS.rewiring_connectivity,
                                                global_step=global_step,
                                                all_trained_var_list=tf.trainable_variables())
    else:
        train_step = optimizer.minimize(loss=loss, global_step=global_step)

# Real-time plotting
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Open an interactive matplotlib window to plot in real time
if FLAGS.interactive_plot:
    plt.ion()

fig, ax_list = plt.subplots(5, figsize=(6, 6))


def update_plot(plot_result_values, batch=0, n_max_neuron_per_raster=300):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # Plot the data, from top to bottom each axe represents: inputs, recurrent and controller
    ax_list[0].set_title("Target: " + str(plot_results_values['targets'][batch]))
    for k_data, data, d_name in zip(range(3),
                                    [plot_result_values['input_spikes'], plot_result_values['z_regular'],
                                     plot_result_values['z_adaptive']],
                                    ['Input', 'R', 'A']):

        ax = ax_list[k_data]
        ax.grid(color='black', alpha=0.15, linewidth=0.4)

        if np.size(data) > 0:
            data = data[batch]
            n_max = min(data.shape[1], n_max_neuron_per_raster)
            cell_select = np.linspace(start=0, stop=data.shape[1] - 1, num=n_max, dtype=int)
            data = data[:, cell_select]  # select a maximum of n_max_neuron_per_raster neurons to plot
            if k_data == 0:
                ax.imshow(data.T, aspect='auto', cmap='Greys')  # plot LSTM activity differently
                ax.set_yticklabels([])
            else:
                raster_plot(ax, data)
            ax.set_ylabel(d_name)
            ax.set_xticklabels([])

    # plot targets
    ax = ax_list[3]
    ax.set_yticks([0, 2, 4, 6, 8])
    classify_out = plot_result_values['out_plot'][batch]
    ax.imshow(classify_out.T, origin='lower', aspect='auto')
    ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('Output')
    ax.set_xticklabels([])

    # debug plot for psp-s or biases
    plot_param = 'b_con'  # or 'psp'
    ax.set_xticklabels([])
    ax = ax_list[-1]
    ax.grid(color='black', alpha=0.08, linewidth=0.3)
    ax.set_ylabel('PSPs' if plot_param == 'psp' else 'Threshold')
    sub_data = plot_result_values[plot_param][batch]
    if plot_param == 'b_con':
        sub_data = sub_data + FLAGS.thr

    presentation_steps = np.arange(int(sub_data.shape[0]))
    ax.plot(sub_data[:, :], color='r', label='Output', alpha=0.6, linewidth=1)
    ax.axis([0, presentation_steps[-1], np.min(sub_data[:, :]), np.max(sub_data[:, :])])

    ax.set_xlabel('Time in ms')
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.interactive_plot:
        plt.draw()
        plt.pause(1)


# Store some results across iterations
test_loss_list = []
test_loss_with_reg_list = []
test_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []

# Dictionaries of tensorflow ops to be evaluated simualtenously by a session
results_tensors = {'loss': loss,
                   'loss_reg': loss_regularization,
                   'loss_recall': loss_recall,
                   'accuracy': accuracy,
                   'final_state': final_state,
                   'av': av,
                   'learning_rate': learning_rate,

                   'w_in_val':cell.w_in_val,
                   'w_rec_val':cell.w_rec_val,
                   'w_out': w_out,
                   'b_out': b_out
                   }


plot_result_tensors = {'input_spikes': input_spikes,
                       'z': z,
                       'psp': psp,
                       'out_plot':out_plot,
                       'Y_predict': Y_predict,
                       'b_con': b,
                       'z_regular': z_regular,
                       'z_adaptive': z_adaptive,
                       'targets': targets}

t_train = 0
for k_iter in range(FLAGS.n_iter):

    # Decaying learning rate
    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0 and mnist.train._epochs_completed > 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Print some values to monitor convergence
    if np.mod(k_iter, FLAGS.print_every) == 0:

        val_dict, input_img = get_data_dict(FLAGS.n_batch, test=True)
        results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)

        # Storage of the results
        test_loss_with_reg_list.append(results_values['loss_reg'])
        test_loss_list.append(results_values['loss_recall'])
        test_error_list.append(results_values['accuracy'])
        training_time_list.append(t_train)

        print(
            '''Iteration {}, epoch {} validation accuracy {:.2g} +- {:.2g} (trial averaged)'''
                .format(k_iter, mnist.train._epochs_completed, np.mean(test_error_list[-FLAGS.print_every:]),
                        np.std(test_error_list[-FLAGS.print_every:])))

        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max

        firing_rate_stats = get_stats(results_values['av'] * 1000)

        # some connectivity statistics
        rewired_ref_list = ['w_in_val', 'w_rec_val', 'w_out']
        non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
        sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
        empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)

        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t average {:.0f} +- std {:.0f} (over neurons)
            connectivity {:.3g} \t Non zeros: W_in {}/{} W_rec {}/{} w_out {}/{}

            classification loss {:.2g} \t regularization loss {:.2g}
            learning rate {:.2g} \t training op. time {:.2g}
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                empirical_connectivity,
                non_zeros[0], sizes[0],
                non_zeros[1], sizes[1],
                non_zeros[2], sizes[2],
                results_values['loss_recall'], results_values['loss_reg'],
                results_values['learning_rate'],t_train,
            ))

        if FLAGS.interactive_plot:
            update_plot(plot_results_values)

    # train
    t0 = time()
    train_dict, input_img = get_data_dict(FLAGS.n_batch)
    final_state_value, _ = sess.run([final_state, train_step], feed_dict=train_dict)
    t_train = time() - t0

update_plot(plot_results_values)
plt.ioff() if FLAGS.interactive_plot else None
plt.show()

# Saving setup
# Get a meaning full fill name and so on
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
cell_name = type(cell).__name__
file_reference = '{}_{}_in{}_R{}_A{}_taua{}_comment{}'.format(
    time_stamp, cell_name, FLAGS.n_in, FLAGS.n_regular, FLAGS.n_adaptive, FLAGS.tau_a, FLAGS.comment)

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
start_time = datetime.datetime.now()

full_path = os.path.join(result_folder, file_reference)
if not os.path.exists(full_path):
    os.makedirs(full_path)


# Save a sample trajectory
if FLAGS.save_data:

    # Save files result
    try:
        flag_dict = FLAGS.flag_values_dict()
    except:
        print(
            'Deprecation WARNING: with next tensorflow versions (>= 1.5) we should use FLAGS.flag_values_dict() to transform flag to dict')
        flag_dict = FLAGS.__flags

    # Save files result
    results = {
        'error': test_error_list[-1],
        'loss': test_loss_list[-1],
        'loss_with_reg': test_loss_with_reg_list[-1],
        'loss_with_reg_list': test_loss_with_reg_list,
        'error_list': test_error_list,
        'loss_list': test_loss_list,
        'time_to_ref': time_to_ref_list,
        'training_time': training_time_list,
        'tau_delay_list': tau_delay_list,
        'flags': flag_dict,
    }

    save_file(flag_dict, full_path, 'flag', file_type='json')
    save_file(results, full_path, 'results', file_type='json')
    save_file(tf_cell_to_savable_dict(cell, sess), full_path, 'network_data', file_type='pickle')

del sess
