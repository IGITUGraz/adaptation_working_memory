from tutorial_storerecall_utils import generate_poisson_noise_np
# from bin.tutorial_temporalXOR_utils import generate_xor_input
from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik
# import matplotlib
# matplotlib.use('Agg')

import datetime
import os
import json
import socket
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from lsnn.guillaume_toolbox.file_saver_dumper_no_h5py import save_file
from bin.tutorial_singen_utils import update_plot

from lsnn.spiking_models import tf_cell_to_savable_dict, placeholder_container_for_rnn_state,\
    feed_dict_with_placeholder_container, exp_convolve, ALIF
from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
##
tf.app.flags.DEFINE_integer('batch_train', 128, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('batch_val', 128, 'batch size of the validation set')
tf.app.flags.DEFINE_integer('batch_test', 128, 'batch size of the testing set')
tf.app.flags.DEFINE_integer('n_charac', 1, 'number of characters in the recall task')
tf.app.flags.DEFINE_integer('n_out', 1, 'number of output neurons (number of target curves)')
tf.app.flags.DEFINE_integer('n_in', 100, 'number of input units.')
tf.app.flags.DEFINE_integer('n_regular', 100, 'number of recurrent units.')
tf.app.flags.DEFINE_integer('n_adaptive', 100, 'number of controller units')
tf.app.flags.DEFINE_integer('f0', 50, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('reg_max_rate', 100, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 2000, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 10, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 3, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('seq_len', 1000, 'Number of character steps')
tf.app.flags.DEFINE_integer('seq_delay', 100, 'Expected delay in character steps. Must be <= seq_len - 2')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('lr_decay_every', 100, 'Decay every')
tf.app.flags.DEFINE_integer('print_every', 20, 'Decay every')
##
tf.app.flags.DEFINE_float('stop_crit', 0.005, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_float('beta', 1.7, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 200, 'Mikolov model alpha - threshold decay')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in LSNN and output neurons')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.8, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 1., 'regularization coefficient')
tf.app.flags.DEFINE_float('rewiring_connectivity', -1, 'possible usage of rewiring with ALIF and LIF (0.1 is default)')
tf.app.flags.DEFINE_float('readout_rewiring_connectivity', -1, '')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring')
tf.app.flags.DEFINE_float('rewiring_temperature', 0, '')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, '')
tf.app.flags.DEFINE_float('stochastic_factor', -1, '')
tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', .01, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('add_current', .0, 'constant current to feed to all neurons')
##
tf.app.flags.DEFINE_bool('tau_a_spread', False, 'Mikolov model spread of alpha - threshold decay')
tf.app.flags.DEFINE_bool('save_data', True, 'Save the data (training, test, network, trajectory for plotting)')
tf.app.flags.DEFINE_bool('do_plot', True, 'Perform plots')
tf.app.flags.DEFINE_bool('monitor_plot', False, 'Perform plots during training')
tf.app.flags.DEFINE_bool('interactive_plot', False, 'Perform plots')
tf.app.flags.DEFINE_bool('device_placement', False, '')
tf.app.flags.DEFINE_bool('verbose', True, '')
tf.app.flags.DEFINE_bool('neuron_sign', True, '')
tf.app.flags.DEFINE_bool('adaptive_reg', False, '')
tf.app.flags.DEFINE_bool('sum_sines', True, '')

# Run asserts to check seq_delay and seq_len relation is ok
assert FLAGS.seq_len > FLAGS.seq_delay * 2 + 200

# Fix the random seed if given as an argument
if FLAGS.seed >= 0:
    seed = FLAGS.seed
else:
    seed = rd.randint(10 ** 6)
rd.seed(seed)
tf.set_random_seed(seed)

# Experiment parameters
dt = 1.
repeat_batch_test = 10
print_every = FLAGS.print_every

# Frequencies
input_f0 = FLAGS.f0 / 1000  # in kHz in coherence with the usgae of ms for time
regularization_f0 = FLAGS.reg_rate / 1000
regularization_f0_max = FLAGS.reg_max_rate / 1000

# Network parameters
tau_v = FLAGS.tau_out
thr = FLAGS.thr

decay = np.exp(-dt / FLAGS.tau_out)  # output layer psp decay, chose value between 15 and 30ms as for tau_v
# Symbol number
n_charac = FLAGS.n_charac  # Number of digit symbols
n_input_symbols = n_charac + 2  # Total number of symbols including recall and store
n_output_symbols = n_charac  # Number of output symbols
recall_symbol = n_input_symbols - 1  # ID of the recall symbol
store_symbol = n_input_symbols - 2  # ID of the store symbol

# Neuron population sizes
input_neuron_split = np.array_split(np.arange(FLAGS.n_in), n_input_symbols)

# Sign of the neurons
if 0 < FLAGS.rewiring_connectivity and FLAGS.neuron_sign:
    n_excitatory_in = int(0.75 * FLAGS.n_in)
    n_inhibitory_in = FLAGS.n_in - n_excitatory_in
    in_neuron_sign = np.concatenate([-np.ones(n_inhibitory_in), np.ones(n_excitatory_in)])
    np.random.shuffle(in_neuron_sign)

    n_excitatory = int(0.75 * (FLAGS.n_regular + FLAGS.n_adaptive))
    n_inhibitory = FLAGS.n_regular + FLAGS.n_adaptive - n_excitatory
    rec_neuron_sign = np.concatenate([-np.ones(n_inhibitory), np.ones(n_excitatory)])
else:
    if not FLAGS.neuron_sign: print('WARNING: Neuron sign is set to None without rewiring but sign is requested')
    in_neuron_sign = None
    rec_neuron_sign = None

# Generate the cell
beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])
if FLAGS.tau_a_spread:
    tau_a_spread = np.random.uniform(size=FLAGS.n_regular+FLAGS.n_adaptive) * FLAGS.tau_a
else:
    tau_a_spread = FLAGS.tau_a
cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v, n_delay=FLAGS.n_delay,
            n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_spread, beta=beta, thr=thr,
            rewiring_connectivity=FLAGS.rewiring_connectivity,
            in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
            dampening_factor=FLAGS.dampening_factor, add_current=FLAGS.add_current
            )

cell_name = type(cell).__name__
print('\n -------------- \n' + cell_name + '\n -------------- \n')
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_reference = '{}_{}_seqdelay{}_in{}_R{}_A{}_lr{}_comment{}'.format(
    time_stamp, cell_name, FLAGS.seq_delay, FLAGS.n_in, FLAGS.n_regular, FLAGS.n_adaptive,
    FLAGS.learning_rate, FLAGS.comment)
file_reference = file_reference + '_taua' + str(FLAGS.tau_a) + '_beta' + str(FLAGS.beta)
print('FILE REFERENCE: ' + file_reference)

# Save parameters and training log
try:
    flag_dict = FLAGS.flag_values_dict()
except:
    print('Deprecation WARNING: with tensorflow >= 1.5 we should use FLAGS.flag_values_dict() to transform to dict')
    flag_dict = FLAGS.__flags
print(json.dumps(flag_dict, indent=4))

# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
target_nums = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_out),
                             name='TargetNums')  # Lists of target characters of the recall task

batch_size_holder = tf.placeholder(dtype=tf.int32, name='BatchSize')  # Int that contains the batch size
init_state_holder = placeholder_container_for_rnn_state(cell.state_size, dtype=tf.float32, batch_size=None)
psp_out_state_holder = tf.placeholder(dtype=tf.float32, shape=(None, FLAGS.n_regular + FLAGS.n_adaptive), name='PspOutState')


frozen_noise_rates = np.zeros((FLAGS.seq_len, FLAGS.n_in))
frozen_noise_rates[:, :] = FLAGS.f0 / 1000
frozen_noise_spikes = generate_poisson_noise_np(frozen_noise_rates)

frozen_noise_spikes1 = np.zeros_like(frozen_noise_spikes)  # no input
frozen_noise_spikes2 = np.zeros_like(frozen_noise_spikes)  # no input

clk_n_in = 80
n_of_steps = 4
input_spike_every = 6
assert FLAGS.seq_len % n_of_steps == 0
step_len = int(FLAGS.seq_len / n_of_steps)
step_group = int(clk_n_in / n_of_steps)
for i in range(n_of_steps):  # clock signal like
    frozen_noise_spikes1[i*step_len:(i+1)*step_len:input_spike_every, i*step_group:(i+1)*step_group] = 1.
    frozen_noise_spikes2[i*step_len:(i+1)*step_len:input_spike_every, i*step_group:(i+1)*step_group] = 1.
frozen_noise_spikes1[::input_spike_every, 80:90] = 1.
frozen_noise_spikes2[::input_spike_every, 90:100] = 1.


def sum_of_sines_target(n_sines=4, periods=[1000, 500, 333, 200], weights=[1., 1., 1., 1.], phases=[0., 0., 0., 0.]):
    if periods is None:
        periods = [np.random.uniform(low=100, high=1000) for i in range(n_sines)]
    assert n_sines == len(periods)
    sines = []
    weights = np.random.uniform(low=0.5, high=2, size=n_sines) if weights is None else weights
    phases = np.random.uniform(low=0., high=np.pi*2, size=n_sines) if phases is None else phases
    for i in range(n_sines):
        sine = np.sin(np.linspace(0+phases[i], np.pi*2*(FLAGS.seq_len//periods[i])+phases[i], FLAGS.seq_len))
        sines.append(sine*weights[i])
    return sum(sines)

ts = [np.expand_dims(sum_of_sines_target(n_sines=3, periods=[1000, 333, 200], weights=None, phases=[0., 0., 0.]),
                     axis=0) for i in range(FLAGS.n_out)]
ts2 = [np.expand_dims(sum_of_sines_target(n_sines=3, periods=[1000, 500, 250], weights=None, phases=[0., 0., 0.]),
                      axis=0) for i in range(FLAGS.n_out)]
target_sine = np.stack(ts, axis=2)
target_sine2 = np.stack(ts2, axis=2)


def get_data_dict(batch_size):
    spikes = np.zeros((batch_size, FLAGS.seq_len, FLAGS.n_in))
    target_data = np.zeros((batch_size, FLAGS.seq_len, 1))
    # frozen_noise_spikes = generate_poisson_noise_np(frozen_noise_rates)
    for b in range(batch_size):
        if rd.random() < 0.5:
            spikes[b] = frozen_noise_spikes1
            target_data[b] = target_sine
        else:
            spikes[b] = frozen_noise_spikes2
            target_data[b] = target_sine2

    psp_out_state = np.zeros(shape=(batch_size, FLAGS.n_regular + FLAGS.n_adaptive))

    data_dict = {input_spikes: spikes, target_nums: target_data, batch_size_holder: batch_size,
                 psp_out_state_holder: psp_out_state}

    return data_dict

# Define the name of spike train for the different models
z_stack, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=init_state_holder, dtype=tf.float32)
z, b_con = z_stack
z_con = []
z_all = z


with tf.name_scope('RecallLoss'):
    target_nums_at_recall = target_nums
    Y = target_nums

    # MTP models do not use controller (modulator) population for output
    out_neurons = z_all
    n_neurons = out_neurons.get_shape()[2]
    psp = exp_convolve(out_neurons, decay=decay, init=psp_out_state_holder)

    if 0 < FLAGS.rewiring_connectivity and 0 < FLAGS.readout_rewiring_connectivity:
        w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, FLAGS.n_out,
                                                         FLAGS.readout_rewiring_connectivity,
                                                         neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, FLAGS.n_out])

    out = einsum_bij_jk_to_bik(psp, w_out)
    # Y_predict = tf.squeeze(out)
    Y_predict = out

    loss_recall = tf.reduce_mean((target_nums - Y_predict) ** 2)

    recall_errors = tf.reduce_mean(tf.abs(target_nums - Y_predict), axis=(0, 1))
    recall_accuracy = 1. - tf.abs(recall_errors)

    _, tmp_diff_var = tf.nn.moments(target_nums - Y_predict, axes=[0, 1])
    _, tmp_target_var = tf.nn.moments(target_nums, axes=[0, 1])
    depasq_err = tmp_diff_var / tmp_target_var
    depasq_err = tf.reduce_mean(depasq_err)


# Target regularization
with tf.name_scope('RegularizationLoss'):
    # Firing rate regularization
    av = tf.reduce_mean(z_all, axis=(0, 1)) / dt
    adaptive_regularization_coeff = tf.Variable(np.ones(n_neurons) * FLAGS.reg, dtype=tf.float32, trainable=False)

    loss_reg = tf.reduce_sum(tf.square(av - regularization_f0) * adaptive_regularization_coeff)

    do_increase_reg = tf.greater(av,regularization_f0_max)
    do_increase_reg = tf.cast(do_increase_reg,dtype=tf.float32)

    new_adaptive_coeff = do_increase_reg * adaptive_regularization_coeff * 1.3 \
                         + (1-do_increase_reg) * adaptive_regularization_coeff * 0.93

    if FLAGS.adaptive_reg:
        update_regularization_coeff = tf.assign(adaptive_regularization_coeff,new_adaptive_coeff)
    else:
        update_regularization_coeff = tf.no_op('SkipAdaptiveRegularization')

# Aggregate the losses
with tf.name_scope('OptimizationScheme'):
    # scaling loss_recall to match order of magnitude of loss from script_recall.py
    # this is needed to keep the same regularization coefficients (reg, regl2) across scripts
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, dtype=tf.float32, trainable=False)
    decay_learning_rate_op = tf.assign(learning_rate, learning_rate * FLAGS.lr_decay)

    loss = loss_reg + loss_recall

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if 0 < FLAGS.rewiring_connectivity:

        rewiring_connectivity_list = [FLAGS.rewiring_connectivity, FLAGS.rewiring_connectivity,
                                      FLAGS.readout_rewiring_connectivity]

        train_step = rewiring_optimizer_wrapper(opt, loss, learning_rate, FLAGS.l1, FLAGS.rewiring_temperature,
                                                rewiring_connectivity_list,
                                                global_step=global_step,
                                                var_list=tf.trainable_variables())
    else:
        train_step = opt.minimize(loss=loss, global_step=global_step)

# Real-time plotting
sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.device_placement))
sess.run(tf.global_variables_initializer())

last_final_state_state_training_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32))]
last_final_state_state_validation_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_val, dtype=tf.float32))]
last_final_state_state_testing_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_test, dtype=tf.float32))]
last_psp_state = np.zeros(shape=(FLAGS.batch_val, FLAGS.n_regular + FLAGS.n_adaptive))

# Open an interactive matplotlib window to plot in real time
if FLAGS.do_plot and FLAGS.interactive_plot:
    plt.ion()
if FLAGS.do_plot:
    fig, ax_list = plt.subplots(5, figsize=(6, 7.5))

    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.set_window_title(socket.gethostname() + ' - ' + FLAGS.comment)

test_loss_list = []
test_loss_with_reg_list = []
validation_error_list = []
tau_delay_list = []
training_time_list = []
time_to_ref_list = []
firing_rates = []
depasq_errs = []
results_tensors = {
    'loss': loss,
    'loss_reg': loss_reg,
    'loss_recall': loss_recall,
    'recall_errors': recall_errors,
    'final_state': final_state,
    'av': av,
    'adaptive_regularization_coeff': adaptive_regularization_coeff,
    'depasq_err': depasq_err,
    'w_in_val': cell.w_in_val,
    'w_rec_val': cell.w_rec_val,
    'w_out': w_out
}

w_in_last = sess.run(cell.w_in_val)
w_rec_last = sess.run(cell.w_rec_val)
w_out_last = sess.run(w_out)

plot_result_tensors = {
    'input_spikes': input_spikes,
    'z': z,
    'z_con': z_con,
    'target_nums': target_nums,
    'psp': psp,
    'out_plot': out,
    'Y': Y,
    'Y_predict': Y_predict,
    'b_con': b_con,
}

t_train = 0
t_ref = time()
for k_iter in range(FLAGS.n_iter):

    if k_iter > 0 and np.mod(k_iter, FLAGS.lr_decay_every) == 0:
        old_lr = sess.run(learning_rate)
        new_lr = sess.run(decay_learning_rate_op)
        print('Decaying learning rate: {:.2g} -> {:.2g}'.format(old_lr, new_lr))

    # Monitor the training with a validation set
    t0 = time()
    val_dict = get_data_dict(FLAGS.batch_val)
    feed_dict_with_placeholder_container(val_dict, init_state_holder, last_final_state_state_validation_pointer[0])
    val_dict[psp_out_state_holder] = last_psp_state

    results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=val_dict)
    last_final_state_state_validation_pointer[0] = results_values['final_state']
    last_final_state_state_testing_pointer[0] = results_values['final_state']
    last_psp_state = plot_results_values['psp'][:, -1, :]
    t_run = time() - t0

    # Storage of the results
    test_loss_with_reg_list.append(results_values['loss_reg'])
    test_loss_list.append(results_values['loss_recall'])
    validation_error_list.append(results_values['recall_errors'])
    depasq_errs.append(results_values['depasq_err'])
    training_time_list.append(t_train)
    time_to_ref_list.append(time() - t_ref)

    if np.mod(k_iter, print_every) == 0:

        print('''Iteration {}, statistics on the validation set average error {:.2g} +- {:.2g} (trial averaged) DePasq Error {:.2g}'''
              .format(k_iter, np.mean(validation_error_list[-print_every:]),
                      np.std(validation_error_list[-print_every:]),
                      results_values['depasq_err']))

        def get_stats(v):
            if np.size(v) == 0:
                return np.nan, np.nan, np.nan, np.nan
            min_val = np.min(v)
            max_val = np.max(v)

            k_min = np.sum(v == min_val)
            k_max = np.sum(v == max_val)

            return np.min(v), np.max(v), np.mean(v), np.std(v), k_min, k_max

        firing_rate_stats = get_stats(results_values['av'] * 1000)
        reg_coeff_stats = get_stats(results_values['adaptive_regularization_coeff'])
        firing_rates.append(firing_rate_stats[2])
        if FLAGS.verbose:
            print('''
            firing rate (Hz)  min {:.0f} ({}) \t max {:.0f} ({}) \t
            average {:.0f} +- std {:.0f} (averaged over batches and time)
            reg. coeff        min {:.2g} \t max {:.2g} \t average {:.2g} +- std {:.2g}

            comput. time (s)  training {:.2g} \t validation {:.2g}
            loss              classif. {:.2g} \t reg. loss  {:.2g}
            '''.format(
                firing_rate_stats[0], firing_rate_stats[4], firing_rate_stats[1], firing_rate_stats[5],
                firing_rate_stats[2], firing_rate_stats[3],
                reg_coeff_stats[0], reg_coeff_stats[1], reg_coeff_stats[2], reg_coeff_stats[3],
                t_train, t_run,
                results_values['loss_recall'], results_values['loss_reg']
            ))

        if 0 < FLAGS.rewiring_connectivity:
            rewired_ref_list = ['w_in_val','w_rec_val','w_out']
            non_zeros = [np.sum(results_values[ref] != 0) for ref in rewired_ref_list]
            sizes = [np.size(results_values[ref]) for ref in rewired_ref_list]
            empirical_connectivity = np.sum(non_zeros) / np.sum(sizes)

            if 0 < FLAGS.rewiring_connectivity:
                assert empirical_connectivity < FLAGS.rewiring_connectivity * 1.1,\
                    'Rewiring error: found connectivity {:.3g}'.format(empirical_connectivity)

            w_in_new = results_values['w_in_val']
            w_rec_new = results_values['w_rec_val']
            w_out_new = results_values['w_out']

            stay_con_in = np.logical_and(w_in_new != 0, w_in_last != 0)
            stay_con_rec = np.logical_and(w_rec_new != 0, w_rec_last != 0)
            stay_con_out = np.logical_and(w_out_new != 0, w_out_last != 0)

            Dw_in = np.linalg.norm(w_in_new[stay_con_in] - w_in_last[stay_con_in])
            Dw_rec = np.linalg.norm(w_rec_new[stay_con_rec] - w_rec_last[stay_con_rec])
            Dw_out = np.linalg.norm(w_out_new[stay_con_out] - w_out_last[stay_con_out])

            if FLAGS.verbose:
                print('''Connectivity {:.3g} \t Non zeros: W_in {}/{} W_rec {}/{} w_out {}/{} \t
                New zeros: W_in {} W_rec {} W_out {}'''.format(
                    empirical_connectivity,
                    non_zeros[0], sizes[0],
                    non_zeros[1], sizes[1],
                    non_zeros[2], sizes[2],
                    np.sum(np.logical_and(w_in_new == 0, w_in_last != 0)),
                    np.sum(np.logical_and(w_rec_new == 0, w_rec_last != 0)),
                    np.sum(np.logical_and(w_out_new == 0, w_out_last != 0))
                ))

                print('Delta W norms: {:.3g} \t {:.3g} \t {:.3g}'.format(Dw_in,Dw_rec,Dw_out))

            w_in_last = results_values['w_in_val']
            w_rec_last = results_values['w_rec_val']
            w_out_last = results_values['w_out']

        if FLAGS.do_plot and FLAGS.monitor_plot:
            update_plot(plt, ax_list, FLAGS, plot_results_values)
            tmp_path = os.path.join(result_folder,
                                    'tmp/figure' + start_time.strftime("%H%M") + '_' +
                                    str(k_iter) + '.pdf')
            if not os.path.exists(os.path.join(result_folder, 'tmp')):
                os.makedirs(os.path.join(result_folder, 'tmp'))
            fig.savefig(tmp_path, format='pdf')

        if np.mean(depasq_errs[-print_every:]) < FLAGS.stop_crit:
            print('LESS THAN ' + str(FLAGS.stop_crit) + ' ERROR ACHIEVED - STOPPING - SOLVED at epoch ' + str(k_iter))
            break

    # train
    train_dict = get_data_dict(FLAGS.batch_train)
    feed_dict_with_placeholder_container(train_dict, init_state_holder, last_final_state_state_training_pointer[0])
    t0 = time()
    final_state_value, _, _ = sess.run([final_state, train_step, update_regularization_coeff], feed_dict=train_dict)
    last_final_state_state_training_pointer[0] = final_state_value
    t_train = time() - t0

print('FINISHED IN {:.2g} s'.format(time() - t_ref))

# Save everything
if FLAGS.save_data:

    # Saving setup
    full_path = os.path.join(result_folder, file_reference)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    # Save the tensorflow graph
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(full_path, 'session'))
    saver.export_meta_graph(os.path.join(full_path, 'graph.meta'))

    results = {
        'error': validation_error_list[-1],
        'loss': test_loss_list[-1],
        'loss_with_reg': test_loss_with_reg_list[-1],
        'loss_with_reg_list': test_loss_with_reg_list,
        'error_list': validation_error_list,
        'loss_list': test_loss_list,
        'time_to_ref': time_to_ref_list,
        'training_time': training_time_list,
        'tau_delay_list': tau_delay_list,
        'flags': flag_dict,
        'firing_rate': firing_rates,
        'depasq_errs': depasq_errs,
    }

    save_file(flag_dict, full_path, 'flags', file_type='json')
    save_file(results, full_path, 'training_results', file_type='json')

    # Save sample trajectory (input, output, etc. for plotting)
    test_errors = []
    test_depasq_err = []
    continuous_plot = []
    for i in range(16):
        test_dict = get_data_dict(FLAGS.batch_test)
        feed_dict_with_placeholder_container(test_dict, init_state_holder, last_final_state_state_testing_pointer[0])
        results_values, plot_results_values = sess.run([results_tensors, plot_result_tensors], feed_dict=test_dict)
        continuous_plot.append(plot_results_values)
        last_final_state_state_testing_pointer[0] = results_values['final_state']
        test_errors.append(results_values['recall_errors'])
        test_depasq_err.append(results_values['depasq_err'])
        if FLAGS.do_plot and FLAGS.monitor_plot:
            update_plot(plt, ax_list, FLAGS, plot_results_values)
            tmp_path = os.path.join(full_path,
                                    'figure_test' + start_time.strftime("%H%M") + '_' +
                                    str(i) + '.pdf')
            fig.savefig(tmp_path, format='pdf')

    print('''Statistics on the test set average error {:.2g} +- {:.2g} (averaged over 16 test batches of size {}) DePasq Error {:.2g}'''
          .format(np.mean(test_errors), np.std(test_errors), FLAGS.batch_test, np.mean(test_depasq_err)))
    save_file(plot_results_values, full_path, 'plot_trajectory_data', 'pickle')

    def stack_to_new_axis(key):
        cont_data = [cp[key] for cp in continuous_plot]
        cont_data = np.stack(cont_data, axis=0)
        print(key, cont_data.shape)
        return cont_data
    cont_input_spikes = stack_to_new_axis('input_spikes')
    cont_z = stack_to_new_axis('z')
    cont_b_con = stack_to_new_axis('b_con')
    cont_target_nums = stack_to_new_axis('target_nums')
    cont_out_plot = stack_to_new_axis('out_plot')
    cont_plot_result_values = {
        'input_spikes': cont_input_spikes, 'z': cont_z, 'b_con': cont_b_con, 'target_nums': cont_target_nums,
        'out_plot': cont_out_plot,
    }
    save_file(cont_plot_result_values, full_path, 'plot_cont_trajectory_data', 'pickle')

    # Save test results
    results = {
        'test_depasq_err': test_depasq_err,
        'test_depasq_err_mean': np.mean(test_depasq_err),
        'test_errors': test_errors,
        'test_errors_mean': np.mean(test_errors),
        'test_errors_std': np.std(test_errors),
    }
    save_file(results, full_path, 'test_results', file_type='json')
    print("saved test_results.json")
    # Save network variables (weights, delays, etc.)
    network_data = tf_cell_to_savable_dict(cell, sess)
    network_data['w_out'] = results_values['w_out']
    save_file(network_data, full_path, 'tf_cell_net_data', file_type='pickle')

del sess
