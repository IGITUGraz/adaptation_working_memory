"""
    Weight perturbation robustness test on trained networks (store-recall task)

    compute for ALIF:
    python3 bin/tutorial_storerecall_perturb_test.py --reproduce=560_ALIF --checkpoint=results/tutorial_storerecall_with_LSNN/2019_06_11_12_28_22_ALIF_seqlen20_seqdelay10_in40_R0_A60_lr0.01_tauchar200_comment_ALIF-DEBUG_taua2000_beta1/model

    compute for ELIF:
    python3 bin/tutorial_storerecall_perturb_test.py --reproduce=560_ELIF --checkpoint=results/tutorial_storerecall_with_LSNN/2019_06_11_12_28_00_ALIF_seqlen20_seqdelay10_in40_R0_A60_lr0.01_tauchar200_comment_ELIF-DEBUG_taua2000_beta-0.5/model
"""

# import matplotlib
# matplotlib.use('Agg')

import datetime
import os
import socket
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
import tensorflow as tf
from lsnn.guillaume_toolbox.tensorflow_einsums.einsum_re_written import einsum_bij_jk_to_bik
from lsnn.guillaume_toolbox.file_saver_dumper_no_h5py import save_file

from tutorial_storerecall_utils import generate_storerecall_data, error_rate, gen_custom_delay_batch,\
    update_plot

from lsnn.guillaume_toolbox.tensorflow_utils import tf_downsample
from lsnn.spiking_models import tf_cell_to_savable_dict, placeholder_container_for_rnn_state, \
    feed_dict_with_placeholder_container, exp_convolve, ALIF, STP
from lsnn.guillaume_toolbox.rewiring_tools import weight_sampler, rewiring_optimizer_wrapper

script_name = os.path.basename(__file__)[:-3]
result_folder = 'results/' + script_name + '/'
FLAGS = tf.app.flags.FLAGS
start_time = datetime.datetime.now()

##
tf.app.flags.DEFINE_string('comment', '', 'comment to retrieve the stored results')
tf.app.flags.DEFINE_string('reproduce', '', 'set flags to reproduce results from paper [560_A, ...]')
tf.app.flags.DEFINE_string('checkpoint', '', 'path to pre-trained model to restore')
##
tf.app.flags.DEFINE_integer('batch_train', 128, 'batch size fo the validation set')
tf.app.flags.DEFINE_integer('batch_val', 1000, 'batch size of the validation set')
tf.app.flags.DEFINE_integer('batch_test', 128, 'batch size of the testing set')
tf.app.flags.DEFINE_integer('n_charac', 2, 'number of characters in the recall task')
tf.app.flags.DEFINE_integer('n_in', 100, 'number of input units.')
tf.app.flags.DEFINE_integer('n_regular', 60, 'number of recurrent units.')
tf.app.flags.DEFINE_integer('n_adaptive', 20, 'number of controller units')
tf.app.flags.DEFINE_integer('f0', 50, 'input firing rate')
tf.app.flags.DEFINE_integer('reg_rate', 10, 'target rate for regularization')
tf.app.flags.DEFINE_integer('reg_max_rate', 100, 'target rate for regularization')
tf.app.flags.DEFINE_integer('n_iter', 200, 'number of iterations')
tf.app.flags.DEFINE_integer('n_delay', 10, 'number of delays')
tf.app.flags.DEFINE_integer('n_ref', 3, 'Number of refractory steps')
tf.app.flags.DEFINE_integer('seq_len', 12, 'Number of character steps')
tf.app.flags.DEFINE_integer('seq_delay', 6, 'Expected delay in character steps. Must be <= seq_len - 2')
tf.app.flags.DEFINE_integer('tau_char', 200, 'Duration of symbols')
tf.app.flags.DEFINE_integer('seed', -1, 'Random seed.')
tf.app.flags.DEFINE_integer('lr_decay_every', 100, 'Decay every')
tf.app.flags.DEFINE_integer('print_every', 20, 'Decay every')
tf.app.flags.DEFINE_integer('perturb_avg', 100, 'Decay every')
##
tf.app.flags.DEFINE_float('stop_crit', 0.05, 'Stopping criterion. Stops training if error goes below this value')
tf.app.flags.DEFINE_float('beta', 1.7, 'Mikolov adaptive threshold beta scaling parameter')
tf.app.flags.DEFINE_float('tau_a', 1200, 'Mikolov model alpha - threshold decay')
tf.app.flags.DEFINE_float('tau_out', 20, 'tau for PSP decay in LSNN and output neurons')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Base learning rate.')
tf.app.flags.DEFINE_float('lr_decay', 0.3, 'Decaying factor')
tf.app.flags.DEFINE_float('reg', 1., 'regularization coefficient')
tf.app.flags.DEFINE_float('rewiring_connectivity', -1, 'possible usage of rewiring with ALIF and LIF (0.1 is default)')
tf.app.flags.DEFINE_float('readout_rewiring_connectivity', -1, '')
tf.app.flags.DEFINE_float('l1', 1e-2, 'l1 regularization that goes with rewiring')
tf.app.flags.DEFINE_float('rewiring_temperature', 0, '')
tf.app.flags.DEFINE_float('dampening_factor', 0.3, '')
tf.app.flags.DEFINE_float('stochastic_factor', -1, '')
tf.app.flags.DEFINE_float('dt', 1., '(ms) simulation step')
tf.app.flags.DEFINE_float('thr', .01, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('thr_min', .005, 'threshold at which the LSNN neurons spike')
tf.app.flags.DEFINE_float('ELIF_to_iLIF', 0.35, 'ELIF motif param')
tf.app.flags.DEFINE_float('iLIF_to_ELIF', -0.15, 'ELIF motif param')
tf.app.flags.DEFINE_float('tauF', 2000, 'STP tau facilitation')
tf.app.flags.DEFINE_float('tauD', 200, 'STP tau depression')
tf.app.flags.DEFINE_float('U', .2, 'STP baseline value of u')
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
tf.app.flags.DEFINE_bool('preserve_state', False, 'preserve network state between training trials')


if FLAGS.reproduce == '560_ELIF':
    print("Using the hyperparameters as in 560 paper: pure ELIF network")
    FLAGS.beta = -0.5
    FLAGS.thr = 0.02
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.tau_a = 2000
    FLAGS.n_in = 40
    FLAGS.stop_crit = 0.0
    FLAGS.n_iter = 400

if FLAGS.reproduce == '560_ALIF':
    print("Using the hyperparameters as in 560 paper: pure ALIF network")
    FLAGS.beta = 1
    FLAGS.thr = 0.01
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.tau_a = 2000
    FLAGS.n_in = 40
    FLAGS.stop_crit = 0.0
    FLAGS.n_iter = 400

if FLAGS.reproduce == '560_STP':
    print("Using the hyperparameters as in 560 paper: pure STP network")
    FLAGS.thr = 0.02
    FLAGS.n_regular = 0
    FLAGS.n_adaptive = 60
    FLAGS.seq_len = 20
    FLAGS.seq_delay = 10
    FLAGS.tauF = 2000
    FLAGS.tauD = 200
    FLAGS.U = .2
    FLAGS.n_in = 40
    FLAGS.stop_crit = 0.0
    FLAGS.n_iter = 400

# Run asserts to check seq_delay and seq_len relation is ok
_ = gen_custom_delay_batch(FLAGS.seq_len, FLAGS.seq_delay, 1)

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
if FLAGS.tau_a_spread:
    tau_a_spread = np.random.uniform(size=FLAGS.n_regular+FLAGS.n_adaptive) * FLAGS.tau_a
else:
    tau_a_spread = FLAGS.tau_a
beta = np.concatenate([np.zeros(FLAGS.n_regular), np.ones(FLAGS.n_adaptive) * FLAGS.beta])

if FLAGS.reproduce == '560_STP':
    cell = STP(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v, n_delay=FLAGS.n_delay,
               n_refractory=FLAGS.n_ref, dt=dt, thr=FLAGS.thr,
               rewiring_connectivity=FLAGS.rewiring_connectivity,
               in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
               dampening_factor=FLAGS.dampening_factor,
               tau_F=FLAGS.tauF, tau_D=FLAGS.tauD, U=FLAGS.U,
               )
else:
    cell = ALIF(n_in=FLAGS.n_in, n_rec=FLAGS.n_regular + FLAGS.n_adaptive, tau=tau_v, n_delay=FLAGS.n_delay,
                n_refractory=FLAGS.n_ref, dt=dt, tau_adaptation=tau_a_spread, beta=beta, thr=FLAGS.thr,
                rewiring_connectivity=FLAGS.rewiring_connectivity,
                in_neuron_sign=in_neuron_sign, rec_neuron_sign=rec_neuron_sign,
                dampening_factor=FLAGS.dampening_factor, thr_min=FLAGS.thr_min
                )

cell_name = type(cell).__name__
print('\n -------------- \n' + cell_name + '\n -------------- \n')
time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_reference = '{}_{}_seqlen{}_seqdelay{}_in{}_R{}_A{}_lr{}_tauchar{}_comment{}'.format(
    time_stamp, cell_name, FLAGS.seq_len, FLAGS.seq_delay, FLAGS.n_in, FLAGS.n_regular, FLAGS.n_adaptive,
    FLAGS.learning_rate, FLAGS.tau_char, FLAGS.comment)
file_reference = file_reference + '_taua' + str(FLAGS.tau_a) + '_beta' + str(FLAGS.beta)
print('FILE REFERENCE: ' + file_reference)

# Generate input
input_spikes = tf.placeholder(dtype=tf.float32, shape=(None, None, FLAGS.n_in),
                              name='InputSpikes')  # MAIN input spike placeholder
input_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                            name='InputNums')  # Lists of input character for the recall task
target_nums = tf.placeholder(dtype=tf.int64, shape=(None, None),
                             name='TargetNums')  # Lists of target characters of the recall task
recall_mask = tf.placeholder(dtype=tf.bool, shape=(None, None),
                             name='RecallMask')  # Binary tensor that points to the time of presentation of a recall

# Other placeholder that are useful for computing accuracy and debuggin
target_sequence = tf.placeholder(dtype=tf.int64, shape=(None, None),
                                 name='TargetSequence')  # The target characters with time expansion
batch_size_holder = tf.placeholder(dtype=tf.int32, name='BatchSize')  # Int that contains the batch size
init_state_holder = placeholder_container_for_rnn_state(cell.state_size, dtype=tf.float32, batch_size=None)
recall_charac_mask = tf.equal(input_nums, recall_symbol, name='RecallCharacMask')


def get_data_dict(batch_size, seq_len=FLAGS.seq_len, batch=None, override_input=None):
    p_sr = 1/(1 + FLAGS.seq_delay)
    spk_data, is_recall_data, target_seq_data, memory_seq_data, in_data, target_data = generate_storerecall_data(
        batch_size=batch_size,
        f0=input_f0,
        sentence_length=seq_len,
        n_character=FLAGS.n_charac,
        n_charac_duration=FLAGS.tau_char,
        n_neuron=FLAGS.n_in,
        prob_signals=p_sr,
        with_prob=True,
        override_input=override_input,
    )
    data_dict = {input_spikes: spk_data, input_nums: in_data, target_nums: target_data, recall_mask: is_recall_data,
                 target_sequence: target_seq_data, batch_size_holder: batch_size}

    return data_dict

# Define the name of spike train for the different models
z_stack, final_state = tf.nn.dynamic_rnn(cell, input_spikes, initial_state=init_state_holder, dtype=tf.float32)
if FLAGS.reproduce == '560_STP':
    z, stp_u, stp_x = z_stack
else:
    z, b_con = z_stack
z_con = []
z_all = z

with tf.name_scope('RecallLoss'):
    target_nums_at_recall = tf.boolean_mask(target_nums, recall_charac_mask)
    Y = tf.one_hot(target_nums_at_recall, depth=n_output_symbols, name='Target')

    # MTP models do not use controller (modulator) population for output
    out_neurons = z_all
    n_neurons = out_neurons.get_shape()[2]
    psp = exp_convolve(out_neurons, decay=decay)

    if 0 < FLAGS.rewiring_connectivity and 0 < FLAGS.readout_rewiring_connectivity:
        w_out, w_out_sign, w_out_var, _ = weight_sampler(FLAGS.n_regular + FLAGS.n_adaptive, n_output_symbols,
                                                         FLAGS.readout_rewiring_connectivity,
                                                         neuron_sign=rec_neuron_sign)
    else:
        w_out = tf.get_variable(name='out_weight', shape=[n_neurons, n_output_symbols])

    out = einsum_bij_jk_to_bik(psp, w_out)
    out_char_step = tf_downsample(out, new_size=FLAGS.seq_len, axis=1)
    Y_predict = tf.boolean_mask(out_char_step, recall_charac_mask, name='Prediction')

    # loss_recall = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_predict))
    loss_recall = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_nums_at_recall,
                                                                                logits=Y_predict))

    with tf.name_scope('PlotNodes'):
        out_plot = tf.nn.softmax(out)
        out_plot_char_step = tf_downsample(out_plot, new_size=FLAGS.seq_len, axis=1)

    _, recall_errors, false_sentence_id_list = error_rate(out_char_step, target_nums, input_nums, n_charac)

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

config = tf.ConfigProto(log_device_placement=FLAGS.device_placement)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if len(FLAGS.checkpoint) > 0:
    saver = tf.train.Saver(tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES))
    saver.restore(sess, FLAGS.checkpoint)
    print("Model restored from ", FLAGS.checkpoint)
else:
    saver = tf.train.Saver()


last_final_state_state_training_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_train, dtype=tf.float32))]
last_final_state_state_validation_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_val, dtype=tf.float32))]
last_final_state_state_testing_pointer = [sess.run(cell.zero_state(batch_size=FLAGS.batch_test, dtype=tf.float32))]

if FLAGS.do_plot:
    # Open an interactive matplotlib window to plot in real time
    if FLAGS.interactive_plot:
        plt.ion()
    fig, ax_list = plt.subplots(nrows=5, figsize=(6, 7.5), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    # re-name the window with the name of the cluster to track relate to the terminal window
    fig.canvas.set_window_title(socket.gethostname() + ' - ' + FLAGS.comment)


results_tensors = {
    'loss': loss,
    'loss_reg': loss_reg,
    'loss_recall': loss_recall,
    'recall_errors': recall_errors,
    'final_state': final_state,
    'av': av,
    'adaptive_regularization_coeff': adaptive_regularization_coeff,
    'w_in_val': cell.w_in_val,
    'w_rec_val': cell.w_rec_val,
    'w_out': w_out,
}

perturbation_std = tf.placeholder(dtype=tf.float32, name='PerturbationStd')  # Int that contains the batch size

w_in_shape = tf.shape(cell.w_in_var)
w_in_noise = tf.random_normal(w_in_shape, mean=0.0, stddev=perturbation_std, dtype=tf.float32)
perturb_in_w = tf.assign_add(cell.w_in_var, w_in_noise)

w_rec_shape = tf.shape(cell.w_rec_var)
w_rec_noise = tf.random_normal(w_rec_shape, mean=0.0, stddev=perturbation_std, dtype=tf.float32)
perturb_rec_w = tf.assign_add(cell.w_rec_var, w_rec_noise)

perturb_stds = [0., 0.001, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.]
perturb_errs = []

# prepare input data
val_dict = get_data_dict(FLAGS.batch_val)
feed_dict_with_placeholder_container(val_dict, init_state_holder, last_final_state_state_validation_pointer[0])

for p_std in perturb_stds:

    # perturb
    val_dict[perturbation_std] = p_std
    all_ptb_errors = []
    for _ in range(FLAGS.perturb_avg):
        # perturb
        sess.run([perturb_in_w, perturb_rec_w], feed_dict=val_dict)

        # compute performance
        results_values = sess.run(results_tensors, feed_dict=val_dict)
        all_ptb_errors.append(results_values['recall_errors'])

        # restore weights
        saver.restore(sess, FLAGS.checkpoint)

        # test
        # restore_values = sess.run(results_tensors, feed_dict=val_dict)
        # assert perturb_errs[0] == restore_values['recall_errors']

    ptb_error = np.mean(all_ptb_errors)
    print("perturbation {} results in error \t {}".format(p_std, ptb_error))
    perturb_errs.append(ptb_error)


full_path = os.path.join(result_folder, file_reference)
if not os.path.exists(full_path):
    os.makedirs(full_path)

results = {
    'perturb_errs': perturb_errs,
    'perturb_stds': perturb_stds,
    'checkpoint': FLAGS.checkpoint,
}

save_file(results, full_path, 'perturb_results', file_type='json')

del sess
