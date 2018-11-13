import numpy as np
import numpy.random as rd
from scipy.stats import norm
from matplotlib import collections as mc, patches
from matplotlib.patches import Patch
from lsnn.guillaume_toolbox.matplotlib_extension import strip_right_top_axis, raster_plot, hide_bottom_axis


def update_plot(plt, ax_list, FLAGS, plot_result_values, batch=0, cont=-1):
    """
    This function iterates the matplotlib figure on every call.
    It plots the data for a fixed sequence that should be representative of the expected computation
    :return:
    """
    ylabel_x = -0.08
    ylabel_y = 0.5
    fs = 10
    plt.rcParams.update({'font.size': fs})

    # Clear the axis to print new plots
    for k in range(ax_list.shape[0]):
        ax = ax_list[k]
        ax.clear()
        strip_right_top_axis(ax)

    # PLOT Input spikes
    ax = ax_list[0]
    data = plot_result_values['input_spikes']
    data = data[batch] if cont == -1 else data[cont, batch]
    raster_plot(ax, data, linewidth=0.4, color='black')
    ax.set_xlim([0, len(data)])
    ax.set_ylabel('input X')
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    ax.add_patch(patches.Rectangle((0, 90), data.shape[0], 10, facecolor="purple", alpha=0.15))
    ax.add_patch(patches.Rectangle((0, 80), data.shape[0], 10, facecolor="orange", alpha=0.15))
    custom_lines = [Patch(facecolor="purple", label="cue 1", alpha=0.25),
                    Patch(facecolor="orange", label="cue 2", alpha=0.25)]
    ax.legend(handles=custom_lines, loc='lower right', fontsize=7, ncol=3)
    hide_bottom_axis(ax)

    # PLOT SPIKES
    ax = ax_list[1]
    data = plot_result_values['z']
    data = data[batch] if cont == -1 else data[cont, batch]
    data = data[:, :FLAGS.n_regular]  # R neurons
    raster_plot(ax, data, linewidth=.3)
    ax.set_ylabel('R', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    hide_bottom_axis(ax)
    ax = ax_list[2]
    data = plot_result_values['z']
    data = data[batch] if cont == -1 else data[cont, batch]
    data = data[:, FLAGS.n_regular:FLAGS.n_regular+FLAGS.n_adaptive]  # A neurons
    raster_plot(ax, data, linewidth=1.)
    ax.set_ylabel('A', fontsize=fs)
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    hide_bottom_axis(ax)

    # debug plot for psp-s or biases
    plot_param = 'b_con'  # or 'psp'
    ax.set_xticklabels([])
    ax = ax_list[-2]
    ax.set_ylabel('threshold of A')
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    sub_data = plot_result_values['b_con']
    sub_data = sub_data[batch] if cont == -1 else sub_data[cont, batch]
    sub_data = sub_data[:, FLAGS.n_regular:FLAGS.n_regular+FLAGS.n_adaptive]
    sub_data = sub_data + FLAGS.thr
    vars = np.var(sub_data, axis=0)
    # cell_with_max_var = np.argsort(vars)[::-1][:n_max_synapses * 3:3]
    cell_with_max_var = np.argsort(vars)[::-1]
    presentation_steps = np.arange(sub_data.shape[0])
    ax.plot(sub_data[:, cell_with_max_var], color='r', label='threshold', alpha=0.4, linewidth=1)
    ax.axis([0, presentation_steps[-1], np.min(sub_data[:, cell_with_max_var]),
                 np.max(sub_data[:, cell_with_max_var])])  # [xmin, xmax, ymin, ymax]
    hide_bottom_axis(ax)

    # PLOT OUTPUT AND TARGET
    ax = ax_list[-1]
    data = plot_result_values['target_nums']
    data = data[batch] if cont == -1 else data[cont, batch]
    presentation_steps = np.arange(data.shape[0])
    for i in range(data.shape[1]):
        line_target, = ax.plot(presentation_steps[:], data[:, i], color='blue', label='target', alpha=0.7)
    output2 = plot_result_values['out_plot']
    output2 = output2[batch] if cont == -1 else output2[cont, batch]
    presentation_steps = np.arange(output2.shape[0])
    ax.set_yticks([-1, 0, 1])
    # ax.grid(color='black', alpha=0.15, linewidth=0.4)
    ax.set_ylabel('output Y')
    ax.get_yaxis().set_label_coords(ylabel_x, ylabel_y)
    for i in range(data.shape[1]):
        line_output2, = ax.plot(presentation_steps, output2[:, i], color='green', label='output', alpha=0.7)
    ax.set_xlim([0, presentation_steps[-1] + 1])
    ax.legend(handles=[line_output2, line_target], loc='lower left', fontsize=7, ncol=3)

    ax.set_xlabel('time in ms')
    # To plot with interactive python one need to wait one second to the time to draw the axis
    if FLAGS.do_plot:
        plt.draw()
        plt.pause(1)


def offline_plot(data_path, custom_plot=False):
    import matplotlib.pyplot as plt
    import datetime
    import pickle
    import json
    import os

    flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    from types import SimpleNamespace
    flags = SimpleNamespace(**flags_dict)

    plot_data = 'plot_cont_trajectory_data.pickle' if custom_plot else 'plot_trajectory_data.pickle'
    plot_result_values = pickle.load(open(os.path.join(data_path, plot_data), 'rb'))

    plt.ion()
    nrows = 5
    height = 7.5
    fig, ax_list = plt.subplots(nrows=nrows, figsize=(6, height), gridspec_kw={'wspace': 0, 'hspace': 0.2})
    if custom_plot:
        for b in range(flags.batch_test):
            for c in range(plot_result_values['z'].shape[0]):
                update_plot(plt, ax_list, flags, plot_result_values, batch=b, cont=c)
                start_time = datetime.datetime.now()
                fig.savefig(
                    os.path.join(data_path, 'figure_cont_test_b' + str(b) + '_c' + str(c) + '_' +
                                 start_time.strftime("%H%M") + '.pdf'), format='pdf')
    else:
        for b in range(flags.batch_test):
            update_plot(plt, ax_list, flags, plot_result_values, batch=b)
            start_time = datetime.datetime.now()
            fig.savefig(os.path.join(data_path, 'figure_test' + str(b) + '_' + start_time.strftime("%H%M") + '.pdf'),
                        format='pdf')
