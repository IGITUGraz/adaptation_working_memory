import numpy as np


def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def autocorr_plot(data_path, plot=True, max_neurons=20):
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr
    import pickle
    import json
    import math
    import os

    flags_dict = json.load(open(os.path.join(data_path, 'flags.json')))
    from types import SimpleNamespace
    FLAGS = SimpleNamespace(**flags_dict)

    if not os.path.exists(os.path.join(data_path, 'autocorr')):
        os.makedirs(os.path.join(data_path, 'autocorr'))

    plot_data = 'plot_trajectory_data.pickle'
    data = pickle.load(open(os.path.join(data_path, plot_data), 'rb'))
    bin_size = 50
    sample_size = 500
    assert sample_size % bin_size == 0
    n_bins = int(sample_size / bin_size)  # == 10
    spikes = data['z'][:, :, FLAGS.n_regular:FLAGS.n_regular+FLAGS.n_adaptive]  # batch x time x neurons
    inferred_taus = []
    inferred_As = []
    inferred_Bs = []
    n_idxs = []
    max_neurons = max_neurons if max_neurons > 0 else FLAGS.n_adaptive
    for n_idx in range(min(FLAGS.n_adaptive, max_neurons)):  # loop over adaptive neurons
        spk_count = spikes[:, :sample_size, n_idx]
        bin_spk_count = bin_ndarray(spk_count, (spikes.shape[0], n_bins), operation='sum')  # batch x n_bins
        # if np.count_nonzero(bin_spk_count) == 0:
        if np.count_nonzero(bin_spk_count) == 0:
            continue  # skip dead neurons
        # calculate correlations across bins
        lags_idxs = [l for l in range(1, n_bins)]  # idy for [l for l in range(50, 500, 50)]
        corrs_neuron = []  # black dots
        for lag_idx in lags_idxs:
            corr_per_lag = []
            for i in range(n_bins - lag_idx):
                pearson_corr_coeff, _ = pearsonr(bin_spk_count[:, i], bin_spk_count[:, i + lag_idx])
                # print(i, i+lag_idx, pearson_corr_coeff)
                corr_per_lag.append(pearson_corr_coeff)
            corrs_neuron.append(corr_per_lag)
        avg_corrs_neuron = [np.mean(c) for c in corrs_neuron]  # red dots
        if np.array([math.isnan(c) for c in avg_corrs_neuron]).any():
            continue  # skip neurons with nan correlation
        # print("---------- neuron ", n_idx)
        # for l in range(len(avg_corrs_neuron)):
        #     print(l*bin_size, avg_corrs_neuron[l])  # print red dots

        # fit curve to black dots
        def func(x, A, B, tau):
            return A * (np.exp(-x / tau) + B)
        xdata = np.arange(len(avg_corrs_neuron))
        popt, pcov = curve_fit(func, xdata, avg_corrs_neuron)
        A, B, tau = tuple(popt)
        tau = tau * bin_size  # convert from bins to ms
        if tau > max(FLAGS.tauas):
            continue
        inferred_As.append(A)
        inferred_Bs.append(B)
        inferred_taus.append(tau)
        n_idxs.append(n_idx)

        if plot:
            plt.cla()
            plt.clf()
            for l in range(len(corrs_neuron)):
                for i in range(len(corrs_neuron[l])):
                    plt.plot(xdata[l], corrs_neuron[l][i], 'ko')  # grid line at zero
            plt.plot(xdata, np.zeros_like(xdata), 'k--')  # grid line at zero
            plt.plot(xdata, avg_corrs_neuron, 'ro')  # avg red dots
            plt.plot(xdata, func(xdata, *popt), 'k-', label='fitted curve')
            plt.title("Fitted exponential curve tau = {:.0f}ms".format(tau))
            # plt.show()
            plt_path = os.path.join(data_path, 'autocorr/autocorr_' + str(n_idx) + '.pdf')
            plt.savefig(plt_path, format='pdf')

    resuls = {
        'tau': inferred_taus,
        'A': inferred_As,
        'B': inferred_Bs,
        'neuron_idx': n_idxs,
    }

    try:
        resuls['taua'] = FLAGS.tauas
        defined_tauas = np.array(FLAGS.tauas)
        defined_tauas = defined_tauas[n_idxs]  # take only entries for the neurons that have inferred tau
        if plot and len(n_idxs) > 5:
            plt.cla()
            plt.clf()
            plt.plot(defined_tauas, inferred_taus, 'ko')
            plt.xlabel('(defined) adaptation time constant')
            plt.ylabel('(inferred) intrinsic time constant')
            # plt.show()
            plt_path = os.path.join(data_path, 'autocorr/tau_comp.pdf')
            plt.savefig(plt_path, format='pdf')
    except AttributeError:
        pass  # skip if the recorded data does not contain the tau_a-s
    with open(os.path.join(data_path, 'autocorr/results.json'), 'w') as fp:
        json.dump(resuls, fp, indent=4)
