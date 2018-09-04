import numpy as np
import numpy.random as rd
from scipy.stats import norm


def generate_xor_input(batch_size, length, pulse_delay=100, pulse_duration=30, out_duration=60):
    input_nums = np.zeros((batch_size, length), dtype=float)
    gocue_nums = np.zeros((batch_size, length), dtype=float)
    targets = np.zeros((batch_size), dtype=int)
    target_nums = np.zeros((batch_size, length), dtype=int)
    target_mask_nums = np.zeros((batch_size, length), dtype=int)

    def pulse_trace():
        return norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), pulse_duration))

    for b in range(batch_size):
        seq = np.zeros(length)
        p1 = rd.choice([-1, 1])
        p2 = rd.choice([-1, 1])
        pulse_1 = pulse_trace() * p1
        pulse_2 = pulse_trace() * p2
        seq[0:pulse_duration] += pulse_1
        seq[pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_2
        input_nums[b, :] = seq

        gocue_seq = np.zeros(length)
        end_pulses = np.nonzero(seq)[0][-1]
        gocue_seq[end_pulses+pulse_delay:end_pulses+pulse_delay+pulse_duration] += pulse_trace()
        gocue_nums[b, :] = gocue_seq

        # target = 1 if p1 != p2 else -1
        target = p1 != p2
        targets[b] = target
        start_cue = end_pulses+pulse_delay
        target_nums[b, start_cue:start_cue+pulse_duration+out_duration] = target

        target_mask_nums[b, start_cue:start_cue+out_duration] = 1
        # target_mask_nums[b, :] = 1

    network_input = np.stack((input_nums, gocue_nums), axis=-1)  # batch x length x 2
    return network_input, target_nums, target_mask_nums, targets
