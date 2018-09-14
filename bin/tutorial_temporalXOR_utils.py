import numpy as np
import numpy.random as rd
from scipy.stats import norm


def generate_xor_input(batch_size, length, expected_delay=100, pulse_duration=30, out_duration=60):
    input_nums = np.zeros((batch_size, length), dtype=float)
    gocue_nums = np.zeros((batch_size, length), dtype=float)
    targets = np.ones((batch_size), dtype=int) * 2
    # target_nums = np.zeros((batch_size, length), dtype=int)  # 2 classes
    target_nums = np.ones((batch_size, length), dtype=int) * 2
    target_mask_nums = np.zeros((batch_size, length), dtype=int)
    delay_prob = 10/expected_delay

    def prob_calc_delay():
        d = 10
        while True:
            if delay_prob > rd.uniform():
                break
            else:
                d += 10
        d = min(d, int(expected_delay))
        return d

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def pulse_trace():
        return normalize(norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), pulse_duration)))

    for b in range(batch_size):
        null_target = 0.2 > rd.uniform()
        pulse_delay = expected_delay if b == 0 else prob_calc_delay()
        seq = np.zeros(length)
        p1 = rd.choice([-1, 1])
        p2 = rd.choice([-1, 1])
        pulse_1 = pulse_trace() * p1
        pulse_2 = pulse_trace() * p2
        seq[0:pulse_duration] += pulse_1
        if not null_target:
            seq[pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_2
        input_nums[b, :] = seq

        gocue_seq = np.zeros(length)
        end_pulses = np.nonzero(seq)[0][-1]
        gocue_seq[end_pulses+pulse_delay:end_pulses+pulse_delay+pulse_duration] += pulse_trace()
        gocue_nums[b, :] = gocue_seq

        # target = 1 if p1 != p2 else -1
        target = p1 != p2
        start_cue = end_pulses+pulse_delay
        if not null_target:
            targets[b] = target
            target_nums[b, start_cue:start_cue+pulse_duration+out_duration] = target

        target_mask_nums[b, start_cue:start_cue+out_duration] = 1
        # target_mask_nums[b, :] = 1

    network_input = np.stack((input_nums, gocue_nums), axis=-1)  # batch x length x 2
    return network_input, target_nums, target_mask_nums, targets


def generate_xor_spike_input(batch_size, length, expected_delay=100, pulse_duration=30, out_duration=60):
    input_nums0 = np.zeros((batch_size, length), dtype=float)
    input_nums1 = np.zeros((batch_size, length), dtype=float)
    gocue_nums = np.zeros((batch_size, length), dtype=float)
    targets = np.zeros((batch_size), dtype=int)
    target_nums = np.zeros((batch_size, length), dtype=int)
    target_mask_nums = np.zeros((batch_size, length), dtype=int)
    delay_prob = 10/expected_delay

    def prob_calc_delay():
        d = 10
        while True:
            if delay_prob > rd.uniform():
                break
            else:
                d += 10
        d = min(d, int(expected_delay))
        return d

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def pulse_trace():
        return normalize(norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), pulse_duration)))

    for b in range(batch_size):
        pulse_delay = prob_calc_delay()
        p1 = rd.choice([0, 1])
        p2 = rd.choice([0, 1])
        if p1 == 0:
            input_nums0[b, 0:pulse_duration] += pulse_trace()
        else:
            input_nums1[b, 0:pulse_duration] += pulse_trace()

        if p2 == 0:
            input_nums0[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()
        else:
            input_nums1[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()

        seq = input_nums0[b] + input_nums1[b]
        end_pulses = np.nonzero(seq)[0][-1]
        gocue_seq = np.zeros(length)
        gocue_seq[end_pulses+pulse_delay:end_pulses+pulse_delay+pulse_duration] += pulse_trace()
        gocue_nums[b, :] = gocue_seq

        # target = 1 if p1 != p2 else -1
        target = p1 != p2
        targets[b] = target
        start_cue = end_pulses+pulse_delay
        target_nums[b, start_cue:start_cue+pulse_duration+out_duration] = target

        target_mask_nums[b, start_cue:start_cue+out_duration] = 1
        # target_mask_nums[b, :] = 1

    network_input = np.stack((input_nums0, input_nums1, gocue_nums), axis=-1)  # batch x length x 2
    return network_input, target_nums, target_mask_nums, targets


def generate_3class_xor_spike_input(batch_size, length, expected_delay=100, pulse_duration=30, out_duration=60):
    input_nums0 = np.zeros((batch_size, length), dtype=float)
    input_nums1 = np.zeros((batch_size, length), dtype=float)
    gocue_nums = np.zeros((batch_size, length), dtype=float)
    targets = np.zeros((batch_size), dtype=int)
    target_nums = np.ones((batch_size, length), dtype=int) * 2
    target_mask_nums = np.zeros((batch_size, length), dtype=int)
    delay_prob = 10/expected_delay

    def prob_calc_delay():
        d = 10
        while True:
            if delay_prob > rd.uniform():
                break
            else:
                d += 10
        d = min(d, int(expected_delay))
        return d

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    def pulse_trace():
        return normalize(norm.pdf(np.linspace(norm.ppf(0.01), norm.ppf(0.99), pulse_duration)))

    delays = []
    for b in range(batch_size):
        pulse_delay = prob_calc_delay()
        delays.append(pulse_delay)
        p1 = rd.choice([0, 1])
        p2 = rd.choice([0, 1])
        if p1 == 0:
            input_nums0[b, 0:pulse_duration] += pulse_trace()
        else:
            input_nums1[b, 0:pulse_duration] += pulse_trace()

        if p2 == 0:
            input_nums0[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()
        else:
            input_nums1[b, pulse_duration+pulse_delay:pulse_duration*2+pulse_delay] += pulse_trace()

        seq = input_nums0[b] + input_nums1[b]
        end_pulses = np.nonzero(seq)[0][-1]
        gocue_seq = np.zeros(length)
        gocue_seq[end_pulses+pulse_delay:end_pulses+pulse_delay+pulse_duration] += pulse_trace()
        gocue_nums[b, :] = gocue_seq

        # target = 1 if p1 != p2 else -1
        target = p1 != p2
        targets[b] = target
        start_cue = end_pulses+pulse_delay
        target_nums[b, start_cue:start_cue+pulse_duration+out_duration] = target

        target_mask_nums[b, start_cue:start_cue+out_duration] = 1
        # target_mask_nums[b, :] = 1

    network_input = np.stack((input_nums0, input_nums1, gocue_nums), axis=-1)  # batch x length x 2
    return network_input, target_nums, target_mask_nums, targets
