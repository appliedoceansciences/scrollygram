#!/usr/bin/env python3
# usage: ssh remotehost shm2pgram.py /cobs_to_shm | ./scroll_gram_from_json.py
import socket
import sys
import threading
import queue
import math
import signal
import base64
import struct
from collections import namedtuple

# This provides the generator function which knows how to extract sensor-agnostic frames of
# acoustic sample data from whatever possibly sensor-specific format they are coming from
from parse_acoustic_packets import yield_acoustic_packets, yield_packet_bytes_from_log_stream

# This provides the generator function which accepts the above sequence of sensor-agnostic
# numpy arrays of acoustic samples, assembles them into 50%-overlapped frames according to
# a desired frequency resolution, does the FFT on each frame as it becomes ready,
# incoherently averages these for a desired time resolution, and yields these to the plotter
from spectrogram_generator import incoherent_fft_frame_generator

from shared_memory_ringbuffer_reader import shared_memory_ringbuffer_generator

import numpy as np

def round_up_to_next_multiple_of(a, q):
    return a + q - a % q if a % q else a

def yield_from_shm_and_strip_logging_header(source):
    for packet_with_logging_header in shared_memory_ringbuffer_generator(source):
        packet_bytes = packet_with_logging_header[8:]
        packet_size, timestamp_lsbs, timestamp_msbs = struct.unpack('<HHI', packet_with_logging_header[0:8])
        logged_timestamp_microseconds = ((timestamp_msbs << 16) | timestamp_lsbs) * 16
        yield packet_with_logging_header[8:], logged_timestamp_microseconds

def incoherent_cqt_frames_generator(bins_per_octave, args_for_incoherent_fft_frames_generator):
    child = incoherent_fft_frame_generator(*args_for_incoherent_fft_frames_generator)

    frame = next(child, None)
    if frame is None: return

    df = frame.df
    dt = frame.dt
    f0 = frame.f0
    C, F = frame.bins.shape

    linear_bins_from_dc = int(math.ceil(bins_per_octave / math.log(2.0)))
    T = 2 * (F + round(f0 / df))
    log_bins = int(np.ceil(np.log2(T / (2.0 * linear_bins_from_dc)) * bins_per_octave))

    X = linear_bins_from_dc + log_bins - 2
    print('last linear bin is %u from dc (%g Hz)' % (linear_bins_from_dc, linear_bins_from_dc * df), file=sys.stderr)

    sigma_hann_bins = 0.600561
    sqrt_2_pi = 2.50663

    dlogf = math.log(2.0) / bins_per_octave
    advance = math.exp(dlogf)

    # precalculate, for each log-spaced output bin, the set of input bins which contribute
    # to it, and the weight of each one, hopefully normalized such that a unit-magnitude
    # tone at any frequency will result in a unit-magnitude output bin
    weights_list = []
    for ibin in range(log_bins):
        # midpoint, in input frequency bins from dc, of this output bin
        iwf_mid = linear_bins_from_dc * pow(2, ibin / bins_per_octave)

        # nominal lower and upper edges of this output bin in units of input bins
        iwf_lo = iwf_mid / advance
        iwf_hi = iwf_mid * advance

        # the same numbers, rounded down and up to the next integer bin indices
        iw_start = math.floor(iwf_lo)
        iw_stop = math.ceil(min(iwf_hi, F))

        # the desired width of a gaussian in absolute frequency support, centered on this bin
        local_band_spacing_in_bins = iwf_mid * dlogf
        sigma_wanted = local_band_spacing_in_bins * sigma_hann_bins

        # narrow it up to correct for the fact that we used a Hann window on the linear
        # frequencies and therefore already have some amount of spreading
        sigma = math.sqrt(sigma_wanted * sigma_wanted - sigma_hann_bins * sigma_hann_bins)

        # vector of distances in units of input bin index of this bin's midpoint from the
        # set of supporting linearly-spaced input bins
        delta = np.arange(iw_start, iw_stop) - iwf_mid

        # vector of Gaussian weights
        weight = np.exp(-0.5 * (delta * delta / (sigma * sigma)))
        weight_sum = np.sum(weight)

        # renormalize the weights to account for their desired integral and the actual sum,
        # and a factor of 1.5 for the Hann window
        weight = weight * sqrt_2_pi * sigma_wanted / (weight_sum * 1.5)

        # save the index limits and the vector of weights so we can apply it to the data
        weights_list.append((iw_start, iw_stop, weight))

    averaged_frame = namedtuple('averaged_frame', ('timestamp_microseconds', 'intensity', 'f0', 'df', 'dt', 'bins_per_octave'))

    while frame is not None:
        out = np.empty((C, X), dtype=np.single)
        out[:, 0:(linear_bins_from_dc - 2)] = frame.bins[:, 2:linear_bins_from_dc]

        for ibin in range(log_bins):
            iw_start, iw_stop, weight = weights_list[ibin]
            out[:, ibin + linear_bins_from_dc - 2] = np.dot(frame.bins[:, iw_start:iw_stop], weight)

        yield averaged_frame(timestamp_microseconds=frame.timestamp_microseconds,
                             intensity=out,
                             f0=0,
                             df=df,
                             dt=dt,
                             bins_per_octave=bins_per_octave)

        frame = next(child, None)


def main():
    # if not None, a tuple of phone indices to keep, starting at zero
    phonemask = None
    df_desired = None
    f0_desired = 0.0
    fh_desired = None
    dt_desired = 0.125
    bins_per_octave = 48
    micropascals_per_lsb = 1

    # support legacy method of specifying input as a single argument
    input_source_string = sys.argv[1] if 2 == len(sys.argv) else None

    # loop over pairs of arguments
    for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
        if key == 'phonemask': phonemask = list(map(int, value.split(',')))
        if key == 'df': df_desired = float(value)
        if key == 'dt': dt_desired = float(value)
        if key == 'input': input_source_string = value

    if input_source_string is not None:
        input_source = input_source_string
        yield_packet_bytes_function = yield_from_shm_and_strip_logging_header
    else:
        input_source = sys.stdin.buffer
        yield_packet_bytes_function = yield_packet_bytes_from_log_stream
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    for packet in incoherent_cqt_frames_generator(bins_per_octave, (df_desired, dt_desired, f0_desired, fh_desired, yield_acoustic_packets, (yield_packet_bytes_function, input_source, phonemask))):

        C = packet.intensity.shape[0]

        chigh = 0
        cstep = 0.75
        clow = chigh - 256.0 * cstep

        bins_rounded = np.round((10.0 * np.log10(np.mean(packet.intensity, axis=0)) - clow) / cstep)
        print('{ "time": %u.%06u, "df": %.3f, "dt": %.3f, "bins_per_octave": %u, "pgram": "%s" } ' % (packet.timestamp_microseconds // 1000000, packet.timestamp_microseconds % 1000000, packet.df, packet.dt, packet.bins_per_octave, base64.b64encode(bins_rounded.astype(np.int8)).decode('utf-8')), flush=True)

main()
