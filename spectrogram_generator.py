#!/usr/bin/env python3
# this is a generic processing pipeline in the form of a series of nested generator
# functions, which accepts numpy arrays of blocks of samples from some sensor-specific
# upstream generator, and in turn yields incoherently averaged fft frames to its caller

import sys
import math
import numpy as np
from collections import namedtuple

# if numpy version is 1.x, rfft is double precision only, try to use scipy instead
rfft = np.fft.rfft
if rfft(np.ones(8, dtype=np.single)).dtype != np.complex64:
    try: import scipy
    except: print('using slower np.fft.rfft', file=sys.stderr)
    else: rfft = scipy.fft.rfft

def cmagsquared(x):
    return np.square(np.real(x)) + np.square(np.imag(x))

def round_to_nearest_five_smooth(x):
    # given x, returns nearest integer y, such that y has no prime factors larger than five
    y_above = 2 * math.ceil(x) if x > 0 else 1
    y_below = 1

    c = 1
    while c <= y_above:
        b = 1
        while c * b <= y_above:
            a = 1
            while c * b * a <= y_above:
                t = c * b * a
                if t >= x: y_above = t
                elif t > y_below: y_below = t
                a *= 2
            b *= 3
        c *= 5

    # choose whichever is logarithmically closer to x
    return y_above if y_above * y_below <= x * x else y_below

acoustic_block_tuple = namedtuple('acoustic_block_tuple', ('samples', 'timestamp', 'sample_rate', 'fullscale', 'timestamp_microseconds'))

def yield_acoustic_blocks(seconds_desired, yield_acoustic_packets_function, yield_acoustic_packets_arguments):
    child = yield_acoustic_packets_function(*yield_acoustic_packets_arguments)

    it_frame = 0
    it_packet = 0
    out = None

    packet = next(child, None)
    if packet is None: return

    C = packet.samples.shape[1]
    sample_rate = packet.fs

    if seconds_desired is None: seconds_desired = 1024 / packet.fs

    # if two blocks are used as an r2c fft input, the length should be a 5-smooth multiple of 32
    T_per_frame = round_to_nearest_five_smooth(sample_rate * seconds_desired / 16.0) * 16

    while True:
        if packet is None:
            packet = next(child, None)
            if packet is None: break

        if out is None:
            out = np.zeros((T_per_frame, C), dtype=packet.samples.dtype)

        T_packet = packet.samples.shape[0]
        T_copy = min(T_per_frame - it_frame, T_packet - it_packet)

        out[it_frame:(it_frame + T_copy), :] = packet.samples[it_packet:(it_packet + T_copy), :]

        it_frame = it_frame + T_copy
        it_packet = it_packet + T_copy

        if it_frame == T_per_frame:
            end_of_frame_microseconds = packet.timestamp_microseconds - round((T_packet - it_packet) * 1e6 / sample_rate)
            yield acoustic_block_tuple(samples=out, timestamp=packet.timestamp, sample_rate=sample_rate, fullscale=packet.fullscale, timestamp_microseconds=end_of_frame_microseconds)
            it_frame = 0
            out = None

        if it_packet == T_packet:
            packet = None
            it_packet = 0

fft_tuple = namedtuple('fft_tuple', ('bins', 'f0', 'df', 'dt', 'timestamp_microseconds'))

def overlapped_fft_frame_generator(df_desired, f0_desired, fh_desired, yield_acoustic_packets_function, yield_acoustic_packets_arguments):
    # start an upstream generator provided by the child
    child = yield_acoustic_blocks(None if df_desired is None else 0.5 / df_desired,
        yield_acoustic_packets_function, yield_acoustic_packets_arguments)

    acoustic_block = next(child, None)
    if acoustic_block is None: return

    Th, C = acoustic_block.samples.shape
    fs = acoustic_block.sample_rate

    T = 2 * Th
    print('fft length %u samples' % T, file=sys.stderr)

    df = fs / (2 * Th)
    dt = Th / fs

    iw_start = round(f0_desired / df)
    f0 = iw_start * df

    if fh_desired is not None:
        iw_stop = round(fh_desired / df)
    else:
        iw_stop = Th + 1
    F = iw_stop - iw_start

    # apply normalization as additional scalar multipliers here
    window = (np.hanning(2 * Th + 1) * 2)[0:(2 * Th)].astype(np.single) / (T * acoustic_block.fullscale)
    if np.complex64 != acoustic_block.samples.dtype: window *= math.sqrt(2)

    # if doing a c2c fft, modify the window such that it affects an fftshift for free
    if np.complex64 == acoustic_block.samples.dtype: window[1:-1:2] *= -1

    # explicit transpose is worth doing because we only have to do it once per framehalf,
    # as opposed to accessing the values in a suboptimal pattern twice per framehalf
    framehalf_this = np.transpose(acoustic_block.samples)

    # loop over consecutive pairs of framehalves, affecting the 50% overlap
    while True:
        framehalf_last = framehalf_this

        acoustic_block = next(child, None)
        if acoustic_block is None: break

        framehalf_this = np.transpose(acoustic_block.samples)

        # do the fft, keep only the first F bins
        out = np.ndarray(dtype=np.complex64, shape=[C, F])
        for ic in range(C):
            out[ic, :] = rfft(np.concatenate((framehalf_last[ic, :], framehalf_this[ic, :])) * window)[iw_start:iw_stop]
        yield fft_tuple(bins=out, f0=f0, df=df, dt=dt, timestamp_microseconds=acoustic_block.timestamp_microseconds)

def incoherent_fft_frame_generator(df_desired, dt_desired, f0_desired, fh_desired, yield_acoustic_packets_function, yield_acoustic_packets_arguments):
    # start a generator function which yields complete fft frames until eof on stdin
    child = overlapped_fft_frame_generator(df_desired, f0_desired, fh_desired, yield_acoustic_packets_function, yield_acoustic_packets_arguments)

    # get first frame from generator
    frame = next(child, None)
    if frame is None: return

    f0 = frame.f0
    df = frame.df
    dt_per_frame = frame.dt

    # number of consecutive fft frames to accumulate in each output pixel vs time
    incoherent_stacking = round(dt_desired / dt_per_frame) if dt_desired > dt_per_frame else 1

    print('stacking %u consecutive 50%% overlapped ffts for %g s spacing ' % (incoherent_stacking, dt_per_frame * incoherent_stacking), file=sys.stderr)

    istack = 0

    # main loop which runs until eof on stdin
    while frame is not None:
        if 0 == istack: acc = cmagsquared(frame.bins)
        else: acc += cmagsquared(frame.bins)

        istack = istack + 1
        if incoherent_stacking == istack:
            out = acc * (1.0 / incoherent_stacking)
            istack = 0
            yield fft_tuple(bins=out, f0=f0, df=df, dt=dt_per_frame * incoherent_stacking, timestamp_microseconds=frame.timestamp_microseconds)

        # loop on output from frame generator until it yields None
        frame = next(child, None)
