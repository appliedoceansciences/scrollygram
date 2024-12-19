#!/usr/bin/env python3

import sys
import time
import numpy as np

def generate_blocks_of_dithered_tone(T_per_block, C, tone_frequency, tone_amplitude, tpdf_amplitude, fs, dtype):
    advance = np.exp(1j * 2.0 * np.pi * tone_frequency / fs)
    carrier = 1.0

    while True:
        carriers = np.empty((T_per_block,), dtype=np.complex64)
        carriers[:] = advance
        carriers[0] *= carrier
        carriers = np.cumprod(carriers)

        carrier = carriers[-1]
        carrier /= np.abs(carrier)

        samples = tone_amplitude * carriers.real

        # if more than one channel, repeat each sample of the scaled carrier C times
        if C > 1:
            samples = np.tile(samples[:, None], (1, C))
        else:
            samples.shape = (T_per_block, 1)

        # add dither which is unique to each channel
        samples += tpdf_amplitude * np.random.triangular(-1,0,1, size=(T_per_block, C))

        if np.int16 == dtype:
            samples = np.round(samples * 32767.0)
        elif np.int32 == dtype:
            samples = np.round(samples * 2147483647.0)

        yield samples.astype(dtype)

fs = 8000
C = 1
dtype = np.single

tone_frequency = 440

tone_amplitude = 32766.0 / 32767.0
tpdf_amplitude = 1.0 / 32767.0

throttle = 0
duration = 0

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'fs': fs = float(value)
    if key == 'C': C = int(value)
    if key == 'tone_frequency': tone_frequency = float(value)
    if key == 'tone_amplitude': tone_amplitude = float(value)
    if key == 'tpdf_amplitude': tone_amplitude = float(value)
    if key == 'throttle': throttle = float(value)
    if key == 'duration': duration = float(value)
    if key == 'dtype': dtype = np.dtype(value)

T_per_block = 1024

T = int(round(duration * fs))
blocks_to_yield = T // T_per_block
if duration > 0 and blocks_to_yield == 0: blocks_to_yield = 1

t0 = time.monotonic()
time_per_block = T_per_block / fs

child = generate_blocks_of_dithered_tone(T_per_block, C, tone_frequency, tone_amplitude, tpdf_amplitude, fs, dtype)

iblock = 0
while iblock < blocks_to_yield or not blocks_to_yield:
    if throttle:
        now = time.monotonic()
        if now < iblock * time_per_block / throttle + t0:
            time.sleep((iblock * time_per_block / throttle + t0) - now)

    samples = next(child)

    sys.stdout.buffer.write(samples.astype(dtype))
    sys.stdout.buffer.flush()

    iblock += 1
