#!/usr/bin/env python3

import sys
import numpy as np

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

advance = np.exp(1j * 2.0 * np.pi * tone_frequency / fs)
carrier = 1.0

iblock = 0
while iblock < blocks_to_yield or not blocks_to_yield:
    carriers = np.empty((T_per_block,), dtype=np.complex64)
    carriers[:] = advance
    carriers[0] *= carrier
    carriers = np.cumprod(carriers)

    carrier = carriers[-1]
    carrier /= np.abs(carrier)

    if 0 == iblock:
        print(np.angle(carriers[0]), file=sys.stderr)

    samples = tone_amplitude * carriers.real

    # if more than one channel, repeat each sample of the scaled carrier C times
    if C > 1:
        samples = np.tile(samples[:, None], (1, C))

    # add dither which is unique to each channel
    samples += tpdf_amplitude * np.random.triangular(-1,0,1, size=(T_per_block, C))

    if np.int16 == dtype:
        samples = np.round(samples * 32767.0)
    elif np.int32 == dtype:
        samples = np.round(samples * 2147483647.0)

    sys.stdout.buffer.write(samples.astype(dtype))

    iblock += 1
