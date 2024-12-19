#!/usr/bin/env python3

import sys
import time
import struct
import numpy as np

from generate_pcm import generate_blocks_of_dithered_tone

def generate_packets_with_dithered_tone(duration, throttle, t, C, tone_frequency, tone_amplitude, tpdf_amplitude, fs, dtype):
    itemsize = 2 if np.int16 == dtype else 4

    # number of samples per packet is maximum number s.t. packet size is not more than 1500 bytes
    T_per_block = (1500 - 16) // (itemsize * C)

    print('T_per_block = %u' % T_per_block, file=sys.stderr)

    blocks_to_yield = int(round(duration * fs)) // T_per_block
    if duration > 0 and blocks_to_yield == 0: blocks_to_yield = 1

    t0 = time.monotonic()
    time_per_block = T_per_block / fs

    child = generate_blocks_of_dithered_tone(T_per_block, C, tone_frequency, tone_amplitude, tpdf_amplitude, fs, dtype)

    packet_size = 16 + itemsize * C * T_per_block
    packet_size_with_padding = (packet_size + 7) & ~7

    seqnum = 0
    flags = 0b01 if np.int32 == dtype else 0b11 if np.single == dtype else 0b00

    samples_yielded = 0
    iblock = 0
    while iblock < blocks_to_yield or not blocks_to_yield:
        if throttle:
            now = time.monotonic()
            if now < iblock * time_per_block / throttle + t0:
                time.sleep((iblock * time_per_block / throttle + t0) - now)

        samples = next(child)

        samples_yielded += T_per_block
        timestamp_ticks = round((t + samples_yielded / fs) * 1e6 / 16) % 281474976710656
        timestamp_lsbs = (timestamp_ticks) & 65535
        timestamp_msbs = (timestamp_ticks) >> 16

        logging_header_bytes = struct.pack('<HHI',
            packet_size, timestamp_lsbs, timestamp_msbs)

        packet_header_bytes = struct.pack('<BBHfHHI',
            0x45, C, seqnum, fs, flags, timestamp_lsbs, timestamp_msbs)

        # to ensure that subsequent packets are 8-byte-aligned with the stream as expected
        padding = b'\0\0\0\0\0\0\0\0'[0:(packet_size_with_padding - packet_size)]

        packet_bytes = packet_header_bytes + samples.tobytes()

        seqnum = (seqnum + 1) % 65536

        yield logging_header_bytes, packet_bytes, padding

        iblock += 1

if __name__ == '__main__':
    fs = 8000
    C = 1
    dtype = np.single

    tone_frequency = 440

    tone_amplitude = 32766.0 / 32767.0
    tpdf_amplitude = 1.0 / 32767.0

    throttle = 0
    duration = 0

    t = 1725898437.0

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
        if key == 't0': t = float(value)

    child = generate_packets_with_dithered_tone(duration, throttle, t, C, tone_frequency, tone_amplitude, tpdf_amplitude, fs, dtype)

    for logging_header_bytes, packet_bytes, padding in child:
        sys.stdout.buffer.write(logging_header_bytes + packet_bytes + padding)
        sys.stdout.buffer.flush()
