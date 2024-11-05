#!/usr/bin/env python3

import struct
import sys
import numpy as np

input_dtype = np.dtype('int32')
sample_rate = 8000.0
C = 4
t = 1725898437.0

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'fs': sample_rate = float(value)
    if key == 'C': C = int(value)
    if key == 'dtype': input_dtype = np.dtype(value)
    if key == 't0': t = float(value)

# number of samples per packet is maximum number s.t. packet size is not more than 1500 bytes
T = (1500 - 16) // (input_dtype.itemsize * C)

packet_size = 16 + input_dtype.itemsize * C * T
packet_size_with_padding = (packet_size + 7) & ~7

seqnum = 0
flags = 0b01 if input_dtype == np.int32 else 0b11 if input_dtype == np.float32 else 0b00

samples_yielded = 0

while True:
    bytes = sys.stdin.buffer.read(input_dtype.itemsize * C * T)
    if len(bytes) < input_dtype.itemsize * C * T: break

    samples = np.ndarray(buffer=bytes, dtype=input_dtype, shape=[T, C])

    samples_yielded += T
    timestamp_ticks = round(samples_yielded * 1e6 / (sample_rate * 16)) % 281474976710656
    timestamp_lsbs = (timestamp_ticks) & 65535
    timestamp_msbs = (timestamp_ticks) >> 16

    logging_header_bytes = struct.pack('<HHI',
        packet_size, timestamp_lsbs, timestamp_msbs)
    sys.stdout.buffer.write(logging_header_bytes)

    packet_header_bytes = struct.pack('<BBHfHHI',
        0x45, C, seqnum, sample_rate, flags, timestamp_lsbs, timestamp_msbs)

    sys.stdout.buffer.write(packet_header_bytes)

    sys.stdout.buffer.write(samples)
    sys.stdout.buffer.flush()

    seqnum = (seqnum + 1) % 65536
