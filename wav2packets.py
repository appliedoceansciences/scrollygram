#!/usr/bin/env python3
# convenience wrapper that reads wav headers and passes the arguments to pcm2packets

import struct
import sys

from pcm2packets import pcm2packets

input_dtype_string = 'int16'
fs = 31250.0
C = 1
t0 = 1725898437.0

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 't0': t0 = float(value)

riff, _, wave = struct.unpack('4sI4s', sys.stdin.buffer.read(12))
if riff != b'RIFF' or wave != b'WAVE': exit(1)

# consume wav header and read dtype, channel count, sample rate from it
while True:
    section, section_length = struct.unpack('4sI', sys.stdin.buffer.read(8))
    if b'fmt ' == section and 16 == section_length:
        dtype_int, C, fs, _, _, bits_per_sample = struct.unpack('HHIIHH', sys.stdin.buffer.read(16))
        section_length -= 16

        if 3 == dtype_int:
            input_dtype_string = 'single'
        elif 1 == dtype_int and 32 == bits_per_sample:
            input_dtype_string = 'int32'
        elif 1 == dtype_int and 16 == bits_per_sample:
            input_dtype_string = 'int16'
        else:
            raise RuntimeError('unhandled wav format')

    if b'data' == section: break
    sys.stdin.buffer.read(section_length)

for logging_header_bytes, packet_bytes, padding in pcm2packets(sys.stdin.buffer, input_dtype_string, C, fs, t0):
    sys.stdout.buffer.write(logging_header_bytes + packet_bytes + padding)
    sys.stdout.buffer.flush()
