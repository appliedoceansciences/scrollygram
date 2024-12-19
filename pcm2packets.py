#!/usr/bin/env python3
# this is a quick and dirty script which ingests raw pcm in the expected format on stdin,
# and emits the same data with logging and acoustic packet headers prepended s.t. packets
# of not more than 1500 bytes are emitted, in the format expected by the dsp

import struct
import sys

def pcm2packets(src, input_dtype_string, C, sample_rate):
    itemsize = 2 if input_dtype_string == 'int16' else 4

    # number of samples per packet is maximum number s.t. packet size is not more than 1500 bytes
    T = (1500 - 16) // (itemsize * C)

    packet_size = 16 + itemsize * C * T
    packet_size_with_padding = (packet_size + 7) & ~7

    seqnum = 0
    flags = 0b01 if input_dtype_string == 'int32' else 0b11 if input_dtype_string == 'single' else 0b00

    samples_yielded = 0

    while True:
        data_segment_bytes = src.read(itemsize * C * T)
        if len(data_segment_bytes) < itemsize * C * T: break

        samples_yielded += T
        timestamp_ticks = round((t + samples_yielded / sample_rate) * 1e6 / 16) % 281474976710656
        timestamp_lsbs = (timestamp_ticks) & 65535
        timestamp_msbs = (timestamp_ticks) >> 16

        logging_header_bytes = struct.pack('<HHI',
            packet_size, timestamp_lsbs, timestamp_msbs)

        packet_header_bytes = struct.pack('<BBHfHHI',
            0x45, C, seqnum, sample_rate, flags, timestamp_lsbs, timestamp_msbs)

        # to ensure that subsequent packets are 8-byte-aligned with the stream as expected
        padding = b'\0\0\0\0\0\0\0\0'[0:(packet_size_with_padding - packet_size)]

        yield logging_header_bytes, packet_header_bytes + data_segment_bytes, padding

        seqnum = (seqnum + 1) % 65536

input_dtype_string = 'single'
sample_rate = 24000.0
C = 20
t = 1725898437.0

# loop over pairs of arguments
for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
    if key == 'fs': sample_rate = float(value)
    if key == 'C': C = int(value)
    if key == 'dtype': input_dtype_string = value
    if key == 't0': t = float(value)

for logging_header_bytes, packet_bytes, padding in pcm2packets(sys.stdin.buffer, input_dtype_string, C, sample_rate):
    sys.stdout.buffer.write(logging_header_bytes + packet_bytes + padding)
    sys.stdout.buffer.flush()
