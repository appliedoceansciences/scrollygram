#!/usr/bin/env python3
# this is a quick and dirty script which ingests raw pcm in the expected format on stdin,
# and emits the same data with logging and acoustic packet headers prepended s.t. packets
# of not more than 1472 bytes are emitted, in the format expected by the dsp

import struct
import sys
import argparse

def pcm2packets(src, input_dtype_string, C, sample_rate, t0):
    itemsize = 2 if input_dtype_string == 'int16' else 4

    # number of samples per packet is maximum number s.t. packet size is not more than 1472 bytes
    T = (1472 - 16) // (itemsize * C)

    packet_size = 16 + itemsize * C * T
    packet_size_with_padding = (packet_size + 7) & ~7

    seqnum = 0
    flags = 0b01 if input_dtype_string == 'int32' else 0b11 if input_dtype_string == 'single' else 0b00

    samples_yielded = 0

    # convert floating point absolute time in unix seconds to integer number of 16-us ticks
    t0_ticks = round(t0 * 1e6 / 16)

    # number of 16-microsecond ticks
    ticks_per_sample = 1e6 / (sample_rate * 16)

    while True:
        data_segment_bytes = src.read(itemsize * C * T)
        if len(data_segment_bytes) < itemsize * C * T: break

        samples_yielded += T

        # construct the unix time as a 48-bit number of 16-microsecond ticks in unix time
        timestamp_ticks = (t0_ticks + round(samples_yielded * ticks_per_sample)) % (1 << 48)

        # break the 48-bit time down into 16 lsbs and 32 msbs
        timestamp_lsbs = timestamp_ticks & 65535
        timestamp_msbs = timestamp_ticks >> 16

        logging_header_bytes = struct.pack('<HHI',
            packet_size, timestamp_lsbs, timestamp_msbs)

        packet_header_bytes = struct.pack('<BBHfHHI',
            0x45, C, seqnum, sample_rate, flags, timestamp_lsbs, timestamp_msbs)

        # to ensure that subsequent packets are 8-byte-aligned with the stream as expected
        padding = b'\0\0\0\0\0\0\0\0'[0:(packet_size_with_padding - packet_size)]

        yield logging_header_bytes, packet_header_bytes + data_segment_bytes, padding

        seqnum = (seqnum + 1) % 65536

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('params', nargs='?', default=None, help='JSON file from which to read parameteres')
    parser.add_argument('--C', default=1, type=int, help='Number of channels')
    parser.add_argument('--fs', default=31250.0, type=float, help='Sample rate')
    parser.add_argument('--dtype', default='int16', help='Data type of the samples')
    parser.add_argument('--t0', default=1725898437.0, type=float, help='Initial timestamp')

    a = parser.parse_args()
    C, sample_rate, input_dtype_string, t0 = a.C, a.fs, a.dtype, a.t0

    if a.params is not None:
        import json
        with open(a.params) as f: params = json.load(f)
        if 'positions' in params: C = len(params['positions'])
        if 'sample_rate' in params: sample_rate = float(params['sample_rate'])
        if 'dtype' in params: input_dtype_string = params['dtype']

    for logging_header_bytes, packet_bytes, padding in pcm2packets(sys.stdin.buffer, input_dtype_string, C, sample_rate, t0):
        sys.stdout.buffer.write(logging_header_bytes + packet_bytes + padding)
        sys.stdout.buffer.flush()

main()
