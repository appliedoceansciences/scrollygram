#!/usr/bin/env python3
# this is a quick and dirty script which ingests raw pcm in the expected format on stdin,
# and emits the same data with logging and acoustic packet headers prepended s.t. packets
# of not more than 1500 bytes are emitted, in the format expected by the dsp

import struct
import sys

def pcm2packets(src, input_dtype_string, C, sample_rate, t0):
    itemsize = 2 if input_dtype_string == 'int16' else 4

    # number of samples per packet is maximum number s.t. packet size is not more than 1500 bytes
    T = (1500 - 16) // (itemsize * C)

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
    input_dtype_string = 'int16'
    sample_rate = 31250.0
    C = 1
    t0 = 1725898437.0

    # loop over pairs of arguments
    for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
        if key == 'fs': sample_rate = float(value)
        if key == 'C': C = int(value)
        if key == 'dtype': input_dtype_string = value
        if key == 't0': t0 = float(value)

    for logging_header_bytes, packet_bytes, padding in pcm2packets(sys.stdin.buffer, input_dtype_string, C, sample_rate, t0):
        sys.stdout.buffer.write(logging_header_bytes + packet_bytes + padding)
        sys.stdout.buffer.flush()

main()
