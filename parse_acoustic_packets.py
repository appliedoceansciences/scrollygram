#!/usr/bin/env python3
# this file provides a parser that understands the packetized element data collected in real
# time, and is used as part of realtime scrolling spectrogram drawing, but can also be used
# to convert previously logged packetized element data to raw pcm, such that ffmpeg or other
# software can be used to convert it to wav files or for other purposes

import struct
import sys
import numpy as np
from datetime import datetime
from collections import namedtuple

packet_tuple = namedtuple('packet_tuple', ('samples', 'timestamp', 'fs', 'seqnum', 'fullscale'))

# this function can also be imported and used on its own to parse a packet received via udp
def parse_acoustic_packet(packet_bytes):
    # if packet size is too small to be an acoustic packet, skip it
    if len(packet_bytes) < 16: return None

    # parse the packet header
    magic, channels, seqnum, sample_rate, flags, timestamp_lsbs, timestamp_msbs = struct.unpack('<BBHfHHI', packet_bytes[0:16])

    # validate packet header, part one
    if magic != 0x45: return None

    # interpret lowest two bits of flags field as the data type
    dtype = 'h' if (flags & 0x3 == 0) else 'i' if (flags & 0x3 == 1) else 'f' if (flags & 0x3) else 'int24'

    sizeof_sample = 2 if (dtype == 'h') else 4 if (dtype == 'i') else 3 if (dtype == 'int24') else 4
    samples_per_channel_per_packet = (len(packet_bytes) - 16) // (channels * sizeof_sample)

    # validate packet header, part two
    if samples_per_channel_per_packet * sizeof_sample * channels + 16 != len(packet_bytes): return None

    # TODO: numpy does not have a native 24 bit data type so gotta do more work
    if dtype == 'int24':
        raise RuntimeError('24-bit sample reassembly not implemented yet')

    # interpret the data segment as a numpy array with the given shape and dtype
    samples = np.ndarray((samples_per_channel_per_packet, channels), buffer=packet_bytes[16:], dtype=dtype)

    fullscale = 32767.0 if (flags & 0x3 == 0) else 2147483647.0 if (flags & 0x3 == 1) else 1.0 if (flags & 0x3) else 8388607.0

    # reassemble timestamp in unix seconds
    timestamp_unix_seconds = ((timestamp_msbs << 16) | timestamp_lsbs) * 16.0 / 1e6
    return packet_tuple(samples=samples, timestamp=timestamp_unix_seconds, fs=sample_rate, seqnum=seqnum, fullscale=fullscale)

# this generator function can be specified as one of the possible upstream sources of
# blocks of bytes used as input to the below generator, and handles all of the details of
# the on-disk (and in-tcp-stream) logging format used for serialized packets
def yield_packet_bytes_from_log_stream(source):
    while True:
        # read eight bytes, or break out of loop if eof
        logging_header_bytes = source.read(8)
        if len(logging_header_bytes) == 0: break

        # interpret these eight bytes as a 16-bit packet size and 48-bit timestamp
        packet_size, timestamp_lsbs, timestamp_msbs = struct.unpack('<HHI', logging_header_bytes)

        # round packet size up to next eight bytes
        packet_size_with_padding = (packet_size + 7) & ~7

        # read all the remaining bytes
        packet_bytes = source.read(packet_size_with_padding)

        # if padding bytes were ingested, discard them before parsing the rest
        if packet_size_with_padding != packet_size:
            packet_bytes = packet_bytes[0:packet_size]

        yield packet_bytes

def datestr_from_unix_time(time_in_unix_seconds):
    microseconds = round(time_in_unix_seconds * 1e6)
    integer_portion = microseconds // 1000000
    remainder = microseconds % 1000000
    return '%s.%06uZ' % (datetime.utcfromtimestamp(integer_portion).strftime('%Y%m%dT%H%M%S'), remainder)

# this generator function can be imported and used by calling code to ingest from log files or udp
def yield_acoustic_packets(yield_packet_bytes_function, source, phonemask):
    seqnum_prev = None
    initial_timestamp = None
    timestamp_prev = None
    samples_yielded = 0
    sample_rate = 0

    for packet_bytes in yield_packet_bytes_function(source):
        # attempt to parse the packet bytes as an acoustic packet
        packet = parse_acoustic_packet(packet_bytes)
        if not packet: continue

        # emit some diagnostic text on the first packet
        if seqnum_prev is None:
            sample_rate = packet.fs
            print('%u channels, %g sps, %u samples per channel per packet' %
                  (packet.samples.shape[1], sample_rate, packet.samples.shape[0]), file=sys.stderr)
            initial_timestamp = packet.timestamp - packet.samples.shape[0] / sample_rate
            print('first packet timestamp %s, implied start time %s ' %
                (datestr_from_unix_time(packet.timestamp),
                datestr_from_unix_time(initial_timestamp)), file=sys.stderr)
        else:
            packets_missing = (packet.seqnum - seqnum_prev - 1) % 65536
            if packets_missing != 0:
                print('warning: expected seqnum %u, got %u, missing %u packets (%g s)' % ((seqnum_prev + 1) % 65536, packet.seqnum, packets_missing, packets_missing * packet.samples.shape[0] / sample_rate), file=sys.stderr)
        seqnum_prev = packet.seqnum
        timestamp_prev = packet.timestamp
        samples_yielded += packet.samples.shape[0]

        if phonemask is not None:
            packet = packet_tuple(samples=np.take(packet.samples, phonemask, axis=1), timestamp=packet.timestamp, fs=packet.fs, seqnum=packet.seqnum, fullscale=packet.fullscale)

        yield packet

    print('incoming data has ended', file=sys.stderr)
    if timestamp_prev is not None:
        print('final packet ends at %s' % (datestr_from_unix_time(timestamp_prev)), file=sys.stderr)
        print('yielded %g seconds according to expected sample rate, %g seconds according to packet timestamps' %
              ( samples_yielded / sample_rate, timestamp_prev - initial_timestamp), file=sys.stderr)

# if running this as a standalone process rather than importing as a module...
if __name__ == '__main__':
    def main():
        phonemask = None

        # loop over pairs of arguments
        for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
            if key == 'phonemask': phonemask = list(map(int, value.split(',')))

        # loop on output of above generator function, dumping samples to stdout as raw pcm
        for packet in yield_acoustic_packets(yield_packet_bytes_from_log_stream, sys.stdin.buffer, phonemask=phonemask):
            sys.stdout.buffer.write(packet.samples)

    main()
