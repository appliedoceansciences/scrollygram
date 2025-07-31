#!/usr/bin/env python3
import sys
import threading
import queue
import socket
import numpy as np
from collections import namedtuple

# This provides the generator function which knows how to extract sensor-agnostic frames of
# acoustic sample data from whatever possibly sensor-specific format they are coming from
from parse_acoustic_packets import yield_acoustic_packets, yield_packet_bytes_from_log_stream

import matplotlib
#matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.ticker as ticker

# if text gets piled on top of other text, try messing with this logic. the same settings do
# not seem to give satisfactory results on all combinations of OS and screen dpi. if someone
# knows what to do here that does the right thing unconditionally lmk
# if matplotlib.get_backend() != 'MacOSX': matplotlib.rcParams['figure.dpi'] = 300

import matplotlib.pyplot as plt

def packet_concatenator(nominal_length, yield_packet_bytes_function, source, phonemask):
    out = None
    out_prior = None

    child = yield_acoustic_packets(yield_packet_bytes_function, source, phonemask)
    packet = next(child, None)
    if not packet: return

    T_to_yield = int(nominal_length * packet.fs)

    while packet is not None:
        out = packet.samples if out is None else np.concatenate((out, packet.samples), axis=0)

        if out.shape[0] >= T_to_yield:
            if out_prior is not None:
                ic_trigger = np.argmax(np.var(out_prior, axis=0))
                full = np.concatenate((out_prior, out), axis=0)
                it_trim = np.argmax(np.diff(out_prior[:, ic_trigger]))
                subset = full[it_trim:(it_trim + T_to_yield), :]

                concatted_tuple = namedtuple('concatted_tuple', ('samples', 'timestamp', 'fs', 'fullscale'))
                yield concatted_tuple(samples=subset, timestamp=packet.timestamp, fs=packet.fs, fullscale=packet.fullscale)

            out_prior = out
            out = None

        packet = next(child, None)

window_closed = False
def on_close(event):
    global window_closed
    window_closed = True

# turns a generator into a child thread which yields functions and arguments to main thread
def child_thread(main_thread_work, nominal_length, yield_packet_bytes_function, source, phonemask):
    for packet in packet_concatenator(nominal_length, yield_packet_bytes_function, source, phonemask):
        if window_closed: break
        main_thread_work.put(packet)

    # inform main thread that child generator has reached eof and no more input is coming
    main_thread_work.put(None)

def yield_packet_bytes_from_udp(source):
    while True:
        yield source.recvfrom(1500)[0]

def get_input_source(input_source_string):
    if input_source_string is not None:
        if 'shm' in input_source_string:
            try:
                from shared_memory_ringbuffer_reader import shared_memory_ringbuffer_generator
            except:
                raise RuntimeError('shared memory input not supported')

            # hack to peel off logging headers
            def yield_from_shm_and_strip_logging_header(source):
                for packet_with_logging_header in shared_memory_ringbuffer_generator(source):
                    yield packet_with_logging_header[8:]

            input_source = input_source_string.split(':')[1] if 'shm:' in input_source_string else '/cobs_to_shm'
            yield_packet_bytes_function = yield_from_shm_and_strip_logging_header
        elif ':' in input_source_string:
            address, port = input_source_string.split(':')
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((address, int(port)))
            input_source = sock.makefile('rb')
            yield_packet_bytes_function = yield_packet_bytes_from_log_stream
            print('connected to %s:%u via tcp' % (address, int(port)), file=sys.stderr)
        else:
            input_source = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            input_source.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF, 4194304) #set the udp recv buffer to 4mb
            input_source.bind(('', int(input_source_string)))
            yield_packet_bytes_function = yield_packet_bytes_from_udp
            print('listening for udp on port %u' % int(input_source_string), file=sys.stderr)
    else:
        if sys.stdin.isatty():
            print('for shm input: %s shm[:/shm_path]' % sys.argv[0], file=sys.stderr)
            print('for tcp input: %s host:port' % sys.argv[0], file=sys.stderr)
            print('for udp input: %s port' % sys.argv[0], file=sys.stderr)
            print('for stdin input: cat whatever*.bin | %s' % sys.argv[0], file=sys.stderr)
            exit(1)
        input_source = sys.stdin.buffer
        yield_packet_bytes_function = yield_packet_bytes_from_log_stream
    return input_source, yield_packet_bytes_function


def main():
    phonemask = None
    plotdata = None
    C = 0
    ax = None
    lines = None
    bg = None
    xdata = None
    nominal_length = 0.03

    # support legacy method of specifying input as a single argument
    input_source_string = sys.argv[1] if 2 == len(sys.argv) else None

    # loop over pairs of arguments
    for key, value in zip(sys.argv[1::2], sys.argv[2::2]):
        if key == 'phonemask': phonemask = list(map(int, value.split(',')))
        if key == 'input': input_source_string = value

    input_source, yield_packet_bytes_function = get_input_source(input_source_string)

    # create an empty figure but don't show it yet
    fig = plt.figure()

    fig.canvas.mpl_connect('close_event', on_close)

    # thread-safe fifo between rx thread and main thread
    main_thread_work = queue.Queue()

    # start a child thread which accept output yielded from one of two possible generators
    # depending on whether stdin is a tty, and safely communicate that generator output
    # and what to do with it back to the main thread via the work queue
    pth = threading.Thread(target=child_thread, args=(main_thread_work, nominal_length, yield_packet_bytes_function, input_source, phonemask))
    pth.start()

    # event loop which dequeues work from other threads that must be done on main thread
    while True:
        try:
            packet = main_thread_work.get(timeout=0.016)
        except queue.Empty:
            fig.canvas.flush_events()
            continue
        if packet is None: break

        # do this setup stuff on the first input
        if not ax:
            C = packet.samples.shape[1]
            xdata = np.arange(0, packet.samples.shape[0]) / packet.fs
            ax = fig.add_subplot(1, 1, 1)

            lines = ax.plot(xdata, packet.samples)

            ax.set_ylim(ymin=-packet.fullscale, ymax=packet.fullscale)
            ax.set(title='data')

            # label the y axis for the subplots on the left side
            ax.set(xlabel='Time (s)')

            # label the x axis for the subplots on the bottom
            ax.set(ylabel='Amplitude')

            fig.tight_layout(pad=1.5)
            fig.canvas.draw()
            fig.show()
            bg = fig.canvas.copy_from_bbox(fig.bbox)

        if main_thread_work.empty():
            for ic in range(C):
                lines[ic].set_ydata(packet.samples[:, ic])
            fig.canvas.draw()
            fig.canvas.flush_events()

    # if we get here, we got to eof on stdin
    pth.join()

main()
