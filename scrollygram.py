#!/usr/bin/env python3
# This is a quick and dirty, use-case-agnostic realtime scrolling spectrogram plotter for
# multi-channel raw acoustic data. This file only implements the parts of the pipeline which
# render the incoming rows of frequency bins, i.e. this file is the downstream end of a
# pipeline which has raw per-channel acoustic timeseries coming into the upstream end, and
# which eventually yields FFT'd, incoherently averaged frames of frequency bins, as they
# become ready, to the plotter in this file.

import socket
import sys
import threading
import queue
import math

# This provides the generator function which knows how to extract sensor-agnostic frames of
# acoustic sample data from whatever possibly sensor-specific format they are coming from
from parse_acoustic_packets import yield_acoustic_packets, yield_packet_bytes_from_log_stream

# This provides the generator function which accepts the above sequence of sensor-agnostic
# numpy arrays of acoustic samples, assembles them into 50%-overlapped frames according to
# a desired frequency resolution, does the FFT on each frame as it becomes ready,
# incoherently averages these for a desired time resolution, and yields these to the plotter
from spectrogram_generator import incoherent_fft_frame_generator

import numpy as np

import matplotlib
matplotlib.rcParams['toolbar'] = 'None'

# if text gets piled on top of other text, try messing with this logic. the same settings do
# not seem to give satisfactory results on all combinations of OS and screen dpi. if someone
# knows what to do here that does the right thing unconditionally lmk
# if matplotlib.get_backend() != 'MacOSX': matplotlib.rcParams['figure.dpi'] = 300

import matplotlib.pyplot as plt

to_rgba_func = matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps['turbo']).to_rgba

def round_up_to_next_multiple_of(a, q):
    return a + q - a % q if a % q else a

# turns a generator into a child thread which yields functions and arguments to main thread
def child_thread(main_thread_work, incoherent_fft_frame_generator_arguments):
    for packet in incoherent_fft_frame_generator(*incoherent_fft_frame_generator_arguments):
        main_thread_work.put(packet)

    # inform main thread that child generator has reached eof and no more input is coming
    main_thread_work.put(None)

def yield_packet_bytes_from_udp(source):
    while True:
        yield source.recvfrom(1500)[0]

def main():
    # constants you might want to fiddle with. TODO: allow main() to modify these
    clim=(-30, 60)
    phonemask = None
    #phonemask = (0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15)
    df_desired = 20.3451
    dt_desired = 0.0

    # thread-safe fifo between rx thread and main thread
    main_thread_work = queue.Queue()

    # global variables with deferred initialization
    nrows = 0
    ncols = 0
    plotdata = None
    C = 0
    X = 0
    Y = 0
    fig = None
    axes = []
    ims = None
    iy = 0

    if len(sys.argv) > 1:
        if 'shm:' in sys.argv[1]:
            try:
                from shared_memory_ringbuffer_reader import shared_memory_ringbuffer_generator
            except:
                raise RuntimeError('shared memory input not supported')

            # hack to peel off logging headers
            def yield_from_shm_and_strip_logging_header(source):
                for packet_with_logging_header in shared_memory_ringbuffer_generator(source):
                    yield packet_with_logging_header[8:]

            input_source = sys.argv[1].split(':')[1]
            yield_packet_bytes_function = yield_from_shm_and_strip_logging_header
        elif ':' in sys.argv[1]:
            address, port = sys.argv[1].split(':')
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((address, int(port)))
            input_source = sock.makefile('rb')
            yield_packet_bytes_function = yield_packet_bytes_from_log_stream
            print('connected to %s:%u via tcp' % (address, int(port)), file=sys.stderr)
        else:
            input_source = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            input_source.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF, 4194304) #set the udp recv buffer to 4mb
            input_source.bind(('', int(sys.argv[1])))
            yield_packet_bytes_function = yield_packet_bytes_from_udp
            print('listening for udp on port %u' % int(sys.argv[1]), file=sys.stderr)
    else:
        print('listening for input on stdin. if udp input is desired, specify a port number to listen on. if tcp is desired, specify and address:port to connect to', file=sys.stderr)
        input_source = sys.stdin.buffer
        yield_packet_bytes_function = yield_packet_bytes_from_log_stream

    # create an empty figure but don't show it yet
    fig = plt.figure()

    # start a child thread which accepts output yielded from one of several possible generators
    # depending on whether stdin is a tty, and safely communicate that generator output
    # and what to do with it back to the main thread via the work queue
    pth = threading.Thread(target=child_thread, args=(main_thread_work, (df_desired, dt_desired, yield_acoustic_packets, (yield_packet_bytes_function, input_source, phonemask))))
    pth.start()

    # event loop which dequeues work from other threads that must be done on main thread
    while True:
        if main_thread_work.empty():
            # there must be a better way to do this
            fig.canvas.start_event_loop(0.016)
            continue
        packet = main_thread_work.get()
        if packet is None: break

        bins_intensity = packet.bins
        f0 = packet.f0
        df = packet.df
        dt = packet.dt

        C = bins_intensity.shape[0]

        # do this setup stuff on the first call
        if not X:
            X = bins_intensity.shape[1]
            Y = (3 * X) // 4 # plot will have an aspect ratio of 3/4
            plotdata = np.zeros([C, 2 * Y, X, 4], dtype=np.uint8)

            # TODO: figure out a more intelligent way to do this
            nrows = 1 if C <= 2 else 2 if C <= 8 else 3 if C <= 12 else 4
            ncols = round_up_to_next_multiple_of(C, nrows) // nrows

            ims = [None for i in range(C)]
            for ichannel in range(0, C):
                axes.append(fig.add_subplot(nrows, ncols, ichannel + 1))
                ax = axes[ichannel]

                ff = f0 + X * df
                xextent = [f0 / 1e3, ff / 1e3]
                yextent = [0, dt * Y]

                ims[ichannel] = ax.imshow(plotdata[ichannel, 0:Y, :, :],
                    origin='lower',
                    extent=[xextent[0], xextent[1], yextent[0], yextent[1]],
                    aspect=(((xextent[1] - xextent[0]) * Y) / ((yextent[1] - yextent[0]) * X)), animated=True)
                ax.set(title='channel ' + str(ichannel))

                # label the y axis for the subplots on the left side
                if (ichannel % ncols) == 0: ax.set(ylabel='Time (s) in past')

                # label the x axis for the subplots on the bottom
                if ((ichannel // ncols) % nrows) == nrows - 1: ax.set(xlabel='Frequency (kHz)')

            # add padding to the figure -- less when the number of channles is larger
            if C <= 8:
                fig.tight_layout(pad=1.5)
            elif (C > 8) & (C <= 12):
                fig.tight_layout(pad=1, w_pad=0.7, h_pad=0.7)
            else:
                fig.tight_layout(pad=0.5, w_pad=0.3, h_pad=0.3)

            fig.show()
            fig.canvas.blit(fig.bbox)
            fig.canvas.draw()

        # if not the first call, sanity check that X has not changed
        elif bins_intensity.shape[1] != X:
            raise RuntimeError('consecutive packets have different numbers of bins (%u != %u)' % bins_intensity.shape[1], X)

        # convert the values in intensit for the new row of pixels from all channels to rgba values
        bins_rgba = to_rgba_func(np.clip((10.0 * np.log10(bins_intensity + 2e-38) - clim[0]) / (clim[1] - clim[0]), 0, 1), bytes=True, norm=False)

        # insert the new row of pixels into two places within the doubled ring buffer, so that a
        # contiguous slice of it can always be plotted, ending at the most recent row
        plotdata[:, iy + 0, :, :] = bins_rgba
        plotdata[:, iy + Y, :, :] = bins_rgba

        # only redraw the screen if we know there are no more updates in the queue
        if main_thread_work.empty():
            for ichannel in range(0, C):
                # update which subset of the doubled ring buffer will be shown
                ims[ichannel].set_data(plotdata[ichannel, iy:(iy + Y), :])

                ax = axes[ichannel]
                ax.draw_artist(ims[ichannel])
                fig.canvas.blit(ax.bbox)

            fig.canvas.flush_events()

        # advance the ring buffer cursor (decrements w/ wraparound, as newest time is at bottom)
        iy = (iy + Y - 1) % Y

    # if we get here, we got to eof on stdin
    pth.join()

main()
