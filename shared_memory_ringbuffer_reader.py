#!/usr/bin/env python3
# for context, there is a C struct in a shared memory segment called "/shm", containing
# an unsigned long writer_cursor after two size_t's, some possible padding for 16-byte
# alignment, and then a ring buffer with extra space past the end, such that variable-size
# writes to the ring buffer of less than some maximum size can be written and read
# contiguously. each slot has a size_t and some padding for 16-byte alignment as a prefix

# this is more or less a direct port of the equivalent C code, including copying the API,
# and accordingly does not use oop stuff

import os
import mmap
import struct
import sys
import time
from types import SimpleNamespace
from _posixshmem import shm_open

def pid_is_still_alive(pid):
    try: os.kill(pid, 0)
    except: return False
    return True

def shared_memory_ringbuffer_reader_init(name):
    writer_cursor_offset = struct.calcsize('NN')
    writer_cursor_size = struct.calcsize('L')

    while True:
        try: fd = shm_open(name, os.O_RDONLY, 0)
        except FileNotFoundError: return None

        m = mmap.mmap(fd, os.fstat(fd).st_size, prot=mmap.PROT_READ)
        os.close(fd)
        view = memoryview(m)

        # initial value of reader cursor is the first value of writer_cursor that we see
        cursor_wrap, max_slot_size, reader_cursor, pid = struct.unpack_from('NNLl', view)

        if 0 == pid or not pid_is_still_alive(pid): return None
        view_of_writer_cursor = view[writer_cursor_offset:(writer_cursor_offset + writer_cursor_size)].cast('L')

        return SimpleNamespace(view = view,
                               cursor_wrap = cursor_wrap,
                               max_slot_size = max_slot_size,
                               reader_cursor = reader_cursor,
                               view_of_writer_cursor = view_of_writer_cursor,
                               pid = pid)

def shared_memory_ringbuffer_reader_has_kept_up(shm):
    return (shm.view_of_writer_cursor[0] - shm.reader_cursor) + shm.max_slot_size <= shm.cursor_wrap

def shared_memory_ringbuffer_reader_recv(shm):
    data_offset = (struct.calcsize('NNLl') + 15) & ~15
    payload_offset_in_slot = (struct.calcsize('N') + 15) & ~15
    size_of_size = struct.calcsize('N')

    writer_cursor_now = shm.view_of_writer_cursor[0]
    if writer_cursor_now == shm.reader_cursor:
        return None

    slot_offset = data_offset + (shm.reader_cursor % shm.cursor_wrap)
    payload_size = shm.view[slot_offset:(slot_offset + size_of_size)].cast('N')[0]

    # AFTER reading size, BEFORE doing anything with it, need to make sure it was not lapped
    writer_cursor_now = shm.view_of_writer_cursor[0]
    if writer_cursor_now + shm.max_slot_size - shm.reader_cursor > shm.cursor_wrap:
        raise RuntimeError('reader lapped while reading size of slot')

    payload_offset = slot_offset + payload_offset_in_slot

    # advance the reader cursor, with awareness of padding
    shm.reader_cursor += (payload_offset_in_slot + payload_size + 15) & ~15

    return shm.view[payload_offset:(payload_offset + payload_size)]

# end direct port of C API stuff, begin utility generator function that can be used as a
# python iterator by calling code

def shared_memory_ringbuffer_generator(shm_name):
    while True:
        shm = shared_memory_ringbuffer_reader_init(shm_name)
        if shm is not None: break
        print('writer is not running', file=sys.stderr)
        time.sleep(1)

    seconds_per_packet_num = 0
    seconds_per_packet_den = 0
    delay = 0.02

    while True:
        payload = shared_memory_ringbuffer_reader_recv(shm)
        if not payload:
            if seconds_per_packet_num > 0 and not pid_is_still_alive(shm.pid):
                print('writer has exited', file=sys.stderr)
                break

            time.sleep(delay)
            seconds_per_packet_num += delay
            continue

        # maintain a rough estimate of the packet rate for optimal delay
        if seconds_per_packet_num > 0 and seconds_per_packet_den > 0:
            delay = delay + 0.25 * (seconds_per_packet_num / seconds_per_packet_den - delay)
            delay = min(max(delay, 0.05), 1.0)
            seconds_per_packet_den = 0

        seconds_per_packet_num = 0
        seconds_per_packet_den += 1

        yield payload

        # we can now detect whether we were lapped while doing something with this slot.
        # ideally, if we were doing some calculation on the slot contents and then emitting
        # the result downstream, we'd do this check in between those steps
        if not shared_memory_ringbuffer_reader_has_kept_up(shm):
            raise RuntimeError('reader lapped while reading slot')

if __name__ == '__main__':
    def main():
        shm_name = '/shm' if len(sys.argv) < 2 else sys.argv[1]

        for payload in shared_memory_ringbuffer_generator(shm_name):
            sys.stdout.buffer.write(payload)
            sys.stdout.flush()
    main()
