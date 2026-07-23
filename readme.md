# `scrollygram`

This repository contains several live visualization tools useful for working with SCARI output.

## Components

### `shm2pgram.py`

Ingests packets containing acoustic samples, either on `stdin` (such as when piped over ssh to a dev laptop from a remote system via a high-bandwidth link), or directly from the shared-memory ringbuffer segment written to by `cobs_to_shm` if running locally, and emits newline-delimited JSON packets in the same PGRAM format emitted by `scari_uart_to_json.py` and expected by `scroll_gram_from_json.py` in the [scari_tools](https://github.com/appliedoceansciences/scari_tools) repository. This is a lin-log spectrogram representation suitable for transmission over low-bandwidth links.

Example remote usage, assuming `shm2pgram.py` and its dependencies have been installed to `/usr/local/bin/` on the remote system:

    ssh [remote ip] shm2pgram.py /cobs_to_shm | ../scari_tools/scroll_gram_from_json.py

### `scrollygram.py`

Ingests packets containing acoustic samples, either on `stdin` (such as when piped over ssh to a dev laptop from a remote system via a high-bandwidth link) or directly from the shared-memory ringbuffer segment written to by `cobs_to_shm` if running locally, and plots a live-scrolling linear-frequency spectrogram of each channel.

Example local usage, with `cobs_to_shm` already running:

    ./scrollygram.py /cobs_to_shm

Example remote usage:

    ssh [remote ip] shm_to_pipe /cobs_to_shm | ./scrollygram.py

or

    ssh [remote ip] shared_memory_ringbuffer_reader.py /cobs_to_shm | ./scrollygram.py

### `scope.py`

Similar to `scrollygram.py`, but shows an oscilloscope-like line plot representation of the raw timeseries.

## Point of contact

Richard Campbell, richard.campbell@appliedoceansciences.com

## License

Unless otherwise specified, the ISC license applies.
