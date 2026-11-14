"""Scope reader-process children with fake backends (importable by spawn).

Each target replaces scope_process._open_backend inside the child and then
runs the real child session code, so the spawn, protocol, child body and
parent consumers are all exercised without hardware.
"""

from __future__ import annotations

import errno
import struct
import threading
from typing import Any

import numpy as np

from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.scope_sums import ScopeSumGeometry

GEOMETRY = ScopeDmaGeometry(
    frame_samples=512, buffer_samples=132, frame_bytes=264,
    waveform_samples=127, sample_decimation=4, padding_bytes=2,
)
INTERVAL_TICKS = 100


def waveform_record(index: int) -> bytes:
    waveform = (np.arange(127, dtype="<i2") + index).tobytes()
    return struct.pack("<Q", 1000 + index * INTERVAL_TICKS) + waveform + b"\x00\x00"


class FakeWaveformBackend:
    """Delivers ``limit`` frames, then blocks until stopped (like a quiet trigger)."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.sent = 0
        self.stopped = threading.Event()

    def prepare_dma_capture(self) -> None:
        pass

    def get_scope_dma_geometry(self) -> ScopeDmaGeometry:
        return GEOMETRY

    def get_frame_samples(self) -> int:
        return GEOMETRY.frame_samples

    def read_dma_raw_frame(self) -> bytes:
        if self.sent < self.limit:
            self.sent += 1
            return waveform_record(self.sent - 1)
        assert self.stopped.wait(10), "never stopped"
        raise InterruptedError(errno.ECANCELED, "test stop")

    def request_dma_stop(self) -> None:
        self.stopped.set()

    def close_dma_capture(self) -> None:
        pass

    def get_scope_dma_runtime_metadata(self) -> dict[str, object]:
        return {"transport": "fixture", "kernel_buffers": 4, "readbuf_batch_frames": 64}

    def dma_fault_is_latched(self) -> bool:
        return False

    def close(self) -> None:
        pass


def waveform_child(control: Any, data: Any, request: Any) -> None:
    from nlab.hardware.digitizer import scope_process

    limit = int(request.uri.rsplit(":", 1)[1])
    setattr(scope_process, "_open_backend", lambda uri, channel: FakeWaveformBackend(limit))
    scope_process.waveform_main(control, data, request)


def sum_packet(sequence: int, total: int = 16 * 32768) -> bytes:
    first = (1 << 60) + 7
    raw = bytearray(4096)
    struct.pack_into("<4Q", raw, 0, first + sequence * 508 * 7, 0x32534D5553504456,
                     16 | 3 << 32 | 508 << 48, sequence)
    for i in range(508):
        struct.pack_into("<Q", raw, 32 + 8 * i, total)
    return bytes(raw)


class FakeSumBackend:
    def __init__(self, count: int) -> None:
        self.raws = iter([sum_packet(i) for i in range(count)])
        self.stopped = threading.Event()

    def prepare_dma_capture(self) -> None:
        pass

    def get_scope_dma_geometry(self) -> ScopeSumGeometry:
        return ScopeSumGeometry(16, 124, 3)

    def read_dma_raw_frame(self) -> bytes:
        try:
            return next(self.raws)
        except StopIteration:
            self.stopped.wait(10)
            raise OSError(errno.ETIMEDOUT, "terminal drain timeout") from None

    def request_scope_sum_stop(self) -> None:
        self.stopped.set()

    def stop_scope_sum_from_reader(self) -> None:
        self.stopped.set()

    def scope_sum_status(self) -> tuple[int, int]:
        return 0, 5

    def close_scope_sum_capture(self) -> None:
        pass

    def scope_sum_identity(self) -> dict[str, object]:
        return {"uri": "fixture", "channel": 0}

    def get_scope_dma_runtime_metadata(self) -> dict[str, object]:
        return {"transport": "fixture"}

    def dma_fault_is_latched(self) -> bool:
        return False

    def close(self) -> None:
        pass


def sum_child(control: Any, data: Any, request: Any) -> None:
    from nlab.hardware.digitizer import scope_process

    count = int(request.uri.rsplit(":", 1)[1])
    setattr(scope_process, "_open_backend", lambda uri, channel: FakeSumBackend(count))
    scope_process.sum_main(control, data, request)
