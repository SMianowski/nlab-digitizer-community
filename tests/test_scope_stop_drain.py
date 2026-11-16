"""Scope STOP semantics: drain until idle, keep the tail, drop pre-START frames.

PetaLinux cde5fe1 user-api.md "STOP semantics" and "Stream carry-over
between sessions": STOP only closes admission, so accepted frames are still
received before close; frames that complete before this session's enable=1
are left over from an earlier one. No hardware access.
"""

from __future__ import annotations

import errno
import struct
import threading
import time
from types import SimpleNamespace

import pytest

import nlab.hardware.digitizer.backends.iio_backend as module
from nlab.hardware.digitizer.backends.iio_backend import (
    IIODigitizerBackend,
    ScopeTailConsumerError,
)
from nlab.hardware.digitizer.dma import IIOScopeDmaStreamer, ScopeFrameBuffer
from nlab.hardware.digitizer.iio_scope_stream import IiodScopeStream
from tests import _scope_fakes as fakes

SAMPLES = 8  # 16-byte frames: u64 timestamp + four samples


def _frame(timestamp: int) -> bytes:
    return struct.pack("<Q", timestamp) + bytes(8)


class _ScriptedStream(IiodScopeStream):
    """Hands out queued frames; with none queued, blocks until cancelled."""

    def __init__(self, frames: list[bytes] | None = None) -> None:  # no socket
        self.frames = list(frames or [])
        self.available = threading.Condition()
        self.cancelled = self.failed = self.closed = False
        self.payload: bytes | None = None

    def push(self, frame: bytes) -> None:
        with self.available:
            self.frames.append(frame)
            self.available.notify_all()

    def refill(self) -> int:
        with self.available:
            while not self.frames and not self.cancelled:
                self.available.wait(5)
            if self.cancelled:
                raise OSError(errno.ECANCELED, "iiod stream was cancelled")
            self.payload = self.frames.pop(0)
            return len(self.payload)

    def cancel(self) -> None:
        with self.available:
            self.cancelled = self.failed = True
            self.available.notify_all()

    def close(self) -> None:
        self.closed = True
        self.cancel()


def _backend(monkeypatch: pytest.MonkeyPatch, stream: _ScriptedStream) -> IIODigitizerBackend:
    monkeypatch.setattr(module, "_SCOPE_CLOSE_IDLE_SECONDS", 0.05)
    monkeypatch.setattr(module, "_SCOPE_CLOSE_POLL_SECONDS", 0.005)
    monkeypatch.setattr(module, "_SCOPE_PRE_START_SETTLE_SECONDS", 0.02)
    backend = object.__new__(IIODigitizerBackend)
    backend._ch = 0
    backend._dma_buf = stream
    backend._dma_buf_frame_samples = SAMPLES
    backend._dma_capture_geometry = None
    backend._dma_fault_latched = False
    backend._dma_last_timestamp = 9
    backend._dma_stop_requested = threading.Event()
    backend._dma_scope = SimpleNamespace(attrs={})
    backend._scope_dma_reader = None
    backend._scope_stop_lock = threading.Lock()
    backend._scope_stop_timer = None
    backend._dma_refill_since = None
    backend._dma_buf_cancelled = False
    backend._dma_stale_frames_discarded = 0
    enable = [False]
    backend._enable_writes = enable  # type: ignore[attr-defined]
    monkeypatch.setattr(backend, "_dma_get_enable", lambda: enable[0])
    monkeypatch.setattr(backend, "_dma_set_enable", lambda value: enable.__setitem__(0, value))
    monkeypatch.setattr(backend, "get_enable", lambda: enable[0])
    monkeypatch.setattr(backend, "set_enable", lambda value: enable.__setitem__(0, value))
    monkeypatch.setattr(backend, "_wait_for_dma_release", lambda: None)
    return backend


def test_close_drains_the_accepted_tail_into_the_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _ScriptedStream([_frame(10), _frame(11), _frame(12)])
    backend = _backend(monkeypatch, stream)
    received: list[bytes] = []

    backend.close_dma_capture(on_frame=received.append)

    assert received == [_frame(10), _frame(11), _frame(12)]
    assert not backend._dma_fault_latched
    assert backend._dma_buf is None and stream.closed


def test_drain_waits_for_a_late_frame_within_the_idle_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _ScriptedStream([_frame(10)])
    backend = _backend(monkeypatch, stream)
    late = threading.Timer(0.03, lambda: stream.push(_frame(11)))  # < idle 0.05 s
    received: list[bytes] = []
    late.start()
    try:
        backend.close_dma_capture(on_frame=received.append)
    finally:
        late.cancel()

    assert received == [_frame(10), _frame(11)]  # a fixed 20 ms grace would lose it


def test_tail_frame_with_a_stale_timestamp_faults_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _ScriptedStream([_frame(10), _frame(5)])
    backend = _backend(monkeypatch, stream)
    received: list[bytes] = []

    with pytest.raises(OSError, match="stale timestamp"):
        backend.close_dma_capture(on_frame=received.append)

    assert received == [_frame(10)]
    assert backend._dma_fault_latched


def test_tail_consumer_failure_is_not_a_dma_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _ScriptedStream([_frame(10), _frame(11)])
    backend = _backend(monkeypatch, stream)

    def reject(_frame: bytes) -> None:
        raise OSError(errno.ENOSPC, "disk full")

    with pytest.raises(ScopeTailConsumerError, match="disk full"):
        backend.close_dma_capture(on_frame=reject)

    assert not backend._dma_fault_latched
    assert backend._dma_buf is None and stream.closed


def test_stop_cancels_only_a_refill_that_stayed_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _ScriptedStream()
    backend = _backend(monkeypatch, stream)
    backend._dma_refill_since = None  # worker busy outside refill

    backend.request_dma_stop()
    time.sleep(0.15)
    assert not stream.cancelled  # never cancel a reader that is consuming

    backend._dma_refill_since = time.monotonic()  # now blocked without data
    deadline = time.monotonic() + 2
    while not stream.cancelled and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stream.cancelled and backend._dma_buf_cancelled
    assert backend._enable_writes == [False]  # type: ignore[attr-defined]

    backend.close_dma_capture(on_frame=lambda _frame: pytest.fail("cancelled: no drain"))
    assert not backend._dma_fault_latched
    assert backend._scope_stop_timer is None


def test_frames_completing_before_enable_are_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = [_frame(900), _frame(901)]
    stream = _ScriptedStream(stale)
    backend = _backend(monkeypatch, stream)
    backend._dma_last_timestamp = None

    def enable(value: bool) -> None:
        backend._enable_writes[0] = value  # type: ignore[attr-defined]
        stream.push(_frame(1))  # the first frame of this session

    monkeypatch.setattr(backend, "_dma_set_enable", enable)

    assert backend._start_reader_then_enable(stream) == len(_frame(1))
    assert stream.read() == _frame(1)
    assert backend._dma_stale_frames_discarded == 2


def test_streamer_publishes_tail_frames_after_stop() -> None:
    backend = fakes.FakeWaveformBackend(limit=3, tail=2)
    stop = threading.Event()
    frames = ScopeFrameBuffer()
    result: list[int] = []
    worker = threading.Thread(target=lambda: result.append(
        IIOScopeDmaStreamer(backend, 0).stream_to_file(None, stop, frame_buffer=frames)
    ))
    worker.start()
    deadline = time.monotonic() + 5
    while backend.sent < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    stop.set()
    backend.request_dma_stop()
    worker.join(5)

    assert result == [5]  # three captured plus the two-frame tail
    published, dropped = frames.drain()
    assert dropped == 0
    assert [frame.timestamp for frame in published] == [
        1000 + index * fakes.INTERVAL_TICKS for index in range(5)
    ]


def test_frame_limited_capture_does_not_publish_a_tail() -> None:
    backend = fakes.FakeWaveformBackend(limit=5, tail=2)
    result = IIOScopeDmaStreamer(backend, 0).stream_to_file(
        None, threading.Event(), n_frames=3,
    )
    assert result == 3
    assert not backend.tail_delivered
