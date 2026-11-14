"""Scope waveform and periodic-sum reader processes (spawned, fake backends)."""

from __future__ import annotations

import threading
import time
from typing import Any, cast

import pytest

from nlab.hardware.digitizer import scope_process
from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator, ScopeCurrentRuntime
from nlab.hardware.digitizer.dma import IIOScopeDmaStreamer, ScopeFrameBuffer
from nlab.hardware.digitizer.reader_process import DataQueue
from nlab.hardware.digitizer.scope_sums import ScopeSumCapture
from tests import _scope_fakes as fakes


class ParentBackend:
    """GUI-side backend stub: only latch state is consulted in process mode."""

    def __init__(self, latched: bool = False) -> None:
        self.latched = latched
        self.adopted: list[bool | None] = []

    def dma_fault_is_latched(self) -> bool:
        return self.latched

    def adopt_scope_dma_session_state(self, fault_latched: bool | None) -> None:
        self.adopted.append(fault_latched)
        self.latched = self.latched or fault_latched is None or fault_latched


def _streamer(monkeypatch: Any, backend: ParentBackend, frames: int) -> IIOScopeDmaStreamer:
    monkeypatch.setattr(scope_process, "waveform_main", fakes.waveform_child)
    monkeypatch.setattr(scope_process, "sum_main", fakes.sum_child)
    # The fake child reads its frame budget from the URI.
    return IIOScopeDmaStreamer(cast(Any, backend), 0, process_uri=f"fake:{frames}")


def _accumulator() -> ScopeCurrentAccumulator:
    accumulator = ScopeCurrentAccumulator()
    runtime = ScopeCurrentRuntime(0, 122, fakes.INTERVAL_TICKS, 228)
    accumulator.start_session(fakes.GEOMETRY, runtime)
    return accumulator


def test_waveforms_reach_accumulator_and_display_with_runtime_metadata(monkeypatch: Any) -> None:
    backend = ParentBackend()
    streamer = _streamer(monkeypatch, backend, frames=50)
    accumulator = _accumulator()
    display = ScopeFrameBuffer()
    ready: list[None] = []

    captured = streamer.stream_to_file(
        None, threading.Event(), n_frames=50, on_ready=lambda: ready.append(None),
        frame_buffer=display, current_accumulator=accumulator,
    )

    snapshot = accumulator.snapshot()
    frames, dropped = display.drain()
    assert captured == 50 and ready == [None]
    assert snapshot.received_frames == snapshot.analyzed_frames == 50
    assert snapshot.skipped_opportunities == 0 and snapshot.protocol_errors == 0
    assert snapshot.latest_raw_sample == 126 + 49
    assert snapshot.runtime is not None and snapshot.runtime.transport == "fixture"
    assert not snapshot.active  # finish_session ran on the parent accumulator
    assert len(frames) == 50 and dropped == 0
    assert frames[-1].timestamp == 1000 + 49 * fakes.INTERVAL_TICKS
    assert frames[0].samples[0] == 0 and not frames[0].samples.flags.writeable
    assert backend.adopted == [False]


def test_waveform_stop_ends_child_and_every_captured_frame_is_analysed(monkeypatch: Any) -> None:
    backend = ParentBackend()
    streamer = _streamer(monkeypatch, backend, frames=30)
    accumulator = _accumulator()
    stop = threading.Event()
    result: dict[str, int] = {}
    worker = threading.Thread(target=lambda: result.update(n=streamer.stream_to_file(
        None, stop, current_accumulator=accumulator,
    )))
    worker.start()
    deadline = time.monotonic() + 20
    while accumulator.snapshot().received_frames < 30 and time.monotonic() < deadline:
        time.sleep(0.02)
    stop.set()
    streamer.request_stop()
    worker.join(20)

    assert not worker.is_alive()
    assert result["n"] == 30 == accumulator.snapshot().analyzed_frames


def test_latched_parent_refuses_to_spawn_a_waveform_reader(monkeypatch: Any) -> None:
    streamer = _streamer(monkeypatch, ParentBackend(latched=True), frames=1)
    with pytest.raises(RuntimeError, match="fault is latched"):
        streamer.stream_to_file(None, threading.Event(), n_frames=1)


def test_periodic_sums_are_analysed_in_the_gui_process(monkeypatch: Any) -> None:
    backend = ParentBackend()
    streamer = _streamer(monkeypatch, backend, frames=12)
    capture = streamer.create_sum_capture()
    assert isinstance(capture, ScopeSumCapture)
    result: dict[str, Any] = {}
    ready: list[None] = []
    worker = threading.Thread(target=lambda: result.update(capture.run(
        None, ready=lambda: ready.append(None),
    )))
    worker.start()
    deadline = time.monotonic() + 20
    while capture.analysis.packet_count < 12 and time.monotonic() < deadline:
        time.sleep(0.02)
    capture.request_stop()
    worker.join(20)

    assert not worker.is_alive()
    assert ready == [None]
    assert capture.analysis.packet_count == 12
    assert capture.analysis.window_count == 12 * 508
    assert result["complete"] is True and result["final_missed_windows"] == 5
    assert backend.adopted == [False]


def test_lossless_queue_blocks_the_producer_until_drained() -> None:
    queue = DataQueue(max_items=2, lossless=True, wake=threading.Event())
    queue.put(b"a")
    queue.put(b"b")
    blocked = threading.Thread(target=queue.put, args=(b"c",))
    blocked.start()
    blocked.join(0.2)
    assert blocked.is_alive()
    assert queue.take(1 << 20) == ([b"a", b"b"], 0)
    blocked.join(2)
    assert not blocked.is_alive() and queue.take(1 << 20) == ([b"c"], 0)


def test_display_queue_drops_oldest_and_counts() -> None:
    queue = DataQueue(max_items=2, lossless=False, wake=threading.Event())
    for item in (b"a", b"b", b"c"):
        queue.put(item)
    assert queue.take(1 << 20) == ([b"b", b"c"], 1)


def test_frame_buffer_upstream_drops_reach_subscribers() -> None:
    primary, subscriber = ScopeFrameBuffer(), ScopeFrameBuffer()
    primary.subscribe(subscriber)
    primary.add_dropped(3)
    assert primary.drain()[1] == 3 and subscriber.drain()[1] == 3
