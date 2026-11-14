"""Scope waveform and periodic-sum DMA in dedicated reader processes.

See reader_process.py for why. The child builds a ``_scope_dma_only``
IIODigitizerBackend (Scope control plus DMA contexts) and runs the unchanged
``IIOScopeDmaStreamer.stream_to_file()`` or ``ScopeSumCapture.run()``
lifecycle, including file recording. The GUI's consumers stay in the parent:

* Waveforms for the Current accumulator travel through a lossless queue with
  backpressure. If the GUI falls behind, the reader slows exactly as the
  in-process accumulator used to slow it; the FPGA then skips triggers and
  the accumulator still sees every captured frame (gaps show as skipped
  opportunities in the hardware timestamps, as before).
* Display-only waveforms use a bounded queue; drops are counted.
* Periodic sums are always lossless: ScopeSumAnalysis needs every packet.

Receive times are stamped in the child with ``time.perf_counter_ns()``. That
clock is system-wide on Windows (QueryPerformanceCounter) and Linux
(CLOCK_MONOTONIC), so GUI-side lag figures remain meaningful.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nlab.hardware.digitizer.reader_process import (
    ChildContext,
    DataQueue,
    Pipe,
    run_child,
)

if TYPE_CHECKING:
    from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
    from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator
    from nlab.hardware.digitizer.dma import ScopeDmaGeometry

log = logging.getLogger(__name__)

# Queue bounds in bytes of payload, converted to items per geometry.
_LOSSLESS_QUEUE_BYTES = 32 * 1024 * 1024
_DISPLAY_QUEUE_BYTES = 8 * 1024 * 1024
_SUM_QUEUE_PACKETS = 4096


@dataclass(frozen=True)
class ScopeWaveformRequest:
    uri: str
    channel: int
    filepath: str | None
    n_frames: int | None
    want_frames: bool
    lossless: bool
    log_level: int


@dataclass(frozen=True)
class ScopeSumRequest:
    uri: str
    channel: int
    filepath: str | None
    max_packets: int | None
    overwrite: bool
    configuration_yaml: str
    log_level: int


def waveform_label(channel: int) -> str:
    return f"Scope DMA reader ch{channel}"


def sum_label(channel: int) -> str:
    return f"Scope sum reader ch{channel}"


def _open_backend(uri: str, channel: int) -> IIODigitizerBackend:
    from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend

    return IIODigitizerBackend(channel, uri, _scope_dma_only=True)


def _close_backend(context: ChildContext, backend: IIODigitizerBackend | None, label: str) -> None:
    if backend is None:
        context.result["fault_latched"] = False  # no buffer ever existed
        return
    try:
        backend.close()
    except BaseException as exc:
        log.error("%s: close failed: %s", label, exc)
        context.result["close_error"] = str(exc)
    context.result["fault_latched"] = backend.dma_fault_is_latched()


# ----------------------------------------------------------------------------
# Raw waveform DMA
# ----------------------------------------------------------------------------


class _ForwardingAccumulator:
    """Child-side stand-in for ScopeCurrentAccumulator.

    stream_to_file() hands it each complete record with its geometry and
    receive time, plus the same rejected/protocol-error/runtime notes the
    real accumulator gets; all of it is forwarded to the parent in order.
    """

    def __init__(self, context: ChildContext, request: ScopeWaveformRequest) -> None:
        self._context = context
        self._request = request
        self._geometry: ScopeDmaGeometry | None = None
        self._queue: DataQueue | None = None

    def append_frame(self, record: bytes, geometry: ScopeDmaGeometry, received_ns: int) -> None:
        if self._queue is None or geometry != self._geometry:
            self._geometry = geometry
            self._context.outbox.note("geometry", geometry)
            budget = _LOSSLESS_QUEUE_BYTES if self._request.lossless else _DISPLAY_QUEUE_BYTES
            if self._queue is None:
                self._queue = self._context.outbox.open_data(
                    max_items=budget // max(1, geometry.frame_bytes),
                    lossless=self._request.lossless,
                )
        self._queue.put((record, received_ns))

    def update_runtime_transport(self, **values: object) -> None:
        self._context.outbox.note("runtime", values)

    def note_rejected_frame(self) -> None:
        self._context.outbox.note("rejected")

    def note_protocol_error(self) -> None:
        self._context.outbox.note("protocol_error")

    def finish_session(self) -> None:
        pass  # the parent finishes its real accumulator after the child ends


def waveform_main(control: Pipe, data: Pipe, request: ScopeWaveformRequest) -> None:
    run_child(
        control, data, label=waveform_label(request.channel), log_level=request.log_level,
        body=lambda context: _waveform_session(context, request),
    )


def _waveform_session(context: ChildContext, request: ScopeWaveformRequest) -> Mapping[str, object]:
    from nlab.hardware.digitizer.dma import IIOScopeDmaStreamer

    backend = None
    try:
        backend = _open_backend(request.uri, request.channel)
        streamer = IIOScopeDmaStreamer(backend, request.channel)
        # request_dma_stop() is safe at any point: it clears enable and
        # cancels a blocked refill; the loop also checks the stop event.
        context.set_stop_hook(streamer.request_stop)
        forwarder = _ForwardingAccumulator(context, request) if request.want_frames else None
        frames = streamer.stream_to_file(
            Path(request.filepath) if request.filepath else None,
            context.stop_event,
            n_frames=request.n_frames,
            on_ready=context.outbox.ready,
            on_progress=context.outbox.progress,
            current_accumulator=cast("ScopeCurrentAccumulator | None", forwarder),
        )
        return {"frames": frames}
    finally:
        _close_backend(context, backend, waveform_label(request.channel))


# ----------------------------------------------------------------------------
# Periodic sums
# ----------------------------------------------------------------------------


def sum_main(control: Pipe, data: Pipe, request: ScopeSumRequest) -> None:
    run_child(
        control, data, label=sum_label(request.channel), log_level=request.log_level,
        body=lambda context: _sum_session(context, request),
    )


def _sum_session(context: ChildContext, request: ScopeSumRequest) -> Mapping[str, object]:
    from nlab.hardware.digitizer.scope_sums import (
        ScopeSumAnalysis,
        ScopeSumCapture,
        ScopeSumPacket,
    )

    packets = context.outbox.open_data(max_items=_SUM_QUEUE_PACKETS, lossless=True)

    class ForwardingAnalysis(ScopeSumAnalysis):
        """Keep the child's counters (run() uses them) and forward packets."""

        def append(self, packet: ScopeSumPacket) -> None:
            super().append(packet)
            packets.put(packet.raw)

    backend = None
    try:
        backend = _open_backend(request.uri, request.channel)
        capture = ScopeSumCapture(
            backend, request.channel, configuration_yaml=request.configuration_yaml,
        )
        capture.analysis = ForwardingAnalysis()
        context.set_stop_hook(capture.request_stop)
        snapshot = cast(dict[str, Any], capture.run(
            Path(request.filepath) if request.filepath else None,
            ready=context.outbox.ready,
            max_packets=request.max_packets,
            overwrite=request.overwrite,
        ))
        return {
            "complete": bool(snapshot.get("complete")),
            "final_missed_windows": int(snapshot.get("final_missed_windows") or 0),
        }
    finally:
        _close_backend(context, backend, sum_label(request.channel))
