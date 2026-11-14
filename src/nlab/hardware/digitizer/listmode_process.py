"""MCA list-mode DMA in a dedicated reader process (see reader_process.py).

The child builds a ``_listmode_only`` IIODigitizerBackend with its own stream
and control contexts and runs the unchanged ``IIOMcaDmaStreamer`` lifecycle,
including the capture writer. Live consumers receive complete 16 KiB frames
through a bounded queue; display drops are counted, never hidden. Capture
continuity never depends on that queue.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nlab.hardware.digitizer.iio_listmode import IIO_LM_EVENT_DTYPE
from nlab.hardware.digitizer.reader_process import (
    ChildContext,
    Pipe,
    ReaderProcessCrashedError,
    ReaderSession,
    raise_child_error,
    run_child,
)

log = logging.getLogger(__name__)

_FRAME_RECORDS = 1024
# ~1 s of frames at ~530 kcps before the oldest display frame is dropped.
_DISPLAY_QUEUE_FRAMES = 512


@dataclass(frozen=True)
class ListModeRequest:
    """Everything the child needs; must stay picklable (spawn start method)."""

    uri: str
    channel: int
    profile: dict[str, object] | None
    filepath: str | None
    output_mode: str
    configuration_yaml: str
    client_record_schema: str | None
    want_events: bool
    log_level: int


@dataclass
class ListModeOutcome:
    records: int = 0
    diagnostics: dict[str, object] | None = None
    # None: the process ended without reporting (crash); state unverified.
    recovery_required: bool | None = None
    events_dropped: int = 0


def _label(channel: int) -> str:
    return f"list-mode reader ch{channel}"


def _reader_main(control: Pipe, data: Pipe, request: ListModeRequest) -> None:
    """Child entry point: one complete list-mode session."""
    run_child(
        control, data, label=_label(request.channel), log_level=request.log_level,
        body=lambda context: _listmode_session(context, request),
    )


def _listmode_session(context: ChildContext, request: ListModeRequest) -> Mapping[str, object]:
    from nlab.hardware.digitizer.acquisition_profile import AcquisitionProfile
    from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
    from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer, McaEventBuffer
    from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode

    events = (
        context.outbox.open_data(max_items=_DISPLAY_QUEUE_FRAMES, lossless=False)
        if request.want_events else None
    )

    class ForwardingEvents(McaEventBuffer):
        """Hand each frame to the outbox instead of a local display queue."""

        def append(self, batch: np.ndarray) -> None:
            assert events is not None
            events.put(batch.tobytes())

    backend: IIODigitizerBackend | None = None
    streamer: IIOMcaDmaStreamer | None = None
    try:
        profile = AcquisitionProfile.from_mapping(request.profile) if request.profile else None
        backend = IIODigitizerBackend(
            request.channel, request.uri, _listmode_only=True, profile=profile,
        )
        streamer = IIOMcaDmaStreamer(backend, request.channel)
        bound = streamer

        def on_ready() -> None:
            # Only an armed session may be stopped through the backend:
            # _create_mca_dma_buffer() clears the backend's stop flag, so an
            # earlier stop is re-applied here by set_stop_hook().
            context.set_stop_hook(bound.request_stop)
            context.outbox.ready()

        if context.stop_event.is_set():
            return {"records": 0}
        records = streamer.stream_events(
            stop_event=context.stop_event,
            filepath=Path(request.filepath) if request.filepath else None,
            event_buffer=ForwardingEvents() if events is not None else None,
            on_ready=on_ready,
            on_progress=context.outbox.progress,
            output_mode=McaDmaOutputMode(request.output_mode),
            configuration_yaml=request.configuration_yaml,
            client_record_schema=request.client_record_schema,
        )
        return {"records": records}
    finally:
        progress = context.outbox.latest_progress
        context.result["records"] = progress if isinstance(progress, int) else 0
        context.result["diagnostics"] = streamer.last_capture_diagnostics if streamer else None
        if backend is not None:
            try:
                backend.close()
            except BaseException as exc:
                log.error("%s: close failed: %s", _label(request.channel), exc)
                context.result["close_error"] = str(exc)
            context.result["recovery_required"] = backend.mca_recovery_required()
        else:
            context.result["recovery_required"] = False  # no buffer ever existed


class ListModeProcessSession:
    """Parent side of one list-mode session."""

    def __init__(
        self,
        request: ListModeRequest,
        *,
        target: Callable[[Pipe, Pipe, ListModeRequest], None] | None = None,
    ) -> None:
        # Tests substitute a module-level fake child (spawn pickles it by name).
        self._session = ReaderSession(
            target or _reader_main, request, label=_label(request.channel),
        )
        self.outcome = ListModeOutcome()

    def request_stop(self) -> None:
        self._session.request_stop()

    def run(
        self,
        stop_event: threading.Event,
        *,
        on_ready: Callable[[], None] | None,
        on_progress: Callable[[int], None] | None,
        on_events: Callable[[np.ndarray], None] | None,
        on_dropped: Callable[[int], None] | None,
    ) -> int:
        dropped_seen = 0

        def data(items: list[bytes], dropped_frames: int) -> None:
            nonlocal dropped_seen
            if items and on_events is not None:
                on_events(np.frombuffer(b"".join(items), dtype=IIO_LM_EVENT_DTYPE))
            if dropped_frames > dropped_seen:
                if on_dropped is not None:
                    on_dropped((dropped_frames - dropped_seen) * _FRAME_RECORDS)
                dropped_seen = dropped_frames

        def progress(value: int) -> None:
            self.outcome.records = int(value)
            if on_progress is not None:
                on_progress(self.outcome.records)

        handlers: dict[str, Callable[..., None]] = {"data": data, "progress": progress}
        if on_ready is not None:
            handlers["ready"] = on_ready
        try:
            finished = self._session.run(stop_event, handlers)
        except ReaderProcessCrashedError:
            self.outcome.recovery_required = None
            raise
        finally:
            self.outcome.events_dropped = dropped_seen * _FRAME_RECORDS
            report = self._session.finished
            if report is not None:
                diagnostics = report.get("diagnostics")
                self.outcome.diagnostics = diagnostics if isinstance(diagnostics, dict) else None
                self.outcome.recovery_required = bool(report.get("recovery_required"))
                records = report.get("records")
                if isinstance(records, int):
                    self.outcome.records = records
        raise_child_error(finished)
        return self.outcome.records
