"""MCA list-mode DMA in a dedicated reader process.

Why a process
-------------
Measured live (v121, ch0, ~530 kcps, online only): the in-process reader kept
up on its own, but one CPU-busy Python thread next to it cut 3 s of reading
from ~1550 to ~60 frames. Every libiio call releases the interpreter lock and
the reader must then win it back from the busy thread, while the eight kernel
blocks (8192 records) cover only ~15 ms at that rate, so the FPGA drops
records. GUI rendering is exactly such a thread. A child process has its own
interpreter lock, so the GUI can no longer starve the reader.

Boundary
--------
The child (``_reader_main``; imports no Qt) builds a ``_listmode_only``
IIODigitizerBackend with its own stream and control contexts and runs the
unchanged ``IIOMcaDmaStreamer.stream_events()`` lifecycle, including the
capture writer, so recording happens next to the reader. One process is one
session: the native buffer, its latch state and its contexts all end with it.

Messages, child to parent (data pipe), all sent by one sender thread so the
reader never blocks on the pipe::

    ("ready",)                          buffer armed, reader waiting, enable=1
    ("progress", records)               throttled
    ("events", payload, dropped_total)  concatenated frames for live consumers
    ("log", record_dict)                child log records for the System Log
    ("finished", result_dict)           always last

Parent to child (control pipe): ``("stop",)``. End of file on the control pipe
(the GUI process died) is treated as a stop: the child still performs STOP,
tail drain and close, and finishes its capture file.

The parent never terminates the child. Like ``QThread.terminate()`` on an
in-process reader, killing it could bypass the driver-ordered close.
"""

from __future__ import annotations

import builtins
import errno
import logging
import multiprocessing
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from nlab.hardware.digitizer.iio_listmode import IIO_LM_EVENT_DTYPE
from nlab.utils.transport_errors import is_fatal_transport_error

log = logging.getLogger(__name__)


class _Pipe(Protocol):
    """The used subset of multiprocessing Connection/PipeConnection."""

    def send(self, obj: Any) -> None: ...

    def recv(self) -> Any: ...

    def poll(self, timeout: float | None = ...) -> bool: ...

    def close(self) -> None: ...

# Set to "1" to run list mode on a thread of the GUI process (diagnostics).
IN_PROCESS_ENV = "NLAB_LISTMODE_IN_PROCESS"

_FRAME_RECORDS = 1024
# Child-side display queue: ~1 s of frames at 530 kcps before the oldest is
# dropped (and reported). Capture-file continuity never depends on it.
_CHILD_EVENT_QUEUE_FRAMES = 512
# One message carries at most this many frames (1 MiB) to bound parent queues:
# the GUI-side McaEventBuffer counts batches, not records.
_MAX_FRAMES_PER_MESSAGE = 64
_SEND_INTERVAL_SECONDS = 0.020
_PROGRESS_INTERVAL_SECONDS = 0.100
_POLL_SECONDS = 0.050
# After "finished" the child only returns; after a stop without "finished"
# it may still be draining (one-second inactivity window) and closing.
_EXIT_JOIN_SECONDS = 5.0
_ORPHAN_WAIT_SECONDS = 15.0


def in_process_requested() -> bool:
    return os.environ.get(IN_PROCESS_ENV, "").strip() == "1"


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
    error: dict[str, object] | None = None
    exit_code: int | None = None
    events_dropped: int = 0


# ----------------------------------------------------------------------------
# Child side
# ----------------------------------------------------------------------------


class _Outbox:
    """Single sender thread; producers only enqueue and never block."""

    def __init__(self, conn: _Pipe, want_events: bool) -> None:
        from nlab.hardware.digitizer.dma import McaEventBuffer

        self._conn = conn
        self.events = McaEventBuffer(max_batches=_CHILD_EVENT_QUEUE_FRAMES) if want_events else None
        self._ready = threading.Event()
        self._ready_sent = False
        self._progress: int | None = None
        self._progress_sent: int | None = None
        self._progress_at = 0.0
        self._dropped_sent = 0
        self._logs: deque[dict[str, object]] = deque(maxlen=1000)
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._broken = False
        self._thread = threading.Thread(target=self._run, name="listmode-outbox", daemon=True)
        self._thread.start()

    @property
    def latest_progress(self) -> int:
        return self._progress or 0

    def ready(self) -> None:
        self._ready.set()
        self._wake.set()

    def progress(self, records: int) -> None:
        self._progress = records

    def log(self, record: dict[str, object]) -> None:
        self._logs.append(record)

    def _send(self, message: tuple[object, ...]) -> None:
        if self._broken:
            return
        try:
            self._conn.send(message)
        except (OSError, EOFError, ValueError):
            # The GUI process is gone. Keep capturing: the control pipe's EOF
            # already requested a normal STOP/drain/close in this process.
            self._broken = True

    def _flush(self, *, final: bool) -> None:
        if self._ready.is_set() and not self._ready_sent:
            self._send(("ready",))
            self._ready_sent = True
        while self._logs:
            self._send(("log", self._logs.popleft()))
        if self.events is not None:
            batches, dropped = self.events.drain()
            for start in range(0, len(batches), _MAX_FRAMES_PER_MESSAGE):
                chunk = batches[start:start + _MAX_FRAMES_PER_MESSAGE]
                payload = np.concatenate(chunk).tobytes() if len(chunk) > 1 else chunk[0].tobytes()
                self._send(("events", payload, dropped))
                self._dropped_sent = dropped
            if dropped != self._dropped_sent:
                self._send(("events", b"", dropped))
                self._dropped_sent = dropped
        now = time.monotonic()
        if self._progress is not None and self._progress != self._progress_sent and (
            final or now - self._progress_at >= _PROGRESS_INTERVAL_SECONDS
        ):
            self._send(("progress", self._progress))
            self._progress_sent = self._progress
            self._progress_at = now

    def _run(self) -> None:
        while True:
            closing = self._closing.is_set()
            self._flush(final=closing)
            if closing:
                return
            self._wake.wait(_SEND_INTERVAL_SECONDS)
            self._wake.clear()

    def finish(self, message: tuple[object, ...]) -> None:
        self._closing.set()
        self._wake.set()
        self._thread.join()
        self._send(message)


class _ForwardingHandler(logging.Handler):
    def __init__(self, outbox: _Outbox) -> None:
        super().__init__()
        self._outbox = outbox

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if record.exc_info:
                message = f"{message}\n{logging.Formatter().formatException(record.exc_info)}"
            self._outbox.log({"name": record.name, "levelno": record.levelno, "msg": message})
        except Exception:
            self.handleError(record)


def _error_info(exc: BaseException) -> dict[str, object]:
    number = getattr(exc, "errno", None)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "errno": number if isinstance(number, int) else None,
        "strerror": getattr(exc, "strerror", None) if isinstance(exc, OSError) else None,
        "notes": [str(note) for note in getattr(exc, "__notes__", [])],
        "fatal_transport": is_fatal_transport_error(exc),
    }


def _reader_main(control: _Pipe, data: _Pipe, request: ListModeRequest) -> None:
    """Child entry point: one complete list-mode session."""
    outbox = _Outbox(data, request.want_events)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(_ForwardingHandler(outbox))
    root.setLevel(request.log_level)

    from nlab.hardware.digitizer.acquisition_profile import AcquisitionProfile
    from nlab.hardware.digitizer.backends.iio_backend import IIODigitizerBackend
    from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer
    from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode

    stop_event = threading.Event()
    streamer: IIOMcaDmaStreamer | None = None
    backend: IIODigitizerBackend | None = None
    started = threading.Event()
    result: dict[str, object] = {}

    def request_stop() -> None:
        stop_event.set()
        if streamer is not None and started.is_set():
            try:
                streamer.request_stop()
            except Exception:
                log.exception("List-mode reader ch%d: stop request failed", request.channel)

    def watch_control() -> None:
        try:
            while control.recv()[0] != "stop":
                pass
        except (EOFError, OSError):
            log.warning("List-mode reader ch%d: GUI process gone; stopping capture",
                        request.channel)
        request_stop()

    def on_ready() -> None:
        started.set()
        # A stop that arrived while the buffer was being armed set only the
        # event: _create_mca_dma_buffer() clears the backend's stop flag.
        if stop_event.is_set():
            request_stop()
        outbox.ready()

    try:
        profile = (
            AcquisitionProfile.from_mapping(request.profile) if request.profile else None
        )
        backend = IIODigitizerBackend(
            request.channel, request.uri, _listmode_only=True, profile=profile,
        )
        streamer = IIOMcaDmaStreamer(backend, request.channel)
        threading.Thread(target=watch_control, name="listmode-control", daemon=True).start()
        if stop_event.is_set():
            result.update(ok=True, records=0)
        else:
            records = streamer.stream_events(
                stop_event=stop_event,
                filepath=Path(request.filepath) if request.filepath else None,
                event_buffer=outbox.events,
                on_ready=on_ready,
                on_progress=outbox.progress,
                output_mode=McaDmaOutputMode(request.output_mode),
                configuration_yaml=request.configuration_yaml,
                client_record_schema=request.client_record_schema,
            )
            result.update(ok=True, records=records)
    except BaseException as exc:
        log.error("List-mode reader ch%d: session failed", request.channel, exc_info=True)
        result.update(ok=False, records=outbox.latest_progress, error=_error_info(exc))
    finally:
        result["diagnostics"] = streamer.last_capture_diagnostics if streamer else None
        if backend is not None:
            try:
                backend.close()
            except BaseException as exc:
                log.error("List-mode reader ch%d: close failed: %s", request.channel, exc)
                result.setdefault("close_error", str(exc))
            result["recovery_required"] = backend.mca_recovery_required()
        else:
            # Construction failed before any buffer existed.
            result["recovery_required"] = False
        outbox.finish(("finished", result))


# ----------------------------------------------------------------------------
# Parent side
# ----------------------------------------------------------------------------


def _rebuild_error(info: dict[str, object]) -> BaseException:
    """Recreate an exception that classifies like the child's original."""
    number = info.get("errno")
    strerror = info.get("strerror")
    message = str(info.get("message", ""))
    name = str(info.get("type", "RuntimeError"))
    error: BaseException
    if info.get("fatal_transport"):
        error = ConnectionError(
            number if isinstance(number, int) else errno.ECONNRESET,
            str(strerror) if strerror is not None else message,
        )
    elif isinstance(number, int) and strerror is not None:
        error = OSError(number, str(strerror))
    elif isinstance(number, int):
        error = OSError(number, message)
    else:
        candidate = getattr(builtins, name, None)
        if (isinstance(candidate, type) and issubclass(candidate, Exception)
                and not issubclass(candidate, OSError)):
            try:
                error = candidate(message)
            except Exception:
                error = RuntimeError(f"{name}: {message}")
        else:
            error = RuntimeError(f"{name}: {message}" if name != "RuntimeError" else message)
    notes = info.get("notes")
    if isinstance(notes, list):
        for note in notes:
            error.add_note(str(note))
    return error


class ListModeProcessSession:
    """Parent-side owner of one reader process (one capture session)."""

    def __init__(
        self,
        request: ListModeRequest,
        *,
        target: Callable[[_Pipe, _Pipe, ListModeRequest], None] | None = None,
    ) -> None:
        self._request = request
        # Tests substitute a module-level fake child; it must be importable
        # by name because the spawn start method pickles it.
        self._target = target or _reader_main
        self._control_lock = threading.Lock()
        self._control: _Pipe | None = None
        self._stop_sent = False
        self.outcome = ListModeOutcome()

    def request_stop(self) -> None:
        with self._control_lock:
            if self._stop_sent or self._control is None:
                return
            try:
                self._control.send(("stop",))
            except (OSError, EOFError, ValueError):
                pass  # already exited; the run loop reports its outcome
            self._stop_sent = True

    def run(
        self,
        stop_event: threading.Event,
        *,
        on_ready: Callable[[], None] | None,
        on_progress: Callable[[int], None] | None,
        on_events: Callable[[np.ndarray], None] | None,
        on_dropped: Callable[[int], None] | None,
    ) -> int:
        context = multiprocessing.get_context("spawn")
        data_recv, data_send = context.Pipe(duplex=False)
        control_recv, control_send = context.Pipe(duplex=False)
        process = context.Process(
            target=self._target,
            args=(control_recv, data_send, self._request),
            name=f"nlab-listmode-ch{self._request.channel}",
            daemon=False,
        )
        try:
            process.start()
        except BaseException:
            for conn in (data_recv, data_send, control_recv, control_send):
                conn.close()
            raise
        # Close the parent's copies so either side sees EOF when the other ends.
        data_send.close()
        control_recv.close()
        with self._control_lock:
            self._control = control_send
        finished: dict[str, object] | None = None
        dropped_seen = 0
        try:
            while finished is None:
                if stop_event.is_set():
                    self.request_stop()
                if data_recv.poll(_POLL_SECONDS):
                    try:
                        message = data_recv.recv()
                    except EOFError:
                        break
                    kind = message[0]
                    if kind == "ready":
                        if on_ready is not None:
                            on_ready()
                    elif kind == "progress":
                        self.outcome.records = int(message[1])
                        if on_progress is not None:
                            on_progress(self.outcome.records)
                    elif kind == "events":
                        payload, dropped = message[1], int(message[2])
                        if payload and on_events is not None:
                            on_events(np.frombuffer(payload, dtype=IIO_LM_EVENT_DTYPE))
                        if dropped > dropped_seen:
                            if on_dropped is not None:
                                on_dropped(dropped - dropped_seen)
                            dropped_seen = dropped
                    elif kind == "log":
                        _relay_log(message[1], self._request.channel)
                    elif kind == "finished":
                        finished = message[1]
                elif not process.is_alive() and not data_recv.poll(0):
                    break
        finally:
            if finished is None and process.is_alive():
                # The parent itself failed mid-session (e.g. a consumer raised).
                # Stop the capture normally and give it time to close.
                self.request_stop()
                _discard_until_exit(data_recv, process, _ORPHAN_WAIT_SECONDS)
            process.join(_EXIT_JOIN_SECONDS)
            if process.is_alive():
                log.error(
                    "List-mode reader process ch%d did not exit; leaving it to finish "
                    "its own driver-ordered close (never terminated)",
                    self._request.channel,
                )
            self.outcome.exit_code = process.exitcode
            with self._control_lock:
                self._control = None
                try:
                    control_send.close()
                except OSError:
                    pass
            data_recv.close()
        self.outcome.events_dropped = dropped_seen
        if finished is None:
            self.outcome.recovery_required = None
            raise RuntimeError(
                f"MCA list-mode reader process ch{self._request.channel} ended unexpectedly "
                f"(exit code {process.exitcode}); target state unverified -- check the "
                "target and acknowledge recovery before re-arming"
            )
        records = finished.get("records")
        self.outcome.records = records if isinstance(records, int) else 0
        diagnostics = finished.get("diagnostics")
        self.outcome.diagnostics = diagnostics if isinstance(diagnostics, dict) else None
        self.outcome.recovery_required = bool(finished.get("recovery_required"))
        error = finished.get("error")
        if isinstance(error, dict):
            self.outcome.error = error
            raise _rebuild_error(error)
        if not finished.get("ok", False):
            raise RuntimeError("MCA list-mode reader process reported failure without detail")
        return self.outcome.records


def _discard_until_exit(conn: _Pipe, process: Any, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while process.is_alive() and time.monotonic() < deadline:
        try:
            if conn.poll(_POLL_SECONDS):
                message = conn.recv()
                if message[0] == "finished":
                    return
        except (EOFError, OSError):
            return


def _relay_log(record: object, channel: int) -> None:
    if not isinstance(record, dict):
        return
    name = str(record.get("name", __name__))
    levelno = record.get("levelno", logging.INFO)
    logging.getLogger(name).log(
        levelno if isinstance(levelno, int) else logging.INFO,
        "[list-mode reader ch%d] %s", channel, record.get("msg", ""),
    )
