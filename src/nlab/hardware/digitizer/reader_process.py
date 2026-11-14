"""Shared core for DMA reader processes (MCA list mode, Scope waveforms and sums).

Why separate processes
----------------------
Every libiio call releases Python's interpreter lock and the reader must then
win it back. Next to a busy GUI thread that wait dominated: measured live on
ch0, list mode fell from ~1550 to ~60 frames in 3 s, raw Scope DMA from 8877
to 474 frames/s, and periodic sums at 1000 samples/window missed 75-85x more
windows. A reader process has its own interpreter lock.

One process is one capture session. The child owns the native buffer, its
contexts, the capture file and the backend's fault latch; all of them end
with it. The parent never terminates a child: like ``QThread.terminate()`` on
an in-process reader, that could bypass the driver-ordered close.

Protocol
--------
Child to parent (data pipe), sent only by the child's outbox thread, so a
reader thread never blocks on the pipe except through a lossless queue's
deliberate backpressure::

    ("ready",)
    ("progress", value)                 throttled
    ("note", key, value)                ordered side information
    ("data", [items], dropped_items)    batched records for live consumers
    ("log", record_dict)                child log records for the System Log
    ("finished", result_dict)           always last

Parent to child (control pipe): ``("stop",)``. End of file on the control pipe
(the GUI process died) is a stop too: the child still performs the normal
STOP/drain/close and completes its capture file.
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
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from nlab.utils.transport_errors import is_fatal_transport_error

log = logging.getLogger(__name__)

# Set to "1" to run DMA readers on threads of the GUI process (diagnostics).
IN_PROCESS_ENV = "NLAB_DMA_IN_PROCESS"

_SEND_INTERVAL_SECONDS = 0.020
_PROGRESS_INTERVAL_SECONDS = 0.100
_MESSAGE_BYTES = 1024 * 1024
_POLL_SECONDS = 0.050
_EXIT_JOIN_SECONDS = 5.0
_ORPHAN_WAIT_SECONDS = 15.0


def in_process_requested() -> bool:
    return os.environ.get(IN_PROCESS_ENV, "").strip() == "1"


class Pipe(Protocol):
    """The used subset of multiprocessing Connection/PipeConnection."""

    def send(self, obj: Any) -> None: ...

    def recv(self) -> Any: ...

    def poll(self, timeout: float | None = ...) -> bool: ...

    def close(self) -> None: ...


def _item_bytes(item: object) -> int:
    if isinstance(item, (bytes, bytearray)):
        return len(item)
    if isinstance(item, tuple) and item and isinstance(item[0], (bytes, bytearray)):
        return len(item[0])
    return 64


# ----------------------------------------------------------------------------
# Child side
# ----------------------------------------------------------------------------


class DataQueue:
    """Child-side record queue drained by the outbox thread.

    ``lossless``: put() blocks while full. The reader then slows exactly as
    an in-process consumer used to slow it, and the FPGA accounts the
    resulting skipped triggers/windows, so exact GUI-side analyses stay
    consistent. Otherwise the oldest item is dropped and counted.
    """

    def __init__(self, *, max_items: int, lossless: bool, wake: threading.Event) -> None:
        self._items: deque[object] = deque()
        self._max_items = max(1, max_items)
        self._lossless = lossless
        self._wake = wake
        self._condition = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put(self, item: object) -> None:
        with self._condition:
            if self._lossless:
                while len(self._items) >= self._max_items and not self._closed:
                    self._wake.set()
                    self._condition.wait(0.05)
                if self._closed:
                    return  # nobody left to deliver to
            elif len(self._items) >= self._max_items:
                self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            if len(self._items) * 4 >= self._max_items:
                self._wake.set()

    def take(self, max_bytes: int) -> tuple[list[object], int]:
        with self._condition:
            batch: list[object] = []
            size = 0
            while self._items and (not batch or size < max_bytes):
                item = self._items.popleft()
                batch.append(item)
                size += _item_bytes(item)
            self._condition.notify_all()
            return batch, self.dropped

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class Outbox:
    """Single sender thread; producers only enqueue."""

    def __init__(self, conn: Pipe) -> None:
        self._conn = conn
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._lock = threading.Lock()
        self._ready = False
        self._ready_sent = False
        self._notes: deque[tuple[str, object]] = deque()
        self._logs: deque[dict[str, object]] = deque(maxlen=1000)
        self._progress: object = None
        self._progress_sent: object = None
        self._progress_at = 0.0
        self._dropped_sent = 0
        self._broken = False
        self.data: DataQueue | None = None
        self._thread = threading.Thread(target=self._run, name="dma-reader-outbox", daemon=True)
        self._thread.start()

    @property
    def latest_progress(self) -> object:
        return self._progress

    def open_data(self, *, max_items: int, lossless: bool) -> DataQueue:
        self.data = DataQueue(max_items=max_items, lossless=lossless, wake=self._wake)
        return self.data

    def ready(self) -> None:
        self._ready = True
        self._wake.set()

    def progress(self, value: object) -> None:
        self._progress = value

    def note(self, key: str, value: object = None) -> None:
        with self._lock:
            self._notes.append((key, value))
        self._wake.set()

    def log(self, record: dict[str, object]) -> None:
        self._logs.append(record)

    def _send(self, message: tuple[object, ...]) -> None:
        if self._broken:
            return
        try:
            self._conn.send(message)
        except (OSError, EOFError, ValueError):
            # The GUI process is gone; the control pipe's EOF has already
            # requested a normal stop. Never block a lossless producer now.
            self._broken = True
            if self.data is not None:
                self.data.close()

    def _flush(self, *, final: bool) -> None:
        if self._ready and not self._ready_sent:
            self._send(("ready",))
            self._ready_sent = True
        while True:
            with self._lock:
                if not self._notes:
                    break
                key, value = self._notes.popleft()
            self._send(("note", key, value))
        while self._logs:
            self._send(("log", self._logs.popleft()))
        data = self.data
        if data is not None:
            while True:
                batch, dropped = data.take(_MESSAGE_BYTES)
                if not batch:
                    break
                self._send(("data", batch, dropped))
                self._dropped_sent = dropped
            if data.dropped != self._dropped_sent:
                self._dropped_sent = data.dropped
                self._send(("data", [], self._dropped_sent))
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
    def __init__(self, outbox: Outbox) -> None:
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


def error_info(exc: BaseException) -> dict[str, object]:
    number = getattr(exc, "errno", None)
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "errno": number if isinstance(number, int) else None,
        "strerror": getattr(exc, "strerror", None) if isinstance(exc, OSError) else None,
        "notes": [str(note) for note in getattr(exc, "__notes__", [])],
        "fatal_transport": is_fatal_transport_error(exc),
    }


class ChildContext:
    """What a child session body gets: stop signal, outbox and result slots."""

    def __init__(self, outbox: Outbox) -> None:
        self.outbox = outbox
        self.stop_event = threading.Event()
        # Written by the body (also from its finally block); merged into the
        # "finished" message even when the body raises.
        self.result: dict[str, object] = {}
        self._hook_lock = threading.Lock()
        self._stop_hook: Callable[[], None] | None = None

    def set_stop_hook(self, hook: Callable[[], None] | None) -> None:
        with self._hook_lock:
            self._stop_hook = hook
        if hook is not None and self.stop_event.is_set():
            self._call(hook)

    def request_stop(self) -> None:
        self.stop_event.set()
        with self._hook_lock:
            hook = self._stop_hook
        if hook is not None:
            self._call(hook)

    @staticmethod
    def _call(hook: Callable[[], None]) -> None:
        try:
            hook()
        except Exception:
            log.exception("DMA reader process: stop request failed")


def run_child(
    control: Pipe,
    data: Pipe,
    *,
    label: str,
    log_level: int,
    body: Callable[[ChildContext], Mapping[str, object]],
) -> None:
    """Child entry boilerplate: logging, control watcher, final report."""
    outbox = Outbox(data)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(_ForwardingHandler(outbox))
    root.setLevel(log_level)
    context = ChildContext(outbox)

    def watch_control() -> None:
        try:
            while control.recv()[0] != "stop":
                pass
        except (EOFError, OSError):
            log.warning("%s: GUI process gone; stopping capture", label)
        context.request_stop()

    threading.Thread(target=watch_control, name="dma-reader-control", daemon=True).start()
    result: dict[str, object] = {}
    try:
        returned = body(context)
        result.update(context.result)
        result.update(returned)
        result["ok"] = True
    except BaseException as exc:
        log.error("%s: session failed", label, exc_info=True)
        result.update(context.result)
        result.update(ok=False, error=error_info(exc))
    finally:
        outbox.finish(("finished", result))


# ----------------------------------------------------------------------------
# Parent side
# ----------------------------------------------------------------------------


def rebuild_error(info: Mapping[str, object]) -> BaseException:
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
    elif isinstance(number, int):
        error = OSError(number, str(strerror) if strerror is not None else message)
    else:
        candidate = getattr(builtins, name, None)
        if (isinstance(candidate, type) and issubclass(candidate, Exception)
                and not issubclass(candidate, OSError)):
            try:
                error = candidate(message)
            except Exception:
                error = RuntimeError(f"{name}: {message}")
        else:
            error = RuntimeError(message if name == "RuntimeError" else f"{name}: {message}")
    notes = info.get("notes")
    if isinstance(notes, list):
        for note in notes:
            error.add_note(str(note))
    return error


def raise_child_error(finished: Mapping[str, object]) -> None:
    error = finished.get("error")
    if isinstance(error, dict):
        raise rebuild_error(error)
    if not finished.get("ok", False):
        raise RuntimeError("DMA reader process reported failure without detail")


class ReaderProcessCrashedError(RuntimeError):
    """The child ended without its final report; target state is unverified."""


class ReaderSession:
    """Parent-side owner of one reader process (one capture session)."""

    def __init__(
        self,
        target: Callable[[Pipe, Pipe, Any], None],
        request: object,
        *,
        label: str,
    ) -> None:
        self._target = target
        self._request = request
        self._label = label
        self._control_lock = threading.Lock()
        self._control: Pipe | None = None
        self._stop_sent = False
        self.finished: dict[str, object] | None = None
        self.exit_code: int | None = None

    def request_stop(self) -> None:
        with self._control_lock:
            if self._stop_sent or self._control is None:
                return
            try:
                self._control.send(("stop",))
            except (OSError, EOFError, ValueError):
                pass  # already exited; run() reports the outcome
            self._stop_sent = True

    def run(
        self,
        stop_event: threading.Event,
        handlers: Mapping[str, Callable[..., None]],
    ) -> dict[str, object]:
        """Run the child to completion; return its final report.

        Raises ReaderProcessCrashedError if the child ended without a report, or
        the first exception a parent-side handler raised (after stopping
        the child and waiting for its normal close).
        """
        context = multiprocessing.get_context("spawn")
        data_recv, data_send = context.Pipe(duplex=False)
        control_recv, control_send = context.Pipe(duplex=False)
        process = context.Process(
            target=self._target,
            args=(control_recv, data_send, self._request),
            name=self._label.replace(" ", "-"),
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
        consumer_error: BaseException | None = None
        try:
            while self.finished is None:
                if stop_event.is_set():
                    self.request_stop()
                if data_recv.poll(_POLL_SECONDS):
                    try:
                        message = data_recv.recv()
                    except EOFError:
                        break
                    kind = message[0]
                    if kind == "finished":
                        self.finished = message[1]
                    elif kind == "log":
                        _relay_log(message[1], self._label)
                    elif consumer_error is None and (handler := handlers.get(kind)):
                        try:
                            handler(*message[1:])
                        except BaseException as exc:
                            # Stop the capture normally; report this error.
                            consumer_error = exc
                            self.request_stop()
                elif not process.is_alive() and not data_recv.poll(0):
                    break
        finally:
            if self.finished is None and process.is_alive():
                self.request_stop()
                self.finished = _discard_until_finished(data_recv, process)
            process.join(_EXIT_JOIN_SECONDS)
            if process.is_alive():
                log.error(
                    "%s did not exit; leaving it to finish its own driver-ordered "
                    "close (never terminated)",
                    self._label,
                )
            self.exit_code = process.exitcode
            with self._control_lock:
                self._control = None
                try:
                    control_send.close()
                except OSError:
                    pass
            data_recv.close()
        if consumer_error is not None:
            child_error = self.finished.get("error") if self.finished is not None else None
            if isinstance(child_error, dict):
                consumer_error.add_note(f"Reader: {child_error.get('message')}")
            raise consumer_error
        if self.finished is None:
            raise ReaderProcessCrashedError(
                f"{self._label} ended unexpectedly (exit code {process.exitcode}); "
                "target state unverified -- check the target and acknowledge "
                "recovery before re-arming"
            )
        return self.finished


def _discard_until_finished(conn: Pipe, process: Any) -> dict[str, object] | None:
    deadline = time.monotonic() + _ORPHAN_WAIT_SECONDS
    while process.is_alive() and time.monotonic() < deadline:
        try:
            if conn.poll(_POLL_SECONDS):
                message = conn.recv()
                if message[0] == "finished":
                    finished: dict[str, object] = message[1]
                    return finished
        except (EOFError, OSError):
            return None
    return None


def _relay_log(record: object, label: str) -> None:
    if not isinstance(record, dict):
        return
    name = str(record.get("name", __name__))
    levelno = record.get("levelno", logging.INFO)
    logging.getLogger(name).log(
        levelno if isinstance(levelno, int) else logging.INFO,
        "[%s] %s", label, record.get("msg", ""),
    )
