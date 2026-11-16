"""Run a BaseWorker on its own QThread, and stop it without ever terminating it.

Every polling owner (MCA, PSU, Global diagnostics and temperature correction,
Current IIR monitor, external devices) uses the same wiring and the same
shutdown policy:

* ``finished -> thread.quit`` is a direct connection: owners wait for the
  thread on the GUI thread, where the QThread object lives, and a queued
  quit() would sit behind that wait for the whole timeout.
* The owner's *on_finished* callback receives the thread that finished. It
  is queued, so it can run after the owner has already started a replacement
  worker (Stop then Start, or a synchronous stop followed by a restart); the
  owner must ignore a thread that is no longer its current one, or it would
  forget the running worker and could never stop or join it.
* ``join_worker_thread`` never calls ``QThread.terminate()``. Termination can
  kill a thread inside a network IIO or Modbus call and leave the transport
  and Python object graph inconsistent, and owners close those connections
  right after joining. Each such call is bounded by its own transport timeout,
  so waiting for it always ends.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QThread, Slot

from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


class _FinishedRelay(QObject):
    """Delivers ``thread.finished`` to the owner's thread with the thread."""

    def __init__(
        self, owner: QObject, thread: QThread, callback: Callable[[QThread], None],
    ) -> None:
        super().__init__(owner)
        self._thread = thread
        self._callback = callback

    @Slot()
    def fire(self) -> None:
        try:
            self._callback(self._thread)
        finally:
            self.deleteLater()


def create_worker_thread(
    owner: QObject,
    worker: BaseWorker,
    *,
    on_finished: Callable[[QThread], None],
    delete_thread_on_finish: bool = True,
) -> QThread:
    """Wire *worker* onto a new QThread owned by *owner*; the caller starts it.

    Connect worker-specific signals before ``thread.start()``. *on_finished*
    runs queued on the owner's thread after the worker's thread has finished
    and receives that thread; ignore it unless it is still the current one.
    """
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    thread.finished.connect(worker.deleteLater)
    if delete_thread_on_finish:
        thread.finished.connect(thread.deleteLater)
    relay = _FinishedRelay(owner, thread, on_finished)
    thread.finished.connect(relay.fire, Qt.ConnectionType.QueuedConnection)
    return thread


def join_worker_thread(
    thread: QThread | None,
    *,
    label: str,
    warn_after_ms: int = 3000,
) -> None:
    """Wait for a worker thread whose stop was already requested; never terminate it."""
    if thread is None:
        return
    try:
        if thread.wait(warn_after_ms):
            return
    except RuntimeError:
        return  # the finished thread was already deleted
    log.warning(
        "%s: waiting for an in-flight hardware call to finish "
        "(worker threads are never terminated)",
        label,
    )
    thread.quit()
    thread.wait()
