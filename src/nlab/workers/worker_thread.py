"""Run a BaseWorker on its own QThread, and stop it without ever terminating it.

Every polling owner (MCA, PSU, Global diagnostics and temperature correction,
Current IIR monitor, external devices) uses the same wiring and the same
shutdown policy:

* ``finished -> thread.quit`` is a direct connection: owners wait for the
  thread on the GUI thread, where the QThread object lives, and a queued
  quit() would sit behind that wait for the whole timeout.
* ``join_worker_thread`` never calls ``QThread.terminate()``. Termination can
  kill a thread inside a network IIO or Modbus call and leave the transport
  and Python object graph inconsistent, and owners close those connections
  right after joining. Each such call is bounded by its own transport timeout,
  so waiting for it always ends.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QThread

from nlab.workers.base_worker import BaseWorker

log = logging.getLogger(__name__)


def create_worker_thread(
    owner: QObject,
    worker: BaseWorker,
    *,
    on_finished: Callable[[], None],
    delete_thread_on_finish: bool = True,
) -> QThread:
    """Wire *worker* onto a new QThread owned by *owner*; the caller starts it.

    Connect worker-specific signals before ``thread.start()``. *on_finished*
    runs queued on the owner's thread after the worker's thread has finished.
    """
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit, Qt.ConnectionType.DirectConnection)
    thread.finished.connect(worker.deleteLater)
    if delete_thread_on_finish:
        thread.finished.connect(thread.deleteLater)
    thread.finished.connect(on_finished, Qt.ConnectionType.QueuedConnection)
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
