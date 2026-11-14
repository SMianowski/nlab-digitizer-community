"""Shared worker-thread wiring and the never-terminate join policy."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from PySide6.QtCore import QObject, QThread
from pytestqt.qtbot import QtBot

from nlab.workers.base_worker import BaseWorker
from nlab.workers.worker_thread import create_worker_thread, join_worker_thread


class _BlockingWorker(BaseWorker):
    """run() blocks like an in-flight transport call, then finishes."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self.release = release
        self.ran_on: QThread | None = None

    def run(self) -> None:
        self.ran_on = QThread.currentThread()
        self.release.wait(5)
        self.finished.emit()


def test_worker_runs_on_its_own_thread_and_owner_is_called_back(qtbot: QtBot) -> None:
    owner = QObject()
    release = threading.Event()
    release.set()
    worker = _BlockingWorker(release)
    finished: list[None] = []
    thread = create_worker_thread(owner, worker, on_finished=lambda: finished.append(None))
    thread.start()
    qtbot.waitUntil(lambda: finished == [None], timeout=5000)
    assert worker.ran_on is not None and worker.ran_on is not owner.thread()


def test_join_waits_for_an_in_flight_call_instead_of_terminating(
    qtbot: QtBot, caplog: Any,
) -> None:
    owner = QObject()
    release = threading.Event()
    worker = _BlockingWorker(release)
    thread = create_worker_thread(owner, worker, on_finished=lambda: None,
                                  delete_thread_on_finish=False)
    thread.start()
    threading.Timer(0.3, release.set).start()  # the "network call" returns later
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="nlab.workers.worker_thread"):
        join_worker_thread(thread, label="test poller", warn_after_ms=50)
    assert time.monotonic() - started >= 0.25  # waited for the call to return
    assert thread.isFinished()
    assert any("never terminated" in record.getMessage() for record in caplog.records)


def test_join_tolerates_missing_or_deleted_threads(qtbot: QtBot) -> None:
    join_worker_thread(None, label="none")
    owner = QObject()
    release = threading.Event()
    release.set()
    worker = _BlockingWorker(release)  # owners keep their worker referenced
    finished: list[None] = []
    thread = create_worker_thread(owner, worker, on_finished=lambda: finished.append(None))
    thread.start()
    qtbot.waitUntil(lambda: finished == [None], timeout=5000)
    qtbot.wait(50)  # let thread.deleteLater run
    join_worker_thread(thread, label="deleted")
