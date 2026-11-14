"""Fake list-mode reader children for spawn tests (must be importable by name)."""

from __future__ import annotations

import errno
import logging
import os
from typing import Any

import numpy as np

from nlab.hardware.digitizer.iio_listmode import IIO_LM_EVENT_DTYPE


def frames(count: int, seed: int = 7) -> np.ndarray:
    data = np.random.default_rng(seed).integers(
        0, 256, size=count * 1024 * IIO_LM_EVENT_DTYPE.itemsize, dtype=np.uint8,
    )
    return data.view(IIO_LM_EVENT_DTYPE)


def streams_then_finishes(control: Any, data: Any, request: Any) -> None:
    data.send(("ready",))
    data.send(("log", {"name": "nlab.fake", "levelno": logging.WARNING, "msg": "hello"}))
    data.send(("events", frames(2).tobytes(), 0))
    data.send(("events", b"", 5))
    data.send(("progress", 2048))
    data.send(("finished", {
        "ok": True, "records": 3072, "recovery_required": False,
        "diagnostics": {"streamed_frames": 3, "completed_frames": 3, "continuity_valid": True},
    }))


def waits_for_stop(control: Any, data: Any, request: Any) -> None:
    data.send(("ready",))
    assert control.recv() == ("stop",)
    data.send(("finished", {"ok": True, "records": 1024, "recovery_required": False}))


def reports_failure(control: Any, data: Any, request: Any) -> None:
    fatal = request.channel == 1
    data.send(("finished", {
        "ok": False, "records": 0, "recovery_required": True,
        "error": {
            "type": "ConnectionResetError" if fatal else "OSError",
            "message": "[Errno 9] Drain: Bad file descriptor",
            "errno": errno.ECONNRESET if fatal else errno.EBADF,
            "strerror": "Drain: Bad file descriptor",
            "notes": ["Capture cleanup: example"],
            "fatal_transport": fatal,
        },
    }))


def crashes(control: Any, data: Any, request: Any) -> None:
    data.send(("ready",))
    os._exit(3)
