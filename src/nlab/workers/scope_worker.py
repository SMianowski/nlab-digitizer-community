from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from nlab.hardware.digitizer.scope import SCOPE_DATAPATH_CLOCK_PERIOD_NS, Scope
from nlab.utils.transport_errors import is_fatal_transport_error

log = logging.getLogger(__name__)

_VIEWER_POINT_PERIOD_NS = SCOPE_DATAPATH_CLOCK_PERIOD_NS


class _FrameSignals(QObject):
    ready = Signal(object)
    # Scalar error data only (message, connection lost): see AGENTS.md.
    failed = Signal(str, bool)


class ScopeWorker(QRunnable):
    """Acquires a single scope frame off the GUI thread.

    For IIO, a lazy provider supplies the viewer-only Scope connection.
    Emits ``signals.ready`` with ``[time_array, frame_array]`` on success,
    or ``None`` on failure.  Auto-deletes after run.
    """

    def __init__(self, scope: Scope | Callable[[], Scope]) -> None:
        super().__init__()
        self.signals = _FrameSignals()
        self._scope = scope
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            # The IIO viewer supplies a lazy, viewer-only Scope. Never use
            # the GUI/config context from this pool thread.
            scope = self._scope() if callable(self._scope) else self._scope
            frame_samples = scope.get_frame_samples()
            raw_frame = scope.acquire_frame()
            frame = raw_frame[: int(frame_samples) // 4]
            raw_time = np.arange(len(frame)) * _VIEWER_POINT_PERIOD_NS
            self.signals.ready.emit([raw_time, frame])
        except Exception as exc:
            connection_lost = is_fatal_transport_error(exc)
            log.exception(
                "Frame acquisition failed; connection lost"
                if connection_lost
                else "Frame acquisition failed"
            )
            self.signals.failed.emit(str(exc), connection_lost)
            self.signals.ready.emit(None)
