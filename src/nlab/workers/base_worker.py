from PySide6.QtCore import QObject, Signal, Slot

# Re-exported: workers import the classifier from here.
from nlab.utils.transport_errors import is_fatal_transport_error as is_fatal_transport_error


class BaseWorker(QObject):
    """Abstract base for all background workers.

    Usage pattern (in a controller):
        thread = QThread()
        worker = ConcreteWorker(...)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(
            controller._on_worker_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        thread.finished.connect(thread.deleteLater)
        thread.start()

    Tying worker deletion to ``QThread.finished`` ensures the thread handles
    its deferred deletion before GUI-side code releases the Python wrapper.
    GUI/controller cleanup handlers must use an explicit queued connection so
    they never manipulate widgets from the thread emitting ``finished``.
    """

    finished = Signal()
    error = Signal(str)
    # Timer-driven pollers (``self._timer`` / ``self._interval_ms``) can be
    # paused while another activity needs the board's iiod to itself.
    # Pausing only skips polls; it never changes hardware state.
    set_paused = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._paused = False
        self.set_paused.connect(self._set_paused)

    def set_initial_paused(self, paused: bool) -> None:
        """Start paused; call before the worker's thread starts."""
        self._paused = paused

    @Slot(bool)
    def _set_paused(self, paused: bool) -> None:
        self._paused = paused
        timer = getattr(self, "_timer", None)
        if timer is None:
            return
        if paused:
            timer.stop()
        elif not timer.isActive() and not getattr(self, "_finished", False):
            timer.start(getattr(self, "_interval_ms", timer.interval()))

    def run(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class PauseRequests:
    """Named pause requests; a poller stays paused while any is active.

    Separate names keep one requester (e.g. one Current channel) from
    resuming polling that another still needs paused.
    """

    def __init__(self) -> None:
        self._reasons: set[str] = set()

    @property
    def active(self) -> bool:
        return bool(self._reasons)

    def update(self, reason: str, paused: bool) -> bool:
        """Record a request; return whether the effective state changed."""
        before = self.active
        if paused:
            self._reasons.add(reason)
        else:
            self._reasons.discard(reason)
        return self.active != before


def pause_requests(owner: object) -> PauseRequests:
    """The owner's PauseRequests, created on first use."""
    requests = getattr(owner, "_readback_pause_requests", None)
    if not isinstance(requests, PauseRequests):
        requests = PauseRequests()
        setattr(owner, "_readback_pause_requests", requests)
    return requests
