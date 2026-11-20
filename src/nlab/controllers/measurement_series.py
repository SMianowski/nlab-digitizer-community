"""Repeated timed measurements ("series") for the Scope, MCA and Current views.

Settings > Enable Series lets one Start run the view's measurement several
times. The series never touches hardware itself: each run is started and
stopped through the view's own lifecycle, so every run repeats the complete
arm sequence and every DMA invariant still applies per run.

* A run that ends on its time limit starts the next run once the view is idle
  again: polling workers have exited and any DMA reader has drained, closed its
  buffer and released its thread (AGENTS.md DMA invariant 6).
* Any other end -- Stop, an error, a lost connection, session replay,
  coincidence, reconnect or shutdown -- ends the series. A failed run is never
  restarted automatically.
* While the option is checked, every recording is named
  ``<core>_NNN<suffix>``. The core name is the file the operator chose, or the
  generated default, fixed once for the whole series.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QLabel, QSpinBox

log = logging.getLogger(__name__)

MAX_SERIES_RUNS = 9999
_IDLE_RETRY_MS = 50
# DMA teardown is finished before a run reports its end; the wait only covers
# a polling worker that is still leaving its thread.
_IDLE_TIMEOUT_S = 30.0

SERIES_TOOLTIP = (
    "Number of runs one Start performs while Settings > Enable Series is checked. "
    "Each run lasts Time [s] and the next one starts after the previous run has "
    "stopped and released the board; Stop, an error or a lost connection ends the "
    "series. Recordings are numbered <name>_001, <name>_002, ..., where <name> is "
    "the chosen file or the generated default."
)
SERIES_DISABLED_TOOLTIP = "Check Settings > Enable Series to repeat measurements."


def time_limit_required_message(runs: int) -> str:
    """Status text for a series that cannot end its runs by itself."""
    return f"Set Time [s] above 0 to run a series of {runs} measurements."


def numbered_path(path: Path, run: int, total: int) -> Path:
    """``<stem>_NNN<suffix>``, zero-padded to at least three digits."""
    width = max(3, len(str(total)))
    return path.with_name(f"{path.stem}_{run:0{width}d}{path.suffix}")


class MeasurementSeries(QObject):
    """Run counter, recording names and next-run scheduling for one view.

    The view reports its lifecycle with :meth:`begin` (operator Start),
    :meth:`mark_completed` (time limit reached, before stopping),
    :meth:`stop_requested` (any stop), :meth:`run_ended` (the run's resources
    are released) and :meth:`cancel` (errors, replay, shutdown).
    """

    def __init__(
        self,
        parent: QObject,
        *,
        spinbox: QSpinBox,
        label: QLabel,
        name: str,
        is_idle: Callable[[], bool],
        start_run: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self._spinbox = spinbox
        self._label = label
        self._idle_text = label.text()
        self._name = name
        self._is_idle = is_idle
        self._start_run = start_run
        self._enabled = False
        self._total = 0
        self._run = 0
        self._completed = False
        self._base_path: Path | None = None
        self._idle_deadline = 0.0
        self._next_timer = QTimer(self)
        self._next_timer.setSingleShot(True)
        self._next_timer.timeout.connect(self._try_start_next)
        spinbox.setRange(1, MAX_SERIES_RUNS)
        spinbox.setValue(1)
        self._sync_widgets()

    @property
    def enabled(self) -> bool:
        """Whether Settings > Enable Series is checked."""
        return self._enabled

    @property
    def active(self) -> bool:
        """Whether a series is running, including the pause between runs."""
        return self._total > 0

    @property
    def run(self) -> int:
        """1-based number of the current run; 0 without an active series."""
        return self._run

    @property
    def total(self) -> int:
        return self._total

    def set_enabled(self, enabled: bool) -> None:
        """Apply the global option; unchecking makes the current run the last."""
        self._enabled = bool(enabled)
        if not self._enabled and self.active:
            if self._next_timer.isActive():
                log.info("%s: series disabled before run %d started", self._name, self._run)
                self.cancel()
                return
            log.info("%s: series disabled; run %d is the last", self._name, self._run)
            self._total = self._run
        self._sync_widgets()

    def set_runs(self, runs: int) -> None:
        """Restore a saved run count; an active series keeps its own."""
        if not self.active:
            self._spinbox.setValue(max(1, min(MAX_SERIES_RUNS, int(runs))))

    def runs(self) -> int:
        return self._spinbox.value()

    def begin(self, time_limit_s: float) -> bool:
        """Prepare for the operator's Start.

        Returns False, leaving no series active, when more than one run is
        requested without a time limit: such a run would never end by itself.
        Without the option checked this only clears a previous series.
        """
        self.cancel()
        if not self._enabled:
            return True
        total = self._spinbox.value()
        if total > 1 and time_limit_s <= 0:
            return False
        self._total = total
        self._run = 1
        log.info("%s: starting a series of %d run(s)", self._name, total)
        self._sync_widgets()
        return True

    def recording_path(self, default: Callable[[], Path]) -> Path | None:
        """This run's numbered recording, or None without an active series.

        *default* supplies the series' core name and is called once per series:
        the operator's chosen file, or a newly generated one.
        """
        if not self.active:
            return None
        if self._base_path is None:
            self._base_path = default()
            log.info("%s: series recordings use %s", self._name, self._base_path)
        return numbered_path(self._base_path, self._run, self._total)

    def mark_completed(self) -> None:
        """The current run reached its time limit; call before stopping it."""
        if self.active:
            self._completed = True

    def stop_requested(self) -> None:
        """A stop that is not the time limit ends the series."""
        if self.active and not self._completed:
            log.info("%s: series stopped during run %d of %d", self._name, self._run, self._total)
            self.cancel()

    def run_ended(self, error: str | None = None) -> None:
        """The run's resources are released; start the next run if it completed."""
        if not self.active or self._next_timer.isActive():
            return
        if error:
            log.warning(
                "%s: series ended after run %d of %d failed: %s",
                self._name, self._run, self._total, error,
            )
            self.cancel()
            return
        if not self._completed:
            self.cancel()
            return
        if self._run >= self._total:
            log.info("%s: series of %d run(s) complete", self._name, self._total)
            self.cancel()
            return
        self._completed = False
        self._run += 1
        self._idle_deadline = time.monotonic() + _IDLE_TIMEOUT_S
        self._sync_widgets()
        self._next_timer.start(0)

    def cancel(self) -> None:
        """End the series; a run in progress continues as an ordinary run."""
        self._next_timer.stop()
        self._total = 0
        self._run = 0
        self._completed = False
        self._base_path = None
        self._sync_widgets()

    def _try_start_next(self) -> None:
        if not self.active:
            return
        if not self._is_idle():
            if time.monotonic() >= self._idle_deadline:
                log.error(
                    "%s: series ended; the view did not become idle for run %d",
                    self._name, self._run,
                )
                self.cancel()
            else:
                self._next_timer.start(_IDLE_RETRY_MS)
            return
        log.info("%s: starting series run %d of %d", self._name, self._run, self._total)
        try:
            self._start_run()
        except Exception:
            log.exception("%s: series run %d failed to start", self._name, self._run)
            self.cancel()
            return
        if self.active and self._is_idle():
            # The view declined the start (interlock, lost connection, ...)
            # and has already reported why.
            log.warning("%s: series ended; run %d did not start", self._name, self._run)
            self.cancel()

    def _sync_widgets(self) -> None:
        self._spinbox.setEnabled(self._enabled and not self.active)
        self._spinbox.setToolTip(SERIES_TOOLTIP if self._enabled else SERIES_DISABLED_TOOLTIP)
        self._label.setToolTip(self._spinbox.toolTip())
        if self.active:
            colon = ":" if self._idle_text.endswith(":") else ""
            self._label.setText(f"Run {self._run}/{self._total}{colon}")
        else:
            self._label.setText(self._idle_text)


def owner_series(owner: object) -> MeasurementSeries | None:
    """The view's series; None for lightweight owners used by unit tests."""
    series = getattr(owner, "_series", None)
    return series if isinstance(series, MeasurementSeries) else None
