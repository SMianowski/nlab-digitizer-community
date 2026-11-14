"""Background analysis of two independently recorded MCA list-mode streams."""

from __future__ import annotations

import logging
import threading
import time

from nlab.analysis.coincidence import (
    CoincidenceAnalyzer,
    CoincidencePeakFit,
    CoincidenceSettings,
    CoincidenceSnapshot,
    fit_coincidence_peak,
)
from nlab.hardware.digitizer.dma import McaEventBuffer

log = logging.getLogger(__name__)

# The Gaussian timing fit (scipy curve_fit) runs here, never on the GUI
# thread; while data streams it is refreshed at most this often.
_FIT_INTERVAL_S = 0.5


def _pair_mode(settings: CoincidenceSettings) -> bool:
    return settings.operator == "AND" and not (settings.not_ch0 or settings.not_ch1)


class CoincidenceAnalysisThread(threading.Thread):
    def __init__(
        self,
        ch0: McaEventBuffer,
        ch1: McaEventBuffer,
        settings: CoincidenceSettings,
    ) -> None:
        super().__init__(name="coincidence-analysis", daemon=False)
        self._buffers = (ch0, ch1)
        self._settings = settings
        self._analyzer = CoincidenceAnalyzer(settings)
        self._snapshot = self._analyzer.snapshot()
        self._timing_fit: CoincidencePeakFit | None = None
        self._fit_stale = False
        self._last_fit_s = float("-inf")
        self._lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._pending_settings: CoincidenceSettings | None = None
        self._error: str | None = None

    def request_settings(self, settings: CoincidenceSettings) -> None:
        """Begin a fresh analysis generation without altering raw recording."""
        with self._lock:
            self._pending_settings = settings
        for buffer in self._buffers:
            buffer.clear()

    def request_stop(self) -> None:
        self._stop_requested.set()

    def result(self) -> tuple[CoincidenceSnapshot, str | None]:
        with self._lock:
            return self._snapshot, self._error

    def timing_fit(self) -> CoincidencePeakFit | None:
        """Latest timing fit (pair mode only); may trail the snapshot by one interval."""
        with self._lock:
            return self._timing_fit

    def _refit(self, snapshot: CoincidenceSnapshot) -> None:
        fit: CoincidencePeakFit | None = None
        if _pair_mode(self._settings):
            try:
                fit = fit_coincidence_peak(snapshot.delay_counts, self._settings)
            except Exception:
                log.exception("Coincidence timing fit failed")
        self._fit_stale = False
        self._last_fit_s = time.monotonic()
        with self._lock:
            self._timing_fit = fit

    def run(self) -> None:
        try:
            while True:
                with self._lock:
                    settings = self._pending_settings
                    self._pending_settings = None
                if settings is not None:
                    self._settings = settings
                    self._analyzer = CoincidenceAnalyzer(settings)
                    with self._lock:
                        self._timing_fit = None
                batches = [buffer.drain() for buffer in self._buffers]
                changed = settings is not None or any(
                    channel_batches for channel_batches, _ in batches
                )
                for channel, (channel_batches, dropped) in enumerate(batches):
                    if dropped:
                        raise RuntimeError(
                            f"channel {channel} live queue dropped {dropped} records; "
                            "coincidence results are invalid"
                        )
                    for events in channel_batches:
                        self._analyzer.add_batch(channel, events)
                if changed:
                    snapshot = self._analyzer.snapshot()
                    with self._lock:
                        self._snapshot = snapshot
                    self._fit_stale = True
                if self._stop_requested.is_set() and not any(batch[0] for batch in batches):
                    self._analyzer.finish()
                    snapshot = self._analyzer.snapshot()
                    with self._lock:
                        self._snapshot = snapshot
                    self._refit(snapshot)
                    return
                if self._fit_stale and time.monotonic() - self._last_fit_s >= _FIT_INTERVAL_S:
                    self._refit(self._snapshot)
                self._stop_requested.wait(0.02)
        except Exception as exc:
            log.exception("Coincidence analysis failed")
            with self._lock:
                self._error = str(exc)
