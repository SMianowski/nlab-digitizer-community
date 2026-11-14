from __future__ import annotations

import math
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.scope import SCOPE_DATAPATH_CLOCK_PERIOD_NS


@dataclass(frozen=True)
class ScopeCurrentBin:
    """One immutable, sample-weighted Scope-current analysis interval."""

    index: int
    first_timestamp: int
    last_timestamp: int
    first_received_ns: int
    last_received_ns: int
    raw_sum: int
    sample_count: int
    frame_count: int
    payload_bytes: int
    coverage_ns: int

    @property
    def raw_mean(self) -> float:
        return self.raw_sum / self.sample_count if self.sample_count else math.nan


@dataclass(frozen=True)
class ScopeCurrentRuntime:
    """Configuration needed to interpret one accumulator session."""

    channel: int
    ip_version: int
    gap_cycles: int
    expected_interval_ticks: int
    transport: str = "pending"
    kernel_buffers: int = 0
    readbuf_batch_frames: int = 0
    queued_blocks: int = 0
    queue_high_watermark: int = 0
    viewer_state: str = "dma-preview"


@dataclass(frozen=True)
class ScopeCurrentPreview:
    """Latest owned DMA record for optional display-only preview."""

    generation: int
    timestamp: int
    received_ns: int
    record: bytes
    geometry: ScopeDmaGeometry


@dataclass(frozen=True)
class ScopeCurrentSnapshot:
    """Immutable receiver/analysis state consumed by the GUI."""

    session_id: int
    generation: int
    active: bool
    bin_width_ns: int
    geometry: ScopeDmaGeometry | None
    runtime: ScopeCurrentRuntime | None
    bins: tuple[ScopeCurrentBin, ...]
    waveform_records: tuple[bytes, ...]
    latest_raw_sample: int | None
    latest_frame_minimum: int | None
    latest_frame_maximum: int | None
    latest_timestamp: int | None
    latest_received_ns: int | None
    received_frames: int
    received_bytes: int
    rejected_frames: int
    analyzed_frames: int
    analyzed_raw_sum: int
    analyzed_sample_count: int
    discarded_analysis_frames: int
    replaced_preview_frames: int
    display_updates: int
    replaced_display_generations: int
    protocol_errors: int
    skipped_opportunities: int
    off_grid_intervals: int
    received_fps: float
    payload_mb_s: float
    median_interval_ns: float
    maximum_interval_ns: int
    observed_coverage_percent: float
    data_age_ns: int
    analysis_queue_depth: int
    analysis_queue_high_water: int
    analysis_lag_ns: int
    maximum_analysis_lag_ns: int
    last_display_duration_ns: int
    display_p95_duration_ns: float


@dataclass
class _ScopeCurrentBinBuilder:
    index: int
    first_timestamp: int
    last_timestamp: int
    first_received_ns: int
    last_received_ns: int
    raw_sum: int = 0
    sample_count: int = 0
    frame_count: int = 0
    payload_bytes: int = 0
    coverage_ns: int = 0

    def freeze(self) -> ScopeCurrentBin:
        return ScopeCurrentBin(
            index=self.index,
            first_timestamp=self.first_timestamp,
            last_timestamp=self.last_timestamp,
            first_received_ns=self.first_received_ns,
            last_received_ns=self.last_received_ns,
            raw_sum=self.raw_sum,
            sample_count=self.sample_count,
            frame_count=self.frame_count,
            payload_bytes=self.payload_bytes,
            coverage_ns=self.coverage_ns,
        )


class ScopeCurrentAccumulator:
    """Receiver-owned analysis and bounded raw display data for Scope DMA.

    Exact signed sums and sample counts are updated on the DMA thread, so
    display cadence, calibration changes and preview replacement cannot alter
    interval statistics.  A short hardware-timestamped window of complete raw
    records is retained separately for waveform display; the GUI never turns
    those records into one averaged point per frame.
    """

    _WAVEFORM_HISTORY_NS = 10_000_000

    def __init__(
        self,
        *,
        bin_width_ms: int = 100,
        history_seconds: int = 10,
    ) -> None:
        if not 1 <= bin_width_ms <= 10_000:
            raise ValueError("bin_width_ms must be 1..10000")
        if history_seconds <= 0:
            raise ValueError("history_seconds must be positive")
        self._lock = threading.Lock()
        self._history_seconds = history_seconds
        self._waveform_history_ticks = (
            self._WAVEFORM_HISTORY_NS // SCOPE_DATAPATH_CLOCK_PERIOD_NS
        )
        self._session_id = 0
        self._display_durations: deque[int] = deque(maxlen=256)
        self._interval_ticks: deque[int] = deque(maxlen=8192)
        self._set_bin_width(bin_width_ms)
        self._reset_locked()

    def _set_bin_width(self, bin_width_ms: int) -> None:
        bin_width_ns = bin_width_ms * 1_000_000
        if bin_width_ns % SCOPE_DATAPATH_CLOCK_PERIOD_NS:
            raise ValueError("analysis bin must align to the Scope timestamp clock")
        self._bin_width_ns = bin_width_ns
        self._bin_ticks = bin_width_ns // SCOPE_DATAPATH_CLOCK_PERIOD_NS
        self._maximum_bins = math.ceil(
            self._history_seconds * 1_000_000_000 / bin_width_ns
        ) + 2

    def configure_bin_width_ms(self, value: int) -> None:
        """Change summary granularity while no DMA session is active."""
        if not 1 <= value <= 10_000:
            raise ValueError("analysis bin must be 1..10000 ms")
        with self._lock:
            if self._active:
                raise RuntimeError("cannot change analysis bins during capture")
            self._set_bin_width(value)

    @property
    def bin_width_ms(self) -> int:
        with self._lock:
            return self._bin_width_ns // 1_000_000

    def _reset_locked(self) -> None:
        self._active = False
        self._geometry: ScopeDmaGeometry | None = None
        self._runtime: ScopeCurrentRuntime | None = None
        self._origin_timestamp: int | None = None
        self._previous_timestamp: int | None = None
        self._current_bin: _ScopeCurrentBinBuilder | None = None
        self._bins: deque[ScopeCurrentBin] = deque()
        self._waveform_records: deque[bytes] = deque()
        self._latest_raw_sample: int | None = None
        self._latest_frame_minimum: int | None = None
        self._latest_frame_maximum: int | None = None
        self._latest_timestamp: int | None = None
        self._latest_received_ns: int | None = None
        self._preview: ScopeCurrentPreview | None = None
        self._preview_generation = 0
        self._consumed_preview_generation = 0
        self._generation = 0
        self._received_frames = 0
        self._received_bytes = 0
        self._rejected_frames = 0
        self._analyzed_frames = 0
        self._analyzed_raw_sum = 0
        self._analyzed_sample_count = 0
        self._discarded_analysis_frames = 0
        self._replaced_preview_frames = 0
        self._display_updates = 0
        self._replaced_display_generations = 0
        self._last_display_generation = 0
        self._protocol_errors = 0
        self._skipped_opportunities = 0
        self._off_grid_intervals = 0
        self._analysis_lag_ns = 0
        self._maximum_analysis_lag_ns = 0
        self._last_display_duration_ns = 0
        self._display_durations.clear()
        self._interval_ticks.clear()

    def start_session(
        self,
        geometry: ScopeDmaGeometry,
        runtime: ScopeCurrentRuntime,
    ) -> int:
        if runtime.expected_interval_ticks <= 0:
            raise ValueError("expected periodic interval must be positive")
        with self._lock:
            if self._active:
                raise RuntimeError("Scope current analysis is already active")
            self._reset_locked()
            self._session_id += 1
            self._active = True
            self._geometry = geometry
            self._runtime = runtime
            return self._session_id

    def update_runtime_transport(
        self,
        *,
        transport: str,
        kernel_buffers: int,
        readbuf_batch_frames: int,
        queued_blocks: int,
        queue_high_watermark: int,
    ) -> None:
        with self._lock:
            if self._runtime is None:
                return
            self._runtime = replace(
                self._runtime,
                transport=transport,
                kernel_buffers=kernel_buffers,
                readbuf_batch_frames=readbuf_batch_frames,
                queued_blocks=queued_blocks,
                queue_high_watermark=queue_high_watermark,
            )

    def append_frame(
        self,
        record: bytes,
        geometry: ScopeDmaGeometry,
        received_ns: int,
    ) -> None:
        """Validate and reduce one complete immutable transport frame."""
        self.append_frames((record,), geometry, (received_ns,))

    def append_frames(
        self,
        records: Sequence[bytes],
        geometry: ScopeDmaGeometry,
        received_ns: Sequence[int],
    ) -> None:
        """Validate and reduce complete transport frames in arrival order.

        Equivalent to append_frame() for each record in turn, but one numpy
        pass per batch: the reader process forwards frames in batches, and
        per-frame Python work here limited Current DMA to ~5200 frames/s next
        to a busy GUI thread. Frames before the first invalid one are
        analysed; that one is counted as rejected and raises ValueError.
        """
        count = len(records)
        if count != len(received_ns):
            raise ValueError("each Scope current frame needs one receive time")
        if count == 0:
            return
        frame_bytes = geometry.frame_bytes
        valid = count
        for index, record in enumerate(records):
            if len(record) != frame_bytes:
                valid = index
                break
        block = np.frombuffer(b"".join(records[:valid]), dtype=np.uint8).reshape(valid, frame_bytes)
        if geometry.padding_bytes and valid:
            padded = np.flatnonzero(block[:, frame_bytes - geometry.padding_bytes :].any(axis=1))
            if padded.size:
                valid = int(padded[0])
        if geometry.waveform_samples <= 0:
            valid = 0
        if valid:
            self._reduce_frames(records[:valid], block[:valid], geometry, received_ns[:valid])
        if valid < count:
            self.note_rejected_frame()
            record = records[valid]
            if len(record) != frame_bytes:
                raise ValueError(
                    f"Scope current frame has {len(record)} bytes, expected {frame_bytes}"
                )
            if geometry.padding_bytes and any(record[-geometry.padding_bytes :]):
                raise ValueError("Scope current frame has nonzero alignment padding")
            raise ValueError("Scope current frame has no waveform samples")

    def _reduce_frames(
        self,
        records: Sequence[bytes],
        block: np.ndarray,
        geometry: ScopeDmaGeometry,
        received_ns: Sequence[int],
    ) -> None:
        count = len(records)
        # Reconcile transport independently of analysis. A snapshot taken
        # during the reduction may briefly show more received than analyzed
        # frames, but a GUI stall cannot alter either total.
        with self._lock:
            if not self._active or self._geometry != geometry:
                self._discarded_analysis_frames += count
                raise RuntimeError("Scope current frame arrived outside its analysis session")
            session_id = self._session_id
            self._received_frames += count
            self._received_bytes += count * geometry.frame_bytes

        stamps = np.ascontiguousarray(block[:, :8]).view("<u8").reshape(count)
        # int64 arithmetic is exact below 2**62 ticks (~1170 years at 8 ns);
        # object arrays keep Python-int semantics for anything larger.
        timestamps = (
            stamps.astype(np.int64) if int(stamps.max()) < 1 << 62
            else np.array([int(value) for value in stamps], dtype=object)
        )
        values = block[:, 8 : 8 + 2 * geometry.waveform_samples].view("<i2")
        sums = values.sum(axis=1, dtype=np.int64)
        samples_per_frame = geometry.waveform_samples
        last = count - 1

        with self._lock:
            if (
                not self._active
                or self._session_id != session_id
                or self._geometry != geometry
            ):
                self._discarded_analysis_frames += count
                raise RuntimeError("Scope current session ended during frame analysis")
            previous = self._previous_timestamp
            deltas = (
                np.diff(timestamps, prepend=previous) if previous is not None
                else np.diff(timestamps)
            )
            if deltas.size:
                self._interval_ticks.extend(deltas.tolist())
                expected = self._runtime.expected_interval_ticks if self._runtime else 0
                if expected:
                    off_grid = (deltas <= 0) | (deltas % expected != 0)
                    on_grid = deltas[~off_grid]
                    self._off_grid_intervals += int(np.count_nonzero(off_grid))
                    if on_grid.size:
                        self._skipped_opportunities += int((on_grid // expected - 1).sum())
                else:
                    self._off_grid_intervals += int(deltas.size)
            self._previous_timestamp = int(timestamps[last])

            if self._origin_timestamp is None:
                self._origin_timestamp = int(timestamps[0])
            bin_indices = (timestamps - self._origin_timestamp) // self._bin_ticks
            starts = [0, *(np.flatnonzero(bin_indices[1:] != bin_indices[:-1]) + 1).tolist()]
            run_sums = np.add.reduceat(sums, starts).tolist()
            ends = [*starts[1:], count]
            for start, end, run_sum in zip(starts, ends, run_sums, strict=True):
                frames = end - start
                bin_index = int(bin_indices[start])
                current = self._current_bin
                if current is None or current.index != bin_index:
                    if current is not None:
                        self._bins.append(current.freeze())
                    current = _ScopeCurrentBinBuilder(
                        index=bin_index,
                        first_timestamp=int(timestamps[start]),
                        last_timestamp=int(timestamps[start]),
                        first_received_ns=int(received_ns[start]),
                        last_received_ns=int(received_ns[start]),
                    )
                    self._current_bin = current
                    while len(self._bins) > self._maximum_bins:
                        self._bins.popleft()
                current.last_timestamp = int(timestamps[end - 1])
                current.last_received_ns = int(received_ns[end - 1])
                current.raw_sum += run_sum
                current.sample_count += samples_per_frame * frames
                current.frame_count += frames
                current.payload_bytes += geometry.frame_bytes * frames
                current.coverage_ns += geometry.capture_duration_ns * frames

            batch_sum = int(sums.sum())
            self._analyzed_frames += count
            self._analyzed_raw_sum += batch_sum
            self._analyzed_sample_count += samples_per_frame * count
            last_values = values[last]
            self._latest_raw_sample = int(last_values[-1])
            self._latest_frame_minimum = int(last_values.min())
            self._latest_frame_maximum = int(last_values.max())
            self._latest_timestamp = int(timestamps[last])
            self._latest_received_ns = int(received_ns[last])

            history = self._waveform_records
            if count == 1 or bool((deltas[-(count - 1):] >= 0).all()):
                # Monotonic batch: trimming once at its newest cutoff equals
                # trimming after every frame.
                history.extend(records)
                cutoff = int(timestamps[last]) - self._waveform_history_ticks
                while history and int.from_bytes(history[0][:8], "little") < cutoff:
                    history.popleft()
            else:
                for index, record in enumerate(records):
                    history.append(record)
                    cutoff = int(timestamps[index]) - self._waveform_history_ticks
                    while history and int.from_bytes(history[0][:8], "little") < cutoff:
                        history.popleft()

            # Every frame replaces the previous preview; within one batch no
            # consumer can take the intermediate ones.
            if self._preview is not None and (
                self._preview.generation > self._consumed_preview_generation
            ):
                self._replaced_preview_frames += 1
            self._replaced_preview_frames += count - 1
            self._preview_generation += count
            self._preview = ScopeCurrentPreview(
                generation=self._preview_generation,
                timestamp=int(timestamps[last]),
                received_ns=int(received_ns[last]),
                record=records[last],
                geometry=geometry,
            )
            now_ns = time.perf_counter_ns()
            self._analysis_lag_ns = max(0, now_ns - int(received_ns[last]))
            self._maximum_analysis_lag_ns = max(
                self._maximum_analysis_lag_ns,
                now_ns - int(min(received_ns)),
            )
            self._generation += count

    def note_rejected_frame(self) -> None:
        with self._lock:
            self._rejected_frames += 1
            self._protocol_errors += 1

    def note_protocol_error(self) -> None:
        with self._lock:
            self._protocol_errors += 1

    def finish_session(self) -> None:
        with self._lock:
            if not self._active:
                return
            if self._current_bin is not None:
                self._bins.append(self._current_bin.freeze())
                self._current_bin = None
            while len(self._bins) > self._maximum_bins:
                self._bins.popleft()
            self._active = False
            self._generation += 1

    def take_latest_preview(self) -> ScopeCurrentPreview | None:
        with self._lock:
            preview = self._preview
            if preview is not None:
                self._consumed_preview_generation = preview.generation
            return preview

    def note_display_update(self, generation: int, duration_ns: int) -> None:
        with self._lock:
            if self._last_display_generation:
                self._replaced_display_generations += max(
                    0, generation - self._last_display_generation - 1
                )
            self._last_display_generation = generation
            self._display_updates += 1
            self._last_display_duration_ns = max(0, duration_ns)
            self._display_durations.append(self._last_display_duration_ns)

    def snapshot(self, *, now_ns: int | None = None) -> ScopeCurrentSnapshot:
        current_time_ns = now_ns if now_ns is not None else 0
        with self._lock:
            bins = list(self._bins)
            if self._current_bin is not None:
                bins.append(self._current_bin.freeze())
            if bins:
                latest_index = bins[-1].index
                bins = [
                    item
                    for item in bins
                    if latest_index - item.index < self._maximum_bins
                ]
            recent = bins[-math.ceil(1_000_000_000 / self._bin_width_ns) - 1 :]
            recent_frames = sum(item.frame_count for item in recent)
            recent_bytes = sum(item.payload_bytes for item in recent)
            received_fps = payload_mb_s = 0.0
            coverage_percent = 0.0
            if recent and recent_frames > 1:
                host_span_ns = recent[-1].last_received_ns - recent[0].first_received_ns
                if host_span_ns > 0:
                    received_fps = (recent_frames - 1) * 1_000_000_000 / host_span_ns
                    first_frame_bytes = recent[0].payload_bytes / recent[0].frame_count
                    payload_mb_s = (
                        max(0.0, recent_bytes - first_frame_bytes)
                        * 1_000
                        / host_span_ns
                    )
                timestamp_span_ns = (
                    recent[-1].last_timestamp - recent[0].first_timestamp
                ) * SCOPE_DATAPATH_CLOCK_PERIOD_NS
                if timestamp_span_ns > 0:
                    coverage_ns = sum(item.coverage_ns for item in recent)
                    first_frame_coverage = recent[0].coverage_ns / recent[0].frame_count
                    coverage_percent = (
                        max(0.0, coverage_ns - first_frame_coverage)
                        * 100
                        / timestamp_span_ns
                    )

            intervals = tuple(self._interval_ticks)
            median_interval_ns = (
                statistics.median(intervals) * SCOPE_DATAPATH_CLOCK_PERIOD_NS
                if intervals
                else 0.0
            )
            maximum_interval_ns = (
                max(intervals) * SCOPE_DATAPATH_CLOCK_PERIOD_NS if intervals else 0
            )
            display_durations = sorted(self._display_durations)
            if display_durations:
                p95_index = math.ceil(0.95 * len(display_durations)) - 1
                display_p95_ns = float(display_durations[max(0, p95_index)])
            else:
                display_p95_ns = 0.0
            age_ns = (
                max(0, current_time_ns - self._latest_received_ns)
                if current_time_ns and self._latest_received_ns is not None
                else 0
            )
            return ScopeCurrentSnapshot(
                session_id=self._session_id,
                generation=self._generation,
                active=self._active,
                bin_width_ns=self._bin_width_ns,
                geometry=self._geometry,
                runtime=self._runtime,
                bins=tuple(bins),
                waveform_records=tuple(self._waveform_records),
                latest_raw_sample=self._latest_raw_sample,
                latest_frame_minimum=self._latest_frame_minimum,
                latest_frame_maximum=self._latest_frame_maximum,
                latest_timestamp=self._latest_timestamp,
                latest_received_ns=self._latest_received_ns,
                received_frames=self._received_frames,
                received_bytes=self._received_bytes,
                rejected_frames=self._rejected_frames,
                analyzed_frames=self._analyzed_frames,
                analyzed_raw_sum=self._analyzed_raw_sum,
                analyzed_sample_count=self._analyzed_sample_count,
                discarded_analysis_frames=self._discarded_analysis_frames,
                replaced_preview_frames=self._replaced_preview_frames,
                display_updates=self._display_updates,
                replaced_display_generations=self._replaced_display_generations,
                protocol_errors=self._protocol_errors,
                skipped_opportunities=self._skipped_opportunities,
                off_grid_intervals=self._off_grid_intervals,
                received_fps=received_fps,
                payload_mb_s=payload_mb_s,
                median_interval_ns=median_interval_ns,
                maximum_interval_ns=maximum_interval_ns,
                observed_coverage_percent=coverage_percent,
                data_age_ns=age_ns,
                # Reduction is synchronous on the receiver thread, so there
                # is deliberately no analysis queue to grow or overflow.
                analysis_queue_depth=0,
                analysis_queue_high_water=0,
                analysis_lag_ns=self._analysis_lag_ns,
                maximum_analysis_lag_ns=self._maximum_analysis_lag_ns,
                last_display_duration_ns=self._last_display_duration_ns,
                display_p95_duration_ns=display_p95_ns,
            )


class CurrentMonitorClient(ABC):
    """Small worker-owned connection for the live FPGA IIR readout.

    The client deliberately exposes only the one operation needed by the
    current monitor.  Keeping it separate from ``Digitizer`` prevents a
    background polling thread from sharing an IIO context with
    GUI/configuration calls.
    """

    @abstractmethod
    def read_raw(self) -> int:
        """Return the current signed IIR output code."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the worker-owned transport."""
        ...
