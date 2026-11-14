"""Batched Scope-current ingest must equal the original per-frame analysis.

``legacy_append_frame`` is the pre-batching ScopeCurrentAccumulator method,
kept verbatim as a reference model operating on the same private state.
"""

from __future__ import annotations

import random
import struct
import time

import numpy as np
import pytest

from nlab.hardware.digitizer.current_monitor import (
    ScopeCurrentAccumulator,
    ScopeCurrentPreview,
    ScopeCurrentRuntime,
    _ScopeCurrentBinBuilder,
)
from nlab.hardware.digitizer.dma import ScopeDmaGeometry, ScopeFrameBuffer

GEOMETRY = ScopeDmaGeometry(
    frame_samples=512, buffer_samples=132, frame_bytes=264,
    waveform_samples=127, sample_decimation=4, padding_bytes=2,
)
INTERVAL = 40_000  # ticks; with 1 ms bins (125000 ticks) bins roll every ~3 frames


def legacy_append_frame(
    self: ScopeCurrentAccumulator,
    record: bytes,
    geometry: ScopeDmaGeometry,
    received_ns: int,
) -> None:
    """Pre-batching append_frame(), verbatim: the reference model."""
    if len(record) != geometry.frame_bytes:
        self.note_rejected_frame()
        raise ValueError(
            f"Scope current frame has {len(record)} bytes, expected {geometry.frame_bytes}"
        )
    if geometry.padding_bytes and any(record[-geometry.padding_bytes :]):
        self.note_rejected_frame()
        raise ValueError("Scope current frame has nonzero alignment padding")
    if geometry.waveform_samples <= 0:
        self.note_rejected_frame()
        raise ValueError("Scope current frame has no waveform samples")

    # Reconcile transport independently of analysis. A snapshot taken
    # during the reduction may briefly show one more received than
    # analyzed frame, but a GUI stall cannot alter either total.
    with self._lock:
        if not self._active or self._geometry != geometry:
            self._discarded_analysis_frames += 1
            raise RuntimeError("Scope current frame arrived outside its analysis session")
        session_id = self._session_id
        self._received_frames += 1
        self._received_bytes += len(record)

    timestamp = int.from_bytes(record[:8], "little")
    values = np.frombuffer(
        record,
        dtype="<i2",
        count=geometry.waveform_samples,
        offset=8,
    )
    raw_sum = int(values.sum(dtype=np.int64))
    sample_count = int(values.size)
    latest_raw_sample = int(values[-1])
    frame_minimum = int(values.min())
    frame_maximum = int(values.max())

    with self._lock:
        if (
            not self._active
            or self._session_id != session_id
            or self._geometry != geometry
        ):
            self._discarded_analysis_frames += 1
            raise RuntimeError("Scope current session ended during frame analysis")
        previous = self._previous_timestamp
        if previous is not None:
            delta = timestamp - previous
            self._interval_ticks.append(delta)
            expected = self._runtime.expected_interval_ticks if self._runtime else 0
            if delta <= 0 or not expected or delta % expected:
                self._off_grid_intervals += 1
            else:
                self._skipped_opportunities += max(0, delta // expected - 1)
        self._previous_timestamp = timestamp

        if self._origin_timestamp is None:
            self._origin_timestamp = timestamp
        bin_index = (timestamp - self._origin_timestamp) // self._bin_ticks
        current = self._current_bin
        if current is None or current.index != bin_index:
            if current is not None:
                self._bins.append(current.freeze())
            current = _ScopeCurrentBinBuilder(
                index=bin_index,
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                first_received_ns=received_ns,
                last_received_ns=received_ns,
            )
            self._current_bin = current
            while len(self._bins) > self._maximum_bins:
                self._bins.popleft()

        current.last_timestamp = timestamp
        current.last_received_ns = received_ns
        current.raw_sum += raw_sum
        current.sample_count += sample_count
        current.frame_count += 1
        current.payload_bytes += len(record)
        current.coverage_ns += geometry.capture_duration_ns
        self._analyzed_frames += 1
        self._analyzed_raw_sum += raw_sum
        self._analyzed_sample_count += sample_count
        self._latest_raw_sample = latest_raw_sample
        self._latest_frame_minimum = frame_minimum
        self._latest_frame_maximum = frame_maximum
        self._latest_timestamp = timestamp
        self._latest_received_ns = received_ns
        self._waveform_records.append(record)
        waveform_cutoff = timestamp - self._waveform_history_ticks
        while (
            self._waveform_records
            and int.from_bytes(self._waveform_records[0][:8], "little")
            < waveform_cutoff
        ):
            self._waveform_records.popleft()

        if self._preview is not None and (
            self._preview.generation > self._consumed_preview_generation
        ):
            self._replaced_preview_frames += 1
        self._preview_generation += 1
        self._preview = ScopeCurrentPreview(
            generation=self._preview_generation,
            timestamp=timestamp,
            received_ns=received_ns,
            record=record,
            geometry=geometry,
        )
        self._analysis_lag_ns = max(0, time.perf_counter_ns() - received_ns)
        self._maximum_analysis_lag_ns = max(
            self._maximum_analysis_lag_ns,
            self._analysis_lag_ns,
        )
        self._generation += 1


def _records(seed: int, count: int) -> tuple[list[bytes], list[int]]:
    rng = random.Random(seed)
    timestamp = 10_000
    records, stamps = [], []
    for index in range(count):
        roll = rng.random()
        if roll < 0.05:
            timestamp += INTERVAL * rng.randint(2, 5)  # skipped opportunities
        elif roll < 0.07:
            timestamp += rng.randint(1, INTERVAL - 1)  # off-grid
        elif roll < 0.08:
            timestamp -= rng.randint(1, 3 * INTERVAL)  # non-monotonic anomaly
        else:
            timestamp += INTERVAL
        waveform = np.array([rng.randint(-32768, 32767) for _ in range(127)], dtype="<i2")
        records.append(struct.pack("<Q", timestamp) + waveform.tobytes() + b"\x00\x00")
        stamps.append(1_000_000 + index * 1000 + rng.randint(0, 999))
    return records, stamps


def _new() -> ScopeCurrentAccumulator:
    accumulator = ScopeCurrentAccumulator(bin_width_ms=1, history_seconds=1)
    accumulator.start_session(GEOMETRY, ScopeCurrentRuntime(0, 124, 0, INTERVAL))
    return accumulator


_TIMING_FIELDS = {"analysis_lag_ns", "maximum_analysis_lag_ns", "data_age_ns"}


def _comparable(accumulator: ScopeCurrentAccumulator) -> dict[str, object]:
    snapshot = accumulator.snapshot(now_ns=5_000_000_000)
    values = {
        name: getattr(snapshot, name)
        for name in snapshot.__dataclass_fields__
        if name not in _TIMING_FIELDS
    }
    values["intervals"] = list(accumulator._interval_ticks)
    preview = accumulator.take_latest_preview()
    values["preview"] = None if preview is None else (preview.generation, preview.record)
    return values


@pytest.mark.parametrize("seed", range(6))
def test_batched_ingest_matches_per_frame_reference(seed: int) -> None:
    records, stamps = _records(seed, 600)
    reference, batched = _new(), _new()
    for record, stamp in zip(records, stamps, strict=True):
        legacy_append_frame(reference, record, GEOMETRY, stamp)
    rng = random.Random(seed + 100)
    start = 0
    while start < len(records):
        size = rng.choice([1, 2, 7, 64, 300])
        batched.append_frames(records[start:start + size], GEOMETRY, stamps[start:start + size])
        if rng.random() < 0.3:
            batched.take_latest_preview()
            reference_preview = reference._preview
            if reference_preview is not None:
                reference._consumed_preview_generation = reference_preview.generation
        start += size
    # Preview consumption timing differs between the two runs by design, so
    # compare replacement counts only for an unconsumed-at-every-step run.
    expected, actual = _comparable(reference), _comparable(batched)
    expected.pop("replaced_preview_frames")
    actual.pop("replaced_preview_frames")
    assert actual == expected


def test_preview_replacement_counts_match_without_consumption() -> None:
    records, stamps = _records(42, 200)
    reference, batched = _new(), _new()
    for record, stamp in zip(records, stamps, strict=True):
        legacy_append_frame(reference, record, GEOMETRY, stamp)
    for start in range(0, 200, 37):
        batched.append_frames(records[start:start + 37], GEOMETRY, stamps[start:start + 37])
    assert _comparable(batched) == _comparable(reference)


def test_invalid_frame_mid_batch_analyses_prefix_then_rejects() -> None:
    records, stamps = _records(3, 10)
    records[6] = records[6][:-1] + b"\x01"  # nonzero alignment padding
    accumulator = _new()
    with pytest.raises(ValueError, match="nonzero alignment padding"):
        accumulator.append_frames(records, GEOMETRY, stamps)
    snapshot = accumulator.snapshot()
    assert snapshot.analyzed_frames == 6
    assert snapshot.rejected_frames == 1 and snapshot.protocol_errors == 1


def test_frame_buffer_batch_publish_equals_per_frame_append() -> None:
    records, stamps = _records(5, 700)
    one_by_one, batched = ScopeFrameBuffer(max_frames=256), ScopeFrameBuffer(max_frames=256)
    batched_subscriber = ScopeFrameBuffer(max_frames=100)
    batched.subscribe(batched_subscriber)
    from nlab.hardware.digitizer.dma import ScopeDmaFrame

    for record, stamp in zip(records[:300], stamps[:300], strict=True):
        one_by_one.append(ScopeDmaFrame(
            int.from_bytes(record[:8], "little"), stamp,
            np.frombuffer(record, "<i2", count=127, offset=8), GEOMETRY,
        ))
    batched.append_records(records[:300], GEOMETRY, stamps[:300])
    expected_frames, expected_dropped = one_by_one.drain()
    actual_frames, actual_dropped = batched.drain()
    assert actual_dropped == expected_dropped == 44
    assert [f.timestamp for f in actual_frames] == [f.timestamp for f in expected_frames]
    assert batched_subscriber.drain()[1] == 200


def test_large_batch_ingest_is_fast() -> None:
    records, stamps = _records(9, 2000)
    accumulator = _new()
    started = time.perf_counter()
    accumulator.append_frames(records, GEOMETRY, stamps)
    # Per-frame ingest took ~13 us/frame; the batch path is several times faster.
    assert time.perf_counter() - started < 2000 * 13e-6
