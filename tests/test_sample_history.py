from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from nlab.utils.sample_history import SampleHistory


def _samples(start: int, count: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(timestamp_ns=t, raw_code=float(t) / 2, read_latency_ns=t + 1)
        for t in range(start, start + count)
    ]


def test_extend_and_trim_keep_columns_aligned() -> None:
    history = SampleHistory(capacity=4)
    history.extend(_samples(0, 10))
    history.trim_before(6)
    assert len(history) == 4
    assert history.timestamps.tolist() == [6, 7, 8, 9]
    assert history.raw_codes.tolist() == [3.0, 3.5, 4.0, 4.5]
    assert history.read_latencies_ns.tolist() == [7, 8, 9, 10]


def test_trim_keeps_samples_at_the_cutoff_and_since_matches() -> None:
    history = SampleHistory()
    history.extend(_samples(0, 5))
    assert history.timestamps[history.since(3)].tolist() == [3, 4]
    history.trim_before(3)
    assert history.timestamps.tolist() == [3, 4]
    history.trim_before(100)
    assert len(history) == 0


def test_long_sliding_window_stays_bounded_and_ordered() -> None:
    history = SampleHistory(capacity=8)
    for start in range(0, 10_000, 7):
        history.extend(_samples(start, 7))
        history.trim_before(start + 7 - 50)
    assert len(history) == 50
    assert np.all(np.diff(history.timestamps) == 1)
    assert history._timestamps.size <= 256  # compacts instead of growing without bound


def test_clear_and_empty_extend() -> None:
    history = SampleHistory()
    history.extend([])
    history.extend(_samples(0, 3))
    history.clear()
    assert len(history) == 0
    assert history.timestamps.size == 0
