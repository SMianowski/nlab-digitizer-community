"""Time-windowed history of scalar current samples, stored as NumPy columns.

The Current monitor keeps ~10 s of IIR samples and redraws them at display
rate. Storing them as columns (instead of a deque of sample objects) lets the
plot, rate and zero computations run vectorized instead of rebuilding Python
lists of every sample each frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt


class _Sample(Protocol):
    @property
    def timestamp_ns(self) -> int: ...

    @property
    def raw_code(self) -> float: ...

    @property
    def read_latency_ns(self) -> int: ...


class SampleHistory:
    """Append-only columns with a moving start; trimming never copies.

    Timestamps must be non-decreasing (host monotonic clock or a recorded,
    time-ordered replay); :meth:`trim_before` relies on that ordering.
    """

    def __init__(self, capacity: int = 4096) -> None:
        self._timestamps = np.empty(capacity, dtype=np.int64)
        self._raw = np.empty(capacity, dtype=np.float64)
        self._latency = np.empty(capacity, dtype=np.int64)
        self._start = 0
        self._stop = 0

    def __len__(self) -> int:
        return self._stop - self._start

    def clear(self) -> None:
        self._start = self._stop = 0

    def extend(self, samples: Sequence[_Sample]) -> None:
        count = len(samples)
        if not count:
            return
        self._reserve(count)
        stop = self._stop + count
        self._timestamps[self._stop:stop] = [sample.timestamp_ns for sample in samples]
        self._raw[self._stop:stop] = [sample.raw_code for sample in samples]
        self._latency[self._stop:stop] = [sample.read_latency_ns for sample in samples]
        self._stop = stop

    def trim_before(self, cutoff_ns: int) -> None:
        """Drop samples older than ``cutoff_ns``."""
        self._start += int(
            np.searchsorted(self.timestamps, cutoff_ns, side="left")
        )

    def since(self, cutoff_ns: int) -> slice:
        """Index range (into the column views) of samples at or after ``cutoff_ns``."""
        return slice(int(np.searchsorted(self.timestamps, cutoff_ns, side="left")), None)

    @property
    def timestamps(self) -> npt.NDArray[np.int64]:
        return self._timestamps[self._start:self._stop]

    @property
    def raw_codes(self) -> npt.NDArray[np.float64]:
        return self._raw[self._start:self._stop]

    @property
    def read_latencies_ns(self) -> npt.NDArray[np.int64]:
        return self._latency[self._start:self._stop]

    def _reserve(self, count: int) -> None:
        size = len(self)
        capacity = self._timestamps.size
        if self._stop + count <= capacity:
            return
        if size + count > capacity // 2:
            capacity = max(capacity * 2, 2 * (size + count))
        live = slice(self._start, self._stop)
        self._timestamps = _moved(self._timestamps[live], capacity)
        self._raw = _moved(self._raw[live], capacity)
        self._latency = _moved(self._latency[live], capacity)
        self._start, self._stop = 0, size


def _moved[T: np.generic](live: npt.NDArray[T], capacity: int) -> npt.NDArray[T]:
    column = np.empty(capacity, dtype=live.dtype)
    column[: live.size] = live
    return column
