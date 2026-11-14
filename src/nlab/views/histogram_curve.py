"""Step histogram curve that draws at most ~2 points per screen pixel.

A full 16,384-bin MCA spectrum in ``stepMode="center"`` is ~32k vertices,
painted on every refresh even when the whole spectrum spans a few hundred
pixels. pyqtgraph's own ``autoDownsample`` cannot be used here: its peak
decimation assumes ``len(x) == len(y)`` and breaks step mode.

When the visible bins outnumber the plot's pixel columns, this item instead
draws, for each group of ``factor`` bins, the group's maximum and minimum
(the same envelope the dense step curve paints into that pixel column), so
peaks and narrow lines are never lost. Groups are aligned to absolute bin
multiples so panning does not shimmer. Zoomed in to ~2 bins per pixel or
less, the exact step histogram is drawn unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg

# Decimate only when there are more than this many visible bins per pixel.
_BINS_PER_PIXEL = 2


class StepHistogramCurve(pg.PlotDataItem):  # type: ignore[misc]
    def __init__(self, **kwargs: Any) -> None:
        self._edges: npt.NDArray[np.float64] | None = None
        self._counts: npt.NDArray[Any] | None = None
        self._layout: tuple[int, int, int] | None = None
        self._refreshing = False
        super().__init__(stepMode="center", **kwargs)

    def setData(self, *args: Any, **kwargs: Any) -> None:  # noqa: N802 - pyqtgraph API
        """Accept ``setData(edges, counts)`` with ``len(edges) == len(counts) + 1``."""
        if len(args) != 2 or kwargs or np.size(args[1]) == 0:
            self._edges = self._counts = None
            self._layout = None
            super().setData(*args, **kwargs)
            return
        edges = np.asarray(args[0], dtype=np.float64)
        counts = np.asarray(args[1])
        if edges.ndim != 1 or counts.ndim != 1 or len(edges) != len(counts) + 1:
            raise ValueError("StepHistogramCurve expects bin edges and one count per bin")
        self._edges, self._counts = edges, counts
        self._layout = None
        self._refresh()

    def viewRangeChanged(  # noqa: N802 - pyqtgraph API
        self, vb: Any = None, ranges: Any = None, changed: Any = None,
    ) -> None:
        super().viewRangeChanged(vb, ranges, changed)
        if changed is None or changed[0]:
            self._refresh()

    def viewTransformChanged(self) -> None:  # noqa: N802 - pyqtgraph API
        super().viewTransformChanged()
        self._refresh()  # the plot may have been resized

    def _plan(self) -> tuple[int, int, int]:
        """(first bin, end bin, bins per group); factor 1 means draw exactly."""
        assert self._edges is not None and self._counts is not None
        count = len(self._counts)
        view = self.getViewBox()
        if view is None or count == 0:
            return 0, count, 1
        width_px = float(view.width())
        left, right = view.viewRange()[0]
        first = max(0, int(np.searchsorted(self._edges, left, side="right")) - 1)
        end = min(count, int(np.searchsorted(self._edges, right, side="left")))
        visible = end - first
        if width_px <= 0 or visible <= _BINS_PER_PIXEL * width_px:
            return 0, count, 1
        factor = int(visible // width_px)
        # One group of margin on both sides keeps the curve running to the edges.
        first = max(0, (first // factor - 1) * factor)
        end = min(count, (-(-end // factor) + 1) * factor)
        return first, end, factor

    def _refresh(self) -> None:
        edges, counts = self._edges, self._counts
        if edges is None or counts is None or self._refreshing:
            return
        layout = self._plan()
        if layout == self._layout:
            return
        self._layout = layout
        self._refreshing = True
        try:
            first, end, factor = layout
            if factor == 1:
                super().setData(edges, counts, stepMode="center")
                return
            starts = np.arange(first, end, factor)
            window = counts[first:end]
            high = np.maximum.reduceat(window, starts - first)
            low = np.minimum.reduceat(window, starts - first)
            stops = np.minimum(starts + factor, len(counts))
            centers = 0.5 * (edges[starts] + edges[stops])
            # Span the exact outer edges so autoranging sees the same extent.
            centers[0], centers[-1] = edges[first], edges[end]
            x = np.repeat(centers, 2)
            y = np.column_stack((high, low)).ravel().astype(np.float64)
            super().setData(x, y, stepMode=None, connect="all")
        finally:
            self._refreshing = False
