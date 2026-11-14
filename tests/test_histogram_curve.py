from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pytestqt.qtbot import QtBot

from nlab.views.histogram_curve import StepHistogramCurve


def _plot(qtbot: QtBot, width: int = 400) -> tuple[pg.PlotWidget, StepHistogramCurve]:
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    widget.resize(width, 300)
    widget.show()
    qtbot.waitExposed(widget)
    curve = StepHistogramCurve()
    widget.addItem(curve)
    return widget, curve


def _spectrum() -> np.ndarray:
    counts = np.full(16_384, 5, dtype=np.uint32)
    counts[1_000] = 9_000  # one-bin line must survive decimation
    counts[12_345] = 0
    return counts


def test_zoomed_out_spectrum_is_a_peak_preserving_envelope(qtbot: QtBot) -> None:
    widget, curve = _plot(qtbot)
    counts = _spectrum()
    curve.setData(np.arange(len(counts) + 1), counts)
    widget.getPlotItem().getViewBox().autoRange()

    x, y = curve.getData()
    view_width = widget.getPlotItem().getViewBox().width()
    assert len(y) <= 2 * (view_width + 4)
    assert y.max() == 9_000
    assert y.min() == 0
    assert x[0] == 0 and x[-1] == len(counts)


def test_zoomed_in_view_draws_the_exact_step_histogram(qtbot: QtBot) -> None:
    widget, curve = _plot(qtbot)
    counts = _spectrum()
    curve.setData(np.arange(len(counts) + 1), counts)
    view = widget.getPlotItem().getViewBox()
    view.setXRange(990, 1_010, padding=0)

    x, y = curve.getData()
    assert len(x) == len(counts) + 1
    np.testing.assert_array_equal(y, counts)

    view.setXRange(0, len(counts), padding=0)
    assert len(curve.getData()[1]) < len(counts)


def test_clear_and_small_histograms_pass_through(qtbot: QtBot) -> None:
    _, curve = _plot(qtbot)
    curve.setData([0, 1, 2], [1, 4])
    x, y = curve.getData()
    np.testing.assert_array_equal(x, [0, 1, 2])
    np.testing.assert_array_equal(y, [1, 4])
    curve.setData([], [])
    assert curve.getData()[1] is None or len(curve.getData()[1]) == 0
