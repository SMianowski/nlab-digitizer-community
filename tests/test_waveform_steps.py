from __future__ import annotations

import struct
import time
from types import SimpleNamespace
from typing import cast

import numpy as np
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot

from nlab import settings_keys
from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.current_monitor import ScopeCurrentAccumulator, ScopeCurrentRuntime
from nlab.hardware.digitizer.dma import ScopeDmaGeometry
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.views.plot_viewbox import (
    WAVEFORM_STEP_MODE,
    DraggableScopeCurve,
    WaveformDrawing,
    follow_waveform_drawing,
    save_waveform_drawing,
    saved_waveform_drawing,
    waveform_step_mode,
)
from nlab.workers.current_monitor_worker import CurrentSample


@pytest.mark.parametrize("draggable", [False, True])
def test_sample_hold_path_preserves_timestamps_and_codes(qtbot: QtBot, draggable: bool) -> None:
    item = DraggableScopeCurve() if draggable else pg.PlotDataItem(stepMode=WAVEFORM_STEP_MODE)
    x = np.array([-16.0, -8.0, 0.0, 8.0])
    y = np.array([10, 20, -5, 7], dtype=np.int16)
    item.setData(x, y)
    actual_x, actual_y = item.getData()
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_y, y)
    curve = item if draggable else item.curve
    path = curve.getPath()
    points = [(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())]
    # Qt/pyqtgraph may retain a duplicate zero-length endpoint.
    points = [
        point for index, point in enumerate(points)
        if not index or point != points[index - 1]
    ]
    assert points == [
        (-16.0, 10.0), (-8.0, 10.0), (-8.0, 20.0), (0.0, 20.0),
        (0.0, -5.0), (8.0, -5.0), (8.0, 7.0),
    ]
    # Updating/clearing does not silently return to diagonal interpolation.
    item.clear()
    item.setData(x, y + 1)
    assert curve.opts["stepMode"] == WAVEFORM_STEP_MODE


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_sample_paths_are_safe(qtbot: QtBot, count: int) -> None:
    curve = DraggableScopeCurve()
    curve.setData(np.arange(count, dtype=float), np.ones(count))
    path = curve.getPath()
    assert path.isEmpty() or path.boundingRect().width() == 0


def test_sample_steps_do_not_bridge_missing_dma_fragments(qtbot: QtBot) -> None:
    item = pg.PlotDataItem(stepMode=WAVEFORM_STEP_MODE, connect="finite")
    item.setData([0, 1, 1, 10, 11], [1, 2, np.nan, 3, 4])
    path = item.curve.getPath()
    for i in range(1, path.elementCount()):
        right, left = path.elementAt(i), path.elementAt(i - 1)
        if not right.isMoveTo():
            assert abs(right.x - left.x) <= 1


def test_current_dma_steps_and_iir_trend_restore_independently(qtbot: QtBot) -> None:
    controller = CurrentMonitorController(
        cast(MultiChannelAnalyzer, SimpleNamespace()), channel=0, auto_start=False,
    )
    qtbot.addWidget(controller)
    geometry = ScopeDmaGeometry.legacy(8)
    accumulator = ScopeCurrentAccumulator()
    accumulator.start_session(geometry, ScopeCurrentRuntime(0, 121, 8, 10))
    for timestamp, samples in ((1, [1, 2, 3, 4]), (100, [5, 6, 7, 8])):
        record = struct.pack("<Q", timestamp) + np.asarray(samples, dtype="<i2").tobytes()
        accumulator.append_frame(record, geometry, time.perf_counter_ns())
    controller._render_dma_plot(accumulator.snapshot(), zero=1, scale=2)
    assert controller._curve.opts["stepMode"] == WAVEFORM_STEP_MODE
    assert controller._curve.opts["connect"] == "finite"
    _x, y = controller._curve.getData()
    np.testing.assert_array_equal(y, [0, 2, 4, 6, np.nan, 8, 10, 12, 14])
    controller._sample_buffer.append(CurrentSample(1, time.perf_counter_ns(), 10, 100))
    controller._render_pending()
    controller._display_timer.stop()
    assert controller._curve.opts["stepMode"] is None
    assert controller._curve.opts["connect"] == "auto"


def _reference_rasterize(buffer: np.ndarray, frame: np.ndarray, decay: float) -> None:
    """The original per-column loop, kept as the behavioural reference."""
    nx, ny = buffer.shape
    buffer *= decay
    n = min(len(frame), nx)
    if not n:
        return
    y = (frame[:n].astype(np.float32) - -32_000) / 64_000 * (ny - 1)
    y_pixels = np.clip(y, 0, ny - 1).astype(np.intp)
    buffer[0, y_pixels[0]] = 1.0
    for i in range(1, n):
        low, high = sorted((y_pixels[i - 1], y_pixels[i]))
        buffer[i, low:high + 1] = 1.0


def test_vectorized_persistence_matches_reference_loop() -> None:
    rng = np.random.default_rng(1234)
    nx, ny = 64, 128
    state = SimpleNamespace(
        persistence=0.75, _display_nx=nx, _display_ny=ny, _Y_MIN=-32_000, _Y_MAX=32_000,
        _persistence_buffer=np.zeros((nx, ny), dtype=np.float32),
    )
    expected = np.zeros((nx, ny), dtype=np.float32)
    for length in (nx, nx // 2, 1, nx + 9, nx):
        # Includes full-scale clipping, partial (prefix) frames and overlong frames.
        frame = rng.integers(-40_000, 40_000, size=length).astype(np.int32)
        ScopeController._rasterize_frame(cast(ScopeController, state), frame)
        _reference_rasterize(expected, frame, 0.75)
        np.testing.assert_array_equal(state._persistence_buffer, expected)


def test_persistence_uses_sample_holds_and_next_sample_transitions() -> None:
    state = SimpleNamespace(
        persistence=0.5, _display_nx=6, _display_ny=5, _Y_MIN=0, _Y_MAX=4,
        _persistence_buffer=np.zeros((6, 5), dtype=np.float32),
    )
    controller = cast(ScopeController, state)
    ScopeController._rasterize_frame(controller, np.array([1, 3, 2]))
    expected = np.zeros((6, 5))
    expected[0, 1] = 1
    expected[1, 1:4] = 1
    expected[2, 2:4] = 1
    np.testing.assert_array_equal(state._persistence_buffer, expected)
    ScopeController._rasterize_frame(controller, np.array([], dtype=np.int16))
    np.testing.assert_array_equal(state._persistence_buffer, expected * 0.5)
    state._persistence_buffer.fill(0)
    ScopeController._rasterize_frame(controller, np.array([2]))
    assert state._persistence_buffer[0, 2] == 1
    assert np.count_nonzero(state._persistence_buffer) == 1
    # Samples beyond the configured frame do not accumulate in its final column.
    state._persistence_buffer.fill(0)
    ScopeController._rasterize_frame(controller, np.array([1] * 6 + [4]))
    np.testing.assert_array_equal(state._persistence_buffer[:, 1], np.ones(6))
    assert not state._persistence_buffer[:, 2:].any()


# ------------------------------------------------ steps or lines (General Settings)


def _path_points(curve: pg.PlotCurveItem) -> list[tuple[float, float]]:
    path = curve.getPath()
    points = [(path.elementAt(i).x, path.elementAt(i).y) for i in range(path.elementCount())]
    return [point for index, point in enumerate(points) if not index or point != points[index - 1]]


def test_waveform_drawing_defaults_to_steps_and_rejects_unknown_values(
    isolated_waveform_drawing: dict[str, object],
) -> None:
    assert saved_waveform_drawing() == "steps"
    assert waveform_step_mode() == WAVEFORM_STEP_MODE
    isolated_waveform_drawing[settings_keys.APPEARANCE_WAVEFORM_DRAWING] = "spline"
    assert saved_waveform_drawing() == "steps"
    with pytest.raises(ValueError, match="unsupported"):
        save_waveform_drawing(cast(WaveformDrawing, "spline"))


@pytest.mark.parametrize("draggable", [False, True])
def test_switching_to_lines_redraws_existing_waveforms_without_changing_samples(
    qtbot: QtBot,
    isolated_waveform_drawing: dict[str, object],
    draggable: bool,
) -> None:
    item: pg.PlotCurveItem | pg.PlotDataItem
    if draggable:
        item = DraggableScopeCurve()
    else:
        item = pg.PlotDataItem()
        follow_waveform_drawing(item)
    curve = item if isinstance(item, pg.PlotCurveItem) else item.curve
    x = np.array([-16.0, -8.0, 0.0, 8.0])
    y = np.array([10, 20, -5, 7], dtype=np.int16)
    item.setData(x, y)
    assert curve.opts["stepMode"] == WAVEFORM_STEP_MODE

    save_waveform_drawing("lines")

    assert isolated_waveform_drawing[settings_keys.APPEARANCE_WAVEFORM_DRAWING] == "lines"
    assert curve.opts["stepMode"] is None
    assert _path_points(curve) == [(-16.0, 10.0), (-8.0, 20.0), (0.0, -5.0), (8.0, 7.0)]
    actual_x, actual_y = item.getData()
    np.testing.assert_array_equal(actual_x, x)
    np.testing.assert_array_equal(actual_y, y)
    # Later data updates keep the chosen drawing.
    item.setData(x, y + 1)
    assert curve.opts["stepMode"] is None

    save_waveform_drawing("steps")
    assert curve.opts["stepMode"] == WAVEFORM_STEP_MODE
    assert len(_path_points(curve)) == 7


def test_empty_waveforms_switch_drawing_and_new_curves_use_the_saved_one(qtbot: QtBot) -> None:
    empty = DraggableScopeCurve()
    save_waveform_drawing("lines")
    assert empty.opts["stepMode"] is None and empty.getData()[1].size == 0
    assert DraggableScopeCurve().opts["stepMode"] is None


def test_mca_debug_traces_follow_the_drawing_but_the_gate_guide_stays_stepped(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    from nlab.controllers.mca_controller import MCAController

    mca = MagicMock()
    mca.get_debug_signal_selectors.return_value = tuple(range(9))
    mca.edge_det_coeff_is_hardware_backed.return_value = False
    monkeypatch.setattr(MCAController, "_load_hardware_state", lambda self: None)
    controller = MCAController(mca, mca_dma=None, channel=0)
    qtbot.addWidget(controller)
    controller._debug1_curve.setData(np.arange(4.0), np.array([1.0, 3.0, 2.0, 4.0]))

    save_waveform_drawing("lines")

    assert controller._debug1_curve.opts["stepMode"] is None
    assert controller._debug2_curve.opts["stepMode"] is None
    np.testing.assert_array_equal(controller._debug1_curve.getData()[1], [1.0, 3.0, 2.0, 4.0])
    assert controller._debug_cc_guide.opts["stepMode"] == WAVEFORM_STEP_MODE


def test_current_dma_trace_uses_the_saved_drawing(qtbot: QtBot) -> None:
    save_waveform_drawing("lines")
    controller = CurrentMonitorController(
        cast(MultiChannelAnalyzer, SimpleNamespace()), channel=0, auto_start=False,
    )
    qtbot.addWidget(controller)
    geometry = ScopeDmaGeometry.legacy(8)
    accumulator = ScopeCurrentAccumulator()
    accumulator.start_session(geometry, ScopeCurrentRuntime(0, 121, 8, 10))
    record = struct.pack("<Q", 1) + np.asarray([1, 2, 3, 4], dtype="<i2").tobytes()
    accumulator.append_frame(record, geometry, time.perf_counter_ns())
    controller._render_dma_plot(accumulator.snapshot(), zero=1, scale=2)
    assert controller._curve.opts["stepMode"] is None
    assert controller._curve.opts["connect"] == "finite"


def test_waveform_analysis_curves_follow_the_drawing(qtbot: QtBot) -> None:
    from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog

    dialog = WaveformAnalysisDialog()
    qtbot.addWidget(dialog)
    curves = (dialog.raw_curve, dialog.oriented_curve, dialog.average_curve)
    assert all(curve.opts["stepMode"] == WAVEFORM_STEP_MODE for curve in curves)
    save_waveform_drawing("lines")
    assert all(curve.opts["stepMode"] is None for curve in curves)
    dialog.close()


def test_general_settings_applies_the_waveform_drawing_on_accept(
    qtbot: QtBot,
    isolated_waveform_drawing: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlab.views import theme
    from nlab.views.general_settings_dialog import GeneralSettingsDialog

    monkeypatch.setattr(theme, "save_theme_mode", lambda _mode: None)
    curve = DraggableScopeCurve()
    dialog = GeneralSettingsDialog("192.0.2.10", {})
    qtbot.addWidget(dialog)
    assert dialog.waveform_drawing == "steps"
    dialog.waveform_drawing_combo.setCurrentIndex(
        dialog.waveform_drawing_combo.findData("lines")
    )
    assert curve.opts["stepMode"] == WAVEFORM_STEP_MODE  # Not until OK.

    dialog.accept()

    assert isolated_waveform_drawing[settings_keys.APPEARANCE_WAVEFORM_DRAWING] == "lines"
    assert curve.opts["stepMode"] is None
    reopened = GeneralSettingsDialog("192.0.2.10", {})
    qtbot.addWidget(reopened)
    assert reopened.waveform_drawing == "lines"
