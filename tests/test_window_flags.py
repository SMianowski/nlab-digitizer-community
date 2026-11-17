"""Every tool that opens in its own window shares one set of title-bar buttons."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from nlab.views.current_capture_dialog import CurrentCaptureDialog
from nlab.views.dose_calibration_dialog import DoseCalibrationChannelSource, DoseCalibrationDialog
from nlab.views.energy_calibration_dialog import EnergyCalibrationDialog
from nlab.views.help_dialog import HelpDialog
from nlab.views.mca_peak_analysis_dialog import McaPeakAnalysisDialog
from nlab.views.mca_roi_history_dialog import McaRoiHistoryDialog
from nlab.views.psd_readback_dialog import PsdReadbackDialog
from nlab.views.timing_validation_dialog import TimingValidationDialog
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog
from nlab.views.window_flags import TOOL_WINDOW_FLAGS
from tests.test_dose_calibration import _mca_snapshot, _McaSource


def _dose() -> QWidget:
    source = DoseCalibrationChannelSource(label="D1:CH0", mca=_McaSource(_mca_snapshot([1])))
    return DoseCalibrationDialog([source])


_TOOL_WINDOWS: dict[str, Callable[[], QWidget]] = {
    "current capture": CurrentCaptureDialog,
    "dose calibration": _dose,
    "energy calibration": lambda: EnergyCalibrationDialog([]),
    "help": HelpDialog,
    "peak analysis": lambda: McaPeakAnalysisDialog([]),
    "ROI history": lambda: McaRoiHistoryDialog(0),
    "PSD readback": PsdReadbackDialog,
    "timing validation": TimingValidationDialog,
    "waveform analysis": WaveformAnalysisDialog,
}


@pytest.mark.parametrize("factory", _TOOL_WINDOWS.values(), ids=_TOOL_WINDOWS.keys())
def test_tool_windows_have_minimize_maximize_and_close(
    qtbot: QtBot, factory: Callable[[], QWidget],
) -> None:
    window = factory()
    qtbot.addWidget(window)
    flags = window.windowFlags()
    assert window.isWindow()
    for hint in (
        Qt.WindowType.WindowMinimizeButtonHint,
        Qt.WindowType.WindowMaximizeButtonHint,
        Qt.WindowType.WindowCloseButtonHint,
    ):
        assert flags & hint, hint
    assert not flags & Qt.WindowType.WindowContextHelpButtonHint
    assert flags == TOOL_WINDOW_FLAGS
