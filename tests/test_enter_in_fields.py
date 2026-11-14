from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDialog, QPushButton
from pytestqt.qtbot import QtBot

from nlab.views.current_capture_dialog import CurrentCaptureDialog
from nlab.views.numeric_controls import keep_enter_in_fields
from nlab.views.timing_validation_dialog import TimingValidationDialog
from nlab.views.waveform_analysis_dialog import WaveformAnalysisDialog


@pytest.mark.parametrize(
    "factory", [CurrentCaptureDialog, TimingValidationDialog, WaveformAnalysisDialog],
)
def test_no_button_is_pressed_by_enter_in_a_field(qtbot: QtBot, factory: type[QDialog]) -> None:
    dialog = factory()
    qtbot.addWidget(dialog)
    dialog.show()
    buttons = dialog.findChildren(QPushButton)
    assert buttons
    assert not [b.text() for b in buttons if b.autoDefault() or b.isDefault()]
    dialog.close()


def test_helper_clears_default_and_auto_default(qtbot: QtBot) -> None:
    dialog = QDialog()
    qtbot.addWidget(dialog)
    button = QPushButton("Go", dialog)
    button.setDefault(True)
    keep_enter_in_fields(dialog)
    assert not button.autoDefault() and not button.isDefault()
