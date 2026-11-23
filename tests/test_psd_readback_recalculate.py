"""PSD readback: asking to rebuild the matrix after its histogram settings change."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QMessageBox, QPushButton
from pytestqt.qtbot import QtBot

from nlab.hardware.digitizer.dma import _LM_EVENT_DTYPE
from nlab.views.psd_readback_dialog import PsdReadbackDialog


@pytest.fixture
def loaded(tmp_path: Path, qtbot: QtBot) -> Iterator[PsdReadbackDialog]:
    path = tmp_path / "readback.h5"
    events = np.zeros(200, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = np.arange(200) % 100 + 100
    events["charge_energy"] = events["trapezoid_energy"] // 2
    with h5py.File(path, "w") as file:
        file.create_dataset("events", data=events)
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.activateWindow()
    qtbot.waitUntil(dialog.isActiveWindow, timeout=5000)
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=5000)
    assert dialog.plot._matrix.shape == (1024, 256)
    yield dialog
    dialog.close_without_prompt()


def _answer(monkeypatch: pytest.MonkeyPatch, button: QMessageBox.StandardButton) -> Mock:
    question = Mock(return_value=button)
    monkeypatch.setattr(QMessageBox, "question", question)
    return question


def _edit(qtbot: QtBot, control: QAbstractSpinBox, text: str) -> None:
    """Type a value and press Enter, as the user commits an edit."""
    control.setFocus()
    qtbot.waitUntil(control.hasFocus, timeout=2000)
    control.selectAll()
    QTest.keyClicks(control, text)
    QTest.keyClick(control, Qt.Key.Key_Return)


def test_changed_setting_asks_and_recalculates_on_yes(
    loaded: PsdReadbackDialog, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _answer(monkeypatch, QMessageBox.StandardButton.Yes)
    loaded.plot.cut_spin.setValue(0.4)

    _edit(qtbot, loaded.ratio_bins, "128")
    qtbot.waitUntil(lambda: question.called, timeout=2000)
    qtbot.waitUntil(lambda: loaded._thread is None, timeout=5000)

    question.assert_called_once()
    assert "readback.h5" in question.call_args.args[2]
    # Enter only committed the field: no push button (Fit, Copy table) is a default.
    assert not any(button.autoDefault() for button in loaded.findChildren(QPushButton))
    assert loaded.plot._matrix.shape == (1024, 128)
    assert int(loaded.plot._matrix.sum()) == 200
    assert loaded.plot.cut_spin.value() == pytest.approx(0.4)  # the cut is kept
    assert not loaded.recalculate_action.isEnabled()


def test_declined_settings_wait_for_recalculate(
    loaded: PsdReadbackDialog, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _answer(monkeypatch, QMessageBox.StandardButton.No)

    _edit(qtbot, loaded.energy_bins, "512")
    qtbot.waitUntil(lambda: question.called, timeout=2000)

    assert loaded.plot._matrix.shape == (1024, 256)  # unchanged
    assert loaded.recalculate_action.isEnabled()
    assert loaded.recalculate_action in loaded.analysis_menu.actions()
    assert "Recalculate PSD matrix" in loaded.status.text()
    # Committing the same declined values again does not ask again.
    QTest.keyClick(loaded.energy_bins, Qt.Key.Key_Return)
    QApplication.processEvents()
    question.assert_called_once()

    loaded.recalculate_action.trigger()
    qtbot.waitUntil(lambda: loaded._thread is None, timeout=5000)
    assert loaded.plot._matrix.shape == (512, 256)
    assert not loaded.recalculate_action.isEnabled()


def test_moving_between_settings_asks_once_when_leaving_them(
    loaded: PsdReadbackDialog, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _answer(monkeypatch, QMessageBox.StandardButton.No)

    loaded.ratio_min.setFocus()
    qtbot.waitUntil(loaded.ratio_min.hasFocus, timeout=2000)
    loaded.ratio_min.setValue(-0.5)
    loaded.ratio_max.setFocus()  # on to the next histogram field
    qtbot.waitUntil(loaded.ratio_max.hasFocus, timeout=2000)
    loaded.ratio_max.setValue(0.8)
    QApplication.processEvents()
    question.assert_not_called()

    loaded.plot.cut_spin.setFocus()  # editing of the settings is finished
    qtbot.waitUntil(lambda: question.called, timeout=2000)
    question.assert_called_once()


def test_unchanged_or_unloaded_settings_never_ask(
    loaded: PsdReadbackDialog, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = _answer(monkeypatch, QMessageBox.StandardButton.Yes)

    _edit(qtbot, loaded.ratio_bins, "256")  # the displayed value
    QApplication.processEvents()
    question.assert_not_called()

    fresh = PsdReadbackDialog()
    qtbot.addWidget(fresh)
    fresh.show()
    fresh.activateWindow()
    qtbot.waitUntil(fresh.isActiveWindow, timeout=5000)
    _edit(qtbot, fresh.ratio_bins, "128")  # no file loaded yet
    QApplication.processEvents()
    question.assert_not_called()
    assert not fresh.recalculate_action.isEnabled()
