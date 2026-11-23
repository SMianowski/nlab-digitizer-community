"""PSD readback: Analysis > FOM vs energy, end to end through a loaded capture."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox
from pytestqt.qtbot import QtBot

import nlab.analysis.psd_report as psd_report_module
from nlab.hardware.digitizer.dma import _LM_EVENT_DTYPE
from nlab.help.context import resolve_context
from nlab.views.psd_readback_dialog import PsdReadbackDialog


def _capture(path: Path, count: int = 80_000) -> None:
    """Events whose upper PSD band moves down with energy, as in liquid scintillator."""
    rng = np.random.default_rng(5)
    long_gate = np.clip(rng.exponential(1500.0, count) + 500.0, 500, 6000)
    second = rng.random(count) < 0.3
    width = 0.025 * np.sqrt(1000.0 / long_gate)
    centers = np.where(second, 0.18 - 0.05 * long_gate / 6000.0, 0.08)
    ratio = rng.normal(centers, np.where(second, 1.3, 1.0) * width)
    events = np.zeros(count, dtype=_LM_EVENT_DTYPE)
    events["trapezoid_energy"] = np.round(long_gate)
    events["charge_energy"] = np.clip(np.round(long_gate * (1.0 - ratio)), 0, 65535)
    with h5py.File(path, "w") as file:
        file.create_dataset("events", data=events)


@pytest.fixture
def readback(tmp_path: Path, qtbot: QtBot) -> Iterator[PsdReadbackDialog]:
    path = tmp_path / "liquid.h5"
    _capture(path)
    dialog = PsdReadbackDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    assert not dialog.energy_fom_action.isEnabled()  # nothing loaded yet
    dialog.open_path(path)
    qtbot.waitUntil(lambda: dialog._thread is None, timeout=10_000)
    dialog.plot.energy_region.setRegion((500.0, 6000.0))
    widget = dialog.plot.fom_widget
    widget.fit_region.setRegion((0.0, 0.32))
    widget.centroid_lines[0].setPos(0.08)
    widget.centroid_lines[1].setPos(0.16)
    yield dialog
    dialog.close_without_prompt()


def test_fom_vs_energy_fits_slices_of_the_loaded_matrix(
    readback: PsdReadbackDialog, qtbot: QtBot,
) -> None:
    assert readback.energy_fom_action.isEnabled()
    readback.energy_fom_action.trigger()
    dialog = readback._energy_fom_dialog
    assert dialog is not None and dialog.isVisible()

    result = dialog.last_result
    assert result is not None
    assert len(result.slices) == 6 and dialog.table.rowCount() == 7
    assert result.request.settings == readback.plot.fom_widget.fit_settings()
    assert result.whole.low <= 500.0 < result.slices[0].high
    assert sum(item.trusted for item in result.slices) >= 5
    second = [item.fit.centroids[1] for item in result.slices if item.trusted and item.fit]
    assert second[0] > second[-1]  # the moving band is followed
    assert "slices (equal counts)" in dialog.summary.text()
    assert resolve_context(dialog.slices_spin).topic == "psd-energy-fom"

    dialog.slices_spin.setValue(3)
    dialog.spacing_combo.setCurrentIndex(dialog.spacing_combo.findData("width"))
    dialog.fit_button.click()
    assert dialog.last_result is not None and len(dialog.last_result.slices) == 3
    assert dialog.table.rowCount() == 4
    copied = dialog.copy_table()
    assert copied is not None and copied.splitlines()[-1].startswith("Whole ROI\t")


def test_csv_report_and_rebuild(
    readback: PsdReadbackDialog, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    readback.energy_fom_action.trigger()
    dialog = readback._energy_fom_dialog
    assert dialog is not None and dialog.last_result is not None

    target = tmp_path / "slices"
    chooser = Mock(return_value=(str(target), ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", chooser)
    dialog.save_button.click()
    assert chooser.call_args.args[2].endswith("liquid_fom_vs_energy.csv")
    assert target.with_suffix(".csv").read_text(encoding="utf-8").startswith(
        "# PSD FOM versus energy"
    )

    # The report carries the fitted slices.
    assert readback.plot.fom_widget.fit_current_projection() is not None
    captured: list[object] = []
    monkeypatch.setattr(
        QFileDialog, "getExistingDirectory", Mock(return_value=str(tmp_path)),
    )
    monkeypatch.setattr(QMessageBox, "information", Mock())
    monkeypatch.setattr(
        psd_report_module, "export_psd_report",
        lambda request, _folder: captured.append(request) or Mock(directory=tmp_path),
    )
    readback._download_report()
    assert captured and captured[0].energy_fom is dialog.last_result  # type: ignore[attr-defined]

    # Rebuilding the matrix forgets slices fitted to the old one.
    readback.ratio_bins.setValue(128)
    readback._recalculate()
    qtbot.waitUntil(lambda: readback._thread is None, timeout=10_000)
    assert dialog.last_result is None
    assert dialog.table.rowCount() == 0
    assert "rebuilt" in dialog.summary.text()
    assert not dialog.save_button.isEnabled()

    readback.close_without_prompt()
    assert not dialog.isVisible()


def test_best_slice_becomes_the_readback_roi_and_is_refitted(
    readback: PsdReadbackDialog, qtbot: QtBot,
) -> None:
    readback.plot.roi_checkbox.setChecked(False)
    readback.energy_fom_action.trigger()
    dialog = readback._energy_fom_dialog
    assert dialog is not None
    best = dialog.best_slice()
    assert best is not None and best.fit is not None and best.trusted
    assert dialog.best_button.isEnabled()
    result = dialog.last_result
    assert result is not None
    trusted_fom = [item.fit.fom for item in result.slices if item.trusted and item.fit]
    assert best.fit.fom == max(trusted_fom)
    row = result.slices.index(best)
    assert dialog.table.item(row, 0).font().bold()

    dialog.best_button.click()

    assert tuple(readback.plot.energy_region.getRegion()) == (best.low, best.high)
    assert readback.plot.roi_checkbox.isChecked()  # the ROI is shown
    refit = readback.plot.fom_widget.last_fit
    assert refit is not None
    # The ROI selects exactly the slice's bins, so the panel reproduces its FOM.
    assert refit.fom == pytest.approx(best.fit.fom, rel=1e-9)
    assert "best slice" in readback.status.text()
    assert f"Sent slice {row + 1}" in dialog.summary.text()


def test_best_slice_button_needs_a_trusted_fit(readback: PsdReadbackDialog) -> None:
    readback.energy_fom_action.trigger()
    dialog = readback._energy_fom_dialog
    assert dialog is not None
    dialog.minimum_spin.setValue(10_000_000)  # no slice has enough events to fit
    dialog.fit_button.click()
    assert dialog.best_slice() is None
    assert not dialog.best_button.isEnabled()
