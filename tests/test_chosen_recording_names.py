"""Start timestamps on recording names the operator chooses (General Settings)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMessageBox
from pytestqt.qtbot import QtBot

from nlab import settings_keys
from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.mca_controller import CoincidenceRole, MCAController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.utils import recording_paths
from nlab.utils.recording_paths import chosen_recording_path

# Bound at import, before the autouse conftest fixture replaces the module's
# getter for every test, so the QSettings read itself stays testable.
from nlab.utils.recording_paths import timestamp_chosen_names as stored_option
from tests.test_measurement_series import _FakeSumScope

STAMP = "20261123_101500"


@pytest.fixture(autouse=True)
def measurement_folder(tmp_path: Path) -> Iterator[Path]:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "ini"))
    folder = tmp_path / "runs"
    folder.mkdir()
    QSettings().setValue(settings_keys.DMA_SAVE_FOLDER, str(folder))
    yield folder
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recording_paths.time, "strftime", lambda _fmt: STAMP)


@pytest.fixture
def stamping(timestamp_chosen_names: dict[str, bool]) -> None:
    timestamp_chosen_names["enabled"] = True


# ------------------------------------------------------------------ the rule


def test_option_is_off_by_default_and_read_from_settings() -> None:
    assert stored_option() is False
    QSettings().setValue(settings_keys.RECORDING_TIMESTAMP_CHOSEN_NAMES, True)
    assert stored_option() is True
    QSettings().setValue(settings_keys.RECORDING_TIMESTAMP_CHOSEN_NAMES, False)
    assert stored_option() is False


def test_chosen_names_are_used_as_given_without_the_option(measurement_folder: Path) -> None:
    chosen = measurement_folder / "detector.bin"
    chosen.write_bytes(b"the operator confirmed replacing this")
    assert chosen_recording_path(chosen) == chosen


def test_chosen_names_get_the_start_time_and_never_replace_a_file(
    measurement_folder: Path, stamping: None,
) -> None:
    chosen = measurement_folder / "detector.bin"
    first = chosen_recording_path(chosen)
    assert first == measurement_folder / f"detector_{STAMP}.bin"
    first.write_bytes(b"earlier capture")
    assert chosen_recording_path(chosen) == measurement_folder / f"detector_{STAMP}_001.bin"
    # A leftover sidecar blocks a name as well.
    (measurement_folder / f"detector_{STAMP}_001.yaml").write_text("settings")
    assert chosen_recording_path(chosen, sidecar_suffixes=(".yaml",)) == (
        measurement_folder / f"detector_{STAMP}_002.bin"
    )
    assert chosen_recording_path(measurement_folder / "run.v2.h5").name == f"run.v2_{STAMP}.h5"
    assert first.read_bytes() == b"earlier capture"


# --------------------------------------------------------------------- Scope


@pytest.fixture
def scope(qtbot: QtBot) -> Iterator[ScopeController]:
    device = create_offline_digitizer(0)
    controller = ScopeController(device.scope)
    qtbot.addWidget(controller)
    try:
        yield controller
    finally:
        controller._series.cancel()
        controller.close_viewer_client()
        device.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_scope_chosen_dma_file_gets_the_start_time_only_when_enabled(
    scope: ScopeController, measurement_folder: Path,
    timestamp_chosen_names: dict[str, bool], enabled: bool,
) -> None:
    timestamp_chosen_names["enabled"] = enabled
    scope._dma_filepath = measurement_folder / "source.bin"
    expected = f"source_{STAMP}.bin" if enabled else "source.bin"
    assert scope._next_dma_filepath() == measurement_folder / expected
    # A blank name keeps its generated default either way.
    scope._dma_filepath = None
    assert scope._next_dma_filepath() == measurement_folder / f"scope_ch0_{STAMP}.bin"


def test_scope_series_numbers_runs_after_the_first_runs_start_time(
    scope: ScopeController, measurement_folder: Path, stamping: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stamps = iter(["20261123_101500", "20261123_101530"])
    monkeypatch.setattr(recording_paths.time, "strftime", lambda _fmt: next(stamps))
    series = scope._series
    series.set_enabled(True)
    scope.ui.spinSeriesRuns.setValue(2)
    scope._dma_filepath = measurement_folder / "source.bin"

    assert series.begin(10)
    assert scope._next_dma_filepath() == measurement_folder / "source_20261123_101500_001.bin"
    series.mark_completed()
    series.run_ended()
    # The series keeps one stem; only the run number changes.
    assert scope._next_dma_filepath() == measurement_folder / "source_20261123_101500_002.bin"


# ----------------------------------------------------------------------- MCA


def _list_mode_owner(chosen: Path, role: CoincidenceRole) -> SimpleNamespace:
    """Just enough of an MCAController to arm list mode on the fake worker."""
    return SimpleNamespace(
        _active_dma_mode=McaDmaOutputMode.BINARY,
        _coincidence_role=role,
        _dma_error=None,
        _dma_filepath=chosen,
        _dma_summary=None,
        _event_buffer=Mock(),
        _channel=0,
        _mca=Mock(),
        _mca_dma=Mock(qualified_record_schema=lambda: "vdpp-zc-calc-q2.14-v1"),
        _measurement_configuration=None,
        _output_mode=lambda: McaDmaOutputMode.BINARY,
        _prepare_psd_capture=lambda: None,
        _psd_capture_enabled=False,
        _set_controls_enabled=Mock(),
        _on_dma_ready=Mock(),
        _on_dma_progress=Mock(),
        _on_connection_lost=Mock(),
        _on_dma_error=Mock(),
        _on_dma_summary=Mock(),
        _on_dma_finished=Mock(),
        status=Mock(),
        ui=SimpleNamespace(spinTimeLimit=SimpleNamespace(value=lambda: 30)),
    )


@pytest.mark.parametrize(
    ("role", "expected"),
    [
        # The YAML sidecar of an earlier capture still blocks the stamped name.
        (CoincidenceRole.NONE, f"source_{STAMP}_001.bin"),
        # A coincidence session names its own files; they are never altered.
        (CoincidenceRole.ACTIVE, "source.bin"),
    ],
)
def test_mca_chosen_list_mode_file_gets_the_start_time(
    monkeypatch: pytest.MonkeyPatch, measurement_folder: Path, stamping: None,
    role: CoincidenceRole, expected: str,
) -> None:
    worker_factory = Mock(return_value=Mock())
    monkeypatch.setattr("nlab.controllers.mca_controller.IIOMcaDmaWorker", worker_factory)
    monkeypatch.setattr("nlab.controllers.mca_controller.QThread", Mock(return_value=Mock()))
    monkeypatch.setattr("nlab.utils.settings_io.write_configuration", Mock())
    monkeypatch.setattr("nlab.controllers.mca_controller.owner_series", lambda _owner: None)
    (measurement_folder / f"source_{STAMP}.yaml").write_text("an earlier capture's settings")
    controller = _list_mode_owner(measurement_folder / "source.bin", role)

    MCAController._start_with_dma(cast(MCAController, controller))

    assert worker_factory.call_args.kwargs["filepath"] == measurement_folder / expected
    assert controller._dma_filepath is None


def test_mca_series_stem_is_the_stamped_chosen_name(
    measurement_folder: Path, stamping: None,
) -> None:
    owner = SimpleNamespace(_dma_filepath=measurement_folder / "source.bin", _channel=0)
    base = MCAController._take_series_base_path(cast(MCAController, owner))
    assert base == measurement_folder / f"source_{STAMP}.bin"
    assert owner._dma_filepath is None


# ------------------------------------------------------------------- Current


@pytest.fixture
def current(qtbot: QtBot) -> Iterator[tuple[CurrentMonitorController, _FakeSumScope]]:
    scope = _FakeSumScope()
    mca = cast(MultiChannelAnalyzer, SimpleNamespace())
    controller = CurrentMonitorController(
        mca, channel=0, auto_start=False, scope_controller=cast(Any, scope),
    )
    qtbot.addWidget(controller)
    controller.ui.comboMode.setCurrentIndex(2)  # Periodic sums
    controller.ui.cbRecordSums.setChecked(True)
    try:
        yield controller, scope
    finally:
        controller._series.cancel()
        controller.stop_monitor_sync()


@pytest.mark.parametrize(
    ("entered", "expected"),
    [
        ("detector", f"runs/detector_{STAMP}.bin"),
        ("nested/detector.h5", f"runs/nested/detector_{STAMP}.h5"),
        ("", f"runs/ch0_current_{STAMP}.bin"),
    ],
)
def test_current_sum_name_gets_the_start_time(
    current: tuple[CurrentMonitorController, _FakeSumScope], tmp_path: Path,
    stamping: None, entered: str, expected: str,
) -> None:
    controller, _scope = current
    controller.ui.editSumFile.setText(entered)
    assert controller._sum_output_path() == tmp_path / expected
    absolute = tmp_path / "elsewhere" / "custom.h5"
    controller.ui.editSumFile.setText(str(absolute))
    assert controller._sum_output_path() == absolute.with_name(f"custom_{STAMP}.h5")


def test_stamped_current_sum_never_asks_to_overwrite(
    current: tuple[CurrentMonitorController, _FakeSumScope], measurement_folder: Path,
    stamping: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = Mock(side_effect=AssertionError("a stamped name must not replace a file"))
    monkeypatch.setattr(QMessageBox, "question", question)
    controller, scope = current
    (measurement_folder / f"detector_{STAMP}.bin").write_bytes(b"earlier recording")
    controller.ui.editSumFile.setText("detector.bin")

    controller.ui.btnStart.click()

    assert scope.outputs == [(measurement_folder / f"detector_{STAMP}_001.bin", False)]
    question.assert_not_called()
    # The entered name is kept for the next Start.
    assert controller.ui.editSumFile.text() == "detector.bin"
