"""Measurement series: repeated timed runs with numbered recordings."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QLabel, QSpinBox, QWidget
from pytestqt.qtbot import QtBot

from nlab import settings_keys
from nlab.controllers.current_monitor_controller import (
    CurrentAcquisition,
    CurrentMonitorController,
)
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.measurement_series import (
    MeasurementSeries,
    numbered_path,
    time_limit_required_message,
)
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.current_monitor import CurrentMonitorClient
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.hardware.digitizer.mca_capture import McaDmaOutputMode
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.utils import recording_paths


@pytest.fixture(autouse=True)
def measurement_folder(tmp_path: Path) -> Iterator[Path]:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "ini"))
    folder = tmp_path / "runs"
    QSettings().setValue(settings_keys.DMA_SAVE_FOLDER, str(folder))
    yield folder
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recording_paths.time, "strftime", lambda _fmt: "20261120_120000")


# --------------------------------------------------------------- helper only


def test_numbered_path_pads_to_three_digits_or_the_run_count() -> None:
    assert numbered_path(Path("d/run.bin"), 1, 5) == Path("d/run_001.bin")
    assert numbered_path(Path("d/run.v2.h5"), 12, 1500) == Path("d/run.v2_0012.h5")


class _Probe:
    def __init__(self, qtbot: QtBot) -> None:
        self.parent = QWidget()
        qtbot.addWidget(self.parent)
        self.spinbox = QSpinBox(self.parent)
        self.label = QLabel("Repeats", self.parent)
        self.idle = True
        self.starts = 0
        self.series = MeasurementSeries(
            self.parent,
            spinbox=self.spinbox,
            label=self.label,
            name="probe",
            is_idle=lambda: self.idle,
            start_run=self._start,
        )

    def _start(self) -> None:
        self.starts += 1
        self.idle = False


def test_disabled_series_is_a_single_unnumbered_run(qtbot: QtBot) -> None:
    probe = _Probe(qtbot)
    series = probe.series
    assert not probe.spinbox.isEnabled()
    assert probe.spinbox.value() == 1

    assert series.begin(0)
    assert not series.active
    assert series.recording_path(lambda: Path("run.bin")) is None

    series.set_enabled(True)
    assert probe.spinbox.isEnabled()


def test_more_than_one_run_requires_a_time_limit(qtbot: QtBot) -> None:
    probe = _Probe(qtbot)
    series = probe.series
    series.set_enabled(True)
    probe.spinbox.setValue(3)

    assert not series.begin(0)
    assert not series.active
    assert time_limit_required_message(3) == (
        "Set Time [s] above 0 to run a series of 3 measurements."
    )
    # A single numbered run needs no time limit.
    probe.spinbox.setValue(1)
    assert series.begin(0)
    assert series.active


def test_completed_runs_start_the_next_one_once_idle(qtbot: QtBot) -> None:
    probe = _Probe(qtbot)
    series = probe.series
    series.set_enabled(True)
    probe.spinbox.setValue(3)
    calls: list[str] = []

    def default() -> Path:
        calls.append("default")
        return Path("runs/ch0_20261120_120000.bin")

    assert series.begin(10)
    probe.idle = False  # the view started run 1
    assert probe.label.text() == "Run 1/3"
    assert not probe.spinbox.isEnabled()
    assert series.recording_path(default) == Path("runs/ch0_20261120_120000_001.bin")

    series.mark_completed()
    series.stop_requested()  # the time-limit stop does not end the series
    assert series.active
    series.run_ended()
    assert probe.label.text() == "Run 2/3"
    qtbot.wait(120)
    assert probe.starts == 0  # still waiting for the view to release its resources
    probe.idle = True
    qtbot.waitUntil(lambda: probe.starts == 1, timeout=2000)
    assert series.recording_path(default) == Path("runs/ch0_20261120_120000_002.bin")

    series.mark_completed()
    probe.idle = True
    series.run_ended()
    qtbot.waitUntil(lambda: probe.starts == 2, timeout=2000)
    assert series.run == 3
    series.mark_completed()
    probe.idle = True
    series.run_ended()
    assert not series.active
    assert probe.label.text() == "Repeats"
    assert probe.spinbox.isEnabled()
    assert calls == ["default"]  # one core name for the whole series


@pytest.mark.parametrize("ending", ["stop", "error", "declined"])
def test_stop_error_or_declined_start_ends_the_series(qtbot: QtBot, ending: str) -> None:
    probe = _Probe(qtbot)
    series = probe.series
    series.set_enabled(True)
    probe.spinbox.setValue(3)
    assert series.begin(10)
    probe.idle = False

    if ending == "stop":
        series.stop_requested()
        series.run_ended()
    elif ending == "error":
        series.mark_completed()
        series.run_ended("DMA fault")
    else:
        probe.idle = True
        probe.series._start_run = lambda: None  # the view refuses to start
        series.mark_completed()
        series.run_ended()
        qtbot.waitUntil(lambda: not series.active, timeout=2000)

    qtbot.wait(60)
    assert not series.active
    assert probe.starts == 0
    assert probe.label.text() == "Repeats"


def test_unchecking_the_option_makes_the_current_run_the_last(qtbot: QtBot) -> None:
    probe = _Probe(qtbot)
    series = probe.series
    series.set_enabled(True)
    probe.spinbox.setValue(5)
    assert series.begin(10)
    probe.idle = False

    series.set_enabled(False)
    assert series.active
    assert probe.label.text() == "Run 1/1"
    series.mark_completed()
    probe.idle = True
    series.run_ended()
    qtbot.wait(60)
    assert not series.active
    assert probe.starts == 0


# --------------------------------------------------------------------- Scope


@pytest.fixture
def scope(qtbot: QtBot) -> Iterator[tuple[ScopeController, Any]]:
    device = create_offline_digitizer(0)
    controller = ScopeController(device.scope)
    qtbot.addWidget(controller)
    try:
        yield controller, device
    finally:
        controller._series.cancel()
        controller._refresh_timer.stop()
        controller._measurement_timer.stop()
        controller.stop_dma_sync()
        controller.close_viewer_client()
        device.close()


def test_scope_series_repeats_timed_runs_until_the_count(
    scope: tuple[ScopeController, Any], qtbot: QtBot,
) -> None:
    controller, _device = scope
    controller.set_measurement_series_enabled(True)
    controller.ui.spinSeriesRuns.setValue(3)
    controller.ui.spinTime.setValue(60)
    activity: list[bool] = []
    controller.measurement_active_changed.connect(activity.append)

    controller.ui.btnStart.click()
    assert controller.measurement_active
    assert controller.ui.labelSeriesRuns.text() == "Run 1/3"
    assert not controller.ui.spinSeriesRuns.isEnabled()

    for run in (2, 3):
        controller._on_measurement_timeout()  # the run's time limit expires
        qtbot.waitUntil(lambda run=run: controller._series.run == run, timeout=2000)
        qtbot.waitUntil(lambda: controller.measurement_active, timeout=2000)
        assert controller._measurement_timer.isActive()

    controller._on_measurement_timeout()
    qtbot.wait(60)
    assert not controller.measurement_active
    assert not controller._series.active
    assert controller.ui.labelSeriesRuns.text() == "Repeats"
    assert activity == [True, False, True, False, True, False]


def test_scope_stop_ends_the_series(scope: tuple[ScopeController, Any], qtbot: QtBot) -> None:
    controller, _device = scope
    controller.set_measurement_series_enabled(True)
    controller.ui.spinSeriesRuns.setValue(3)
    controller.ui.spinTime.setValue(60)

    controller.ui.btnStart.click()
    controller.ui.btnStop.click()
    qtbot.wait(60)

    assert not controller.measurement_active
    assert not controller._series.active
    assert controller.ui.btnStart.isEnabled()


def test_scope_series_without_time_limit_is_refused(
    scope: tuple[ScopeController, Any],
) -> None:
    controller, _device = scope
    controller.set_measurement_series_enabled(True)
    controller.ui.spinSeriesRuns.setValue(2)
    controller.ui.spinTime.setValue(0)

    controller.ui.btnStart.click()

    assert not controller.measurement_active
    assert not controller.ui.btnStart.isChecked()
    assert controller.status.text() == time_limit_required_message(2)


def test_scope_dma_recordings_number_the_chosen_or_default_name(
    scope: tuple[ScopeController, Any], measurement_folder: Path, frozen_clock: None,
) -> None:
    controller, _device = scope
    series = controller._series
    series.set_enabled(True)
    controller.ui.spinSeriesRuns.setValue(2)

    assert series.begin(10)
    default = measurement_folder / "scope_ch0_20261120_120000_001.bin"
    assert controller._next_dma_filepath() == default

    chosen = measurement_folder / "source.bin"
    (measurement_folder / "source_001.bin").write_bytes(b"earlier series")
    controller._dma_filepath = chosen
    assert series.begin(10)
    # An earlier recording is never replaced.
    assert controller._next_dma_filepath() == measurement_folder / "source_001_001.bin"
    series.mark_completed()
    series.run_ended()
    controller._dma_filepath = None
    assert controller._next_dma_filepath() == measurement_folder / "source_002.bin"


def test_scope_without_series_keeps_its_unnumbered_names(
    scope: tuple[ScopeController, Any], measurement_folder: Path, frozen_clock: None,
) -> None:
    controller, _device = scope
    assert controller._next_dma_filepath() == measurement_folder / "scope_ch0_20261120_120000.bin"


# ----------------------------------------------------------------------- MCA


@pytest.fixture
def mca(qtbot: QtBot) -> Iterator[MCAController]:
    device = create_offline_digitizer(0)
    controller = MCAController(device.mca, mca_dma=None, channel=0)
    qtbot.addWidget(controller)
    try:
        yield controller
    finally:
        controller._series.cancel()
        controller.stop_worker_sync()
        controller.stop_dma_sync()
        device.close()


def _complete_mca_run(controller: MCAController, qtbot: QtBot) -> None:
    """Mirror MCAWorker at the hardware time limit: stop polling, then report."""
    readbacks: list[object] = []
    controller.session_readback.connect(lambda _channel, readback: readbacks.append(readback))
    qtbot.waitUntil(lambda: bool(readbacks), timeout=5000)  # the worker is polling
    worker = controller._worker
    assert worker is not None
    worker.request_stop.emit()
    controller._on_measurement_done()


def test_mca_series_saves_each_runs_spectrum_as_numbered_csv(
    mca: MCAController, qtbot: QtBot, measurement_folder: Path, frozen_clock: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Polling reads the same histogram; derive it from the run being saved.
    monkeypatch.setattr(
        mca._mca,
        "acquire_spectrum",
        lambda: np.arange(8, dtype=np.uint32) + 8 * max(0, mca._series.run - 1),
    )
    monkeypatch.setattr(mca._mca.statistics, "get_elapsed_time", lambda: 50)
    mca.set_measurement_series_enabled(True)
    mca.ui.spinSeriesRuns.setValue(2)
    mca.ui.spinTimeLimit.setValue(5)

    mca.ui.btnStart.click()
    assert mca.measurement_active
    assert mca.ui.labelSeriesRuns.text() == "Run 1/2"
    _complete_mca_run(mca, qtbot)
    qtbot.waitUntil(lambda: mca._series.run == 2 and mca.measurement_active, timeout=5000)
    _complete_mca_run(mca, qtbot)
    qtbot.waitUntil(lambda: not mca._series.active, timeout=5000)

    first = measurement_folder / "ch0_20261120_120000_001.csv"
    second = measurement_folder / "ch0_20261120_120000_002.csv"
    assert sorted(measurement_folder.glob("*.csv")) == [first, second]
    lines = first.read_text().splitlines()
    assert lines[:3] == [
        "# MCA channel CH0 spectrum export",
        "# elapsed_s=5.0",
        "# source=stopped hardware readout",
    ]
    assert lines[3] == "channel,counts"
    assert lines[4:6] == ["0,0", "1,1"]
    assert second.read_text().splitlines()[4] == "0,8"


def test_mca_stopped_series_run_keeps_its_spectrum(
    mca: MCAController, measurement_folder: Path, frozen_clock: None,
) -> None:
    mca.set_measurement_series_enabled(True)
    mca.ui.spinSeriesRuns.setValue(3)
    mca.ui.spinTimeLimit.setValue(5)
    mca._dma_filepath = measurement_folder / "source.bin"

    mca.ui.btnStart.click()
    mca.ui.btnStop.click()

    assert not mca._series.active
    assert (measurement_folder / "source_001.csv").is_file()
    assert mca._dma_filepath is None  # the chosen name was used for this series


def test_mca_spectrum_falls_back_to_the_last_live_readback(
    mca: MCAController, measurement_folder: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable() -> np.ndarray:
        raise RuntimeError("histogram attribute is not discoverable")

    monkeypatch.setattr(mca._mca, "acquire_spectrum", unreadable)
    mca._last_histogram = np.array([3, 4], dtype=np.uint32)
    mca._last_elapsed_s = 2.5
    path = measurement_folder / "fallback.csv"
    measurement_folder.mkdir(parents=True)
    mca._series_spectrum_path = path

    assert mca._save_series_spectrum()
    assert path.read_text().splitlines() == [
        "# MCA channel CH0 spectrum export",
        "# elapsed_s=2.5",
        "# source=last live readback",
        "channel,counts",
        "0,3",
        "1,4",
    ]


def test_mca_list_mode_series_pairs_the_spectrum_with_the_recording(
    monkeypatch: pytest.MonkeyPatch, measurement_folder: Path,
) -> None:
    worker_factory = Mock(return_value=Mock())
    monkeypatch.setattr("nlab.controllers.mca_controller.IIOMcaDmaWorker", worker_factory)
    monkeypatch.setattr("nlab.controllers.mca_controller.QThread", Mock(return_value=Mock()))
    monkeypatch.setattr("nlab.utils.settings_io.write_configuration", Mock())
    measurement_folder.mkdir(parents=True)
    (measurement_folder / "source_001.csv").write_text("an earlier spectrum")
    base = measurement_folder / "source.bin"
    series = SimpleNamespace(recording_path=lambda default: base.with_name("source_001.bin"))
    controller = SimpleNamespace(
        _active_dma_mode=McaDmaOutputMode.BINARY,
        _dma_error=None,
        _dma_filepath=None,
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
        _take_series_base_path=Mock(return_value=base),
        _on_dma_ready=Mock(),
        _on_dma_progress=Mock(),
        _on_connection_lost=Mock(),
        _on_dma_error=Mock(),
        _on_dma_summary=Mock(),
        _on_dma_finished=Mock(),
        status=Mock(),
        ui=SimpleNamespace(spinTimeLimit=SimpleNamespace(value=lambda: 30)),
    )
    monkeypatch.setattr(
        "nlab.controllers.mca_controller.owner_series",
        lambda owner: series if owner is controller else None,
    )

    MCAController._start_with_dma(cast(MCAController, controller))

    # source_001.csv exists, so the whole run moves to a free stem.
    recording = measurement_folder / "source_001_001.bin"
    assert worker_factory.call_args.kwargs["filepath"] == recording
    assert controller._series_spectrum_path == recording.with_suffix(".csv")


class _FakeListModeWorker(QObject):
    """Stands in for IIOMcaDmaWorker: ready on its thread, writes the file on close."""

    ready = Signal()
    progress = Signal(object)
    summary = Signal(object)
    connection_lost = Signal(str)
    error = Signal(str)
    finished = Signal()
    created: list[_FakeListModeWorker] = []

    def __init__(self, *, filepath: Path | None, **_kwargs: object) -> None:
        super().__init__()
        self.filepath = filepath
        self._stopped = threading.Event()
        _FakeListModeWorker.created.append(self)

    def run(self) -> None:
        self.ready.emit()
        self._stopped.wait(10)
        if self.filepath is not None:
            self.filepath.write_bytes(b"NDMA list-mode capture")
        self.finished.emit()

    def stop(self) -> None:
        self._stopped.set()


def test_mca_list_mode_series_writes_a_numbered_capture_every_run(
    qtbot: QtBot, measurement_folder: Path, frozen_clock: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeListModeWorker.created = []
    monkeypatch.setattr(
        "nlab.controllers.mca_controller.IIOMcaDmaWorker", _FakeListModeWorker
    )
    device = create_offline_digitizer(0)
    streamer = SimpleNamespace(qualified_record_schema=lambda: "vdpp-zc-calc-q2.14-v1")
    controller = MCAController(device.mca, mca_dma=cast(Any, streamer), channel=0)
    qtbot.addWidget(controller)
    try:
        controller.set_measurement_series_enabled(True)
        controller.ui.spinSeriesRuns.setValue(3)
        controller.ui.spinTimeLimit.setValue(5)
        controller.ui.cbDmaEnable.setChecked(True)

        controller.ui.btnStart.click()
        for run in (1, 2, 3):
            qtbot.waitUntil(
                lambda run=run: len(_FakeListModeWorker.created) == run
                and controller.measurement_active,
                timeout=5000,
            )
            _complete_mca_run(controller, qtbot)  # hardware time limit ends the run
        qtbot.waitUntil(
            lambda: not controller._series.active and controller._dma_thread is None,
            timeout=5000,
        )

        stem = "ch0_20261120_120000"
        captures = [measurement_folder / f"{stem}_{run:03d}.bin" for run in (1, 2, 3)]
        assert [worker.filepath for worker in _FakeListModeWorker.created] == captures
        for capture in captures:
            assert capture.read_bytes() == b"NDMA list-mode capture"
            assert capture.with_suffix(".yaml").is_file()  # settings snapshot
            assert capture.with_suffix(".csv").is_file()  # spectrum after the close
        assert controller.ui.btnStart.isEnabled()
    finally:
        controller._series.cancel()
        controller.stop_worker_sync()
        controller.stop_dma_sync()
        device.close()


# ------------------------------------------------------------------- Current


class _FakeCurrentClient(CurrentMonitorClient):
    def read_raw(self) -> int:
        return 7

    def close(self) -> None:
        pass


def test_current_series_repeats_timed_iir_runs(qtbot: QtBot) -> None:
    mca = cast(
        MultiChannelAnalyzer,
        SimpleNamespace(create_current_monitor_client=_FakeCurrentClient),
    )
    controller = CurrentMonitorController(mca, channel=0, auto_start=False)
    qtbot.addWidget(controller)
    try:
        controller.set_measurement_series_enabled(True)
        controller.ui.spinSeriesRuns.setValue(2)
        controller.ui.spinTime.setValue(60)

        controller.ui.btnStart.click()
        assert controller._acquisition is CurrentAcquisition.IIR
        assert controller._measurement_timer.isActive()
        assert not controller.ui.spinTime.isEnabled()

        controller._on_measurement_timeout()
        qtbot.waitUntil(
            lambda: controller._series.run == 2
            and controller._acquisition is CurrentAcquisition.IIR,
            timeout=5000,
        )
        controller._on_measurement_timeout()
        qtbot.waitUntil(lambda: controller._worker_thread is None, timeout=5000)
        assert not controller._series.active
        assert controller._acquisition is CurrentAcquisition.IDLE
        assert controller.ui.spinTime.isEnabled()
    finally:
        controller.stop_monitor_sync()


class _FakeSumWorker:
    def __init__(self) -> None:
        self.done = False
        self.error = ""
        self.message = "Stopped."
        self.capture = object()
        self.analysis = SimpleNamespace(
            snapshot=lambda: {"packets": 0, "points": np.empty((0, 2)), "mean": None}
        )

    def request_stop(self) -> None:
        self.done = True

    def join(self) -> None:
        pass


class _FakeSumScope(QObject):
    current_dma_state_changed = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.outputs: list[tuple[Path | None, bool]] = []
        self.last_dma_error: str | None = None

    def current_sum_supported(self) -> bool:
        return True

    def current_dma_supported(self) -> bool:
        return False

    def start_current_sum_monitor(
        self, *, gap_cycles: int, frame_samples: int | None = None,
        output: Path | None = None, overwrite: bool = False,
    ) -> _FakeSumWorker:
        self.outputs.append((output, overwrite))
        return _FakeSumWorker()

    def finish_current_sum_monitor(self) -> None:
        pass


def test_current_series_numbers_sum_recordings(
    qtbot: QtBot, measurement_folder: Path,
) -> None:
    scope = _FakeSumScope()
    mca = cast(MultiChannelAnalyzer, SimpleNamespace())
    controller = CurrentMonitorController(
        mca, channel=0, auto_start=False, scope_controller=cast(Any, scope),
    )
    qtbot.addWidget(controller)
    controller.ui.comboMode.setCurrentIndex(2)  # Periodic sums
    controller.ui.cbRecordSums.setChecked(True)
    controller.ui.editSumFile.setText("source.bin")
    measurement_folder.mkdir(parents=True)
    (measurement_folder / "source.bin").write_bytes(b"single recording")
    controller.set_measurement_series_enabled(True)
    controller.ui.spinSeriesRuns.setValue(2)
    controller.ui.spinTime.setValue(60)

    controller.ui.btnStart.click()  # no overwrite question for numbered runs
    controller._on_measurement_timeout()
    controller._render_sum_snapshot()
    qtbot.waitUntil(lambda: len(scope.outputs) == 2, timeout=2000)
    controller._on_measurement_timeout()
    controller._render_sum_snapshot()

    assert scope.outputs == [
        (measurement_folder / "source_001.bin", False),
        (measurement_folder / "source_002.bin", False),
    ]
    assert not controller._series.active
    assert controller._acquisition is CurrentAcquisition.IDLE
