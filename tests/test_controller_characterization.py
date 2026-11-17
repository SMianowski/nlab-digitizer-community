"""Characterization of the real Scope, MCA and PSU controllers.

Built on a recording Offline backend and observed only through public
surfaces (widgets, signals, settings, backend writes). These pin current
behaviour before the controllers are restructured; they must keep passing
unchanged through extractions and renames of private members.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget
from pytestqt.qtbot import QtBot

from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import DisplayMode, ScopeController
from nlab.hardware.digitizer.scope import TriggerMode
from nlab.ui.ui_scope_view import Ui_ScopeView
from nlab.views.general_settings_dialog import GeneralSettingsDialog
from tests._controller_harness import recording_digitizer, signal_log


@pytest.fixture(autouse=True)
def isolated_qsettings(tmp_path: Path) -> Iterator[None]:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    yield
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


def _scope(qtbot: QtBot, **backend_values: object) -> tuple[ScopeController, object]:
    device, backend, _ids = recording_digitizer()
    for name, value in backend_values.items():
        getattr(backend, f"set_{name}")(value)
    backend.writes.clear()
    controller = ScopeController(device.scope, scope_dma=None, channel=0)
    qtbot.addWidget(controller)
    return controller, backend


def _mca(qtbot: QtBot) -> tuple[MCAController, object]:
    device, backend, _ids = recording_digitizer()
    controller = MCAController(device.mca)
    qtbot.addWidget(controller)
    backend.writes.clear()
    return controller, backend


# --------------------------------------------------------------------- Scope


def test_scope_construction_writes_hardware_defaults_and_shows_them(qtbot: QtBot) -> None:
    controller, backend = _scope(qtbot, trigger_mode=int(TriggerMode.RISING_EDGE),
                                 frame_samples=64)
    # The view resets the Scope to its spec defaults rather than adopting
    # whatever the hardware held before.
    assert [name for name, _ in backend.writes] == [
        "set_edge", "set_trigger_level", "set_dac_value", "set_pretrigger_samples",
        "set_frame_samples", "set_frame_period_cycles",
    ]
    assert controller.ui.comboTriggerMode.currentData() == backend.written("set_edge")[0]
    assert controller.ui.spinFrameSamples.value() == backend.written("set_frame_samples")[0] * 2
    assert controller.ui.spinPretrigger.value() == backend.written("set_pretrigger_samples")[0] * 2


def test_scope_display_defaults_to_raw_without_saved_settings(qtbot: QtBot) -> None:
    # Saved per-channel QSettings override this at construction, so check the
    # form default and the configuration fallbacks directly.
    view = QWidget()
    qtbot.addWidget(view)
    ui = Ui_ScopeView()
    ui.setupUi(view)
    assert ui.comboDisplayMode.currentIndex() == DisplayMode.RAW
    assert ui.comboDisplayMode.currentText() == "Raw"
    dialog = GeneralSettingsDialog("192.0.2.10", {})
    qtbot.addWidget(dialog)
    assert dialog.settings["scope_display_mode"] == DisplayMode.RAW


def test_scope_trigger_mode_choice_is_written_to_the_backend(qtbot: QtBot) -> None:
    controller, backend = _scope(qtbot)
    backend.writes.clear()
    combo = controller.ui.comboTriggerMode
    combo.setCurrentIndex(combo.findData(int(TriggerMode.RISING_EDGE)))
    assert backend.writes == [("set_edge", int(TriggerMode.RISING_EDGE))]


def test_scope_start_and_stop_drive_enable_and_report_activity(qtbot: QtBot) -> None:
    controller, backend = _scope(qtbot, trigger_mode=int(TriggerMode.FALLING_EDGE))
    activity = signal_log(controller.measurement_active_changed)
    try:
        controller.ui.btnStart.click()
        assert controller.measurement_active
        assert backend.written("set_enable")[-1] is True
        assert controller.ui.btnStop.isEnabled()
        controller.ui.btnStop.click()
        assert not controller.measurement_active
        assert backend.written("set_enable")[-1] is False
    finally:
        controller.stop_dma_sync()
        controller.close_viewer_client()
    assert activity == [(True,), (False,)]


def test_scope_view_settings_round_trip(qtbot: QtBot) -> None:
    first, _ = _scope(qtbot)
    first.apply_configuration_settings({
        "display_mode": 1, "persistence": 7, "refresh_rate_hz": 12,
        "measurement_time_s": 30, "dma_enabled": True,
    })
    saved = first.configuration_settings()
    second, _ = _scope(qtbot)
    second.apply_configuration_settings(saved)
    assert second.configuration_settings() == saved


def test_scope_replay_interlock_blocks_start(qtbot: QtBot) -> None:
    controller, backend = _scope(qtbot)
    controller.set_replay_hardware_interlock(True)
    assert not controller.ui.btnStart.isEnabled()
    controller.ui.btnStart.click()
    assert backend.written("set_enable") == []
    controller.set_replay_hardware_interlock(False)
    assert controller.ui.btnStart.isEnabled()


# ----------------------------------------------------------------------- MCA


def test_mca_start_polls_and_stop_disarms(qtbot: QtBot) -> None:
    controller, backend = _mca(qtbot)
    readbacks = signal_log(controller.session_readback)
    activity = signal_log(controller.measurement_active_changed)
    try:
        controller.ui.btnStart.click()
        assert controller.measurement_active
        # The pulse processor is disarmed, the time limit written, then armed.
        assert backend.written("set_global_enable")[-2:] == [False, True]
        assert backend.written("set_time_limit")
        qtbot.waitUntil(lambda: len(readbacks) >= 2, timeout=5000)
        controller.ui.btnStop.click()
        assert backend.written("set_global_enable")[-1] is False
        assert not controller.measurement_active
        settled = len(readbacks)
        qtbot.wait(400)
        assert len(readbacks) <= settled + 1  # at most one in-flight poll
    finally:
        controller.stop_worker_sync()
    assert activity == [(True,), (False,)]


def test_mca_readback_pause_holds_polling_without_touching_hardware(qtbot: QtBot) -> None:
    controller, backend = _mca(qtbot)
    readbacks = signal_log(controller.session_readback)
    try:
        controller.ui.btnStart.click()
        qtbot.waitUntil(lambda: len(readbacks) >= 1, timeout=5000)
        writes_before = list(backend.writes)
        controller.set_readbacks_paused("test", True)
        qtbot.wait(150)
        held = len(readbacks)
        qtbot.wait(400)
        assert len(readbacks) == held
        assert backend.writes == writes_before
        controller.set_readbacks_paused("test", False)
        qtbot.waitUntil(lambda: len(readbacks) > held, timeout=5000)
        controller.ui.btnStop.click()
    finally:
        controller.stop_worker_sync()


def test_mca_view_settings_round_trip(qtbot: QtBot) -> None:
    first, _ = _mca(qtbot)
    first.apply_configuration_settings({"refresh_rate_hz": 7, "roi": [100.0, 250.0],
                                        "dma_enabled": True})
    saved = first.configuration_settings()
    second, _ = _mca(qtbot)
    second.apply_configuration_settings(saved)
    assert second.configuration_settings() == saved


def test_mca_replay_interlock_blocks_start(qtbot: QtBot) -> None:
    controller, backend = _mca(qtbot)
    controller.set_replay_hardware_interlock(True)
    assert not controller.ui.btnStart.isEnabled()
    controller.ui.btnStart.click()
    assert backend.written("set_global_enable") == []
    controller.set_replay_hardware_interlock(False)
    assert controller.ui.btnStart.isEnabled()


# ----------------------------------------------------------------------- PSU


def test_psu_monitor_reads_back_until_stopped(qtbot: QtBot) -> None:
    device, _backend, _ids = recording_digitizer()
    assert device.hv is not None
    controller = PSUController(device.hv)
    qtbot.addWidget(controller)
    readbacks = signal_log(controller.session_readback)
    try:
        if not controller.ui.btnStartMonitor.isEnabled():
            controller.ui.btnStopMonitor.click()
            controller.stop_monitor_sync()
        controller.ui.spinRefreshRate.setValue(controller.ui.spinRefreshRate.minimum())
        controller.ui.btnStartMonitor.click()
        qtbot.waitUntil(lambda: len(readbacks) >= 2, timeout=5000)
        controller.ui.btnStopMonitor.click()
        assert controller.ui.btnStartMonitor.isEnabled()
        # Readbacks polled before Stop can still be queued for the GUI thread:
        # in the CI container a ~280 ms GUI stall delayed the click itself, and
        # three readbacks polled before it were delivered after it. The worker
        # must still go quiet; one that kept polling every 100 ms never would.
        deadline = time.monotonic() + 5
        while True:
            before = len(readbacks)
            qtbot.wait(300)
            if len(readbacks) == before:
                break
            assert time.monotonic() < deadline, "PSU readbacks continued after Stop"
        assert controller.ui.btnStartMonitor.isEnabled()
    finally:
        controller.stop_monitor_sync()


def test_mca_defaults_start_the_cfd_window_at_the_trigger(qtbot: QtBot) -> None:
    controller, _ = _mca(qtbot)
    ui = controller.ui
    # PetaLinux cde5fe1 P2: a window opening inside the pretrigger history
    # reports false offset-0 crossings, so defaults keep low >= pretrigger.
    assert ui.spinCfdTwLow.value() >= ui.spinPretrigger.value()
    ui.cbCfdEnable.setChecked(True)
    assert "Starts before the trigger" not in ui.spinCfdTwLow.toolTip()

    ui.spinCfdTwLow.setValue(0)
    assert "Starts before the trigger" in ui.spinCfdTwLow.toolTip()
    assert ui.spinCfdTwLow.styleSheet()

    ui.cbCfdEnable.setChecked(False)  # CFD off: the window does not matter
    assert "Starts before the trigger" not in ui.spinCfdTwLow.toolTip()
    assert not ui.spinCfdTwLow.styleSheet()
