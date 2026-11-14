"""Current captures can pause MCA, HV and temperature readback polling (opt-in)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from pytestqt.qtbot import QtBot

from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.main_window_controller import MainWindowController
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psu_controller import PSUController
from nlab.hardware.digitizer.mca import MultiChannelAnalyzer
from nlab.workers.base_worker import PauseRequests
from nlab.workers.global_diagnostics_worker import GlobalDiagnosticsWorker
from nlab.workers.mca_worker import MCAWorker
from nlab.workers.psu_worker import PSUWorker
from nlab.workers.temperature_correction_worker import TemperatureCorrectionWorker


def test_worker_pause_stops_and_resume_restarts_its_timer(qtbot: QtBot) -> None:
    worker = MCAWorker(cast(MultiChannelAnalyzer, SimpleNamespace()), interval_ms=50)
    worker.run()
    timer = worker._timer
    assert timer is not None and timer.isActive()

    worker._set_paused(True)
    assert not timer.isActive()
    worker._set_interval(80)  # an interval change while paused is kept, not applied
    assert not timer.isActive()
    worker._set_paused(False)
    assert timer.isActive() and timer.interval() == 80
    worker._stop()


def test_mca_worker_created_paused_does_not_poll_until_resumed(qtbot: QtBot) -> None:
    worker = MCAWorker(cast(MultiChannelAnalyzer, SimpleNamespace()), interval_ms=50, paused=True)
    worker.run()
    assert worker._timer is not None and not worker._timer.isActive()
    worker._set_paused(False)
    assert worker._timer.isActive()
    worker._stop()


def test_psu_worker_honours_an_initial_pause(qtbot: QtBot) -> None:
    worker = PSUWorker(Mock(), read_sipm=False, interval_ms=40)
    worker.set_initial_paused(True)
    worker.run()
    assert worker._timer is not None and not worker._timer.isActive()
    worker._set_interval(70)
    worker._set_paused(False)
    assert worker._timer.isActive() and worker._timer.interval() == 70
    worker._stop()


@pytest.mark.parametrize("kind", ["diagnostics", "temperature"])
def test_global_workers_skip_their_first_read_when_started_paused(qtbot: QtBot, kind: str) -> None:
    worker: Any
    if kind == "diagnostics":
        worker = GlobalDiagnosticsWorker(Mock(), interval_ms=40)
    else:
        worker = TemperatureCorrectionWorker(Mock(), [Mock()], 0.0, 0, interval_ms=40)
    ticks = Mock()
    worker._tick = ticks
    worker.set_initial_paused(True)
    worker.run()
    ticks.assert_not_called()
    assert worker._timer is not None and not worker._timer.isActive()
    worker._set_paused(False)
    assert worker._timer.isActive()
    worker._stop()


def test_pause_requests_are_tracked_per_requester() -> None:
    requests = PauseRequests()
    assert requests.update("Current 0", True)
    assert not requests.update("Current 1", True)
    assert not requests.update("Current 0", False)  # channel 1 still needs it
    assert requests.active
    assert requests.update("Current 1", False)
    assert not requests.active


def _with_worker(**extra: Any) -> Any:
    owner = SimpleNamespace(
        _channel=0,
        _worker=SimpleNamespace(set_paused=SimpleNamespace(emit=Mock())),
        _measurement_active=True,
        status=SimpleNamespace(setText=Mock()),
        **extra,
    )
    return owner


def test_mca_controller_pauses_its_worker_and_reports_it() -> None:
    owner = _with_worker()
    MCAController.set_readbacks_paused(owner, "Current 0", True)
    MCAController.set_readbacks_paused(owner, "Current 1", True)
    MCAController.set_readbacks_paused(owner, "Current 0", False)
    assert owner._worker.set_paused.emit.call_args_list == [((True,),)]
    MCAController.set_readbacks_paused(owner, "Current 1", False)
    assert owner._worker.set_paused.emit.call_args_list == [((True,),), ((False,),)]
    assert "resumed" in owner.status.setText.call_args.args[0]


def test_psu_controller_pauses_its_monitor() -> None:
    owner = _with_worker()
    PSUController.set_readbacks_paused(owner, "Current 0", True)
    owner._worker.set_paused.emit.assert_called_once_with(True)


def test_global_controller_pauses_diagnostics_and_temperature_loop() -> None:
    temperature = SimpleNamespace(set_paused=SimpleNamespace(emit=Mock()))
    owner = _with_worker(_temperature_worker=temperature)
    GlobalController.set_readbacks_paused(owner, "Current 0", True)
    owner._worker.set_paused.emit.assert_called_once_with(True)
    temperature.set_paused.emit.assert_called_once_with(True)


def test_main_window_fans_a_pause_request_out_to_every_poller() -> None:
    window = object.__new__(MainWindowController)
    window._mca_controllers = [Mock(), Mock()]
    window._psu_controllers = [Mock()]
    window._global_controllers = [Mock()]
    window._set_board_readbacks_paused("Current 0", True)
    for controller in (*window._mca_controllers, *window._psu_controllers,
                       *window._global_controllers):
        controller.set_readbacks_paused.assert_called_once_with("Current 0", True)


def _current(qtbot: QtBot) -> CurrentMonitorController:
    controller = CurrentMonitorController(
        cast(MultiChannelAnalyzer, SimpleNamespace()), channel=0, auto_start=False,
    )
    qtbot.addWidget(controller)
    controller._display_timer.stop()
    return controller


def test_current_checkbox_requests_pause_only_while_running(qtbot: QtBot) -> None:
    controller = _current(qtbot)
    requests: list[bool] = []
    controller.readback_pause_changed.connect(requests.append)

    controller.ui.cbPauseReadbacks.setChecked(True)
    assert requests == []  # nothing is running yet
    controller._set_running_controls(True)
    assert requests == [True]
    controller.ui.cbPauseReadbacks.setChecked(False)  # unticked mid-capture
    assert requests == [True, False]
    controller.ui.cbPauseReadbacks.setChecked(True)
    assert requests == [True, False, True]
    controller._set_running_controls(False)
    assert requests == [True, False, True, False]


def test_session_replay_never_pauses_live_polling(qtbot: QtBot) -> None:
    controller = _current(qtbot)
    requests: list[bool] = []
    controller.readback_pause_changed.connect(requests.append)
    controller.ui.cbPauseReadbacks.setChecked(True)
    controller.set_replay_hardware_interlock(True)
    controller._set_running_controls(True)
    assert requests == []


def test_pause_option_round_trips_through_configuration(qtbot: QtBot) -> None:
    controller = _current(qtbot)
    assert controller.configuration_settings()["pause_readbacks"] is False
    controller.apply_configuration_settings({"pause_readbacks": True})
    assert controller.ui.cbPauseReadbacks.isChecked()
    assert controller.configuration_settings()["pause_readbacks"] is True
