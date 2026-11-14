"""Characterize session replay end to end before it moves out of the main window.

A real session is recorded with SessionRecorder, then replayed in a real
offline MainAppWindow. Assertions use only public surfaces: the window's
replay API, Qt's findChildren, and what the views show. Moving the replay
code into its own coordinator must leave these tests unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow
from nlab.controllers.current_monitor_controller import CurrentMonitorController
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.psu_controller import PSUController
from nlab.controllers.scope_controller import ScopeController
from nlab.session.recorder import SessionRecorder
from nlab.workers.current_monitor_worker import CurrentSample, CurrentSampleBatch
from nlab.workers.mca_worker import MCAReadback

_PSU_READBACK = {
    "timestamp": 1.0, "hv_voltage": 42.5, "hv_compens_output": 42.0, "temp_analog": 21.0,
    "temp_digital": 21.5, "temp_digital_status": 1, "ads_temp": 22.0,
}


@pytest.fixture(autouse=True)
def isolated_qsettings(tmp_path: Path) -> Iterator[None]:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "ini"))
    yield
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


def _record(path: Path) -> Path:
    recorder = SessionRecorder(path, {"format_version": 4}, application_version="test")
    recorder.record_scope_viewer(0, np.array([5, 6, 7, 8], dtype=np.int16), sample_period_ns=8)
    time.sleep(0.01)
    recorder.record_mca_readback(0, MCAReadback(
        histogram=np.arange(16, dtype=np.uint32), debug1=np.zeros(2, dtype=np.int16),
        debug2=np.zeros(2, dtype=np.int16), count_rate=12, events_lost=None,
    ))
    recorder.record_current_samples(0, CurrentSampleBatch(
        samples=(CurrentSample(sequence=1, timestamp_ns=10, raw_code=3.0,
                               read_latency_ns=100, hardware_timestamp=None),),
        dropped_samples=0, error=None,
    ))
    recorder.record_telemetry("psu.ch0", {"readback": dict(_PSU_READBACK)})
    time.sleep(0.01)
    recorder.stop(final_configuration={"format_version": 4})
    return path


def _only(window: MainAppWindow, kind: type) -> object:
    views = window.findChildren(kind)
    assert len(views) == 1, (kind.__name__, len(views))
    return views[0]


@pytest.fixture
def offline_window(qtbot: QtBot) -> Iterator[MainAppWindow]:
    window = MainAppWindow(channels=1, offline=True)
    qtbot.addWidget(window)
    yield window
    window._controller.clear_session_replay()


def test_replay_drives_every_view_then_clears(
    qtbot: QtBot, tmp_path: Path, offline_window: MainAppWindow,
) -> None:
    path = _record(tmp_path / "run.nlab-session")
    controller = offline_window._controller
    scope = _only(offline_window, ScopeController)
    psu = _only(offline_window, PSUController)
    assert isinstance(scope, ScopeController) and isinstance(psu, PSUController)

    replay = controller.replay_session(path)

    assert controller.session_replay is replay
    assert controller.session_replay_path == path
    assert scope.session_replay_active
    qtbot.waitUntil(lambda: "Scope replay complete" in scope.status.text(), timeout=10_000)
    assert "1 frame(s)" in scope.status.text()
    qtbot.waitUntil(lambda: psu.ui.lblHvVoltage.text().startswith("42.5"), timeout=10_000)

    controller.clear_session_replay()

    assert controller.session_replay is None
    assert controller.session_replay_path is None
    assert not scope.session_replay_active


def test_replay_keeps_hardware_interlocked_in_offline_mode(
    qtbot: QtBot, tmp_path: Path, offline_window: MainAppWindow,
) -> None:
    path = _record(tmp_path / "run.nlab-session")
    controller = offline_window._controller
    mca = _only(offline_window, MCAController)
    current = _only(offline_window, CurrentMonitorController)
    assert isinstance(mca, MCAController) and isinstance(current, CurrentMonitorController)

    controller.replay_session(path)
    assert not mca.ui.btnStart.isEnabled()
    assert not current.ui.btnStart.isEnabled()
    controller.clear_session_replay()
    # Offline mode never releases the interlock: there is no hardware.
    assert not mca.ui.btnStart.isEnabled()


def test_replaying_again_replaces_the_previous_replay(
    qtbot: QtBot, tmp_path: Path, offline_window: MainAppWindow,
) -> None:
    first = _record(tmp_path / "first.nlab-session")
    second = _record(tmp_path / "second.nlab-session")
    controller = offline_window._controller
    controller.replay_session(first)
    replay = controller.replay_session(second)
    assert controller.session_replay is replay
    assert controller.session_replay_path == second


def test_replay_refuses_channels_the_offline_workspace_lacks(
    tmp_path: Path, offline_window: MainAppWindow,
) -> None:
    path = tmp_path / "two.nlab-session"
    recorder = SessionRecorder(path, {"format_version": 4}, application_version="test")
    recorder.record_scope_viewer(1, np.array([1, 2], dtype=np.int16), sample_period_ns=8)
    recorder.stop(final_configuration={"format_version": 4})
    controller = offline_window._controller
    with pytest.raises(RuntimeError, match="restart Offline Mode with at least 2 channels"):
        controller.replay_session(path)
    assert controller.session_replay is None
