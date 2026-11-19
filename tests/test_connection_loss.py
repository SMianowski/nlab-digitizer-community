"""A lost board connection stops polling once and offers a single reconnect."""

from __future__ import annotations

import errno
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock

import pytest
from pytestqt.qtbot import QtBot

from nlab.app import MainAppWindow
from nlab.controllers.global_controller import GlobalController
from nlab.controllers.main_window_controller import MainWindowController
from nlab.controllers.mca_controller import MCAController
from nlab.controllers.scope_controller import ScopeController
from nlab.hardware.digitizer.offline import create_offline_digitizer
from nlab.hardware.digitizer.scope import Scope
from nlab.workers.scope_worker import ScopeWorker

RESET = ConnectionResetError(10054, "Unknown error")


@pytest.mark.parametrize(
    ("exc", "connection_lost"),
    [(RESET, True), (OSError(errno.EIO, "I/O error"), False)],
)
def test_scope_worker_reports_scalar_failure_and_whether_the_connection_is_gone(
    exc: Exception,
    connection_lost: bool,
) -> None:
    scope = Mock()
    scope.get_frame_samples.side_effect = exc
    # Production passes a lazy viewer-connection provider.
    worker = ScopeWorker(lambda: cast(Scope, scope))
    failures: list[tuple[str, bool]] = []
    frames: list[object] = []
    worker.signals.failed.connect(lambda message, lost: failures.append((message, lost)))
    worker.signals.ready.connect(frames.append)

    worker.run()

    assert failures == [(str(exc), connection_lost)]
    assert frames == [None]


@pytest.fixture
def scope(qtbot: QtBot) -> Iterator[ScopeController]:
    device = create_offline_digitizer(0)
    controller = ScopeController(device.scope)
    qtbot.addWidget(controller)
    try:
        yield controller
    finally:
        controller._refresh_timer.stop()
        controller._measurement_timer.stop()
        controller.stop_dma_sync()
        device.close()


def test_scope_stops_live_view_without_hardware_writes_when_the_connection_is_lost(
    scope: ScopeController,
) -> None:
    scope._start_polling_only()
    scope._refresh_timer.stop()
    scope._refresh_timer.start(10_000)  # Active, but no background frame requests.
    scope.ui.btnStart.setChecked(True)
    scope._scope = MagicMock(wraps=scope._scope)
    lost: list[str] = []
    scope.connection_lost.connect(lost.append)

    scope._on_viewer_frame_failed("transient", False)
    assert scope._refresh_timer.isActive() and lost == []

    scope._on_viewer_frame_failed("[Errno 10054] Unknown error", True)

    assert not scope._refresh_timer.isActive()
    assert not scope.measurement_active
    assert scope.ui.btnStart.isEnabled() and not scope.ui.btnStart.isChecked()
    assert not scope.ui.btnStop.isEnabled()
    assert "Reconnect Device" in scope.status.text()
    assert lost == ["[Errno 10054] Unknown error"]
    scope._scope.stop.assert_not_called()
    # A late failure from a frame already in flight does not report twice.
    scope._on_viewer_frame_failed("[Errno 10054] Unknown error", True)
    assert len(lost) == 1


def test_mca_reports_its_connection_loss(
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mca = MagicMock()
    mca.get_debug_signal_selectors.return_value = tuple(range(9))
    mca.edge_det_coeff_is_hardware_backed.return_value = False
    monkeypatch.setattr(MCAController, "_load_hardware_state", lambda self: None)
    controller = MCAController(mca, mca_dma=None, channel=0)
    qtbot.addWidget(controller)
    lost: list[str] = []
    controller.connection_lost.connect(lost.append)

    controller._on_connection_lost("[Errno 10054] Unknown error")

    assert lost == ["[Errno 10054] Unknown error"]


@pytest.mark.parametrize("worker", ["diagnostics", "temperature"])
@pytest.mark.parametrize("requested", [False, True])
def test_global_workers_report_only_an_unrequested_stop(worker: str, requested: bool) -> None:
    thread = object()
    state: Any = SimpleNamespace(connection_lost=Mock())
    if worker == "diagnostics":
        state._worker = object()
        state._worker_thread = thread
        state._worker_stop_requested = requested
        state._last_diagnostics_error = "[Errno 10054] Unknown error"
        GlobalController._on_diagnostics_finished(cast(GlobalController, state), thread)
        assert state._worker is None and state._worker_thread is None
    else:
        state._temperature_worker = object()
        state._temperature_worker_thread = thread
        state._temperature_worker_stop_requested = requested
        state._last_temperature_error = "[Errno 10054] Unknown error"
        GlobalController._on_temperature_finished(cast(GlobalController, state), thread)
        assert state._temperature_worker is None
    if requested:
        state.connection_lost.emit.assert_not_called()
    else:
        state.connection_lost.emit.assert_called_once_with("[Errno 10054] Unknown error")


def test_main_window_controller_forwards_panel_losses_to_the_window() -> None:
    window = SimpleNamespace(notify_connection_lost=Mock())
    controller: Any = SimpleNamespace(_window=window)

    MainWindowController._report_connection_lost(
        cast(MainWindowController, controller), "PSU CH0", "connection reset",
    )
    MainWindowController._report_connection_lost(
        cast(MainWindowController, SimpleNamespace(_window=object())), "PSU CH0", "x",
    )

    window.notify_connection_lost.assert_called_once_with("PSU CH0", "connection reset")


def _window(**overrides: object) -> Any:
    state: dict[str, object] = {
        "_offline": False,
        "_connection_busy": False,
        "_modbus_refresh_running": False,
        "_connection_loss_prompted": False,
        "_offer_reconnect_after_loss": Mock(),
    }
    state.update(overrides)
    return SimpleNamespace(**state)


def test_window_prompts_once_per_lost_connection(qtbot: QtBot) -> None:
    window = _window()

    MainAppWindow.notify_connection_lost(window, "PSU CH0", "[Errno 10054] Unknown error")
    MainAppWindow.notify_connection_lost(window, "MCA CH0", "[Errno 10054] Unknown error")

    qtbot.waitUntil(lambda: window._offer_reconnect_after_loss.called, timeout=2_000)
    window._offer_reconnect_after_loss.assert_called_once_with(
        "PSU CH0: [Errno 10054] Unknown error"
    )


@pytest.mark.parametrize(
    "state",
    [{"_offline": True}, {"_connection_busy": True}, {"_connection_loss_prompted": True}],
)
def test_window_does_not_prompt_offline_during_reconnect_or_twice(
    qtbot: QtBot,
    state: dict[str, object],
) -> None:
    window = _window(**state)

    MainAppWindow.notify_connection_lost(window, "Scope CH0", "lost")
    qtbot.wait(50)

    window._offer_reconnect_after_loss.assert_not_called()


def test_reconnect_prompt_retries_after_a_failed_attempt_and_stops_when_declined() -> None:
    window: Any = SimpleNamespace(
        _ask_reconnect_after_loss=Mock(side_effect=[True, True, False]),
        _reconnect_device=Mock(side_effect=["board unreachable", "still down"]),
    )

    MainAppWindow._offer_reconnect_after_loss(window, "PSU CH0: connection reset")

    assert [entry.args[0] for entry in window._ask_reconnect_after_loss.call_args_list] == [
        "PSU CH0: connection reset",
        "Reconnect failed: board unreachable",
        "Reconnect failed: still down",
    ]
    assert window._reconnect_device.call_count == 2


def test_successful_reconnect_ends_the_prompt() -> None:
    window: Any = SimpleNamespace(
        _ask_reconnect_after_loss=Mock(return_value=True),
        _reconnect_device=Mock(return_value=None),
    )

    MainAppWindow._offer_reconnect_after_loss(window, "Global D1: connection reset")

    window._ask_reconnect_after_loss.assert_called_once()
    window._reconnect_device.assert_called_once_with()
