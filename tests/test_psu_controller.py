from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from pytestqt.qtbot import QtBot

from nlab.controllers.psu_controller import PSUController
from nlab.hardware.digitizer.backends.base import IDSBackend
from nlab.hardware.digitizer.hv import HVSupply


@pytest.fixture
def psu(qtbot: QtBot) -> tuple[PSUController, MagicMock]:
    backend = MagicMock(spec=IDSBackend)
    backend.get_sipm_overload.return_value = -1
    backend.get_hv_adc_voltage.return_value = 500.0
    backend.get_hv_compens_output.return_value = 500.0
    backend.get_temp_analog.return_value = 20.0
    backend.get_temp_digital.return_value = 20.0
    backend.get_temp_digital_status.return_value = 1
    backend.get_ads_temp.return_value = 25.0
    controller = PSUController(HVSupply(backend))
    qtbot.addWidget(controller)
    controller.show()
    backend.reset_mock()
    return controller, backend


@pytest.mark.parametrize("finish", ["enter", "focus_out"])
@pytest.mark.parametrize(
    ("text", "expected"),
    [("0", 100.0), ("99", 100.0), ("100", 100.0), ("1250", 1250.0)],
)
def test_hv_voltage_edit_commits_a_supported_value(
    psu: tuple[PSUController, MagicMock],
    text: str,
    expected: float,
    finish: str,
) -> None:
    controller, backend = psu
    spinbox = controller.ui.spinHvVoltage
    QTest.mouseClick(spinbox, Qt.MouseButton.LeftButton)
    spinbox.selectAll()
    QTest.keyClicks(spinbox, text)

    if finish == "enter":
        QTest.keyClick(spinbox, Qt.Key.Key_Return)
    else:
        QTest.mouseClick(controller.ui.spinHvCompensCt, Qt.MouseButton.LeftButton)

    assert spinbox.value() == expected
    assert controller.hardware_configuration_settings()["hv_voltage"] == expected
    backend.set_hv_voltage.assert_called_once_with(expected)
    # ADC feedback is independent of the corrected request.
    assert controller.ui.lblHvVoltage.text() == "500.0"


def test_hv_voltage_cannot_step_below_minimum(
    psu: tuple[PSUController, MagicMock],
) -> None:
    controller, backend = psu
    spinbox = controller.ui.spinHvVoltage
    spinbox.setValue(100.0)

    QTest.keyClick(spinbox, Qt.Key.Key_Down)
    QTest.keyClick(spinbox, Qt.Key.Key_Return)

    assert spinbox.value() == 100.0
    backend.set_hv_voltage.assert_called_once_with(100.0)


def _sipm_backend() -> MagicMock:
    backend = MagicMock(spec=IDSBackend)
    backend.get_sipm_overload.return_value = 0
    for name in (
        "get_hv_adc_voltage",
        "get_hv_compens_output",
        "get_temp_analog",
        "get_temp_digital",
        "get_ads_temp",
        "get_sipm_adc_voltage",
        "get_sipm_adc_current",
        "get_sipm_compens_output",
    ):
        getattr(backend, name).return_value = 30.0
    backend.get_temp_digital_status.return_value = 1
    return backend


def test_construction_sends_spec_defaults(qtbot: QtBot) -> None:
    backend = _sipm_backend()
    controller = PSUController(HVSupply(backend))
    qtbot.addWidget(controller)

    backend.set_hv_voltage.assert_called_once_with(100.0)
    backend.set_hv_compens_ct.assert_called_once_with(0.0)
    backend.set_hv_compens_tref.assert_called_once_with(20.0)
    backend.set_hv_compens_mode.assert_called_once_with(0)
    backend.set_temp_digital_enable.assert_called_once_with(0)
    backend.set_sipm_enable.assert_called_once_with(0)
    backend.set_sipm_voltage.assert_called_once_with(30.0)
    backend.set_sipm_compens_ct.assert_called_once_with(0.0)
    backend.set_sipm_compens_tref.assert_called_once_with(20.0)
    backend.set_sipm_compens_mode.assert_called_once_with(0)


def _edit_every_ids_control(controller: PSUController) -> None:
    ui = controller.ui
    ui.cbSipmEnable.setChecked(True)
    ui.comboSipmCompensMode.setCurrentIndex(2)
    ui.spinSipmCompensCt.setValue(0.05)
    ui.spinSipmCompensCt.editingFinished.emit()
    ui.spinSipmCompensTref.setValue(25.0)
    ui.spinSipmCompensTref.editingFinished.emit()
    ui.comboHvCompensMode.setCurrentIndex(1)
    ui.spinHvCompensCt.setValue(-0.03)
    ui.spinHvCompensCt.editingFinished.emit()
    ui.spinHvCompensTref.setValue(30.0)
    ui.spinHvCompensTref.editingFinished.emit()
    ui.comboTempDigitalEnable.setCurrentIndex(1)


def test_ids_controls_write_their_setting(qtbot: QtBot) -> None:
    backend = _sipm_backend()
    controller = PSUController(HVSupply(backend))
    qtbot.addWidget(controller)
    backend.reset_mock()

    _edit_every_ids_control(controller)

    backend.set_sipm_enable.assert_called_once_with(1)
    backend.set_sipm_compens_mode.assert_called_once_with(2)
    backend.set_sipm_compens_ct.assert_called_once_with(pytest.approx(0.05))
    backend.set_sipm_compens_tref.assert_called_once_with(pytest.approx(25.0))
    backend.set_hv_compens_mode.assert_called_once_with(1)
    backend.set_hv_compens_ct.assert_called_once_with(pytest.approx(-0.03))
    backend.set_hv_compens_tref.assert_called_once_with(pytest.approx(30.0))
    backend.set_temp_digital_enable.assert_called_once_with(1)


def test_replay_interlock_blocks_ids_control_writes(qtbot: QtBot) -> None:
    backend = _sipm_backend()
    controller = PSUController(HVSupply(backend))
    qtbot.addWidget(controller)
    controller.set_replay_hardware_interlock(True)
    backend.reset_mock()

    _edit_every_ids_control(controller)

    for name in (
        "set_sipm_enable",
        "set_sipm_compens_mode",
        "set_sipm_compens_ct",
        "set_sipm_compens_tref",
        "set_hv_compens_mode",
        "set_hv_compens_ct",
        "set_hv_compens_tref",
        "set_temp_digital_enable",
    ):
        getattr(backend, name).assert_not_called()


@pytest.mark.parametrize("voltage", [0.0, 99.9, 1250.1])
def test_hv_api_rejects_voltage_outside_operating_range(voltage: float) -> None:
    backend = MagicMock(spec=IDSBackend)

    with pytest.raises(ValueError, match="hv_voltage: .* out of range"):
        HVSupply(backend).set_hv_voltage(voltage)

    backend.set_hv_voltage.assert_not_called()


def test_monitor_stops_and_reports_when_the_board_connection_is_lost(
    psu: tuple[PSUController, MagicMock],
    qtbot: QtBot,
) -> None:
    controller, backend = psu
    backend.get_hv_adc_voltage.side_effect = ConnectionResetError(10054, "Unknown error")
    lost: list[str] = []
    controller.connection_lost.connect(lost.append)

    controller.ui.btnStartMonitor.click()
    qtbot.waitUntil(lambda: controller._worker_thread is None, timeout=5_000)

    assert backend.get_hv_adc_voltage.call_count == 1
    assert controller.ui.btnStartMonitor.isEnabled()
    assert not controller.ui.btnStopMonitor.isEnabled()
    assert "connection to the board was lost" in controller.status.text()
    assert lost == ["[Errno 10054] Unknown error"]


def test_requested_monitor_stop_does_not_report_a_lost_connection(
    psu: tuple[PSUController, MagicMock],
    qtbot: QtBot,
) -> None:
    controller, backend = psu
    lost: list[str] = []
    controller.connection_lost.connect(lost.append)
    controller.ui.btnStartMonitor.click()
    # Stop only once the worker is polling; run() connects request_stop.
    qtbot.waitUntil(lambda: backend.get_hv_adc_voltage.called, timeout=5_000)

    controller.ui.btnStopMonitor.click()
    qtbot.waitUntil(lambda: controller._worker_thread is None, timeout=5_000)

    assert controller.status.text() == ""
    assert lost == []
    assert controller.ui.btnStartMonitor.isEnabled()
