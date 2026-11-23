import math
from types import SimpleNamespace
from typing import Any, cast

from nlab_modbus.core.base_modbus_device import BaseModbusDevice
from nlab_modbus.core.register_specs import RegisterType
from nlab_modbus.devices.geiger import GeigerDevice
from nlab_modbus.devices.sipm import SiPMDevice
from nlab_modbus.maps.sipm_map import SIPM_REGISTER_MAP
from PySide6.QtWidgets import QScrollArea
from pytestqt.qtbot import QtBot

from nlab.controllers.external_device_controller import ExternalDeviceController


def _sipm_device() -> BaseModbusDevice:
    return cast(
        BaseModbusDevice,
        SimpleNamespace(
            REGISTER_MAP=SIPM_REGISTER_MAP,
            READOUT_START=3,
            READOUT_STOP=22,
            device_type=SimpleNamespace(name="SiPM"),
            device_id=1,
            connection_info=lambda: "test connection",
        ),
    )


def test_external_tables_fit_default_view_without_horizontal_scrolling(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ExternalDeviceController, "_load_settings", lambda self: None)
    monkeypatch.setattr(ExternalDeviceController, "_on_start_monitor", lambda self: None)
    controller = ExternalDeviceController(_sipm_device())
    qtbot.addWidget(controller)
    controller.resize(900, 600)
    controller.show()
    qtbot.waitExposed(controller)

    assert controller.ui.settingsTable.font().pointSizeF() < controller.font().pointSizeF()
    assert controller.ui.settingsTable.width() == controller.ui.telemetryTable.width()
    for table in (controller.ui.settingsTable, controller.ui.telemetryTable):
        last_column = table.columnCount() - 1
        last_column_right = (
            table.columnViewportPosition(last_column) + table.columnWidth(last_column)
        )
        assert last_column_right <= table.viewport().width()
        assert table.horizontalScrollBar().maximum() == 0
        assert table.width() < 400

    controls = controller.findChild(QScrollArea, "externalControlScroll")
    assert controls is not None
    assert controls.horizontalScrollBar().maximum() == 0


def test_long_external_register_labels_have_full_tooltips(
    qtbot: QtBot,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ExternalDeviceController, "_load_settings", lambda self: None)
    monkeypatch.setattr(ExternalDeviceController, "_on_start_monitor", lambda self: None)
    controller = ExternalDeviceController(_sipm_device())
    qtbot.addWidget(controller)

    row = controller._row_by_name["telemetry:ext_amp_supply_fault"]
    item = controller.ui.telemetryTable.item(row, 0)
    assert item is not None
    assert item.toolTip() == (
        "External Amplifier Supply Fault\nModbus register: ext_amp_supply_fault"
    )


# --- nlab-modbus 0.3 devices against a fake board ------------------------------


class _FakeBoard:
    """A Modbus client answering like firmware with the given table lengths.

    Reads past a table end get the firmware's exception 2 (illegal data
    address), which nlab-modbus 0.3 uses to detect older, shorter tables.
    """

    def __init__(self, input_words: list[int], holding_words: list[int]) -> None:
        self.input_words = input_words
        self.holding_words = holding_words
        self.writes: list[tuple[int, int]] = []

    @staticmethod
    def _answer(table: list[int], address: int, count: int) -> object:
        if address + count > len(table):
            return SimpleNamespace(isError=lambda: True, exception_code=2)
        words = table[address:address + count]
        return SimpleNamespace(isError=lambda: False, registers=words)

    def read_input_registers(self, *, address: int, count: int, device_id: int) -> object:
        return self._answer(self.input_words, address, count)

    def read_holding_registers(self, *, address: int, count: int, device_id: int) -> object:
        return self._answer(self.holding_words, address, count)

    def write_register(self, *, address: int, value: int, device_id: int) -> object:
        self.writes.append((address, value))
        return SimpleNamespace(isError=lambda: False)


def _board_device(
    device_class: type[BaseModbusDevice], *, input_size: int | None = None,
    holding_size: int | None = None,
) -> tuple[BaseModbusDevice, _FakeBoard]:
    sizes = device_class._map_sizes
    inputs = [0] * (sizes[RegisterType.INPUT] if input_size is None else input_size)
    holdings = [0] * (sizes[RegisterType.HOLDING] if holding_size is None else holding_size)
    for name, spec in device_class.REGISTER_MAP.items():
        if spec.reg_type is RegisterType.HOLDING and spec.address < len(holdings):
            holdings[spec.address] = spec.default if spec.default is not None else spec.min
    board = _FakeBoard(inputs, holdings)
    return device_class(cast(Any, board), 1), board


def _rows(controller: ExternalDeviceController, table: str) -> set[str]:
    prefix = f"{table}:"
    return {key[len(prefix):] for key in controller._row_by_name if key.startswith(prefix)}


def test_reset_command_and_identity_words_are_not_listed(qtbot: QtBot) -> None:
    device, board = _board_device(SiPMDevice)
    controller = ExternalDeviceController(device, auto_start=False)
    qtbot.addWidget(controller)

    settings, telemetry = _rows(controller, "settings"), _rows(controller, "telemetry")
    assert "reset_device_command" in device.REGISTER_MAP  # nlab-modbus 0.3 maps it
    assert "reset_device_command" not in settings
    identity = {"hardware_version", "firmware_version", "modbus_reg_version",
                "serial_number_hi", "serial_number_lo"}
    assert identity <= set(device.REGISTER_MAP)
    assert not identity & telemetry
    # 0.3 polls the whole input table, so this flag is now refreshed and shown.
    assert "vout_overload" in telemetry

    # Saving and restoring a configuration never touches the reset command.
    saved = controller.hardware_configuration_settings()
    assert "reset_device_command" not in saved
    reset_address = device.REGISTER_MAP["reset_device_command"].address
    controller.apply_hardware_configuration_settings({**saved, "reset_device_command": 1})
    assert reset_address not in {address for address, _value in board.writes}


def test_registers_beyond_older_firmware_tables_are_not_listed(qtbot: QtBot) -> None:
    full, _ = _board_device(GeigerDevice)
    newest = ExternalDeviceController(full, auto_start=False)
    qtbot.addWidget(newest)
    width = GeigerDevice.REGISTER_MAP["pwm_pulse_width_ns"]
    assert "pwm_pulse_width_ns" in _rows(newest, "telemetry")
    last_setting = max(
        _rows(newest, "settings"), key=lambda name: GeigerDevice.REGISTER_MAP[name].address,
    )

    older, _ = _board_device(
        GeigerDevice, input_size=width.address,
        holding_size=GeigerDevice.REGISTER_MAP[last_setting].address,
    )
    controller = ExternalDeviceController(older, auto_start=False)
    qtbot.addWidget(controller)

    assert "pwm_pulse_width_ns" not in _rows(controller, "telemetry")
    assert last_setting not in _rows(controller, "settings")
    assert _rows(controller, "telemetry") == _rows(newest, "telemetry") - {"pwm_pulse_width_ns"}


def test_unreachable_board_still_lists_the_register_map(qtbot: QtBot) -> None:
    device, board = _board_device(SiPMDevice)

    def no_reply(**_request: int) -> object:
        raise ConnectionError("no response")

    board.read_input_registers = no_reply  # type: ignore[method-assign]
    board.read_holding_registers = no_reply  # type: ignore[method-assign]
    controller = ExternalDeviceController(device, auto_start=False)
    qtbot.addWidget(controller)

    assert "board_temp" in _rows(controller, "telemetry")
    assert "reset_device_command" not in _rows(controller, "settings")


def test_missing_temperature_sensor_reads_as_not_available(qtbot: QtBot) -> None:
    device, board = _board_device(SiPMDevice)
    temperature = device.REGISTER_MAP["board_temp"]
    board.input_words[temperature.address] = -30000 & 0xFFFF  # firmware: no I2C sensor
    controller = ExternalDeviceController(device, auto_start=False)
    qtbot.addWidget(controller)

    snapshot = device.read_snapshot()
    assert math.isnan(snapshot["board_temp"])
    controller._display_readback(0.0, snapshot)

    item = controller.ui.telemetryTable.item(controller._row_by_name["telemetry:board_temp"], 1)
    assert item is not None and item.text() == "n/a"
