from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import cast

from nlab_modbus.core.base_modbus_device import BaseModbusDevice
from nlab_modbus.core.enums import DeviceType
from nlab_modbus.discovery.scan import scan_remote_modbus_devices
from nlab_modbus.services.manager import DeviceManager

log = logging.getLogger(__name__)

# ser2net TCP ports the digitizer bridges its RS-485 Modbus bus to (one bus,
# multiple daisy-chained devices distinguished by Modbus device_id).
REMOTE_PORTS: tuple[int, ...] = (5001, 5002)


@dataclass(frozen=True)
class ExternalDeviceScan:
    """A fresh manager and the devices it found during one complete scan.

    The manager owns every returned device connection.  Until the GUI adopts
    the result, the caller is responsible for calling ``manager.close()`` if
    the result is discarded.
    """

    manager: ExternalDevices
    devices: tuple[BaseModbusDevice, ...]


class ExternalDevices:
    """Discovers and owns the digitizer's onboard Modbus instruments.

    The SiPM bias board, Geiger-Mueller probe, and PMT HV supply share the
    digitizer's RS-485 bus, bridged to TCP via ser2net — same host as the
    IIO digitizer connection, different port(s). Wraps nlab_modbus's
    DeviceManager so the rest of the app only deals with discovered device
    instances and never imports nlab_modbus or touches connection details
    directly.
    """

    def __init__(self) -> None:
        self._manager = DeviceManager()

    @classmethod
    def scan_new(
        cls,
        host: str,
        ports: tuple[int, ...] = REMOTE_PORTS,
    ) -> ExternalDeviceScan:
        """Scan into a new owner so an active device set remains untouched.

        Runtime refresh uses this operation from a background thread after the
        GUI has stopped old polling and closed its ser2net client. Opening a
        second connection while polling can steal a single-client bridge.
        """
        owner = cls()
        try:
            devices = tuple(owner.discover(host, ports))
        except BaseException:
            owner.close()
            raise
        return ExternalDeviceScan(manager=owner, devices=devices)

    def discover(self, host: str, ports: tuple[int, ...] = REMOTE_PORTS) -> list[BaseModbusDevice]:
        """Scan the digitizer host for Modbus devices and connect to all found.

        Each ser2net port is a separate bridge, so all ports are probed
        concurrently (one client per port): an absent device costs the full
        probe timeout for each of its 16 IDs, and sequential ports doubled
        that (~1.6 s on a board with no modules). Responding IDs are then
        registered with the manager in port order, keeping device order
        deterministic. A port with nothing attached yields no devices.
        Returns every device discovered so far (cumulative across calls).
        """
        responding = _probe_ports(host, ports)
        for port in ports:
            device_ids = responding.get(port)
            if not device_ids:
                continue
            try:
                self._manager.scan_remote(host, port, candidate_ids=device_ids)
            except Exception:
                log.exception("Modbus scan failed on %s:%d", host, port)
        log.info(
            "Discovered %d external Modbus device(s) on %s",
            len(self._manager.all_devices),
            host,
        )
        return cast(list[BaseModbusDevice], self._manager.all_devices)

    def by_type(self, device_type: DeviceType) -> list[BaseModbusDevice]:
        return cast(list[BaseModbusDevice], self._manager.by_type(device_type))

    def close(self) -> None:
        self._manager.close_all()


def _probe_ports(host: str, ports: tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    """Return the responding Modbus IDs per port, probing ports in parallel."""

    def probe(port: int) -> tuple[int, ...]:
        try:
            found = scan_remote_modbus_devices(host, port)
        except Exception:
            log.exception("Modbus scan failed on %s:%d", host, port)
            return ()
        return tuple(int(entry["device_id"]) for entry in found)

    if not ports:
        return {}
    with ThreadPoolExecutor(max_workers=len(ports), thread_name_prefix="modbus-probe") as pool:
        return dict(zip(ports, pool.map(probe, ports), strict=True))
