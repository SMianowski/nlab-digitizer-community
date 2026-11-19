from __future__ import annotations

import errno
from typing import cast
from unittest.mock import Mock

from nlab.hardware.digitizer.hv import HVSupply
from nlab.workers.psu_worker import PSUWorker


def _hv(**reads: object) -> Mock:
    hv = Mock()
    hv.get_hv_adc_voltage.return_value = 500.0
    hv.get_hv_compens_output.return_value = 500.0
    hv.get_temp_analog.return_value = 20.0
    hv.get_temp_digital.return_value = 20.0
    hv.get_temp_digital_status.return_value = 1
    hv.get_ads_temp.return_value = 25.0
    for name, value in reads.items():
        setattr(hv, name, value)
    return hv


def test_connection_reset_reports_once_and_stops_polling() -> None:
    # libiio raises WSAECONNRESET once iiod drops the TCP connection; the
    # context never recovers, so retrying every tick only floods the log.
    hv = _hv(get_hv_adc_voltage=Mock(side_effect=ConnectionResetError(10054, "Unknown error")))
    worker = PSUWorker(cast(HVSupply, hv), read_sipm=False)
    errors: list[str] = []
    finished = Mock()
    readback = Mock()
    worker.error.connect(errors.append)
    worker.finished.connect(finished)
    worker.readback.connect(readback)

    worker._tick()
    worker._tick()
    worker._stop()

    assert errors == ["[Errno 10054] Unknown error"]
    finished.assert_called_once_with()
    readback.assert_not_called()
    hv.get_hv_adc_voltage.assert_called_once_with()


def test_transient_readback_error_is_reported_and_polling_continues() -> None:
    hv = _hv(get_ads_temp=Mock(side_effect=[OSError(errno.EIO, "I/O error"), 25.0]))
    worker = PSUWorker(cast(HVSupply, hv), read_sipm=False)
    errors: list[str] = []
    finished = Mock()
    readback = Mock()
    worker.error.connect(errors.append)
    worker.finished.connect(finished)
    worker.readback.connect(readback)

    worker._tick()
    worker._tick()

    assert errors == ["[Errno 5] I/O error"]
    finished.assert_not_called()
    readback.assert_called_once()
