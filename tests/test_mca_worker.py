from unittest.mock import Mock

import numpy as np

from nlab.workers.mca_worker import MCAWorker


def test_connection_reset_reports_once_and_stops_readback() -> None:
    mca = Mock()
    mca.acquire_spectrum.side_effect = ConnectionResetError(10054, "connection reset")
    worker = MCAWorker(mca)
    errors: list[str] = []
    finished = Mock()
    worker.error.connect(errors.append)
    worker.finished.connect(finished)

    worker._tick()
    worker._tick()

    assert errors == ["[Errno 10054] connection reset"]
    finished.assert_called_once_with()
    mca.acquire_spectrum.assert_called_once_with()


def test_connection_reset_during_status_read_stops_readback() -> None:
    mca = Mock()
    mca.acquire_spectrum.return_value = np.zeros(4, dtype=np.uint32)
    mca.acquire_waveforms.return_value = (
        np.zeros(4, dtype=np.int16),
        np.zeros(4, dtype=np.int16),
    )
    statistics = mca.statistics
    statistics.get_pulse_deadtime_ms.return_value = 0.0
    for name in (
        "get_count_rate",
        "get_events_lost",
        "get_elapsed_time",
        "get_pulse_overrange",
        "get_pulse_pileup",
        "get_energy_overrange",
        "get_energy_estimation_error",
        "get_throughput_error",
    ):
        getattr(statistics, name).return_value = 0
    mca.get_measurement_in_progress.side_effect = ConnectionResetError(
        10054, "connection reset"
    )
    worker = MCAWorker(mca)
    errors: list[str] = []
    readbacks: list[object] = []
    finished = Mock()
    worker.error.connect(errors.append)
    worker.readback.connect(readbacks.append)
    worker.finished.connect(finished)

    worker._tick()

    assert len(errors) == 1
    assert "connection reset" in errors[0]
    assert readbacks == []
    finished.assert_called_once_with()


def test_debug_waveforms_are_read_only_while_wanted() -> None:
    mca = Mock()
    first = (np.arange(4, dtype=np.int16), np.arange(4, dtype=np.int16) * 2)
    mca.acquire_waveforms.return_value = first
    worker = MCAWorker(mca)

    assert worker._acquire_waveforms() == first
    worker.read_debug_waveforms = False
    mca.acquire_waveforms.return_value = (np.zeros(4), np.zeros(4))

    # Hidden plot and no session: no 8 KiB debug read, last banks retained.
    assert worker._acquire_waveforms() == first
    mca.acquire_waveforms.assert_called_once_with()
