"""MCA list-mode reader process: parent session, child entry and wiring.

Spawn tests use fake children (tests/_listmode_fakes.py), so no hardware.
"""

from __future__ import annotations

import errno
import logging
import multiprocessing
import subprocess
import sys
import threading
import time
from typing import Any, cast

import numpy as np
import pytest

from nlab.hardware.digitizer.dma import IIOMcaDmaStreamer, McaEventBuffer
from nlab.hardware.digitizer.iio_listmode import IIO_LM_EVENT_DTYPE
from nlab.hardware.digitizer.listmode_process import (
    ListModeProcessSession,
    ListModeRequest,
    _reader_main,
)
from nlab.utils.transport_errors import is_fatal_transport_error
from tests import _listmode_fakes as fakes


def _request(channel: int = 0, *, want_events: bool = True) -> ListModeRequest:
    return ListModeRequest(
        uri="ip:fake", channel=channel, profile=None, filepath=None,
        output_mode="binary", configuration_yaml="", client_record_schema=None,
        want_events=want_events, log_level=logging.INFO,
    )


def _run(session: ListModeProcessSession, stop: threading.Event | None = None,
         **callbacks: Any) -> int:
    return session.run(
        stop or threading.Event(),
        on_ready=callbacks.get("on_ready"),
        on_progress=callbacks.get("on_progress"),
        on_events=callbacks.get("on_events"),
        on_dropped=callbacks.get("on_dropped"),
    )


def test_session_relays_ready_events_drops_progress_and_logs(caplog: Any) -> None:
    ready: list[None] = []
    progress: list[int] = []
    events: list[np.ndarray] = []
    dropped: list[int] = []
    session = ListModeProcessSession(_request(), target=fakes.streams_then_finishes)

    with caplog.at_level(logging.WARNING, logger="nlab.fake"):
        records = _run(
            session,
            on_ready=lambda: ready.append(None),
            on_progress=progress.append,
            on_events=events.append,
            on_dropped=dropped.append,
        )

    assert records == 3072
    assert ready == [None] and progress == [2048] and dropped == [5]
    assert len(events) == 1 and events[0].dtype == IIO_LM_EVENT_DTYPE
    np.testing.assert_array_equal(events[0], fakes.frames(2))
    assert session.outcome.recovery_required is False
    assert session.outcome.diagnostics == {
        "streamed_frames": 3, "completed_frames": 3, "continuity_valid": True,
    }
    assert any("[list-mode reader ch0] hello" in r.getMessage() for r in caplog.records)


def test_session_forwards_stop_event_to_the_child() -> None:
    stop = threading.Event()
    session = ListModeProcessSession(_request(), target=fakes.waits_for_stop)

    assert _run(session, stop, on_ready=stop.set) == 1024


@pytest.mark.parametrize("channel", [0, 1])
def test_session_rebuilds_child_errors_with_errno_notes_and_fatality(channel: int) -> None:
    session = ListModeProcessSession(_request(channel), target=fakes.reports_failure)

    with pytest.raises(OSError) as caught:
        _run(session)

    if channel == 0:
        assert caught.value.errno == errno.EBADF
        assert str(caught.value) == "[Errno 9] Drain: Bad file descriptor"
        assert not is_fatal_transport_error(caught.value)
    else:
        assert isinstance(caught.value, ConnectionError)
        assert is_fatal_transport_error(caught.value)
    assert caught.value.__notes__ == ["Capture cleanup: example"]
    assert session.outcome.recovery_required is True


def test_crashed_child_is_reported_and_leaves_recovery_unverified() -> None:
    session = ListModeProcessSession(_request(), target=fakes.crashes)

    with pytest.raises(RuntimeError, match="ended unexpectedly .exit code 3"):
        _run(session)
    assert session.outcome.recovery_required is None


def test_streamer_process_mode_latches_backend_after_crash(monkeypatch: Any) -> None:
    import nlab.hardware.digitizer.listmode_process as module

    class Backend:
        latched: bool | None = False

        def mca_recovery_required(self) -> bool:
            return bool(self.latched)

        def acquisition_profile(self) -> dict[str, object]:
            return {}

        def adopt_mca_dma_session_state(self, recovery_required: bool | None) -> None:
            self.latched = recovery_required is None or recovery_required

    original = module.ListModeProcessSession

    def crashing_session(request: ListModeRequest) -> ListModeProcessSession:
        return original(request, target=fakes.crashes)

    monkeypatch.setattr(module, "ListModeProcessSession", crashing_session)
    backend = Backend()
    streamer = IIOMcaDmaStreamer(cast(Any, backend), 0, process_uri="ip:fake")

    with pytest.raises(RuntimeError, match="ended unexpectedly"):
        streamer.stream_events(threading.Event(), event_buffer=McaEventBuffer())
    assert backend.latched is True
    with pytest.raises(RuntimeError, match="fault is latched"):
        streamer.stream_events(threading.Event())


def test_event_buffer_drops_reported_upstream_reach_subscribers() -> None:
    primary, subscriber = McaEventBuffer(), McaEventBuffer()
    primary.subscribe(subscriber)

    primary.add_dropped(7)

    assert primary.drain()[1] == 7 and subscriber.drain()[1] == 7


@pytest.fixture
def isolated_root_logger() -> Any:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


def test_reader_main_stops_capture_when_the_gui_process_disappears(
    monkeypatch: Any, isolated_root_logger: Any,
) -> None:
    import nlab.hardware.digitizer.backends.iio_backend as backend_module
    import nlab.hardware.digitizer.dma as dma_module

    stops: list[None] = []

    class FakeBackend:
        def __init__(self, channel: int, uri: str, *, _listmode_only: bool,
                     profile: object) -> None:
            assert _listmode_only and channel == 0 and uri == "ip:fake"
            self.closed = False

        def close(self) -> None:
            self.closed = True

        def mca_recovery_required(self) -> bool:
            return False

    class FakeStreamer:
        last_capture_diagnostics = {"continuity_valid": True}

        def __init__(self, backend: FakeBackend, channel: int) -> None:
            self.backend = backend

        def request_stop(self) -> None:
            stops.append(None)

        def stream_events(self, stop_event: threading.Event, filepath: object,
                          event_buffer: McaEventBuffer, on_ready: Any,
                          on_progress: Any, **_: object) -> int:
            on_ready()
            count = 0
            while not stop_event.is_set():
                event_buffer.append(fakes.frames(1, seed=count))
                count += 1
                on_progress(count * 1024)
                time.sleep(0.002)
                assert count < 5000, "capture never stopped"
            return count * 1024

    monkeypatch.setattr(backend_module, "IIODigitizerBackend", FakeBackend)
    monkeypatch.setattr(dma_module, "IIOMcaDmaStreamer", FakeStreamer)
    control_recv, control_send = multiprocessing.Pipe(duplex=False)
    data_recv, data_send = multiprocessing.Pipe(duplex=False)
    child = threading.Thread(target=_reader_main, args=(control_recv, data_send, _request()))
    child.start()

    messages: list[tuple[Any, ...]] = []
    while not any(m[0] == "events" for m in messages):
        assert data_recv.poll(5.0)
        messages.append(data_recv.recv())
    control_send.close()  # the GUI process is gone
    while messages[-1][0] != "finished":
        assert data_recv.poll(5.0)
        messages.append(data_recv.recv())
    child.join(5.0)

    assert messages[0] == ("ready",)
    finished = messages[-1][1]
    assert finished["ok"] is True and finished["recovery_required"] is False
    assert stops == [None]
    received = sum(len(m[1]) for m in messages if m[0] == "events")
    assert received == finished["records"] * IIO_LM_EVENT_DTYPE.itemsize
    assert finished["diagnostics"] == {"continuity_valid": True}


def test_reader_process_modules_do_not_import_qt() -> None:
    probe = (
        "import sys\n"
        "import nlab.hardware.digitizer.listmode_process\n"
        "import nlab.hardware.digitizer.backends.iio_backend\n"
        "import nlab.hardware.digitizer.dma\n"
        "print(any(m.startswith('PySide6') for m in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            check=True, text=True, timeout=120)
    assert result.stdout.strip().splitlines()[-1] == "False"
