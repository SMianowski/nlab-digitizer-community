"""Real controllers on a recording, transport-free backend.

Characterization tests build the production ScopeController, MCAController
and PSUController on top of the Offline-mode backend and observe them only
through public surfaces (widgets, signals, settings and backend writes), so
internal refactoring cannot break them by renaming private attributes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from nlab.hardware.digitizer.backends.base import DigitizerBackend, IDSBackend
from nlab.hardware.digitizer.digitizer import Digitizer
from nlab.hardware.digitizer.offline import _OfflineDigitizerBackend, _OfflineIDSBackend


class RecordingBackend(_OfflineDigitizerBackend):
    """Offline backend that records every hardware write in order."""

    def __init__(self, channel: int = 0) -> None:
        self.writes: list[tuple[str, object]] = []
        super().__init__(channel)
        self.writes.clear()  # construction defaults are not test actions

    def __getattr__(self, name: str) -> Callable[..., object]:
        attribute = super().__getattr__(name)
        if not name.startswith("set_"):
            return attribute

        def record(value: object) -> None:
            self.writes.append((name, value))
            attribute(value)

        return record

    def configure_mca_parameters(self, values: dict[str, int]) -> None:
        if hasattr(self, "writes"):
            self.writes.append(("configure_mca_parameters", dict(values)))
        super().configure_mca_parameters(values)

    def written(self, name: str) -> list[object]:
        """Values written through ``name`` (e.g. ``"set_enable"``), in order."""
        return [value for setter, value in self.writes if setter == name]


class RecordingIDSBackend(_OfflineIDSBackend):
    def __init__(self) -> None:
        self.writes: list[tuple[str, object]] = []
        super().__init__()

    def __getattr__(self, name: str) -> Callable[..., object]:
        attribute = super().__getattr__(name)
        if not name.startswith("set_"):
            return attribute

        def record(value: object) -> None:
            self.writes.append((name, value))
            attribute(value)

        return record


def recording_digitizer(
    channel: int = 0,
) -> tuple[Digitizer, RecordingBackend, RecordingIDSBackend]:
    backend = RecordingBackend(channel)
    ids = RecordingIDSBackend()
    device = Digitizer(
        cast(DigitizerBackend, backend), cast(IDSBackend, ids), mca_dma=None,
    )
    return device, backend, ids


def signal_log(signal: Any) -> list[tuple[Any, ...]]:
    """Collect a Qt signal's emissions as argument tuples."""
    received: list[tuple[Any, ...]] = []
    signal.connect(lambda *args: received.append(args))
    return received
