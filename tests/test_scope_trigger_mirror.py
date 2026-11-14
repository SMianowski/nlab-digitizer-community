from __future__ import annotations

import pytest

from nlab.hardware.digitizer.scope import Scope, TriggerMode


class _EdgeBackend:
    """Shared fake hardware register behind independent client transports."""

    def __init__(self, register: list[int] | None = None) -> None:
        self.register = register if register is not None else [int(TriggerMode.ANY_BELOW)]
        self.reads = 0
        self.fail_next_write = False

    def get_edge(self) -> int:
        self.reads += 1
        return self.register[0]

    def set_edge(self, value: int) -> None:
        if self.fail_next_write:
            self.fail_next_write = False
            raise OSError(16, "busy")
        self.register[0] = value

    def get_scope_trigger_modes(self) -> tuple[int, ...]:
        return tuple(int(mode) for mode in TriggerMode)

    def create_isolated_scope_backend(self) -> _EdgeBackend:
        return _EdgeBackend(self.register)


def test_cached_mode_reads_hardware_once_then_serves_the_mirror() -> None:
    backend = _EdgeBackend()
    scope = Scope(backend)  # type: ignore[arg-type]

    assert scope.cached_trigger_mode() is TriggerMode.ANY_BELOW
    assert scope.cached_trigger_mode() is TriggerMode.ANY_BELOW
    assert backend.reads == 1


def test_isolated_client_write_is_visible_without_a_read() -> None:
    backend = _EdgeBackend()
    scope = Scope(backend)  # type: ignore[arg-type]
    assert scope.cached_trigger_mode() is TriggerMode.ANY_BELOW
    auto_setup = scope.create_isolated_client()

    auto_setup.set_trigger_mode(TriggerMode.PERIODIC_SUM)

    assert scope.cached_trigger_mode() is TriggerMode.PERIODIC_SUM
    assert backend.reads == 1


def test_explicit_read_refreshes_the_mirror() -> None:
    backend = _EdgeBackend()
    scope = Scope(backend)  # type: ignore[arg-type]
    scope.set_trigger_mode(TriggerMode.TIMED)
    backend.register[0] = int(TriggerMode.RISING_EDGE)  # changed outside nlab

    assert scope.get_trigger_mode() is TriggerMode.RISING_EDGE
    assert scope.cached_trigger_mode() is TriggerMode.RISING_EDGE


def test_failed_write_forces_the_next_cached_lookup_to_read() -> None:
    backend = _EdgeBackend()
    scope = Scope(backend)  # type: ignore[arg-type]
    scope.set_trigger_mode(TriggerMode.TIMED)
    backend.fail_next_write = True

    with pytest.raises(OSError):
        scope.set_trigger_mode(TriggerMode.PERIODIC_SUM)

    assert scope.cached_trigger_mode() is TriggerMode.TIMED
    assert backend.reads == 1
