"""One collision-free naming rule for automatic recordings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from nlab import settings_keys
from nlab.utils import recording_paths
from nlab.utils.recording_paths import available_path, measurement_folder, unique_recording_path


@pytest.fixture(autouse=True)
def isolated_qsettings(tmp_path: Path) -> Iterator[None]:
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "ini"))
    yield
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recording_paths.time, "strftime", lambda _fmt: "20261114_120000")


def test_names_use_prefix_and_timestamp_in_the_configured_folder(
    tmp_path: Path, frozen_clock: None,
) -> None:
    QSettings().setValue(settings_keys.DMA_SAVE_FOLDER, str(tmp_path / "runs"))
    path = unique_recording_path("scope_ch0", ".bin")
    assert path == tmp_path / "runs" / "scope_ch0_20261114_120000.bin"
    assert path.parent.is_dir()
    assert measurement_folder() == tmp_path / "runs"


def test_existing_recordings_are_never_reused(tmp_path: Path, frozen_clock: None) -> None:
    first = unique_recording_path("scope_ch0", ".bin", folder=tmp_path)
    first.write_bytes(b"earlier capture")
    second = unique_recording_path("scope_ch0", ".bin", folder=tmp_path)
    second.write_bytes(b"next capture")
    third = unique_recording_path("scope_ch0", ".bin", folder=tmp_path)
    assert [p.name for p in (first, second, third)] == [
        "scope_ch0_20261114_120000.bin",
        "scope_ch0_20261114_120000_001.bin",
        "scope_ch0_20261114_120000_002.bin",
    ]
    assert first.read_bytes() == b"earlier capture"


def test_a_leftover_sidecar_also_blocks_a_name(tmp_path: Path) -> None:
    (tmp_path / "ch0.run.json").write_text("{}")
    assert available_path(tmp_path / "ch0.bin", sidecar_suffixes=(".run.json",)).name == (
        "ch0_001.bin"
    )


def test_scope_auto_names_never_overwrite_an_earlier_recording(
    qtbot: object, tmp_path: Path, frozen_clock: None,
) -> None:
    from nlab.controllers.scope_controller import ScopeController
    from tests._controller_harness import recording_digitizer

    QSettings().setValue(settings_keys.DMA_SAVE_FOLDER, str(tmp_path))
    device, _backend, _ids = recording_digitizer()
    names = []
    for _ in range(2):  # two app launches in the same second
        controller = ScopeController(device.scope, scope_dma=None, channel=0)
        path = controller._generate_filepath()
        path.write_bytes(b"capture")
        names.append(path.name)
        controller.deleteLater()
    assert names == ["scope_ch0_20261114_120000.bin", "scope_ch0_20261114_120000_001.bin"]
