"""One naming rule for every automatically named recording.

``<measurement folder>/<prefix>_<YYYYmmdd_HHMMSS><suffix>``; if that name or
any of its sidecar files exists, ``_001``, ``_002``, ... is appended. An
existing measurement is never replaced. (Scope previously used a per-launch
counter without an existence check, so a restart within the same second could
overwrite an earlier recording.)
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSettings

from nlab import settings_keys


def measurement_folder() -> Path:
    """The configured recording folder (General Settings > DMA folder)."""
    value = QSettings().value(settings_keys.DMA_SAVE_FOLDER, settings_keys.DMA_SAVE_FOLDER_DEFAULT)
    return Path(str(value)).expanduser()


def available_path(path: Path, *, sidecar_suffixes: Sequence[str] = ()) -> Path:
    """*path*, or the first ``<stem>_NNN<suffix>`` whose file and sidecars are free."""
    candidate = path
    counter = 1
    while candidate.exists() or any(
        candidate.with_suffix(sidecar).exists() for sidecar in sidecar_suffixes
    ):
        candidate = path.with_name(f"{path.stem}_{counter:03d}{path.suffix}")
        counter += 1
    return candidate


def unique_recording_path(
    prefix: str,
    suffix: str,
    *,
    folder: Path | None = None,
    sidecar_suffixes: Sequence[str] = (),
    create_folder: bool = True,
) -> Path:
    """A new, collision-free, timestamped recording path."""
    target = measurement_folder() if folder is None else folder
    if create_folder:
        target.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return available_path(target / f"{prefix}_{stamp}{suffix}", sidecar_suffixes=sidecar_suffixes)
