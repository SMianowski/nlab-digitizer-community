"""One naming rule for every automatically named recording.

``<measurement folder>/<prefix>_<YYYYmmdd_HHMMSS><suffix>``; if that name or
any of its sidecar files exists, ``_001``, ``_002``, ... is appended. An
existing measurement is never replaced. (Scope previously used a per-launch
counter without an existence check, so a restart within the same second could
overwrite an earlier recording.)

A name the operator chooses is used as given unless General Settings >
Acquisition files asks for timestamps on chosen names too; it then becomes
``<stem>_<YYYYmmdd_HHMMSS><suffix>`` under the same never-replace rule.
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


def timestamp_chosen_names() -> bool:
    """Whether operator-chosen recording names also get the start timestamp."""
    return bool(
        QSettings().value(settings_keys.RECORDING_TIMESTAMP_CHOSEN_NAMES, False, type=bool)
    )


def recording_timestamp() -> str:
    """The local-time stamp every recording name uses: ``YYYYmmdd_HHMMSS``."""
    return time.strftime("%Y%m%d_%H%M%S")


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


def chosen_recording_path(path: Path, *, sidecar_suffixes: Sequence[str] = ()) -> Path:
    """The operator's chosen name for a recording that is starting now.

    Unchanged by default. With timestamps on chosen names enabled, the start
    time goes before the extension and the result never replaces a file.
    """
    if not timestamp_chosen_names():
        return path
    stamped = path.with_name(f"{path.stem}_{recording_timestamp()}{path.suffix}")
    return available_path(stamped, sidecar_suffixes=sidecar_suffixes)


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
    return available_path(
        target / f"{prefix}_{recording_timestamp()}{suffix}",
        sidecar_suffixes=sidecar_suffixes,
    )
