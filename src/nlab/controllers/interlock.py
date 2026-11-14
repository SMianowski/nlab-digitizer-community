"""The one replay hardware-interlock contract every workspace controller implements."""

from __future__ import annotations

from typing import Protocol


class ReplayInterlock(Protocol):
    def set_replay_hardware_interlock(
        self, enabled: bool, *, resume_background: bool = True,
    ) -> None:
        """Block (``enabled``) or release every hardware action of this view.

        ``resume_background``: when releasing, restart background polling that
        the interlock paused. Views without background polling ignore it.
        """
