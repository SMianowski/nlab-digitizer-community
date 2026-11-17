"""Window chrome shared by every tool that opens in its own window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

# A normal top-level window with the native title bar's minimize, maximize and
# close buttons and no "?" context-help button. Floating channel docks and
# every tool window (calibration, analysis, readback, help) use the same set.
TOOL_WINDOW_FLAGS = (
    Qt.WindowType.Window
    | Qt.WindowType.CustomizeWindowHint
    | Qt.WindowType.WindowTitleHint
    | Qt.WindowType.WindowSystemMenuHint
    | Qt.WindowType.WindowMinMaxButtonsHint
    | Qt.WindowType.WindowCloseButtonHint
)


def make_tool_window(widget: QWidget) -> None:
    """Give a tool window the shared title-bar buttons; call before showing it."""
    widget.setWindowFlags(TOOL_WINDOW_FLAGS)
