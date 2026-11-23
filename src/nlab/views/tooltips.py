"""Readable plain-text tooltips without changing their meaning or F1 help."""

from __future__ import annotations

import textwrap

from PySide6 import QtGui
from PySide6.QtCore import QEvent, QObject, QRect
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QMenu, QTabBar, QToolTip, QWidget

from nlab.views.app_events import application_event_hub

TOOLTIP_LINE_LENGTH = 68

# These show a tooltip per tab or menu entry before falling back to their own,
# and an application filter runs first: leave their tooltip events to Qt.
_PER_ITEM_TOOLTIP_WIDGETS = (QTabBar, QMenu)
_ROUTE_OWNER_NAME = "nlabTooltipWrapping"


def wrap_tooltip(text: str) -> str:
    """Wrap prose and long paths, preserving explicit line/paragraph breaks.

    Keep plain text rather than HTML so F1 explanations do not expose markup.
    Character-based wrapping follows the user's font scaling without forcing a
    fixed-pixel tooltip size. Short hints and empty tooltips remain unchanged.
    """
    return "\n".join(
        textwrap.fill(line, width=TOOLTIP_LINE_LENGTH, break_on_hyphens=False,
                      replace_whitespace=False)
        for line in text.split("\n")
    )


def wrap_widget_tooltips(root: QWidget) -> None:
    """Apply at the end of UI construction; safe for already-wrapped children."""
    for widget in (root, *root.findChildren(QWidget)):
        if widget.toolTip():
            widget.setToolTip(wrap_tooltip(widget.toolTip()))


def _show_wrapped_tooltip(watched: QObject, event: QEvent) -> bool:
    """Show a long plain-text widget tooltip wrapped (application event hub).

    Qt word-wraps rich-text tooltips but keeps plain text on one line until it
    is wider than the whole screen. Only the displayed copy is wrapped: the
    widget keeps its tooltip, so F1 help and code read the original text.
    """
    if not isinstance(watched, QWidget) or not isinstance(event, QHelpEvent):
        return False
    if isinstance(watched, _PER_ITEM_TOOLTIP_WIDGETS):
        return False
    text = watched.toolTip()
    if not text or QtGui.Qt.mightBeRichText(text):
        return False
    wrapped = wrap_tooltip(text)
    if wrapped == text:
        return False
    QToolTip.showText(event.globalPos(), wrapped, watched, QRect(), watched.toolTipDuration())
    return True


def install_tooltip_wrapping() -> None:
    """Wrap long tooltips of every existing and future widget when shown."""
    application = QApplication.instance()
    event_hub = application_event_hub()
    if application is None or event_hub is None:
        return
    if application.findChild(QObject, _ROUTE_OWNER_NAME) is not None:
        return
    # The hub keeps one route per owner; the application itself already owns
    # the numeric-input route, so this route gets an owner of its own.
    owner = QObject(application)
    owner.setObjectName(_ROUTE_OWNER_NAME)
    event_hub.register(owner, _show_wrapped_tooltip, events=(QEvent.Type.ToolTip,))
