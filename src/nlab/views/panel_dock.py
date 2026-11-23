"""Dock panels with standard native floating-window title-bar behavior."""

from __future__ import annotations

from typing import override

from PySide6.QtCore import QEvent
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QAbstractButton, QApplication, QDockWidget, QWidget

UNDOCK_TOOLTIP = "Undock current tab"
# The title-bar slot Qt lays out for the float button: a 16 px small icon plus
# its 2 px button margin (QCommonStyle::subElementRect(SE_DockWidgetFloatButton)).
FLOAT_BUTTON_SIZE = 18


class PanelDockWidget(QDockWidget):
    def __init__(self, title: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        button = self.findChild(QAbstractButton, "qt_dockwidget_floatbutton")
        if button is not None:
            button.setToolTip(UNDOCK_TOOLTIP)
            # Under a style sheet Qt sizes this button from a 10 px icon and
            # fixes its height to that, leaving an 18x10 sliver in the 18 px
            # slot. Fill the slot so the themed icon and hover area fit.
            button.setMinimumSize(FLOAT_BUTTON_SIZE, FLOAT_BUTTON_SIZE)

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        owner = self.parentWidget()
        if not self.isFloating() or owner is None:
            super().closeEvent(event)
            return
        # Application shutdown already stops the panel's workers. Let its
        # floating windows close without changing the saved dock layout.
        if QApplication.closingDown() or getattr(owner.window(), "_closing", False):
            event.accept()
            return
        # The floating caption's X means "dock back". Reject the actual close
        # so QWidget cannot hide or delete the panel after it has been docked.
        event.ignore()
        if self.isMaximized() or self.isMinimized():
            normal_geometry = self.normalGeometry()
            self.showNormal()
            # Restore the geometry before Qt records it for the next undock;
            # a window-manager resize may otherwise arrive after redocking.
            if normal_geometry.isValid():
                self.setGeometry(normal_geometry)
        self.setFloating(False)
        self.show()
        self.raise_()

    def event(self, event: QEvent) -> bool:
        if self.isFloating() and event.type() == QEvent.Type.NonClientAreaMouseButtonDblClick:
            # QDockWidget otherwise redocks on the same double-click that the
            # native window manager handles as maximize/restore. Leave the
            # native action alone and prevent the duplicate docking action.
            return True
        return super().event(event)
