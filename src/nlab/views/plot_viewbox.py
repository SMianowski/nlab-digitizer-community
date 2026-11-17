from __future__ import annotations

from typing import Any

from pyqtgraph import PlotCurveItem, PlotWidget, ViewBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QMenu, QWidget

from nlab.views import theme

# pyqtgraph's "right" path holds y[i] from x[i] to x[i+1], then
# transitions vertically at the next timestamp. Keep x/y arrays unexpanded so
# cursor/drag calculations still use the original samples (covered by tests).
WAVEFORM_STEP_MODE = "right"


# Fixed size of a frame-only axis: room for the neighbouring axis's end tick
# label (half its text height) without clipping. Being the same for every
# plot, it keeps linked projection plots' view-box edges identical.
FRAME_AXIS_SIZE = 10


def set_frame_only(axis: Any, frame_only: bool) -> None:
    """Hide (or restore) an axis's values and title, keeping its frame line."""
    axis.setStyle(showValues=not frame_only)
    size = FRAME_AXIS_SIZE if frame_only else None
    if axis.orientation in ("top", "bottom"):
        axis.setHeight(size)
    else:
        axis.setWidth(size)
    if frame_only:
        axis.showLabel(False)


def show_frame_axes(plot: Any, *sides: str) -> None:
    """Close the plot frame on ``sides`` (default top and right) without labels.

    These axes mirror the opposite axis, so repeating their tick values and
    title only adds noise. The axis stays shown for the frame line and tick
    marks, at ``FRAME_AXIS_SIZE`` on every plot.
    """
    for side in sides or ("top", "right"):
        plot.showAxis(side)
        set_frame_only(plot.getAxis(side), True)


def style_plot_widget(widget: PlotWidget) -> None:
    """Match the plot surround to Qt while retaining a light data canvas."""
    interface_background = widget.palette().color(QPalette.ColorRole.Window)
    widget.setBackground(interface_background)
    widget.getViewBox().setBackgroundColor(theme.colors().plot_background)


class ModifierZoomViewBox(ViewBox):  # type: ignore[misc]
    """ViewBox with modifier-key axis-locked scrolling.

    * Shift + wheel  → horizontal zoom only
    * Ctrl  + wheel  → vertical zoom only
    * Plain wheel    → default pyqtgraph behaviour (zoom both)
    """

    def wheelEvent(self, ev: Any, axis: int | None = None) -> None:  # noqa: N802
        mods = ev.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            axis = 0
        elif mods & Qt.KeyboardModifier.ControlModifier:
            axis = 1
        super().wheelEvent(ev, axis=axis)


class HistogramViewBox(ModifierZoomViewBox):
    """Histogram canvas with its own menu rather than generic plot transforms."""

    def set_histogram_menu(self, menu: QMenu) -> None:
        self._histogram_menu = menu

    def raiseContextMenu(self, ev: Any) -> None:  # noqa: N802
        self._histogram_menu.popup(ev.screenPos().toPoint())


class NLabPlotWidget(PlotWidget):  # type: ignore[misc]
    """PlotWidget that uses ModifierZoomViewBox by default.

    Promote QWidget to this class in Qt Designer with header
    ``nlab.views.plot_viewbox``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent, viewBox=ModifierZoomViewBox())


class DraggableScopeCurve(PlotCurveItem):  # type: ignore[misc]
    """Raw Scope trace with a narrow grab area; ordinary plot drags still zoom/pan."""

    drag_started = Signal()
    drag_moved = Signal(float, float)
    drag_finished = Signal(bool)  # True when cancelled.

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("stepMode", WAVEFORM_STEP_MODE)
        super().__init__(*args, **kwargs)
        self.setClickable(True, width=14)
        self._drag_enabled = True
        self._drag_origin = None

    def setDragEnabled(self, enabled: bool) -> None:  # noqa: N802
        if not enabled:
            self.cancelDrag()
        self._drag_enabled = enabled
        self.setClickable(enabled, width=14)
        self.unsetCursor()

    def cancelDrag(self) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self._drag_origin = None
            self.unsetCursor()
            self.drag_finished.emit(True)

    def hoverEvent(self, ev: Any) -> None:  # noqa: N802
        if self._drag_origin is not None:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif (
            self._drag_enabled
            and not ev.isExit()
            and self.mouseShape().contains(ev.pos())
            and ev.acceptDrags(Qt.MouseButton.LeftButton)
        ):
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.unsetCursor()

    def mouseDragEvent(self, ev: Any) -> None:  # noqa: N802
        if not self._drag_enabled or ev.button() != Qt.MouseButton.LeftButton:
            return
        if ev.isStart():
            if not self.mouseShape().contains(ev.buttonDownPos()):
                return
            self._drag_origin = ev.buttonDownPos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.drag_started.emit()
        if self._drag_origin is None:
            return
        ev.accept()
        delta = ev.pos() - self._drag_origin
        self.drag_moved.emit(float(delta.x()), float(delta.y()))
        if ev.isFinish():
            self._drag_origin = None
            self.unsetCursor()
            self.drag_finished.emit(False)

    def mouseClickEvent(self, ev: Any) -> None:  # noqa: N802
        if self._drag_origin is not None and ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            self.cancelDrag()
            return
        super().mouseClickEvent(ev)
