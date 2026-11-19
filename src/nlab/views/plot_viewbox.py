from __future__ import annotations

from typing import Any, Literal

from pyqtgraph import PlotCurveItem, PlotDataItem, PlotWidget, ViewBox
from PySide6.QtCore import QObject, QSettings, Qt, Signal, Slot
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QMenu, QWidget

from nlab import settings_keys
from nlab.views import theme

# pyqtgraph's "right" path holds y[i] from x[i] to x[i+1], then
# transitions vertically at the next timestamp. Keep x/y arrays unexpanded so
# cursor/drag calculations still use the original samples (covered by tests).
WAVEFORM_STEP_MODE = "right"

# How sampled waveforms (Scope trace, MCA debug traces, Waveform Analysis) are
# drawn: held steps between samples, or straight lines joining them. Both use
# the same unexpanded x/y arrays, so switching never changes the data.
WaveformDrawing = Literal["steps", "lines"]
WAVEFORM_DRAWINGS: tuple[WaveformDrawing, ...] = ("steps", "lines")
WAVEFORM_DRAWING_LABELS: dict[WaveformDrawing, str] = {
    "steps": "Steps (hold each sample)",
    "lines": "Lines (join samples directly)",
}
DEFAULT_WAVEFORM_DRAWING: WaveformDrawing = "steps"
_STEP_MODES: dict[WaveformDrawing, str | None] = {"steps": WAVEFORM_STEP_MODE, "lines": None}


def saved_waveform_drawing() -> WaveformDrawing:
    """The waveform drawing chosen in General Settings, or the step default."""
    value = str(
        QSettings().value(settings_keys.APPEARANCE_WAVEFORM_DRAWING, DEFAULT_WAVEFORM_DRAWING)
    )
    return value if value in WAVEFORM_DRAWINGS else DEFAULT_WAVEFORM_DRAWING


def waveform_step_mode(drawing: WaveformDrawing | None = None) -> str | None:
    """pyqtgraph ``stepMode`` for a waveform drawing (the saved one by default)."""
    return _STEP_MODES[drawing if drawing is not None else saved_waveform_drawing()]


class _WaveformDrawingNotifier(QObject):
    changed = Signal(str)


_notifier: _WaveformDrawingNotifier | None = None


def _waveform_drawing_notifier() -> _WaveformDrawingNotifier:
    global _notifier
    if _notifier is None:
        _notifier = _WaveformDrawingNotifier()
    return _notifier


def save_waveform_drawing(drawing: WaveformDrawing) -> None:
    """Persist a waveform drawing and redraw every followed waveform with it."""
    if drawing not in WAVEFORM_DRAWINGS:
        raise ValueError(f"unsupported waveform drawing {drawing!r}")
    changed = drawing != saved_waveform_drawing()
    QSettings().setValue(settings_keys.APPEARANCE_WAVEFORM_DRAWING, drawing)
    if changed:
        _waveform_drawing_notifier().changed.emit(drawing)


def apply_waveform_drawing(
    item: PlotDataItem | PlotCurveItem,
    drawing: WaveformDrawing | None = None,
) -> None:
    """Redraw one waveform curve as steps or lines, keeping its data."""
    mode = waveform_step_mode(drawing)
    if item.opts.get("stepMode") == mode:
        return
    x, y = item.xData, item.yData
    # pyqtgraph replaces the data when stepMode is set alone: resend it.
    if x is None or y is None or len(y) == 0:
        item.setData(stepMode=mode)
    else:
        item.setData(x, y, stepMode=mode)


class _WaveformDrawingFollower(QObject):
    """Child of a curve, so Qt drops the connection when the curve is deleted."""

    def __init__(self, item: PlotDataItem | PlotCurveItem) -> None:
        super().__init__(item)
        self._item = item

    @Slot(str)
    def apply(self, drawing: str) -> None:
        if drawing in WAVEFORM_DRAWINGS:
            apply_waveform_drawing(self._item, drawing)


def follow_waveform_drawing(item: PlotDataItem | PlotCurveItem) -> None:
    """Draw a waveform curve with the saved style now and after every change."""
    apply_waveform_drawing(item)
    follower = _WaveformDrawingFollower(item)
    _waveform_drawing_notifier().changed.connect(follower.apply)


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
        kwargs.setdefault("stepMode", waveform_step_mode())
        super().__init__(*args, **kwargs)
        follow_waveform_drawing(self)
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
