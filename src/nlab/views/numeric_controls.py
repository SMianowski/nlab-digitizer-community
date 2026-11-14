"""Clamp numeric edits before Qt commits them or a controller writes them."""

from __future__ import annotations

import math
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from nlab.views.app_events import application_event_hub

_STEP_PROPERTY = "nlab_snap_to_step"
_BOUND_PROPERTY = "nlab_numeric_input_bound"


def nearest_step_value(
    value: int | float, minimum: int | float, maximum: int | float, step: int | float
) -> float:
    """Nearest in-range grid point; decimal halfway values select the larger."""
    lower, upper = Decimal(str(minimum)), Decimal(str(maximum))
    quantum = Decimal(str(step))
    bounded = max(lower, min(upper, Decimal(str(value))))
    last = ((upper - lower) / quantum).to_integral_value(rounding=ROUND_FLOOR)
    index = ((bounded - lower) / quantum).to_integral_value(rounding=ROUND_HALF_UP)
    return float(lower + min(last, index) * quantum)


def normalize_numeric_control(
    spinbox: QSpinBox | QDoubleSpinBox,
    value: int | float | None = None,
    *,
    snap_to_step: bool | None = None,
) -> None:
    """Clamp to bounds, optionally snapping to a grid anchored at the minimum.

    Hardware controllers opt into step snapping explicitly. A normal spin box's
    arrow increment does not make intermediate analysis or calibration values
    invalid. Halfway values on discrete grids select the larger value.
    """
    if value is None:
        # Read the pending edit, not the last accepted value. In particular,
        # Qt's default fixup can restore the old value for out-of-range text.
        parsed, valid = spinbox.locale().toDouble(spinbox.cleanText())
        value = parsed if valid and math.isfinite(parsed) else spinbox.value()
    minimum, maximum = spinbox.minimum(), spinbox.maximum()
    value = max(minimum, min(maximum, value))
    if snap_to_step is None:
        snap_to_step = bool(spinbox.property(_STEP_PROPERTY))
    if snap_to_step and spinbox.singleStep() > 0:
        # Decimal arithmetic keeps decimal hardware steps (e.g. 0.008 us)
        # exact at endpoints and halfway values, including negative ranges.
        value = nearest_step_value(value, minimum, maximum, spinbox.singleStep())
    if isinstance(spinbox, QSpinBox):
        spinbox.setValue(int(value))
    else:
        spinbox.setValue(float(value))


def bind_numeric_control(spinbox: QSpinBox | QDoubleSpinBox, *, snap_to_step: bool = False) -> None:
    """Normalize before subsequently connected editingFinished write slots."""
    install_numeric_input_policy()
    spinbox.setProperty(_STEP_PROPERTY, snap_to_step)
    if not spinbox.property(_BOUND_PROPERTY):
        spinbox.setProperty(_BOUND_PROPERTY, True)
        spinbox.editingFinished.connect(lambda: normalize_numeric_control(spinbox))


def _numeric_input_filter(watched: QObject, event: QEvent) -> bool:
    """Normalize a spin box on Enter or focus loss (application event hub)."""
    if event.type() == QEvent.Type.KeyPress and (
        not isinstance(event, QKeyEvent)
        or event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
    ):
        return False
    spinbox = watched.parent() if isinstance(watched, QLineEdit) else watched
    if isinstance(spinbox, QSpinBox | QDoubleSpinBox) and not spinbox.isReadOnly():
        normalize_numeric_control(spinbox)
    # Let Qt deliver its normal commit signal exactly once.
    return False


def keep_enter_in_fields(dialog: QDialog) -> None:
    """Stop Enter in a field from clicking a button in a dialog without OK/Cancel.

    QDialog presses its default (or first auto-default) push button on
    Enter, so Enter in a spin box also "clicked" e.g. Open, Browse or Start.
    Call after the dialog's buttons are built; Enter then only commits the
    field, and buttons still respond to Space and clicks.
    """
    for button in dialog.findChildren(QPushButton):
        button.setAutoDefault(False)
        button.setDefault(False)


def install_numeric_input_policy() -> None:
    """Cover existing and future numeric widgets, including modeless dialogs."""
    application = QApplication.instance()
    event_hub = application_event_hub()
    if application is None or event_hub is None:
        return
    if getattr(application, "_nlab_numeric_input_policy", False):
        return
    event_hub.register(
        application, _numeric_input_filter,
        events=(QEvent.Type.KeyPress, QEvent.Type.FocusOut),
    )
    setattr(application, "_nlab_numeric_input_policy", True)
