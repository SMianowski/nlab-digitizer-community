"""Bind hardware parameter specs to Qt input widgets, in one place.

Range, step and default come from the same spec table the hardware layer
validates against, so a widget can neither clamp a valid readback nor offer
a value the driver rejects. Callers convert units with ``RangeSpec.scaled``.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QSlider, QSpinBox

from nlab.hardware.digitizer.scope import RangeSpec
from nlab.views.numeric_controls import bind_numeric_control


def bind_spinbox(spinbox: QSpinBox, spec: RangeSpec, *, suffix: str | None = None) -> None:
    """Integer hardware field: range, step, default; edits snap to the step."""
    spinbox.setRange(int(spec.min_val), int(spec.max_val))
    spinbox.setSingleStep(int(spec.step) or 1)
    spinbox.setValue(int(spec.default))
    if suffix is not None:
        spinbox.setSuffix(suffix)
    bind_numeric_control(spinbox, snap_to_step=True)


def bind_double_spinbox(spinbox: QDoubleSpinBox, spec: RangeSpec, *, snap_to_step: bool) -> None:
    """Float field. Snap only where the hardware has a grid; physical or
    coefficient settings accept intermediate values."""
    spinbox.setRange(float(spec.min_val), float(spec.max_val))
    spinbox.setSingleStep(float(spec.step) or 1.0)
    spinbox.setValue(float(spec.default))
    bind_numeric_control(spinbox, snap_to_step=snap_to_step)


def bind_slider(slider: QSlider, spec: RangeSpec, *, page_step: bool = False) -> None:
    """Slider over the spec's range; ``page_step`` makes a page one step."""
    slider.setRange(int(spec.min_val), int(spec.max_val))
    slider.setSingleStep(int(spec.step) or 1)
    if page_step:
        slider.setPageStep(max(int(spec.step), 1))
    slider.setValue(int(spec.default))
