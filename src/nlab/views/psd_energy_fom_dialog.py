"""FOM versus energy for the PSD event readback workbench."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nlab.analysis.psd_energy_fom import (
    ENERGY_FOM_HEADERS,
    MAX_SLICES,
    PsdEnergyFomRequest,
    PsdEnergyFomResult,
    PsdEnergySlice,
    best_slice,
    energy_fom_rows,
    fit_energy_slices,
    write_energy_fom_csv,
)
from nlab.views import theme
from nlab.views.numeric_controls import keep_enter_in_fields
from nlab.views.tooltips import wrap_widget_tooltips
from nlab.views.window_flags import make_tool_window

log = logging.getLogger(__name__)

_FIRST_COLOR = "#277da1"
_SECOND_COLOR = "#f8961e"
_FOM_COLOR = "#2a9d8f"
# Compact on-screen headers; the CSV and the header tooltips use the full names.
_TABLE_HEADERS = (
    "Slice", "Energy low", "Energy high", "Events", "Centroid 1", "Centroid 2",
    "FWHM 1", "FWHM 2", "FOM", "Deviance/dof", "Status",
)


class PsdEnergyFomDialog(QDialog):
    """Fit the readback matrix in energy slices with the FOM panel's settings.

    ``source`` returns the current matrix, energy ROI and fit settings each
    time the slices are fitted, so the result always reflects what is shown.
    """

    # (low, high) of the best slice, for the readback's FOM energy ROI.
    energy_roi_requested = Signal(float, float)

    def __init__(
        self,
        source: Callable[[], PsdEnergyFomRequest],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        make_tool_window(self)
        self.setModal(False)
        self.setWindowTitle("PSD FOM vs Energy")
        self.resize(980, 700)
        self._source = source
        self.last_result: PsdEnergyFomResult | None = None
        self.default_csv_path: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.slices_spin = QSpinBox(self)
        self.slices_spin.setRange(2, MAX_SLICES)
        self.slices_spin.setValue(6)
        self.slices_spin.setToolTip("Number of energy slices the FOM energy ROI is divided into.")
        self.spacing_combo = QComboBox(self)
        self.spacing_combo.addItem("Equal counts", "counts")
        self.spacing_combo.addItem("Equal width", "width")
        self.spacing_combo.setToolTip(
            "Equal counts gives every slice about the same number of events, so a "
            "falling spectrum does not leave the high-energy slices nearly empty. "
            "Equal width gives slices of the same energy width."
        )
        self.minimum_spin = QSpinBox(self)
        self.minimum_spin.setRange(10, 100_000_000)
        self.minimum_spin.setSingleStep(100)
        self.minimum_spin.setValue(200)
        self.minimum_spin.setToolTip(
            "Slices with fewer events are listed but not fitted."
        )
        self.fit_button = QPushButton("Fit slices", self)
        self.fit_button.setToolTip(
            "Fit every slice, and the whole energy ROI, with the current PSD matrix, "
            "energy ROI and FOM fit settings of the readback."
        )
        for text, field in (
            ("Slices:", self.slices_spin),
            ("Spacing:", self.spacing_combo),
            ("Minimum events:", self.minimum_spin),
        ):
            label = QLabel(text, self)
            label.setBuddy(field)
            controls.addWidget(label)
            controls.addWidget(field)
        controls.addWidget(self.fit_button)
        controls.addStretch(1)
        root.addLayout(controls)

        self.summary = QLabel(self)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.summary)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.plots = pg.GraphicsLayoutWidget(splitter)
        self.plots.setToolTip(
            "Top: FOM of each slice; bars span the slice, hollow markers are fits "
            "flagged as unreliable, the dashed line is the whole-ROI FOM. Bottom: "
            "population centroids with bars of +/- half the FWHM."
        )
        self.fom_plot = self.plots.addPlot(row=0, col=0)
        self.fom_plot.setLabel("left", "FOM")
        self.fom_plot.showGrid(x=True, y=True, alpha=0.2)
        self.centroid_plot = self.plots.addPlot(row=1, col=0)
        self.centroid_plot.setLabel("left", "PSD ratio")
        self.centroid_plot.showGrid(x=True, y=True, alpha=0.2)
        self.centroid_plot.setXLink(self.fom_plot)
        for plot in (self.fom_plot, self.centroid_plot):
            for side in ("left", "bottom"):
                # FOM and ratio are dimensionless and raw energy has no SI scale:
                # pyqtgraph would otherwise label them "(x0.001)" or "kraw".
                plot.getAxis(side).enableAutoSIPrefix(False)
        self.table = QTableWidget(0, len(ENERGY_FOM_HEADERS), splitter)
        for column, (short, full) in enumerate(
            zip(_TABLE_HEADERS, ENERGY_FOM_HEADERS, strict=True)
        ):
            header = QTableWidgetItem(short)
            header.setToolTip(full)
            self.table.setHorizontalHeaderItem(column, header)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.plots)
        splitter.addWidget(self.table)
        splitter.setSizes([420, 220])
        root.addWidget(splitter, 1)

        buttons = QHBoxLayout()
        self.copy_button = QPushButton("Copy table", self)
        self.copy_button.setToolTip("Copy the slice table as tab-separated text.")
        self.save_button = QPushButton("Save CSV...", self)
        self.save_button.setToolTip("Save the slice table with its energy ROI and fit settings.")
        self.best_button = QPushButton("Use best slice as ROI", self)
        self.best_button.setToolTip(
            "Set the readback's FOM energy ROI to the slice with the highest FOM among "
            "fits that converged without warnings, and refit the FOM panel there. "
            "The slice is marked in the plot and shown in bold in the table."
        )
        close_button = QPushButton("Close", self)
        buttons.addWidget(self.copy_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.best_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        root.addLayout(buttons)

        self.fit_button.clicked.connect(self.refit)
        self.copy_button.clicked.connect(self.copy_table)
        self.save_button.clicked.connect(self._save_csv)
        self.best_button.clicked.connect(self._send_best_slice)
        close_button.clicked.connect(self.close)
        self._set_result(None, "Press Fit slices to fit the energy slices.")
        keep_enter_in_fields(self)
        wrap_widget_tooltips(self)

    def refit(self) -> bool:
        """Fit the slices of the readback's current matrix; False if unavailable."""
        try:
            request = replace(
                self._source(),
                slices=self.slices_spin.value(),
                spacing="width" if self.spacing_combo.currentData() == "width" else "counts",
                minimum_events=self.minimum_spin.value(),
            )
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                result = fit_energy_slices(request)
            finally:
                QApplication.restoreOverrideCursor()
        except ValueError as error:
            self._set_result(None, f"FOM versus energy is unavailable: {error}")
            return False
        self._set_result(result)
        return True

    def invalidate(self, reason: str) -> None:
        """Forget a result that no longer matches the readback's matrix."""
        self._set_result(None, reason)

    def copy_table(self) -> str | None:
        if self.last_result is None:
            return None
        text = "\n".join("\t".join(row) for row in energy_fom_rows(self.last_result))
        QApplication.clipboard().setText(text)
        return text

    def best_slice(self) -> PsdEnergySlice | None:
        return None if self.last_result is None else best_slice(self.last_result)

    def _send_best_slice(self) -> None:
        best = self.best_slice()
        result = self.last_result
        if best is None or best.fit is None or result is None:
            return
        self.energy_roi_requested.emit(best.low, best.high)
        number = result.slices.index(best) + 1
        self.summary.setText(
            f"{self._describe(result)}\nSent slice {number} ({best.low:.6g}-{best.high:.6g}, "
            f"FOM {best.fit.fom:.4g}) to the readback's FOM energy ROI."
        )

    def _save_csv(self) -> None:
        result = self.last_result
        if result is None:
            return
        default = str(self.default_csv_path) if self.default_csv_path is not None else ""
        path_text, _ = QFileDialog.getSaveFileName(
            self, "Save FOM vs Energy", default, "CSV files (*.csv);;All files (*)",
        )
        if not path_text:
            return
        path = Path(path_text)
        if not path.suffix:
            path = path.with_suffix(".csv")
        try:
            write_energy_fom_csv(path, result)
        except OSError as error:
            log.exception("Could not save FOM versus energy to %s", path)
            QMessageBox.critical(self, "Save Failed", str(error))
            return
        self.summary.setText(f"{self._describe(result)}\nSaved to {path}.")

    def _set_result(self, result: PsdEnergyFomResult | None, message: str = "") -> None:
        self.last_result = result
        self.copy_button.setEnabled(result is not None)
        self.save_button.setEnabled(result is not None)
        self.best_button.setEnabled(result is not None and best_slice(result) is not None)
        self.fom_plot.clear()
        self.centroid_plot.clear()
        self.table.setRowCount(0)
        if result is None:
            self.summary.setText(message)
            return
        self.summary.setText(self._describe(result))
        self._fill_table(result)
        self._draw(result)

    @staticmethod
    def _describe(result: PsdEnergyFomResult) -> str:
        request = result.request
        settings = request.settings
        low, high = result.energy_roi
        if settings.mode == "separate" and settings.peak_rois is not None:
            windows = " and ".join(f"{a:.4g}-{b:.4g}" for a, b in settings.peak_rois)
            model = f"separate Gaussians in {windows}"
        else:
            model = f"double Gaussian in {settings.roi[0]:.4g}-{settings.roi[1]:.4g}"
        offset = "with" if settings.use_background else "without"
        trusted = sum(item.trusted for item in result.slices)
        spacing = "equal counts" if request.spacing == "counts" else "equal width"
        return (
            f"{request.energy_label} {low:.6g}-{high:.6g} in {len(result.slices)} slices "
            f"({spacing}); {model}, {offset} y0 offset. {trusted} of {len(result.slices)} "
            "slice fits converged without warnings."
        )

    def _fill_table(self, result: PsdEnergyFomResult) -> None:
        rows = energy_fom_rows(result)[1:]
        items = [*result.slices, result.whole]
        warning = theme.colors().status_warning
        best = best_slice(result)
        self.table.setRowCount(len(rows))
        for row_index, (row, item) in enumerate(zip(rows, items, strict=True)):
            for column, text in enumerate(row):
                cell = QTableWidgetItem(text)
                if column < len(row) - 1:
                    cell.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                if not item.trusted:
                    cell.setForeground(pg.mkColor(warning))
                if item is best:
                    font = cell.font()
                    font.setBold(True)
                    cell.setFont(font)
                    cell.setToolTip("Best slice: the highest FOM among trusted fits.")
                self.table.setItem(row_index, column, cell)

    def _draw(self, result: PsdEnergyFomResult) -> None:
        label = result.request.energy_label
        self.fom_plot.setLabel("bottom", f"{label} (raw)")
        self.centroid_plot.setLabel("bottom", f"{label} (raw)")
        fitted = [item for item in result.slices if item.fit is not None]
        if fitted:
            self._draw_fom(fitted)
            self._draw_centroids(fitted)
        best = best_slice(result)
        if best is not None and best.fit is not None:
            self.fom_plot.addItem(pg.ScatterPlotItem(
                x=[best.center], y=[best.fit.fom], size=18, symbol="o",
                pen=pg.mkPen(_FOM_COLOR, width=2), brush=pg.mkBrush(None),
            ))
        whole = result.whole.fit
        if whole is not None:
            self.fom_plot.addItem(pg.InfiniteLine(
                pos=whole.fom, angle=0,
                pen=pg.mkPen(_FOM_COLOR, width=1.5, style=Qt.PenStyle.DashLine),
                label=f"Whole ROI {whole.fom:.3g}",
                labelOpts={"position": 0.9, "color": _FOM_COLOR},
            ))
        low, high = result.energy_roi
        self.fom_plot.setXRange(low, high, padding=0.02)

    def _draw_fom(self, fitted: list[PsdEnergySlice]) -> None:
        centers = np.array([item.center for item in fitted])
        half = np.array([(item.high - item.low) / 2 for item in fitted])
        fom = np.array([item.fit.fom for item in fitted if item.fit is not None])
        self.fom_plot.addItem(pg.ErrorBarItem(
            x=centers, y=fom, left=half, right=half, beam=0.0,
            pen=pg.mkPen(_FOM_COLOR, width=1.2),
        ))
        trusted = np.array([item.trusted for item in fitted])
        self.fom_plot.addItem(pg.ScatterPlotItem(
            x=centers[trusted], y=fom[trusted], size=8,
            pen=pg.mkPen(_FOM_COLOR), brush=pg.mkBrush(_FOM_COLOR),
        ))
        self.fom_plot.addItem(pg.ScatterPlotItem(
            x=centers[~trusted], y=fom[~trusted], size=8,
            pen=pg.mkPen(theme.colors().status_warning, width=1.5), brush=pg.mkBrush(None),
        ))

    def _draw_centroids(self, fitted: list[PsdEnergySlice]) -> None:
        centers = np.array([item.center for item in fitted])
        for index, color in ((0, _FIRST_COLOR), (1, _SECOND_COLOR)):
            centroids = np.array([item.fit.centroids[index] for item in fitted if item.fit])
            half_widths = np.array([item.fit.fwhms[index] / 2 for item in fitted if item.fit])
            self.centroid_plot.addItem(pg.ErrorBarItem(
                x=centers, y=centroids, height=2 * half_widths, beam=0.0,
                pen=pg.mkPen(color, width=1.2),
            ))
            self.centroid_plot.plot(
                centers, centroids, pen=pg.mkPen(color, width=1.5), symbol="o",
                symbolSize=6, symbolBrush=color, symbolPen=color, name=f"Population {index + 1}",
            )
