"""PSD FOM in energy slices: slicing, band tracking, failures, CSV and report."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader
from scipy.stats import norm

from nlab.analysis.psd_energy_fom import (
    ENERGY_FOM_HEADERS,
    PsdEnergyFomRequest,
    energy_fom_rows,
    fit_energy_slices,
    slice_bounds,
    write_energy_fom_csv,
)
from nlab.analysis.psd_fom import PsdFitSettings, fit_psd_fom_lm
from nlab.analysis.psd_report import PsdReportRequest, export_psd_report

ENERGY_RANGE = (0.0, 6000.0)
RATIO_RANGE = (0.0, 1.0)
SETTINGS = PsdFitSettings("double", roi=(0.0, 0.32), centroid_seeds=(0.08, 0.16))


def drifting_matrix(seed: int = 3) -> np.ndarray:
    """A falling spectrum; population 2 moves 0.18 -> 0.13 and both narrow with energy."""
    energy_edges = np.linspace(*ENERGY_RANGE, 61)
    energy = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    ratio_edges = np.linspace(*RATIO_RANGE, 129)
    events = 6000.0 * np.exp(-energy / 2000.0)
    width = 0.025 * np.sqrt(1000.0 / energy)
    second = 0.18 - 0.05 * energy / 6000.0
    expected = np.empty((len(energy), len(ratio_edges) - 1))
    for row, (count, sigma, center) in enumerate(zip(events, width, second, strict=True)):
        first_part = np.diff(norm.cdf(ratio_edges, 0.08, sigma))
        second_part = np.diff(norm.cdf(ratio_edges, center, 1.3 * sigma))
        expected[row] = count * (0.7 * first_part + 0.3 * second_part)
    return np.random.default_rng(seed).poisson(expected).astype(np.uint64)


def request(**changes: object) -> PsdEnergyFomRequest:
    base = PsdEnergyFomRequest(
        matrix=drifting_matrix(),
        energy_range=ENERGY_RANGE,
        ratio_range=RATIO_RANGE,
        energy_roi=(500.0, 6000.0),
        settings=SETTINGS,
        slices=5,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def test_equal_count_slices_hold_similar_events_and_equal_width_similar_bins() -> None:
    totals = np.array([500, 400, 300, 220, 160, 120, 90, 60, 40, 30, 20, 10] * 2, dtype=float)

    by_counts = slice_bounds(totals, 4, "counts")
    events = [totals[low:high].sum() for low, high in by_counts]
    assert by_counts[0][0] == 0 and by_counts[-1][1] == len(totals)
    assert all(a[1] == b[0] for a, b in zip(by_counts, by_counts[1:], strict=False))
    assert max(events) / min(events) < 1.6

    by_width = slice_bounds(totals, 4, "width")
    assert [high - low for low, high in by_width] == [6, 6, 6, 6]
    # More slices than bins: one bin each, never an empty group.
    assert slice_bounds(totals[:3], 10, "counts") == [(0, 1), (1, 2), (2, 3)]


def test_slices_follow_the_moving_band_that_the_whole_roi_merges() -> None:
    result = fit_energy_slices(request())

    assert len(result.slices) == 5
    assert all(item.trusted for item in result.slices), [s.status for s in result.slices]
    # Contiguous, bin-aligned slices that exactly tile the whole ROI.
    assert result.slices[0].low == result.whole.low == 500.0
    assert result.slices[-1].high == result.whole.high == 6000.0
    assert all(a.high == b.low for a, b in zip(result.slices, result.slices[1:], strict=False))
    assert sum(item.events for item in result.slices) == result.whole.events
    second = [item.fit.centroids[1] for item in result.slices if item.fit is not None]
    assert all(later < earlier for earlier, later in zip(second, second[1:], strict=False))
    whole = result.whole.fit
    assert whole is not None
    assert whole.fom < max(item.fit.fom for item in result.slices if item.fit is not None)


def test_sparse_slices_are_listed_with_the_reason() -> None:
    result = fit_energy_slices(request(spacing="width", minimum_events=5_000))

    sparse = [item for item in result.slices if item.fit is None]
    assert sparse
    assert all("at least 5,000" in item.status for item in sparse)
    assert not any(item.trusted for item in sparse)


def test_separate_gaussians_use_their_peak_windows() -> None:
    settings = PsdFitSettings(
        "separate", roi=(0.0, 0.32), centroid_seeds=(0.08, 0.16),
        peak_rois=((0.0, 0.115), (0.125, 0.32)),
    )
    result = fit_energy_slices(request(settings=settings, slices=3))

    assert all(item.fit is not None and item.fit.fit_mode == "separate" for item in result.slices)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"slices": 0}, "between 1 and"),
        ({"energy_roi": (7000.0, 8000.0)}, "no histogram bins"),
        ({"minimum_events": 0}, "must be positive"),
    ],
)
def test_invalid_requests_are_rejected(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_energy_slices(request(**changes))


def test_table_rows_and_csv_record_the_inputs(tmp_path: Path) -> None:
    result = fit_energy_slices(request(slices=3))
    rows = energy_fom_rows(result)
    assert rows[0] == ENERGY_FOM_HEADERS
    assert [row[0] for row in rows[1:]] == ["1", "2", "3", "Whole ROI"]

    path = tmp_path / "fom.csv"
    write_energy_fom_csv(path, result)
    lines = path.read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if line.startswith("#")]
    assert any("Energy ROI (bin edges): 500 to 6000" in line for line in comments)
    assert any("Fit model: Double Gaussian" in line for line in comments)
    table = list(csv.reader(line for line in lines if not line.startswith("#")))
    assert table == [list(row) for row in rows]


def test_report_adds_the_energy_page_csv_and_metadata(tmp_path: Path) -> None:
    result = fit_energy_slices(request(slices=4))
    matrix = drifting_matrix()
    ratio = np.linspace(0.0, 1.0, 129)[:-1] + 0.5 / 128
    fit = fit_psd_fom_lm(
        ratio, matrix[5:].sum(axis=0), roi=(0.0, 0.32), centroid_seeds=(0.08, 0.16),
    )
    source = tmp_path / "capture.root"
    source.write_bytes(b"capture")
    report = PsdReportRequest(
        source_path=source, title="PSD Event Readback Report", matrix=matrix,
        energy_range=ENERGY_RANGE, ratio_range=RATIO_RANGE, energy_label="Long-gate energy",
        cut=0.12, energy_roi=(500.0, 6000.0), matrix_scale="log", energy_log_y=True,
        fit=fit, energy_fom=result,
    )

    bundle = export_psd_report(report, tmp_path)

    assert bundle.fom_vs_energy_csv is not None and bundle.fom_vs_energy_csv.is_file()
    metadata = json.loads(bundle.metadata_json.read_text(encoding="utf-8"))
    assert metadata["files"]["fom_vs_energy_csv"] == "fom_vs_energy.csv"
    assert len(metadata["energy_resolved_fom"]["slices"]) == 4
    assert metadata["energy_resolved_fom"]["settings"]["Slices"] == "4 (equal counts)"
    pages = [page.extract_text() or "" for page in PdfReader(bundle.pdf).pages]
    assert len(pages) == 4
    assert "PSD FOM versus energy" in pages[2]
    assert "Whole ROI" in pages[2]

    without = export_psd_report(replace(report, energy_fom=None), tmp_path)
    assert without.fom_vs_energy_csv is None
    assert len(PdfReader(without.pdf).pages) == 3


def test_best_slice_is_the_highest_fom_among_trusted_fits() -> None:
    from types import SimpleNamespace

    from nlab.analysis.psd_energy_fom import PsdEnergyFomResult, PsdEnergySlice, best_slice

    def item(low: float, fom: float, *, flagged: bool) -> PsdEnergySlice:
        fit = SimpleNamespace(fom=fom, poor_convergence=flagged)
        return PsdEnergySlice(low, low + 1.0, 1000, fit, "")  # type: ignore[arg-type]

    flagged_peak = item(2.0, 1.45, flagged=True)  # inflated FOM from a flagged fit
    best = item(1.0, 0.75, flagged=False)
    slices = (item(0.0, 0.55, flagged=False), best, flagged_peak)
    result = PsdEnergyFomResult(request(), slices, item(0.0, 0.6, flagged=False))
    assert best_slice(result) is best

    none_trusted = PsdEnergyFomResult(request(), (flagged_peak,), flagged_peak)
    assert best_slice(none_trusted) is None
