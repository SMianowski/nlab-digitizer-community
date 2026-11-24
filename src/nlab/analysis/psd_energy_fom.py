"""PSD figure of merit in energy slices of a PSD matrix.

The PSD bands move and narrow with energy. One ratio projection over a wide
energy ROI therefore merges different band positions: its peaks widen, the
Gaussian model stops describing them, and the FOM understates the separation
reached at each energy. Fitting slices of the ROI with the same fit settings
shows how separation develops with energy and where the fit can be trusted.

Narrow energy cuts keep each projection close to Gaussian; Grodzicka-Kobylka
et al. (2020) use +/-10 % energy cuts on zero-crossing-time spectra for this
reason. Sénoville et al. (2020) measure the FOM in narrow energy intervals and
show that its energy dependence is dominated by the square root of the total
light, and Graham et al. (2026) adapt the integration gates to light output.

References
----------
Grodzicka-Kobylka, M., Szczęśniak, T., Moszyński, M., et al. (2020). Fast
    neutron and gamma ray pulse shape discrimination in EJ-276 and EJ-276G
    plastic scintillators. J. Instrum. 15, P03030.
    doi:10.1088/1748-0221/15/03/P03030
Sénoville, M., Delaunay, F., Pârlog, M., Achouri, N. L., Orr, N. A.
    (2020). Neutron-gamma discrimination with organic scintillators:
    intrinsic pulse shape and light yield contributions. Nucl. Instrum.
    Methods Phys. Res. A 971, 164080. doi:10.1016/j.nima.2020.164080
Graham, C., Todd, E., Wilson, C., et al. (2026). Light output-dependent
    integration gating for optimized pulse shape discrimination. Nucl.
    Instrum. Methods Phys. Res. A 1088, 171521.
    doi:10.1016/j.nima.2026.171521
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from nlab.analysis.psd_fom import PsdFitSettings, PsdFomFit, fit_psd_projection

SliceSpacing = Literal["counts", "width"]
MAX_SLICES = 64

ENERGY_FOM_HEADERS = (
    "Slice",
    "Energy low",
    "Energy high",
    "Events",
    "Population 1 centroid",
    "Population 2 centroid",
    "Population 1 FWHM",
    "Population 2 FWHM",
    "FOM",
    "Reduced Poisson deviance",
    "Status",
)


@dataclass(frozen=True)
class PsdEnergyFomRequest:
    """A PSD matrix, the energy ROI to divide, and the fit to apply to each part."""

    matrix: np.ndarray
    energy_range: tuple[float, float]
    ratio_range: tuple[float, float]
    energy_roi: tuple[float, float]
    settings: PsdFitSettings
    energy_label: str = "Long-gate energy"
    slices: int = 6
    spacing: SliceSpacing = "counts"
    minimum_events: int = 200


@dataclass(frozen=True)
class PsdEnergySlice:
    """One energy slice (bin-aligned edges, half-open) and its fit, if any."""

    low: float
    high: float
    events: int
    fit: PsdFomFit | None
    status: str

    @property
    def center(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def trusted(self) -> bool:
        return self.fit is not None and not self.fit.poor_convergence


@dataclass(frozen=True)
class PsdEnergyFomResult:
    """Per-slice fits plus the same fit over the whole ROI, for comparison."""

    request: PsdEnergyFomRequest
    slices: tuple[PsdEnergySlice, ...]
    whole: PsdEnergySlice

    @property
    def energy_roi(self) -> tuple[float, float]:
        """The bin-aligned energy range actually divided."""
        return (self.whole.low, self.whole.high)


def energy_bin_range(edges: np.ndarray, region: tuple[float, float]) -> tuple[int, int]:
    """Half-open bin range of an energy region, as PsdAccumulator.ratio_projection uses."""
    bins = len(edges) - 1
    low, high = sorted(map(float, region))
    first = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, bins))
    last = int(np.clip(np.searchsorted(edges, high, side="left"), 0, bins))
    return first, last


def slice_bounds(events: np.ndarray, slices: int, spacing: SliceSpacing) -> list[tuple[int, int]]:
    """Divide consecutive energy bins into at most ``slices`` contiguous groups.

    ``counts`` places the boundaries so each group holds about the same number
    of events (a falling spectrum would otherwise leave the high-energy groups
    nearly empty); ``width`` gives groups of equal bin count. Boundaries stay on
    bin edges, so a group can never hold less than one bin.
    """
    totals = np.asarray(events, dtype=np.float64)
    bins = len(totals)
    groups = max(1, min(int(slices), bins))
    total = float(totals.sum())
    if spacing == "counts" and total > 0.0:
        cumulative = np.cumsum(totals)
        targets = total * np.arange(1, groups) / groups
        # The bin whose cumulative count first reaches each target: put the
        # boundary before or after it, whichever lands closer to the target.
        crossing = np.searchsorted(cumulative, targets, side="left")
        before = np.where(crossing > 0, cumulative[np.maximum(crossing - 1, 0)], 0.0)
        after = cumulative[np.minimum(crossing, bins - 1)]
        inner = np.where(targets - before < after - targets, crossing, crossing + 1)
    else:
        inner = np.rint(np.linspace(0, bins, groups + 1)[1:-1]).astype(int)
    edges = np.unique(np.clip(np.concatenate(([0], inner, [bins])), 0, bins))
    return [(int(low), int(high)) for low, high in zip(edges[:-1], edges[1:], strict=True)]


def fit_energy_slices(request: PsdEnergyFomRequest) -> PsdEnergyFomResult:
    """Fit every slice of the energy ROI, and the whole ROI, with one fit setting."""
    matrix = np.asarray(request.matrix)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("FOM versus energy needs a non-empty PSD matrix")
    if not 1 <= request.slices <= MAX_SLICES:
        raise ValueError(f"choose between 1 and {MAX_SLICES} energy slices")
    if request.minimum_events < 1:
        raise ValueError("the minimum number of events per slice must be positive")
    if request.spacing not in ("counts", "width"):
        raise ValueError(f"unknown slice spacing {request.spacing!r}")
    energy_edges = np.linspace(*request.energy_range, matrix.shape[0] + 1)
    ratio_edges = np.linspace(*request.ratio_range, matrix.shape[1] + 1)
    ratio_centers = 0.5 * (ratio_edges[:-1] + ratio_edges[1:])
    first, last = energy_bin_range(energy_edges, request.energy_roi)
    if last <= first:
        raise ValueError("the energy ROI contains no histogram bins")
    events_per_bin = matrix[first:last].sum(axis=1, dtype=np.float64)

    def fit_bins(low: int, high: int) -> PsdEnergySlice:
        counts = matrix[low:high].sum(axis=0, dtype=np.float64)
        events = int(counts.sum())
        edges = (float(energy_edges[low]), float(energy_edges[high]))
        if events < request.minimum_events:
            status = f"{events:,} events; at least {request.minimum_events:,} are needed"
            return PsdEnergySlice(*edges, events, None, status)
        try:
            fit = fit_psd_projection(ratio_centers, counts, request.settings)
        except (RuntimeError, TypeError, ValueError) as error:
            return PsdEnergySlice(*edges, events, None, f"Fit unavailable: {error}")
        if fit.poor_convergence:
            status = "; ".join(fit.warnings) or fit.message
        else:
            status = "Converged"
        return PsdEnergySlice(*edges, events, fit, status)

    slices = tuple(
        fit_bins(first + low, first + high)
        for low, high in slice_bounds(events_per_bin, request.slices, request.spacing)
    )
    return PsdEnergyFomResult(request, slices, fit_bins(first, last))


def best_slice(result: PsdEnergyFomResult) -> PsdEnergySlice | None:
    """The slice with the highest FOM among fits that converged without warnings.

    A flagged fit (too few counts, a width below the bin size, a poor Gaussian
    description) can report an inflated FOM, so it is never chosen.
    """
    trusted = [item for item in result.slices if item.trusted and item.fit is not None]
    if not trusted:
        return None
    return max(trusted, key=lambda item: item.fit.fom if item.fit is not None else -np.inf)


def energy_fom_rows(result: PsdEnergyFomResult) -> list[tuple[str, ...]]:
    """Header plus one row per slice and a final whole-ROI row, as text."""
    rows: list[tuple[str, ...]] = [ENERGY_FOM_HEADERS]
    labelled = [(str(index), item) for index, item in enumerate(result.slices, start=1)]
    labelled.append(("Whole ROI", result.whole))
    for label, item in labelled:
        fit = item.fit
        fitted: tuple[str, ...] = ("",) * 6
        if fit is not None:
            fitted = (
                _number(fit.centroids[0]),
                _number(fit.centroids[1]),
                _number(fit.fwhms[0]),
                _number(fit.fwhms[1]),
                _number(fit.fom),
                _number(fit.reduced_deviance),
            )
        rows.append(
            (label, _number(item.low), _number(item.high), str(item.events), *fitted, item.status)
        )
    return rows


def energy_fom_settings(result: PsdEnergyFomResult) -> list[tuple[str, str]]:
    """The inputs that reproduce a result: histogram, ROI, slicing and fit."""
    request = result.request
    settings = request.settings
    shape = np.shape(request.matrix)
    low, high = result.energy_roi
    if settings.mode == "separate" and settings.peak_rois is not None:
        model = "Separate Gaussians"
        windows = "; ".join(_range(window) for window in settings.peak_rois)
    else:
        model = "Double Gaussian"
        windows = _range(settings.roi)
    spacing = "equal counts" if request.spacing == "counts" else "equal width"
    first_seed, second_seed = settings.centroid_seeds
    return [
        ("Energy axis", request.energy_label),
        ("Energy range", _range(request.energy_range)),
        ("PSD ratio range", _range(request.ratio_range)),
        ("Matrix bins (energy x ratio)", f"{shape[0]} x {shape[1]}"),
        ("Energy ROI (bin edges)", _range((low, high))),
        ("Slices", f"{len(result.slices)} ({spacing})"),
        ("Minimum events per fitted slice", str(request.minimum_events)),
        ("Fit model", model),
        ("Fit window(s)", windows),
        ("Centroid seeds", f"{_number(first_seed)}, {_number(second_seed)}"),
        ("y0 offset", "fitted" if settings.use_background else "fixed at zero"),
    ]


def write_energy_fom_csv(path: Path, result: PsdEnergyFomResult) -> None:
    """Write the slice table, preceded by ``#`` lines with its inputs."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("# PSD FOM versus energy\n")
        for key, value in energy_fom_settings(result):
            stream.write(f"# {key}: {value}\n")
        csv.writer(stream).writerows(energy_fom_rows(result))


def _number(value: float) -> str:
    return f"{float(value):.6g}"


def _range(bounds: tuple[float, float]) -> str:
    return f"{_number(bounds[0])} to {_number(bounds[1])}"
