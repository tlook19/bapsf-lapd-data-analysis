"""Noise-aware T_e at the far experiment-set-4 ports, without the window family.

Why a second estimator exists
-----------------------------
The pipeline's T_e comes from a log-slope fit of the electron-retarding branch
over a window whose bounds are set RELATIVE to the sweep's own electron
saturation level: a lower bound at a percentile of the positive electron
current (floored at 1.5 % of the maximum) and an upper bound at a fraction of
that maximum.  ``scripts/refit_sweep_windows.py`` varies both bounds over a
5 x 5 family and reports the spread; ``scripts/refit_window_band.py`` does the
same cell by cell.  At the far set-4 ports that family REFUSES -- the spread
across the family exceeds the criterion by a wide margin, so no value in it is
authoritative.

The reason the family refuses is a fixed additive floor, not a fit convention.
The residual of the ion-line fit on the 3 ohm sense channel has essentially the
same RMS on every run of this circuit configuration, while the electron
saturation level falls steeply from the near port to the far one.  A relative
window therefore reaches ever further DOWN into that fixed floor as the signal
shrinks.  The bias this produces has a closed form: with a measured current
``I_meas = I_e + d`` for an additive offset ``d``,

    d ln(I_meas) / dV = (1 / T_e) * 1 / (1 + d / I_e),

so a local slope fitted at electron current ``I_e`` reads

    T_measured = T_e * (1 + d / I_e).

The bias is unbounded as ``I_e`` falls toward the floor and vanishes well above
it.  A window family anchored to a fraction of ``I_max`` cannot escape it once
``I_max`` is only a few tens of floors: every low window in the family sits
where ``d / I_e`` is order one, and reads high.

The two estimators here
-----------------------
Both are independent of the window family.  Both use the pipeline's own sweep
conditioning: the ramp clip, the Butterworth low-pass at the per-run cutoff, the
ion-line fit with a non-negative slope over the low-voltage quantile, and that
line subtracted from the raw current.  Every step calls the same helper the
window products call -- ``_cutoff_hz_for_run`` and ``_cycle_start_times`` from
``process_langmuir_sweeps``, ``butterworth_lowpass``, ``_robust_line_fit`` and
``_contiguous_true_region`` from the package -- but the SEQUENCE is written out
here rather than called, because ``refit_sweep_windows.refit_sweep`` returns
only ``T_e`` and does not expose the conditioned branch or its fit residual.
The unit tests hold the two in step by running both over one synthetic sweep.

(i) LOCAL SLOPE over a noise-referenced current range.  Fit ``ln I_e`` against
    ``V`` over the contiguous samples with

        I_e in [SLOPE_LOW_FLOORS * noise_floor,  SLOPE_HIGH_FRACTION * I_max]

    and take ``T_e = 1 / slope``.  The lower bound is set by the MEASURED noise
    floor, not by a fraction of ``I_max``, so it does not migrate downward as
    the signal falls: the worst-case local bias inside the range is
    ``1 + 1 / SLOPE_LOW_FLOORS`` at the very bottom sample and less everywhere
    above it.  The price is range: as ``I_max`` falls the range narrows toward
    nothing, and the estimator does NOT refuse on its own -- it goes on fitting
    a sliver as long as ``SLOPE_MIN_SAMPLES`` contiguous samples survive in it,
    and a fit over a sliver returns a plausible-looking number carrying no
    information.  The range width in decades is what says whether the number
    means anything, which is why it is printed for every port and is the first
    pre-registered gate.

(ii) FLOATING-POTENTIAL ratio.  For a Maxwellian retarding branch
     ``I_e(V) = I_es * exp((V - V_p) / T_e)`` that saturates at the plasma
     potential, the floating potential is where the electron current balances
     the ion current, so

        T_e = (V_p - V_f) / ln(I_es / I_i).

     ``V_f`` is the raw filtered current's first negative-to-positive zero
     crossing; ``I_i`` is the magnitude of the fitted ion line there; ``V_p`` is
     the KNEE of the electron-collection branch, located where ``dI_e/dV`` is
     largest between ``V_f`` and the sweep's own current maximum; and ``I_es``
     is the electron current AT that knee, so the two halves of the formula are
     read at the same point.  The saturation ratio is MEASURED on each sweep --
     single digits on this magnetized column -- and is not replaced by the
     unmagnetized planar-probe value near 34.  This estimator reads three points
     of the curve and never enters the low-current region at all, so it fails
     differently from (i); agreement between them is the second pre-registered
     gate.

     ``V_p`` must be the knee and not the current maximum.  The collection
     branch keeps climbing for tens of volts above the knee, so reading ``V_p``
     at the maximum inflates ``V_p - V_f`` by an order of magnitude while
     ``ln(I / I_i)`` grows only logarithmically, and the estimator returns
     values near 10 eV on a sub-eV plasma.  The current-maximum reading is
     reported beside the knee reading at every port, and its size is the
     measure of how much this estimator depends on that convention.

Coverage
--------
Ports 21, 29 and 41 of experiment set 4, plateau cycles 10--19.5 ms, every shot,
core cells ``|x| <= CORE_MAX_CM`` (the boundary the filled T_e product's core
uses).  Both probe faces are read where both are usable: port 21 rot-0 and
rot-180, port 29 rot-0 and rot-180, port 41 rot-0 only.  The port-41 rot-180 run
is EXCLUDED and reported as such -- its ion-saturation state carries the x1.76
rot-180 offset the saturation screen flags, and its cells are almost all bad.

Aggregation
-----------
Two aggregations are reported for every cell, because they weight the sweeps
differently and there is no reason to prefer one blindly:

  ``per-sweep``  estimate each of the 200 sweeps separately, then take the
                 median over sweeps (the aggregation the window products use);
  ``pooled``     pool the in-range samples of all 200 sweeps of a cell into one
                 least-squares fit (estimator (i)), or evaluate the estimator
                 once on the per-sweep medians of its four inputs
                 (estimator (ii)).

The port value is the mean over the admitted core cells, matching the rule the
band product's port T_e uses; the median over cells is printed beside it.  The
pipeline's physical-bounds guard (drop ``T_e <= TE_MIN_EV`` or
``T_e > TE_MAX_EV``) is applied so the numbers are comparable with the window
products; the fraction of sweeps it drops is printed, and the unguarded port
medians are printed beside the guarded ones.

Pre-registered gate (printed, not interpreted)
----------------------------------------------
  * a clean range of at least ``GATE_MIN_DECADES`` decade of current;
  * the two estimators within ``GATE_RATIO`` of each other;
  * at port 29, the two probe faces within ``GATE_RATIO`` of each other.

Pre-registered bins on the port-41 value
----------------------------------------
  T_e <= 0.4 eV      ->  "the T_e chain confirmed at p41"
  0.4 < T_e < 0.8 eV ->  "partial closure -- a x1.2-1.5 residual OPEN"
  T_e >= 0.8 eV      ->  "the collection deficit owns p41"

Control: port 21 must reproduce 0.66 +/- 0.1 eV.

Status
------
This is a REGISTERED ALTERNATIVE estimator, not a replacement.  The window-family
fit remains the pipeline's T_e and nothing here writes a pipeline product; the
banked window-family value is read alongside and printed beside this one so the
two can be compared directly.  ``--output`` defaults outside ``processed/``.

Inputs
------
  config/may2026_run_manifest.toml            run configurations
  data/                                       the raw run files the manifest names
  processed/langmuir_sweeps.hdf5              the x grid and the set-4 run roster
  --window-band <path>                        the banked per-cell window-family
                                              product, read only for the
                                              comparison column; optional

Usage
-----
  PYTHONPATH=src python scripts/es4_te_noise_aware.py \
      --window-band /path/to/window_refit_band_es4_core_rot0.hdf5 \
      --output es4_te_noise_aware.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bapsf_lapd import ChannelKind, LapdDataset  # noqa: E402
from bapsf_lapd.filtering import butterworth_lowpass  # noqa: E402
from bapsf_lapd.langmuir import (  # noqa: E402
    _contiguous_true_region,
    _robust_line_fit,
)


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pls = _load_script_module("pls", ROOT / "scripts" / "process_langmuir_sweeps.py")

MANIFEST = ROOT / "config" / "may2026_run_manifest.toml"
SWEEPS_H5 = ROOT / "processed" / "langmuir_sweeps.hdf5"

#: Experiment set and ports covered.
EXPERIMENT_SET = 4
PORTS = (21, 29, 41)

#: Runs excluded from the pass, with the reason printed in the report.
EXCLUDED_RUNS = {
    "47": (
        "port-41 rot-180: the ion-saturation state carries the x1.76 rot-180 "
        "offset the saturation screen flags, and nearly every cell is bad"
    ),
}

#: Core cells, the boundary the filled T_e product's core uses.
CORE_MAX_CM = 10.0

#: Plateau window and ramp clip, the pipeline's own.
PLATEAU_MS = (10.0, 19.5)
CLIP_US = 10.0
FILTER_ORDER = 4
BASE_CUTOFF_HZ = 100.0e3

#: Ion-branch domain: samples at or below this quantile of the sweep voltage.
ION_CUT_QUANTILE = 0.22

#: Estimator (i): the current range, in units of the measured noise floor
#: (lower) and of the electron saturation level (upper).
SLOPE_LOW_FLOORS = 5.0
SLOPE_HIGH_FRACTION = 0.3

#: Minimum samples inside the range for a slope fit to be attempted.
SLOPE_MIN_SAMPLES = 8

#: Minimum samples on the collection branch for the knee search to be attempted.
KNEE_MIN_SAMPLES = 5

#: Physical-bounds guard, the one the window products apply.
TE_MIN_EV = 0.05
TE_MAX_EV = 30.0

#: Pre-registered gate thresholds.
GATE_MIN_DECADES = 1.0
GATE_RATIO = 1.3

#: Pre-registered bin edges on the port-41 value, in eV.
BIN_CONFIRMED_MAX_EV = 0.4
BIN_PARTIAL_MAX_EV = 0.8

#: Pre-registered control on port 21, in eV.
CONTROL_TE_EV = 0.66
CONTROL_TOLERANCE_EV = 0.1

#: The port whose two faces the third gate clause compares.
FACE_GATE_PORT = 29

#: The port the bins are read on, and the port carrying the control.
BIN_PORT = 41
CONTROL_PORT = 21

CSV_FIELDS = (
    "port",
    "run_id",
    "rotation_deg",
    "x_cm",
    "n_sweeps",
    "noise_floor_a",
    "i_max_a",
    "range_decades",
    "snr_at_range_bottom",
    "te_slope_per_sweep_ev",
    "te_slope_pooled_ev",
    "te_vf_per_sweep_ev",
    "te_vf_pooled_ev",
    "te_vf_i_max_convention_ev",
    "saturation_ratio",
    "i_knee_a",
    "v_plasma_v",
    "v_float_v",
    "guard_dropped_fraction",
    "te_window_family_ev",
    "te_window_default_ev",
)


class SweepConditioningError(ValueError):
    """A sweep that cannot be conditioned into an electron-retarding branch."""


class ConditionedSweep:
    """One filtered sweep reduced to the quantities both estimators read.

    Attributes
    ----------
    voltage, current:
        The sweep sorted by voltage; ``current`` is the raw filtered current,
        ion line included.
    electron_current:
        ``current`` minus the fitted ion line, at every sample.
    noise_floor_a:
        RMS of the ion-line fit residual over the ion-branch domain, in amperes.
        This is the additive floor that biases a low-current log-slope fit.
    i_max_a:
        The largest positive ``electron_current`` anywhere on the sweep.  This
        is the level estimator (i)'s upper range bound is a fraction of.  It is
        NOT the electron saturation level at the knee: the collection branch
        keeps climbing above the knee, so this maximum sits near the ramp end.
    v_i_max_v:
        The voltage at which ``electron_current`` attains ``i_max_a``.
    v_plasma_v:
        The KNEE: the voltage of the steepest rise of ``electron_current``
        between the floating potential and ``v_i_max_v``.  This is the plasma
        potential estimator (ii) reads.
    i_knee_a:
        ``electron_current`` at ``v_plasma_v`` -- the electron saturation level
        that pairs with it, and the numerator of estimator (ii)'s ratio.
    v_float_v:
        The first negative-to-positive zero crossing of the raw ``current``.
    i_ion_a:
        The magnitude of the fitted ion line at ``v_float_v``.
    """

    __slots__ = (
        "voltage",
        "current",
        "electron_current",
        "noise_floor_a",
        "i_max_a",
        "v_i_max_v",
        "v_plasma_v",
        "i_knee_a",
        "v_float_v",
        "i_ion_a",
    )

    def __init__(
        self,
        voltage,
        current,
        electron_current,
        noise_floor_a,
        i_max_a,
        v_i_max_v,
        v_plasma_v,
        i_knee_a,
        v_float_v,
        i_ion_a,
    ):
        self.voltage = voltage
        self.current = current
        self.electron_current = electron_current
        self.noise_floor_a = noise_floor_a
        self.i_max_a = i_max_a
        self.v_i_max_v = v_i_max_v
        self.v_plasma_v = v_plasma_v
        self.i_knee_a = i_knee_a
        self.v_float_v = v_float_v
        self.i_ion_a = i_ion_a

    @property
    def saturation_ratio(self) -> float:
        """The measured ``I_knee / I_i``, the ratio estimator (ii) reads."""
        if not (self.i_ion_a > 0.0):
            return float("nan")
        return float(self.i_knee_a / self.i_ion_a)


def condition_sweep(voltage, current) -> ConditionedSweep:
    """Reduce one filtered sweep the way the pipeline's window fit does.

    Sorts by voltage, fits the ion line over ``V <= the ION_CUT_QUANTILE
    quantile`` with a non-negative slope, subtracts it, and measures the
    residual RMS of that fit -- the noise floor both estimators are referenced
    to.  Raises ``SweepConditioningError`` when the sweep has too few finite
    samples or no usable positive electron branch.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    finite = np.isfinite(voltage) & np.isfinite(current)
    v = voltage[finite]
    i = current[finite]
    if v.size < 20:
        raise SweepConditioningError("fewer than 20 finite samples")
    order = np.argsort(v)
    v = v[order]
    i = i[order]

    ion_cut = float(np.nanquantile(v, ION_CUT_QUANTILE))
    domain = v <= ion_cut
    if domain.sum() < 3:
        raise SweepConditioningError("ion-branch domain has fewer than 3 samples")
    fit = _robust_line_fit(v, i, domain, slope_min=0.0)
    line = fit.slope * v + fit.intercept
    electron_current = i - line
    noise_floor = float(np.sqrt(np.mean((i[domain] - line[domain]) ** 2)))

    positive = electron_current > 0
    if positive.sum() < SLOPE_MIN_SAMPLES:
        raise SweepConditioningError("positive electron branch too short")
    index_max = int(np.argmax(np.where(positive, electron_current, -np.inf)))
    i_max = float(electron_current[index_max])
    v_i_max = float(v[index_max])

    crossings = np.flatnonzero((i[:-1] <= 0.0) & (i[1:] > 0.0))
    if crossings.size:
        index_float = int(crossings[0])
        v_float = float(v[index_float])
        i_ion = float(abs(line[index_float]))
    else:
        v_float = float("nan")
        i_ion = float("nan")

    v_plasma, i_knee = knee_point(v, electron_current, v_float, v_i_max)

    return ConditionedSweep(
        voltage=v,
        current=i,
        electron_current=electron_current,
        noise_floor_a=noise_floor,
        i_max_a=i_max,
        v_i_max_v=v_i_max,
        v_plasma_v=v_plasma,
        i_knee_a=i_knee,
        v_float_v=v_float,
        i_ion_a=i_ion,
    )


def knee_point(voltage, electron_current, v_float_v, v_i_max_v):
    """Return (V_p, I_e(V_p)) at the knee of the electron-collection branch.

    The plasma potential is located where the collected electron current rises
    most steeply -- the standard knee criterion, ``argmax dI_e/dV``.  The search
    spans the collection branch, from the floating potential up to the sweep's
    own current maximum, so it carries no window constant of its own; when there
    is no floating potential the search starts at the first positive sample.

    Locating ``V_p`` at the current maximum instead is WRONG for estimator (ii)
    and the difference is not subtle: the collection branch keeps climbing well
    above the knee, so that convention puts ``V_p`` near the ramp end and
    inflates ``V_p - V_f`` by more than an order of magnitude while
    ``ln(I / I_i)`` grows only logarithmically.  Both are reported, and the
    difference between them is the estimator's convention sensitivity.
    """
    voltage = np.asarray(voltage, dtype=float)
    electron_current = np.asarray(electron_current, dtype=float)
    lower = v_float_v if np.isfinite(v_float_v) else -np.inf
    window = (voltage > lower) & (voltage <= v_i_max_v)
    if window.sum() < KNEE_MIN_SAMPLES:
        return float("nan"), float("nan")
    slope = np.gradient(electron_current, voltage)
    index = int(np.flatnonzero(window)[int(np.argmax(slope[window]))])
    return float(voltage[index]), float(electron_current[index])


def slope_range_bounds(noise_floor_a: float, i_max_a: float) -> tuple[float, float]:
    """Return the (low, high) electron-current bounds of the clean range."""
    return SLOPE_LOW_FLOORS * float(noise_floor_a), SLOPE_HIGH_FRACTION * float(i_max_a)


def range_decades(noise_floor_a: float, i_max_a: float) -> float:
    """Decades of current between the range bounds; negative when it is closed."""
    low, high = slope_range_bounds(noise_floor_a, i_max_a)
    if not (low > 0.0) or not (high > 0.0):
        return float("nan")
    return float(np.log10(high / low))


def slope_range_mask(voltage, electron_current, noise_floor_a, i_max_a):
    """Boolean mask of the contiguous samples inside the clean current range."""
    low, high = slope_range_bounds(noise_floor_a, i_max_a)
    electron_current = np.asarray(electron_current, dtype=float)
    mask = (electron_current >= low) & (electron_current <= high)
    if mask.sum() < SLOPE_MIN_SAMPLES:
        return np.zeros(electron_current.shape, dtype=bool)
    mask = _contiguous_true_region(mask)
    if mask.sum() < SLOPE_MIN_SAMPLES:
        return np.zeros(electron_current.shape, dtype=bool)
    return mask


def local_slope_te_ev(voltage, electron_current, noise_floor_a, i_max_a) -> float:
    """T_e from the log-slope over the noise-referenced current range.

    The range's lower bound is ``SLOPE_LOW_FLOORS`` times the MEASURED noise
    floor, so the residual bias from an additive floor ``d`` is at most
    ``1 + 1 / SLOPE_LOW_FLOORS`` at the bottom sample and falls off as
    ``1 + d / I_e`` above it.  Returns NaN when the range holds too few
    contiguous samples or the fitted slope is not positive.
    """
    voltage = np.asarray(voltage, dtype=float)
    electron_current = np.asarray(electron_current, dtype=float)
    mask = slope_range_mask(voltage, electron_current, noise_floor_a, i_max_a)
    if not mask.any():
        return float("nan")
    slope, _ = np.polyfit(voltage[mask], np.log(electron_current[mask]), 1)
    if not (slope > 0.0):
        return float("nan")
    return float(1.0 / slope)


def floating_potential_te_ev(
    v_plasma_v: float, v_float_v: float, i_max_a: float, i_ion_a: float
) -> float:
    """T_e = (V_p - V_f) / ln(I_max / I_i), with the MEASURED saturation ratio.

    Exact for a Maxwellian retarding branch ``I_e(V) = I_max exp((V - V_p)/T_e)``
    balanced against a constant ion current at the floating potential.  The
    saturation ratio is the one the sweep itself shows; no unmagnetized
    planar-probe value is substituted.  Returns NaN on a non-positive ratio, a
    non-positive potential difference, or a non-finite input.
    """
    if not np.isfinite([v_plasma_v, v_float_v, i_max_a, i_ion_a]).all():
        return float("nan")
    if not (i_ion_a > 0.0) or not (i_max_a > i_ion_a):
        return float("nan")
    delta_v = float(v_plasma_v) - float(v_float_v)
    if not (delta_v > 0.0):
        return float("nan")
    return float(delta_v / np.log(float(i_max_a) / float(i_ion_a)))


def guarded(values):
    """Apply the pipeline's physical-bounds guard, returning a NaN-filled copy."""
    values = np.array(values, dtype=float)
    values[(values <= TE_MIN_EV) | (values > TE_MAX_EV)] = np.nan
    return values


def _run_roster(sweeps_h5: Path) -> list[tuple[str, int, int]]:
    """Return (run_id, port, rotation_deg) for the covered set-4 ports."""
    roster = []
    with h5py.File(sweeps_h5, "r") as hdf:
        group = hdf[f"experiment_sets/{EXPERIMENT_SET}"]
        for run_id in sorted(group, key=int):
            attrs = group[run_id].attrs
            port = int(attrs["port"])
            if port not in PORTS:
                continue
            roster.append((run_id, port, int(attrs["rotation_deg"])))
    return sorted(roster, key=lambda row: (row[1], row[2]))


def _core_cells(x_cm: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.abs(x_cm) <= CORE_MAX_CM)


def measure_run(dataset: LapdDataset, run_id: str, cells: np.ndarray, x_cm: np.ndarray) -> dict:
    """Both estimators, at every core cell of one run, over every plateau sweep."""
    started = time.time()
    run = dataset.run(run_id)
    cycle_start_s = _pls._cycle_start_times(run)
    in_plateau = np.flatnonzero(
        (cycle_start_s * 1e3 >= PLATEAU_MS[0]) & (cycle_start_s * 1e3 <= PLATEAU_MS[1])
    )
    core_cutoff_hz, _ = _pls._cutoff_hz_for_run(run, BASE_CUTOFF_HZ)
    sample_rate_hz = run.config.acquisition.sample_rate_hz
    i_offset = run.default_zero_offset_v(ChannelKind.I_SWEEP)
    v_offset = run.default_zero_offset_v(ChannelKind.V_SWEEP)
    ramp_slices = run.sweep_ramp_sample_slices(clip_s=CLIP_US * 1e-6)

    per_sweep = {int(cell): [] for cell in cells}
    pooled_points = {int(cell): [] for cell in cells}
    for cycle in in_plateau:
        ramp = ramp_slices[cycle]
        i_all = run.langmuir_traces(ChannelKind.I_SWEEP, ramp, zero_offset_v=i_offset)
        v_all = run.langmuir_traces(ChannelKind.V_SWEEP, ramp, zero_offset_v=v_offset)
        for cell in cells:
            cell = int(cell)
            i_cell = butterworth_lowpass(
                i_all[cell],
                sample_rate_hz=sample_rate_hz,
                cutoff_hz=core_cutoff_hz,
                order=FILTER_ORDER,
            )
            v_cell = butterworth_lowpass(
                v_all[cell],
                sample_rate_hz=sample_rate_hz,
                cutoff_hz=core_cutoff_hz,
                order=FILTER_ORDER,
            )
            for shot in range(i_cell.shape[0]):
                try:
                    sweep = condition_sweep(v_cell[shot], i_cell[shot])
                except (SweepConditioningError, ValueError):
                    continue
                mask = slope_range_mask(
                    sweep.voltage,
                    sweep.electron_current,
                    sweep.noise_floor_a,
                    sweep.i_max_a,
                )
                if mask.any():
                    pooled_points[cell].append(
                        (sweep.voltage[mask], np.log(sweep.electron_current[mask]))
                    )
                    snr_bottom = float(
                        sweep.electron_current[mask].min() / sweep.noise_floor_a
                    )
                    te_slope = local_slope_te_ev(
                        sweep.voltage,
                        sweep.electron_current,
                        sweep.noise_floor_a,
                        sweep.i_max_a,
                    )
                else:
                    snr_bottom = float("nan")
                    te_slope = float("nan")
                per_sweep[cell].append(
                    (
                        te_slope,
                        floating_potential_te_ev(
                            sweep.v_plasma_v,
                            sweep.v_float_v,
                            sweep.i_knee_a,
                            sweep.i_ion_a,
                        ),
                        sweep.noise_floor_a,
                        sweep.i_max_a,
                        snr_bottom,
                        sweep.v_plasma_v,
                        sweep.v_float_v,
                        sweep.i_ion_a,
                        sweep.i_knee_a,
                        floating_potential_te_ev(
                            sweep.v_i_max_v,
                            sweep.v_float_v,
                            sweep.i_max_a,
                            sweep.i_ion_a,
                        ),
                    )
                )
        del i_all, v_all

    rows = []
    for cell in cells:
        cell = int(cell)
        table = np.asarray(per_sweep[cell], dtype=float).reshape(-1, 10)
        noise_floor = float(np.nanmedian(table[:, 2])) if table.size else float("nan")
        i_max = float(np.nanmedian(table[:, 3])) if table.size else float("nan")
        te_slope_all = table[:, 0] if table.size else np.array([], dtype=float)
        te_vf_all = table[:, 1] if table.size else np.array([], dtype=float)
        te_slope_guarded = guarded(te_slope_all)
        te_vf_guarded = guarded(te_vf_all)
        finite_before = int(np.isfinite(te_slope_all).sum())
        finite_after = int(np.isfinite(te_slope_guarded).sum())
        dropped = (
            (finite_before - finite_after) / finite_before if finite_before else float("nan")
        )

        if pooled_points[cell]:
            v_pool = np.concatenate([point[0] for point in pooled_points[cell]])
            ln_pool = np.concatenate([point[1] for point in pooled_points[cell]])
            slope, _ = np.polyfit(v_pool, ln_pool, 1)
            te_slope_pooled = float(1.0 / slope) if slope > 0 else float("nan")
        else:
            te_slope_pooled = float("nan")
        te_slope_pooled = float(guarded([te_slope_pooled])[0])

        v_plasma_med = float(np.nanmedian(table[:, 5])) if table.size else float("nan")
        v_float_med = float(np.nanmedian(table[:, 6])) if table.size else float("nan")
        i_ion_med = float(np.nanmedian(table[:, 7])) if table.size else float("nan")
        i_knee_med = float(np.nanmedian(table[:, 8])) if table.size else float("nan")
        te_vf_pooled = float(
            guarded(
                [
                    floating_potential_te_ev(
                        v_plasma_med, v_float_med, i_knee_med, i_ion_med
                    )
                ]
            )[0]
        )
        te_vf_argmax_all = table[:, 9] if table.size else np.array([], dtype=float)
        te_vf_argmax_guarded = guarded(te_vf_argmax_all)

        rows.append(
            dict(
                cell=cell,
                x_cm=float(x_cm[cell]),
                n_sweeps=int(table.shape[0]),
                noise_floor_a=noise_floor,
                i_max_a=i_max,
                range_decades=range_decades(noise_floor, i_max),
                snr_at_range_bottom=_nanmedian_or_nan(table[:, 4] if table.size else None),
                te_slope_per_sweep_ev=float(np.nanmedian(te_slope_guarded))
                if finite_after
                else float("nan"),
                te_slope_per_sweep_unguarded_ev=float(np.nanmedian(te_slope_all))
                if finite_before
                else float("nan"),
                te_slope_pooled_ev=te_slope_pooled,
                te_vf_per_sweep_ev=float(np.nanmedian(te_vf_guarded))
                if np.isfinite(te_vf_guarded).any()
                else float("nan"),
                te_vf_per_sweep_unguarded_ev=float(np.nanmedian(te_vf_all))
                if np.isfinite(te_vf_all).any()
                else float("nan"),
                te_vf_pooled_ev=te_vf_pooled,
                te_vf_argmax_ev=float(np.nanmedian(te_vf_argmax_guarded))
                if np.isfinite(te_vf_argmax_guarded).any()
                else float("nan"),
                saturation_ratio=(i_knee_med / i_ion_med)
                if i_ion_med > 0
                else float("nan"),
                i_knee_a=i_knee_med,
                v_plasma_v=v_plasma_med,
                v_float_v=v_float_med,
                guard_dropped_fraction=dropped,
            )
        )
    return dict(run_id=run_id, rows=rows, n_plateau=len(in_plateau), wall_s=time.time() - started)


def _nanmedian_or_nan(values) -> float:
    """Median over the finite entries; NaN when a cell has none at all."""
    if values is None:
        return float("nan")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmedian(values))


def _port_value(rows: list[dict], key: str) -> tuple[float, float]:
    """Return (mean, median) over the core cells of one per-cell column."""
    values = np.array([row[key] for row in rows], dtype=float)
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    return float(np.nanmean(values)), float(np.nanmedian(values))


def _load_window_band(path: Path | None) -> dict[int, tuple[float, float]]:
    """Return {port: (family median mean, default-window mean)} over core cells."""
    if path is None or not path.exists():
        return {}
    out = {}
    with h5py.File(path, "r") as hdf:
        group = hdf.get(f"set{EXPERIMENT_SET}")
        if group is None:
            return {}
        for name in group:
            record = group[name]
            port = int(record.attrs["port"])
            x_cm = record["x_cm"][:]
            core = np.abs(x_cm) <= CORE_MAX_CM
            family = record["te_window_family_median_ev"][:]
            default = record["te_default_med_ev"][:]
            out[port] = (
                float(np.nanmean(family[core])),
                float(np.nanmean(default[core])),
            )
    return out


def _bin_label(te_ev: float) -> str:
    if not np.isfinite(te_ev):
        return "no value"
    if te_ev <= BIN_CONFIRMED_MAX_EV:
        return "the T_e chain confirmed at p41"
    if te_ev < BIN_PARTIAL_MAX_EV:
        return "partial closure -- a x1.2-1.5 residual OPEN"
    return "the collection deficit owns p41"


def _ratio(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return float("nan")
    return float(max(a, b) / min(a, b))


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--window-band",
        type=Path,
        default=None,
        help="banked per-cell window-family product, read only for the comparison column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("es4_te_noise_aware.csv"),
        help="per-cell CSV; defaults outside processed/, which holds pipeline products",
    )
    parser.add_argument("--sweeps-hdf5", type=Path, default=SWEEPS_H5)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    with h5py.File(args.sweeps_hdf5, "r") as hdf:
        x_cm = hdf["x_cm"][:]
    cells = _core_cells(x_cm)
    roster = _run_roster(args.sweeps_hdf5)
    dataset = LapdDataset.from_manifest(args.manifest)
    band = _load_window_band(args.window_band)

    print("noise-aware T_e at the far set-4 ports -- two window-free estimators")
    print("=" * 96)
    print(
        f"  set {EXPERIMENT_SET}, ports {PORTS}; plateau {PLATEAU_MS[0]}-{PLATEAU_MS[1]} ms, "
        f"every shot; core cells |x| <= {CORE_MAX_CM:g} cm ({cells.size} cells)"
    )
    print(
        f"  conditioning: {CLIP_US:g} us ramp clip, order-{FILTER_ORDER} Butterworth at the "
        f"per-run cutoff, ion line fitted with slope >= 0 over V <= the "
        f"{ION_CUT_QUANTILE:.2f} quantile"
    )
    print(
        f"  estimator (i)  local slope over I_e in [{SLOPE_LOW_FLOORS:g} x noise floor, "
        f"{SLOPE_HIGH_FRACTION:g} x I_max]"
    )
    print(
        "  estimator (ii) T_e = (V_p - V_f) / ln(I_es / I_i) at the KNEE, "
        "measured saturation ratio"
    )
    print(
        f"  guard: T_e outside ({TE_MIN_EV:g}, {TE_MAX_EV:g}] eV dropped, as in the window products"
    )
    for run_id, reason in sorted(EXCLUDED_RUNS.items()):
        print(f"  EXCLUDED run {run_id}: {reason}")
    print()

    results = {}
    csv_rows = []
    for run_id, port, rotation in roster:
        if run_id in EXCLUDED_RUNS:
            continue
        record = measure_run(dataset, run_id, cells, x_cm)
        record.update(port=port, rotation=rotation)
        results[(port, rotation)] = record
        family, default = band.get(port, (float("nan"), float("nan")))
        for row in record["rows"]:
            csv_rows.append(
                {
                    "port": port,
                    "run_id": run_id,
                    "rotation_deg": rotation,
                    "x_cm": f"{row['x_cm']:.6g}",
                    "n_sweeps": row["n_sweeps"],
                    "noise_floor_a": f"{row['noise_floor_a']:.6e}",
                    "i_max_a": f"{row['i_max_a']:.6e}",
                    "range_decades": f"{row['range_decades']:.6g}",
                    "snr_at_range_bottom": f"{row['snr_at_range_bottom']:.6g}",
                    "te_slope_per_sweep_ev": f"{row['te_slope_per_sweep_ev']:.6g}",
                    "te_slope_pooled_ev": f"{row['te_slope_pooled_ev']:.6g}",
                    "te_vf_per_sweep_ev": f"{row['te_vf_per_sweep_ev']:.6g}",
                    "te_vf_pooled_ev": f"{row['te_vf_pooled_ev']:.6g}",
                    "te_vf_i_max_convention_ev": f"{row['te_vf_argmax_ev']:.6g}",
                    "saturation_ratio": f"{row['saturation_ratio']:.6g}",
                    "i_knee_a": f"{row['i_knee_a']:.6e}",
                    "v_plasma_v": f"{row['v_plasma_v']:.6g}",
                    "v_float_v": f"{row['v_float_v']:.6g}",
                    "guard_dropped_fraction": f"{row['guard_dropped_fraction']:.6g}",
                    "te_window_family_ev": f"{family:.6g}",
                    "te_window_default_ev": f"{default:.6g}",
                }
            )
        print(
            f"  measured run {run_id} (port {port}, rot {rotation}): "
            f"{record['n_plateau']} plateau cycles, {cells.size} cells, "
            f"{record['wall_s']:.1f} s"
        )
    print()

    print("per-port table (mean over core cells; median in parentheses)")
    print("-" * 96)
    header = (
        f"{'port':>4} {'run':>4} {'rot':>4} {'floor/mA':>9} {'I_max/mA':>9} "
        f"{'decades':>8} {'SNR_bot':>8} {'Te(i)/eV':>16} {'Te(i)pool':>10} "
        f"{'Te(ii)/eV':>16} {'ratio':>6} {'family':>7} {'default':>8}"
    )
    print(header)
    summary = {}
    for (port, rotation), record in sorted(results.items()):
        rows = record["rows"]
        floor_mean, _ = _port_value(rows, "noise_floor_a")
        imax_mean, _ = _port_value(rows, "i_max_a")
        dec_mean, dec_med = _port_value(rows, "range_decades")
        decades_per_cell = np.array([row["range_decades"] for row in rows], dtype=float)
        cells_open = int(np.count_nonzero(decades_per_cell > 0.0))
        cells_wide = int(np.count_nonzero(decades_per_cell >= GATE_MIN_DECADES))
        snr_mean, _ = _port_value(rows, "snr_at_range_bottom")
        slope_mean, slope_med = _port_value(rows, "te_slope_per_sweep_ev")
        pooled_mean, _ = _port_value(rows, "te_slope_pooled_ev")
        vf_mean, vf_med = _port_value(rows, "te_vf_per_sweep_ev")
        vf_pooled_mean, _ = _port_value(rows, "te_vf_pooled_ev")
        slope_unguarded, _ = _port_value(rows, "te_slope_per_sweep_unguarded_ev")
        vf_unguarded, _ = _port_value(rows, "te_vf_per_sweep_unguarded_ev")
        sat_mean, _ = _port_value(rows, "saturation_ratio")
        vf_argmax_mean, _ = _port_value(rows, "te_vf_argmax_ev")
        knee_mean, _ = _port_value(rows, "i_knee_a")
        vplasma_mean, _ = _port_value(rows, "v_plasma_v")
        vfloat_mean, _ = _port_value(rows, "v_float_v")
        drop_mean, _ = _port_value(rows, "guard_dropped_fraction")
        family, default = band.get(port, (float("nan"), float("nan")))
        summary[(port, rotation)] = dict(
            run_id=record["run_id"],
            decades=dec_mean,
            decades_median=dec_med,
            cells_open=cells_open,
            cells_wide=cells_wide,
            cells_total=len(rows),
            slope=slope_mean,
            slope_median=slope_med,
            slope_pooled=pooled_mean,
            vf=vf_mean,
            vf_median=vf_med,
            vf_pooled=vf_pooled_mean,
            slope_unguarded=slope_unguarded,
            vf_unguarded=vf_unguarded,
            floor=floor_mean,
            i_max=imax_mean,
            snr=snr_mean,
            saturation_ratio=sat_mean,
            vf_i_max_convention=vf_argmax_mean,
            i_knee=knee_mean,
            v_plasma=vplasma_mean,
            v_float=vfloat_mean,
            guard_dropped=drop_mean,
            family=family,
            default=default,
        )
        print(
            f"{port:>4} {record['run_id']:>4} {rotation:>4} "
            f"{floor_mean * 1e3:9.3f} {imax_mean * 1e3:9.2f} "
            f"{dec_mean:8.2f} {snr_mean:8.1f} "
            f"{slope_mean:7.3f} ({slope_med:6.3f}) {pooled_mean:10.3f} "
            f"{vf_mean:7.3f} ({vf_med:6.3f}) {_ratio(slope_mean, vf_mean):6.2f} "
            f"{family:7.3f} {default:8.3f}"
        )
    print()
    print("  floor/mA  = ion-line residual RMS, the additive noise floor")
    print("  decades   = log10(range top / range bottom) of the clean current range")
    print("  SNR_bot   = the lowest fitted electron current in units of the noise floor")
    print("  ratio     = max/min of estimator (i) and estimator (ii) at that face")
    print("  family    = banked window-family median T_e, mean over the same core cells")
    print("  default   = banked default-window T_e, mean over the same core cells")
    print()

    print("unguarded medians and guard bookkeeping")
    print("-" * 96)
    for (port, rotation), item in sorted(summary.items()):
        print(
            f"  port {port} rot {rotation:>3} run {item['run_id']}: "
            f"Te(i) unguarded {item['slope_unguarded']:.3f} eV, "
            f"Te(ii) unguarded {item['vf_unguarded']:.3f} eV, "
            f"guard dropped {item['guard_dropped'] * 100:.1f} % of sweeps"
        )
        print(
            f"      V_f {item['v_float']:+.2f} V, V_p (knee) {item['v_plasma']:+.2f} V, "
            f"I_e(knee) {item['i_knee'] * 1e3:.2f} mA, measured I_knee/I_i "
            f"{item['saturation_ratio']:.2f}; the same estimator read at the "
            f"CURRENT-MAXIMUM convention instead gives "
            f"{item['vf_i_max_convention']:.3f} eV"
        )
    print()

    print("pre-registered gate")
    print("-" * 96)
    for (port, rotation), item in sorted(summary.items()):
        wide = np.isfinite(item["decades"]) and item["decades"] >= GATE_MIN_DECADES
        print(
            f"  clean range >= {GATE_MIN_DECADES:g} decade   port {port} rot {rotation:>3}: "
            f"{item['decades']:.2f} decades (cell median {item['decades_median']:.2f}; "
            f"{item['cells_wide']}/{item['cells_total']} cells at or above the "
            f"threshold, {item['cells_open']}/{item['cells_total']} with a range "
            f"open at all)  "
            f"-> {_verdict(wide)}"
        )
    for (port, rotation), item in sorted(summary.items()):
        ratio = _ratio(item["slope"], item["vf"])
        close = np.isfinite(ratio) and ratio <= GATE_RATIO
        print(
            f"  estimators within x{GATE_RATIO:g}  port {port} rot {rotation:>3}: "
            f"Te(i) {item['slope']:.3f} eV vs Te(ii) {item['vf']:.3f} eV, "
            f"x{ratio:.2f}  -> {_verdict(close)}"
        )
    faces = sorted(rot for (port, rot) in summary if port == FACE_GATE_PORT)
    if len(faces) == 2:
        left = summary[(FACE_GATE_PORT, faces[0])]
        right = summary[(FACE_GATE_PORT, faces[1])]
        for name, key in (("Te(i)", "slope"), ("Te(ii)", "vf")):
            ratio = _ratio(left[key], right[key])
            close = np.isfinite(ratio) and ratio <= GATE_RATIO
            print(
                f"  p{FACE_GATE_PORT} faces within x{GATE_RATIO:g}  {name}: "
                f"rot {faces[0]} {left[key]:.3f} eV vs rot {faces[1]} {right[key]:.3f} eV, "
                f"x{ratio:.2f}  -> {_verdict(close)}"
            )
    else:
        print(f"  p{FACE_GATE_PORT} faces within x{GATE_RATIO:g}: only one face measured")
    print()

    print(f"pre-registered bins on port {BIN_PORT}")
    print("-" * 96)
    for (port, rotation), item in sorted(summary.items()):
        if port != BIN_PORT:
            continue
        print(
            f"  rot {rotation:>3} run {item['run_id']}  estimator (i)  "
            f"{item['slope']:.3f} eV  ->  {_bin_label(item['slope'])}"
        )
        print(
            f"  rot {rotation:>3} run {item['run_id']}  estimator (ii) "
            f"{item['vf']:.3f} eV  ->  {_bin_label(item['vf'])}"
        )
    print()

    print(f"control: port {CONTROL_PORT} must reproduce "
          f"{CONTROL_TE_EV:g} +/- {CONTROL_TOLERANCE_EV:g} eV")
    print("-" * 96)
    for (port, rotation), item in sorted(summary.items()):
        if port != CONTROL_PORT:
            continue
        for name, key in (("estimator (i) ", "slope"), ("estimator (ii)", "vf")):
            value = item[key]
            inside = (
                np.isfinite(value)
                and abs(value - CONTROL_TE_EV) <= CONTROL_TOLERANCE_EV
            )
            print(
                f"  rot {rotation:>3} run {item['run_id']}  {name} {value:.3f} eV  "
                f"->  CONTROL {_verdict(inside)}"
            )
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {args.output} ({len(csv_rows)} cell rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
