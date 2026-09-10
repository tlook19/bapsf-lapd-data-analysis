"""Channel-scale factor between the ES4 and ES3 sweep rest biases.

What this measures
------------------
Between voltage ramps the Langmuir sweep supply parks the probe at a fixed
negative rest bias, and it is that dead-time current -- not the ramp -- that
every ion-saturation product is built from.  ES4 runs 42-48 sweep a nominal
+-20 V ramp on a 3 ohm sense resistor while ES3 sweeps +-75 V on 1 ohm, so
those two sets park at DIFFERENT rest biases (run 41, ES4's p11 run, sweeps
+-75 V on 1 ohm like its ES3 control run 31, and is not part of the RUN_PAIRS
this script processes).  Ion current on a probe grows with the sheath,
so a dead-time current collected at the shallower bias is not on the same scale
as one collected at the deeper bias, and comparing the two sets cell for cell
needs a factor

  F = |I_i(V_rest, ES3)| / |I_i(V_rest, ES4)|                         (>= 1)

This script measures F.  The reciprocal 1/F = |I_i(ES4)| / |I_i(ES3)| (<= 1) is
the same statement read the other way and is printed beside it; both
conventions appear in the table so a reader cannot mistake one for the other.

ES3 is the comparison set because it is the only defensible one: it shares
ES4's 100 V bank and differs only in puff.  ES1 and ES2 move the bank voltage
AND sweep a 250 us ramp instead of 500 us, so they are refused rather than
silently averaged in.

Where the bias numbers come from -- READ THIS BEFORE QUOTING THEM
-----------------------------------------------------------------
The absolute level of both rest biases is NOT a free measurement.  It is set by
this repository's zero-offset convention (``LapdRun.zero_offset_target_v`` in
``src/bapsf_lapd/reader.py``): the ``v_sweep`` trace is offset so that its
post-sweep parked DC equals the manifest's configured ``sweep.voltage_start``.
That anchor is -75 V for ES3 and -20 V for ES4, and the offsets it applies are
large -- about -16.55 V and -5.07 V respectively, printed per run below.

What this script genuinely measures is therefore the DEPARTURE of the dead-time
rest level from that parked anchor: about +0.78 V on ES3 and +0.69 V on ES4.
Anchor plus departure gives the biases the factor is quoted between, and the
table prints all three so the split is visible.

Residual systematic.  A common error in the anchor moves both biases together
and does not cancel in F.  Displacing both by -1 V moves F_direct by +0.2 to
+4.4 percent, and by +1 V by -0.6 to -4.0 percent, depending on the pair.  That
is larger than the cell-to-cell spread of the tightest pair (about +-1.0
percent), so the anchor -- not counting statistics -- is the dominant
uncertainty on the factor.  Nothing here tests the convention itself.

Two frames, and why -90 V and -24 V are also right.  In the RAW digitizer frame
(no zero offset applied) the same settled rest levels read about -90.8 V on ES3
and -24.4 V on ES4; in the offset frame used throughout this script they read
about -74.2 V and -19.3 V.  Both describe one measurement, and the table prints
each rest bias in both frames.  Separately, the deepest single ``v_sweep``
SAMPLE anywhere in a trace (about -92.9 V on ES3 and -22.4 V on ES4 in the
offset frame) is a per-shot noise excursion, not a bias level; the deepest
shot-averaged dead-window level is about -80.4 V and -21.4 V, reached in the
ringing that follows each ramp.  That ringing overshoots POSITIVE first -- it
starts near +80 V on ES3 and +21 V on ES4 -- which is why the rest bias is
measured only past a settling margin.

Two independent routes to the same factor
-----------------------------------------
(i) DIRECT, from the ES3 ramp.  The +-75 V ramp sweeps through BOTH rest
    biases, so the ratio can simply be read off one measured ion branch with no
    functional form at all:

      F_direct = |I(V_ES3)| / |I(V_ES4)|   interpolated on that ramp

    This is the least model-dependent number the data can give.

(ii) EXTRAPOLATED, from the ES4 ramp.  The +-20 V ramp never reaches the ES3
    bias, so its ion branch must be extended.  Three extensions are fitted over
    the same window and each is evaluated as a ratio between the same two
    biases:

      orbital-motion sheath expansion   |I| = A (V_f - V)^(3/4)
      the same form with the exponent free    |I| = A (V_f - V)^p
      a straight line                          |I| = a + b V

    The 3/4 form and the straight line are the two ends the bracket is drawn
    between; the free exponent says what power the branch actually shows.

What the fits can and cannot resolve
------------------------------------
The ES4 ion branch spans only a few volts of sheath, so its current changes by
a few percent across the whole fit window.  Every one of the three forms can be
drawn through that with residuals of the same size, and the printed per-model
RMS says so directly.  The consequence is that ``V_f`` and a free exponent are
only weakly determined -- the 3/4 form buys its shape by placing ``V_f``
wherever it must, including far above the swept range, which is how a form with
a 3/4 power ends up nearly straight over the window.  The fitted ``V_f`` and
exponent are reported for exactly that reason: they are the diagnostic of how
much of each extrapolation is the data and how much is the form.  They are
fitted shape parameters, not measurements of a floating potential or of a
sheath scaling.

Pinning the sheath reference to a measurement (``--pin-vf measured``)
---------------------------------------------------------------------
Because the fitted reference potential above is free to run far above the top
of the ES4 ramp, the 3/4 form has no discrimination left: the reference
absorbs whatever shape the branch demands, and the form is then fitting itself
rather than the data.  The optional ``--pin-vf measured`` mode removes that
freedom.  It anchors the sheath base to a potential this repository has
already MEASURED on the very same run -- the shot-averaged plasma potential in
``processed/langmuir_sweeps.hdf5``, averaged over the same plateau window and
the same core cells the factor itself is measured on, one value per port and
face -- and leaves only the amplitude and the power free:

  |I_i| = A (V_p - V) ** p          V_p pinned, A and p free

The pinned quantity is the PLASMA potential, not a floating potential.  The
sheath-expansion form measures its potential drop from V_p; ``V_f`` is the
name this script gives the form's own base parameter, and pinning it to a
measured V_p is what makes the form testable.  The product carries THREE
plasma-potential estimators -- ``vp_derivative_v`` (the knee of dI/dV),
``vp_log_v`` and ``vp_exp_v``.  This mode pins to ``vp_derivative_v`` and
prints all three beside it, so a reader can see how much of the answer is the
choice of estimator.

Holding the exponent at 3/4 as well leaves nothing free that can move the
ratio: with V_p pinned and p = 3/4 the factor is

  ((V_p - V_ES3) / (V_p - V_ES4)) ** 0.75

and no measured current enters it at all.  That number is printed too, as
``F_pinned(p=3/4)``, because it is the literal 3/4 form evaluated at a measured
reference and it says how far the branch is from that form.  It is a printed
diagnostic; the gate below is on the two-parameter pinned fit.

Second pre-registered gate (pinned mode only)
---------------------------------------------
  F_pinned inside [F_direct, 1.15 * F_direct]
      -> "inside -- the pinned fit supports the direct factor"
  F_pinned outside it
      -> "outside -- the direct factor stands alone", with the value
  the pinned fit converges on no core cell of the pair
      -> "REFUSED (no fit)"

The pinned fit's RMS residual is printed against the FREE fit's, and the free
fit is the control: it carries one more free parameter and can only do better,
so a pinned residual close to it means the measured reference costs the branch
nothing, while a materially worse one is itself the finding rather than a
failure of the mode.

The default invocation does not open the plasma-potential product, and its CSV
and printed report are byte-for-byte what they were without this mode.

The transfer is an ASSUMPTION
-----------------------------
Route (i) reads its ratio off a DIFFERENT plasma from the one the factor is
applied to: the ES3 ramp belongs to its own experiment set, at its own puff.
Using it as ES4's factor assumes the shape of the ion branch -- not its level
-- transfers between the two plasmas.  Nothing in this data proves that.  It is
stated here because the whole comparison rests on it, and it is why route (ii),
which stays inside ES4's own ramp, is computed at all.

Pre-registered gate
-------------------
With the bracket drawn between the direct ratio and the linear extrapolation:

  the 3/4 ratio lands INSIDE the bracket at every port/face
      -> "the factor is bracketed"
  it lands OUTSIDE at any port/face
      -> "the bracket is the claim and no product is rescaled"

THE GATE DOES NOT DISCRIMINATE, and the reader meets that fact beside the
number: the 3/4 ratio sits 83 to 92 percent of the way from the direct end of
the bracket to the linear end at all six pairs, and below the linear ratio in
every core cell, because a 3/4 form with a free ``V_f`` can mimic the nearly
straight branch the linear fit draws.  The finding is not which side of the
bracket it falls on.  The finding is that the direct route (1.07 to 1.25) and
both extrapolations (1.27 to 1.69) disagree by more than either one's spread.
The bracket is the claim.

NO PRODUCT IS RESCALED BY THIS SCRIPT EITHER WAY -- it writes one CSV of
numbers and nothing else, and it refuses an output path inside ``processed/``
so it cannot enter the scoring chain.

Inputs
------
  config/may2026_run_manifest.toml    sweep schedule, channels, calibration
  data/may2026/*.hdf5                 raw runs, opened read-only
  processed/langmuir_sweeps.hdf5      shot-averaged plasma potential,
                                      read ONLY under --pin-vf measured

All are read from the ``repo-root`` given on the command line, which is what
lets this run from a worktree against a checkout that holds the raw data.

Usage
-----
  python scripts/es4_sweep_rest_bias_factor.py <repo-root> [--output out.csv]
  python scripts/es4_sweep_rest_bias_factor.py <repo-root> --pin-vf measured
"""

import argparse
import csv
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit

from bapsf_lapd.density import inter_sweep_sample_slices
from bapsf_lapd.manifest import load_run_manifest
from bapsf_lapd.reader import LapdRun

# The deep-bias set is ES3 and only ES3: it shares ES4's bank voltage and ramp
# duration and differs only in puff.  ES1 and ES2 move the bank and sweep a
# shorter ramp.
DEEP_EXPERIMENT_SET = 3
SHALLOW_EXPERIMENT_SET = 4

# Port/face pairings: each ES4 sweep run beside the ES3 run at the same port
# and probe rotation.
RUN_PAIRS = (
    (21, 0, "32", "42"),
    (21, 180, "33", "43"),
    (29, 0, "34", "44"),
    (29, 180, "35", "45"),
    (41, 0, "36", "46"),
    (41, 180, "37", "47"),
)

# Stable plasma plateau; only whole ramp or dead-time windows inside it are used.
PLATEAU_T_MIN_MS = 15.0
PLATEAU_T_MAX_MS = 19.5

# Dead-time samples to drop before averaging the rest bias.  The supply's
# flyback rings for a few microseconds after each ramp; this clears it with a
# wide margin, and the measured flyback duration is reported so the margin can
# be checked against the data.
REST_SETTLE_US = 25.0

# A cell counts as core if its ion current at the deepest measured bias is at
# least this fraction of the profile peak, on BOTH runs of a pair.
CORE_FRACTION = 0.5

# Upper edge of the ion-branch fit window.  Above this the electron current is
# no longer negligible against the ion current on these branches.
FIT_V_MAX = -10.0

# Voltage tolerance for calling the post-ramp flyback settled.
FLYBACK_SETTLED_V = 0.5

# Directory this instrument must never write into: it is a one-shot measurement,
# not a member of the product chain.
FORBIDDEN_OUTPUT_DIR = "processed"

# --pin-vf measured only.  The shot-averaged Langmuir product, relative to the
# repo-root given on the command line, and the plasma-potential estimators it
# carries.  The pinned mode anchors the sheath base to the knee (dI/dV)
# estimator and reports every estimator the product holds beside it.
VP_PRODUCT_RELPATH = Path("processed/langmuir_sweeps.hdf5")
VP_ESTIMATOR = "vp_derivative_v"
VP_ESTIMATORS_REPORTED = ("vp_derivative_v", "vp_log_v", "vp_exp_v")

# Half-width of the pinned gate, as a fraction above F_direct.
PINNED_GATE_TOLERANCE = 0.15


@dataclass(frozen=True)
class PowerLawFit:
    """|I_i| = amplitude * (v_float - V) ** exponent, fitted on one cell."""

    amplitude: float
    v_float: float
    exponent: float
    rms: float

    def evaluate(self, v: np.ndarray | float) -> np.ndarray | float:
        return self.amplitude * (self.v_float - np.asarray(v)) ** self.exponent

    def ratio(self, v_deep: float, v_shallow: float) -> float:
        """|I| at the deep bias over |I| at the shallow bias, both from the fit."""
        return float(self.evaluate(v_deep) / self.evaluate(v_shallow))


@dataclass(frozen=True)
class LinearFit:
    """|I_i| = intercept + slope * V, fitted on one cell."""

    intercept: float
    slope: float
    rms: float

    def evaluate(self, v: np.ndarray | float) -> np.ndarray | float:
        return self.intercept + self.slope * np.asarray(v)

    def ratio(self, v_deep: float, v_shallow: float) -> float:
        return float(self.evaluate(v_deep) / self.evaluate(v_shallow))


def fit_power_law(
    v: np.ndarray,
    i_abs: np.ndarray,
    *,
    exponent: float | None = None,
    v_float: float | None = None,
) -> PowerLawFit:
    """Least-squares fit of A (V_f - V)**p to one ion branch.

    ``exponent`` fixes p and leaves (A, V_f) free; ``None`` frees p as well.
    V_f is bounded above the fit window so the base stays positive; it is a
    fitted shape parameter of the chosen form, not an independently measured
    floating potential, and a branch flatter than the form predicts will push
    it far outside the swept range.

    ``v_float`` PINS that base potential instead of fitting it, which is what
    the ``--pin-vf measured`` mode does with a measured plasma potential.  With
    ``v_float`` pinned and ``exponent`` free the fit has two free parameters,
    the amplitude and the power; with both pinned only the amplitude is free
    and the resulting ratio between two biases carries no measured current at
    all.  A pin at or below the top of the fit window is refused, because the
    base ``(V_f - V)`` would not stay positive across the branch.
    """
    v = np.asarray(v, dtype=float)
    i_abs = np.asarray(i_abs, dtype=float)
    v_top = float(v.max())
    v_float_lower = v_top + 0.5

    if v_float is not None:
        if not float(v_float) > v_top:
            raise ValueError(
                f"pinned V_f = {float(v_float):.3f} V is not above the top of the fit "
                f"window ({v_top:.3f} V); the sheath base (V_f - V) would not stay "
                "positive across the branch"
            )
        v_float = float(v_float)
        base = v_float - v
        if exponent is not None:

            def model(x, amplitude):
                return amplitude * (v_float - x) ** exponent

            p0 = [float(i_abs.mean() / base.mean() ** exponent)]
            bounds = ([0.0], [np.inf])
            popt, _ = curve_fit(model, v, i_abs, p0=p0, bounds=bounds, maxfev=200000)
            amplitude, power = float(popt[0]), float(exponent)
        else:

            def model(x, amplitude, power_):
                return amplitude * (v_float - x) ** power_

            p0 = [float(i_abs.mean() / base.mean() ** 0.75), 0.75]
            bounds = ([0.0, 0.0], [np.inf, 5.0])
            popt, _ = curve_fit(model, v, i_abs, p0=p0, bounds=bounds, maxfev=200000)
            amplitude, power = float(popt[0]), float(popt[1])
        fit = PowerLawFit(amplitude, v_float, power, 0.0)
        rms = float(np.sqrt(np.mean((np.asarray(fit.evaluate(v)) - i_abs) ** 2)))
        return PowerLawFit(amplitude, v_float, power, rms)

    if exponent is not None:

        def model(x, amplitude, v_float):
            return amplitude * (v_float - x) ** exponent

        p0 = [float(i_abs.mean()), v_top + 20.0]
        bounds = ([0.0, v_float_lower], [np.inf, 1.0e5])
        popt, _ = curve_fit(model, v, i_abs, p0=p0, bounds=bounds, maxfev=200000)
        amplitude, v_float = float(popt[0]), float(popt[1])
        power = float(exponent)
    else:

        def model(x, amplitude, v_float, power_):
            return amplitude * (v_float - x) ** power_

        p0 = [float(i_abs.mean()), v_top + 20.0, 0.75]
        bounds = ([0.0, v_float_lower, 0.0], [np.inf, 1.0e5, 5.0])
        popt, _ = curve_fit(model, v, i_abs, p0=p0, bounds=bounds, maxfev=200000)
        amplitude, v_float, power = float(popt[0]), float(popt[1]), float(popt[2])

    fit = PowerLawFit(amplitude, v_float, power, 0.0)
    rms = float(np.sqrt(np.mean((np.asarray(fit.evaluate(v)) - i_abs) ** 2)))
    return PowerLawFit(amplitude, v_float, power, rms)


def fit_linear(v: np.ndarray, i_abs: np.ndarray) -> LinearFit:
    """Least-squares straight line through one ion branch."""
    v = np.asarray(v, dtype=float)
    i_abs = np.asarray(i_abs, dtype=float)
    slope, intercept = np.polyfit(v, i_abs, 1)
    fit = LinearFit(float(intercept), float(slope), 0.0)
    rms = float(np.sqrt(np.mean((np.asarray(fit.evaluate(v)) - i_abs) ** 2)))
    return LinearFit(float(intercept), float(slope), rms)


@dataclass(frozen=True)
class PinnedVFloat:
    """The measured plasma potential one pinned fit anchors its sheath base to.

    ``v_float`` is the value actually pinned, taken from ``estimator``;
    ``by_estimator`` carries every plasma-potential estimator the product holds
    over the same cells and time bins, so the estimator choice is visible
    beside the answer rather than buried in it.
    """

    v_float: float
    estimator: str
    n_cells: int
    n_time_bins: int
    cell_std: float
    by_estimator: dict = field(default_factory=dict)


def read_pinned_v_float(
    product_path: Path,
    run_id: str,
    cells: np.ndarray,
    *,
    n_positions: int,
    experiment_set: int = SHALLOW_EXPERIMENT_SET,
    estimator: str = VP_ESTIMATOR,
) -> PinnedVFloat:
    """Measured plasma potential of one run, over the plateau and the core cells.

    Averaged over exactly the cells and the plateau window the factor itself is
    measured on, so the pinned potential and the fitted branch describe the same
    plasma.  Every failure is a refusal naming what was missing: this mode must
    not fall back to a guessed potential.
    """
    product_path = Path(product_path)
    if not product_path.exists():
        raise ValueError(
            f"--pin-vf measured needs the shot-averaged Langmuir product at "
            f"{product_path}, which is not in this checkout; the pinned mode reads a "
            "measured plasma potential and has no fallback"
        )
    cells = np.asarray(cells, dtype=int)
    if cells.size == 0:
        raise ValueError(f"run {run_id}: no core cells to average the plasma potential over")
    group = f"experiment_sets/{experiment_set}/{run_id}"
    with h5py.File(product_path, "r") as handle:
        if group not in handle:
            raise ValueError(
                f"{product_path} has no {group}; the pinned mode needs the plasma "
                f"potential of run {run_id} in experiment set {experiment_set}"
            )
        node = handle[group]
        x_cm = np.asarray(handle["x_cm"][:], dtype=float)
        if x_cm.size != n_positions:
            raise ValueError(
                f"{product_path} is on a {x_cm.size}-position grid while run {run_id} "
                f"sweeps {n_positions} positions; the core-cell indices do not "
                "transfer between the two"
            )
        t_ms = np.asarray(node["cycle_time_s"][:], dtype=float) * 1e3
        available = [name for name in VP_ESTIMATORS_REPORTED if name in node]
        if estimator not in available:
            raise ValueError(
                f"{group} in {product_path} carries {available or 'no'} plasma-potential "
                f"estimator(s) and not {estimator}"
            )
        grids = {name: np.asarray(node[name][:], dtype=float) for name in available}

    in_window = (t_ms >= PLATEAU_T_MIN_MS - 1e-9) & (t_ms <= PLATEAU_T_MAX_MS + 1e-9)
    if not in_window.any():
        raise ValueError(
            f"{group}: no cycle-time bin lies inside the {PLATEAU_T_MIN_MS}-"
            f"{PLATEAU_T_MAX_MS} ms plateau the factor is measured over"
        )
    index = np.ix_(cells, np.flatnonzero(in_window))

    def cell_means(grid: np.ndarray) -> np.ndarray:
        block = grid[index]
        if not np.isfinite(block).any():
            raise ValueError(
                f"{group}: every plasma-potential sample over the plateau and the "
                f"{cells.size} core cells is NaN"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(block, axis=1)

    by_estimator = {}
    for name, grid in grids.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            by_estimator[name] = float(np.nanmean(cell_means(grid)))
    pinned_cells = cell_means(grids[estimator])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cell_std = float(np.nanstd(pinned_cells))
    return PinnedVFloat(
        v_float=by_estimator[estimator],
        estimator=estimator,
        n_cells=int(cells.size),
        n_time_bins=int(in_window.sum()),
        cell_std=cell_std,
        by_estimator=by_estimator,
    )


def pinned_gate_band(direct: float) -> tuple[float, float]:
    """The pre-registered acceptance band for the pinned factor."""
    low = float(direct)
    return low, low * (1.0 + PINNED_GATE_TOLERANCE)


def pinned_gate_verdict(direct: float, pinned: float | None) -> str:
    """The pinned gate's one-line verdict; never a NaN and never a blank bin."""
    low, high = pinned_gate_band(direct)
    if pinned is None or not np.isfinite(pinned):
        return "REFUSED (no fit)"
    if low <= float(pinned) <= high:
        return "inside -- the pinned fit supports the direct factor"
    return (
        f"outside -- the direct factor stands alone (F_pinned {float(pinned):.4f} "
        f"vs [{low:.4f}, {high:.4f}])"
    )


def bracket_contains(direct: float, linear: float, candidate: float) -> bool:
    """Whether the candidate factor lies between the two bracket ends."""
    low, high = sorted((float(direct), float(linear)))
    return low <= float(candidate) <= high


def bracket_position(direct: float, linear: float, candidate: float) -> float:
    """Where the candidate sits on the direct-to-linear bracket, 0 at direct.

    1 puts it on the linear end.  This is the number that says whether the gate
    discriminated between the two ends or merely tracked one of them.
    """
    span = float(linear) - float(direct)
    if span == 0.0:
        return float("nan")
    return (float(candidate) - float(direct)) / span


def uniform_ramp_length(slices, run_id: str) -> int:
    """Common sample count of a run's ramp slices, or a clear refusal.

    A ramp whose duration is not a whole number of samples produces slices that
    alternate in length by one -- ES1 and ES2 sweep 250 us at 0.16 us per
    sample, which is 1562.5 -- and those cannot be stacked into one branch.
    """
    lengths = {s.stop - s.start for s in slices}
    if len(lengths) != 1:
        raise ValueError(
            f"Run {run_id}: ramp windows are {sorted(lengths)} samples long, so they "
            "cannot be averaged into one branch; the ramp duration is not a whole "
            "number of samples for this run"
        )
    return lengths.pop()


def plateau_cycle_indices(sweep, *, window: str) -> list[int]:
    """Cycles whose ramp (or dead-time) window lies wholly inside the plateau."""
    if window not in {"ramp", "dead"}:
        raise ValueError("window must be 'ramp' or 'dead'")
    indices = []
    for k in range(sweep.n_cycles):
        cycle_start_s = sweep.t0_s + k * sweep.tau_cycle_s
        if window == "ramp":
            start_s, stop_s = cycle_start_s, cycle_start_s + sweep.tau_ramp_s
        else:
            start_s = cycle_start_s + sweep.tau_ramp_s
            stop_s = cycle_start_s + sweep.tau_cycle_s
        if start_s * 1e3 >= PLATEAU_T_MIN_MS - 1e-9 and stop_s * 1e3 <= PLATEAU_T_MAX_MS + 1e-9:
            indices.append(k)
    return indices


@dataclass
class RunSweep:
    """One run's plateau-averaged ramp branch and measured rest bias."""

    run_id: str
    port: int
    rotation_deg: float
    experiment_set: int
    v_ramp: np.ndarray            # (n_ramp_samples,) mean ramp voltage
    i_ramp_abs: np.ndarray        # (n_positions, n_ramp_samples) mean |ion current|
    v_rest: float
    v_rest_std: float
    v_parked: float               # the configured anchor the offset pins the trace to
    v_offset_applied: float       # the offset the convention subtracts, in probe volts
    n_ramps: int
    n_dead_windows: int
    n_shots_per_position: int
    flyback_us: float

    @property
    def v_departure(self) -> float:
        """Measured rise of the dead-time rest level above the parked anchor."""
        return self.v_rest - self.v_parked

    @property
    def v_rest_raw_frame(self) -> float:
        """The same rest level read in the raw digitizer frame, before the offset."""
        return self.v_rest + self.v_offset_applied


def read_run_sweep(config) -> RunSweep:
    """Average one run's plateau ramps and measure its dead-time rest bias."""
    run = LapdRun(config)
    dt = config.acquisition.sample_dt_s
    ramp_slices = run.sweep_ramp_sample_slices()
    dead_slices = inter_sweep_sample_slices(config.sweep, config.acquisition)
    ramp_k = plateau_cycle_indices(config.sweep, window="ramp")
    dead_k = plateau_cycle_indices(config.sweep, window="dead")
    if not ramp_k or not dead_k:
        raise ValueError(f"Run {config.run_id} has no whole cycle inside the plateau")
    uniform_ramp_length([ramp_slices[k] for k in ramp_k], config.run_id)

    # One contiguous read per channel spanning every window this run needs, with
    # the zero offset resolved once instead of per slice.
    span_start = min(ramp_slices[ramp_k[0]].start, dead_slices[dead_k[0]].start)
    span_stop = max(ramp_slices[ramp_k[-1]].stop, dead_slices[dead_k[-1]].stop)
    span = slice(span_start, span_stop)
    v_offset_pre = run.default_zero_offset_v("v_sweep")
    v_all = run.langmuir_traces("v_sweep", span, zero_offset_v=v_offset_pre)
    i_all = run.langmuir_traces(
        "i_sweep", span, zero_offset_v=run.default_zero_offset_v("i_sweep")
    )

    def local(s: slice) -> slice:
        return slice(s.start - span_start, s.stop - span_start)

    settle = int(round(REST_SETTLE_US * 1e-6 / dt))
    rest_means = [float(v_all[:, :, local(dead_slices[k])][:, :, settle:].mean()) for k in dead_k]

    v_ramp = np.mean([v_all[:, :, local(ramp_slices[k])].mean(axis=(0, 1)) for k in ramp_k], axis=0)
    i_ramp = np.mean([i_all[:, :, local(ramp_slices[k])].mean(axis=1) for k in ramp_k], axis=0)

    # The supply flies back before the configured ramp window closes; drop the
    # samples past the ramp peak so the branch is the swept part only.
    peak = int(np.argmax(v_ramp))
    v_ramp = v_ramp[: peak + 1]
    i_ramp_abs = -i_ramp[:, : peak + 1]

    v_rest = float(np.mean(rest_means))
    flyback_us = _flyback_duration_us(
        v_all[:, :, local(ramp_slices[ramp_k[0]])].mean(axis=(0, 1)),
        v_all[:, :, local(dead_slices[dead_k[0]])].mean(axis=(0, 1)),
        v_rest,
        dt,
    )
    return RunSweep(
        run_id=config.run_id,
        port=config.probe.port,
        rotation_deg=config.probe.rotation_deg,
        experiment_set=config.experiment_set.id,
        v_ramp=v_ramp,
        i_ramp_abs=i_ramp_abs,
        v_rest=v_rest,
        v_rest_std=float(np.std(rest_means)),
        v_parked=float(config.sweep.voltage_start),
        v_offset_applied=float(v_offset_pre * config.channel("v_sweep").multiplier),
        n_ramps=len(ramp_k),
        n_dead_windows=len(dead_k),
        n_shots_per_position=config.acquisition.n_shots_per_position,
        flyback_us=flyback_us,
    )


def _flyback_duration_us(
    v_ramp_full: np.ndarray, v_dead: np.ndarray, v_rest: float, dt: float
) -> float:
    """Microseconds from the ramp peak until the bias settles at the rest level."""
    tail = np.concatenate([v_ramp_full[int(np.argmax(v_ramp_full)) :], v_dead])
    settled = np.flatnonzero(np.abs(tail - v_rest) < FLYBACK_SETTLED_V)
    if settled.size == 0:
        return float("nan")
    return float(settled[0] * dt * 1e6)


def direct_ratio(sweep: RunSweep, cell: int, v_deep: float, v_shallow: float) -> float:
    """Ratio read straight off one measured ramp branch, with no fitted form."""
    order = np.argsort(sweep.v_ramp)
    v = sweep.v_ramp[order]
    i_abs = sweep.i_ramp_abs[cell, order]
    return float(np.interp(v_deep, v, i_abs) / np.interp(v_shallow, v, i_abs))


def core_cells(deep: RunSweep, shallow: RunSweep) -> np.ndarray:
    """Positions whose ion current is within CORE_FRACTION of peak on both runs."""
    def mask(sweep: RunSweep) -> np.ndarray:
        peak_profile = sweep.i_ramp_abs[:, 0]
        return peak_profile >= CORE_FRACTION * peak_profile.max()

    return np.flatnonzero(mask(deep) & mask(shallow))


def analyse_pair(
    deep: RunSweep,
    shallow: RunSweep,
    fit_v_min: float | None,
    pin_product: Path | None = None,
) -> dict:
    """Measure the factor both ways for one port/face pair.

    ``pin_product`` is the shot-averaged Langmuir product to read a measured
    plasma potential from; ``None`` is the default invocation, which reads no
    product and returns exactly the columns it always returned.
    """
    if deep.experiment_set != DEEP_EXPERIMENT_SET:
        raise ValueError(
            f"Run {deep.run_id} is in experiment set {deep.experiment_set}; the deep-bias "
            f"run must be in set {DEEP_EXPERIMENT_SET}, the only set that shares ES4's "
            "bank voltage and ramp duration"
        )
    if shallow.experiment_set != SHALLOW_EXPERIMENT_SET:
        raise ValueError(
            f"Run {shallow.run_id} is in experiment set {shallow.experiment_set}; the "
            f"shallow-bias run must be in set {SHALLOW_EXPERIMENT_SET}"
        )
    v_deep = deep.v_rest
    v_shallow = shallow.v_rest
    cells = core_cells(deep, shallow)
    window_low = float(shallow.v_ramp.min()) if fit_v_min is None else float(fit_v_min)
    in_window = (shallow.v_ramp >= window_low) & (shallow.v_ramp <= FIT_V_MAX)
    if in_window.sum() < 4:
        raise ValueError(
            f"Run {shallow.run_id}: fit window [{window_low}, {FIT_V_MAX}] V holds "
            f"{int(in_window.sum())} ramp samples; the branch does not reach it"
        )
    v_fit = shallow.v_ramp[in_window]

    pin = None
    if pin_product is not None:
        pin = read_pinned_v_float(
            pin_product,
            shallow.run_id,
            cells,
            n_positions=int(shallow.i_ramp_abs.shape[0]),
            experiment_set=shallow.experiment_set,
        )

    # Per-cell pinned results, collected only when the mode is on.  Every list
    # is indexed by CONVERGED cell, and the free fit's residual is carried
    # alongside so the control comparison is over exactly those cells.
    per_cell = {
        key: []
        for key in ("ratio", "ratio_p075", "exponent", "rms", "rms_p075", "rms_free")
    }
    n_cells_pin_failed = 0

    direct, eta34, free, linear = [], [], [], []
    v_float_34, v_float_free, exponent_free = [], [], []
    rms_34, rms_free, rms_linear, slope = [], [], [], []
    for cell in cells:
        direct.append(direct_ratio(deep, cell, v_deep, v_shallow))
        i_fit = shallow.i_ramp_abs[cell, in_window]

        fit34 = fit_power_law(v_fit, i_fit, exponent=0.75)
        eta34.append(fit34.ratio(v_deep, v_shallow))
        v_float_34.append(fit34.v_float)
        rms_34.append(fit34.rms)

        fitfree = fit_power_law(v_fit, i_fit, exponent=None)
        free.append(fitfree.ratio(v_deep, v_shallow))
        v_float_free.append(fitfree.v_float)
        exponent_free.append(fitfree.exponent)
        rms_free.append(fitfree.rms)

        fitlin = fit_linear(v_fit, i_fit)
        linear.append(fitlin.ratio(v_deep, v_shallow))
        slope.append(fitlin.slope)
        rms_linear.append(fitlin.rms)

        if pin is not None:
            try:
                fitpin = fit_power_law(v_fit, i_fit, v_float=pin.v_float)
                fitpin75 = fit_power_law(
                    v_fit, i_fit, exponent=0.75, v_float=pin.v_float
                )
            except (RuntimeError, ValueError):
                n_cells_pin_failed += 1
            else:
                per_cell["ratio"].append(fitpin.ratio(v_deep, v_shallow))
                per_cell["exponent"].append(fitpin.exponent)
                per_cell["rms"].append(fitpin.rms)
                per_cell["rms_free"].append(fitfree.rms)
                per_cell["ratio_p075"].append(fitpin75.ratio(v_deep, v_shallow))
                per_cell["rms_p075"].append(fitpin75.rms)

    def stat(values):
        return float(np.mean(values)), float(np.std(values))

    f_direct, f_direct_std = stat(direct)
    f_eta34, f_eta34_std = stat(eta34)
    f_free, f_free_std = stat(free)
    f_linear, f_linear_std = stat(linear)
    low, high = sorted((f_direct, f_linear))
    row = {
        "port": shallow.port,
        "rotation_deg": int(shallow.rotation_deg),
        "run_deep": deep.run_id,
        "run_shallow": shallow.run_id,
        "experiment_set_deep": deep.experiment_set,
        "experiment_set_shallow": shallow.experiment_set,
        "v_parked_deep_v": deep.v_parked,
        "v_offset_applied_deep_v": deep.v_offset_applied,
        "v_departure_deep_v": deep.v_departure,
        "v_rest_deep_v": v_deep,
        "v_rest_deep_std_v": deep.v_rest_std,
        "v_rest_deep_raw_frame_v": deep.v_rest_raw_frame,
        "v_parked_shallow_v": shallow.v_parked,
        "v_offset_applied_shallow_v": shallow.v_offset_applied,
        "v_departure_shallow_v": shallow.v_departure,
        "v_rest_shallow_v": v_shallow,
        "v_rest_shallow_std_v": shallow.v_rest_std,
        "v_rest_shallow_raw_frame_v": shallow.v_rest_raw_frame,
        "ramp_min_deep_v": float(deep.v_ramp.min()),
        "ramp_max_deep_v": float(deep.v_ramp.max()),
        "ramp_min_shallow_v": float(shallow.v_ramp.min()),
        "ramp_max_shallow_v": float(shallow.v_ramp.max()),
        "flyback_us_deep": deep.flyback_us,
        "flyback_us_shallow": shallow.flyback_us,
        "fit_v_min": window_low,
        "fit_v_max": FIT_V_MAX,
        "n_fit_samples": int(in_window.sum()),
        "n_ramps_deep": deep.n_ramps,
        "n_ramps_shallow": shallow.n_ramps,
        "n_dead_windows_deep": deep.n_dead_windows,
        "n_dead_windows_shallow": shallow.n_dead_windows,
        "n_cells": int(cells.size),
        "n_shots_per_cell_deep": deep.n_shots_per_position * deep.n_ramps,
        "n_shots_per_cell_shallow": shallow.n_shots_per_position * shallow.n_ramps,
        "f_direct_up": f_direct,
        "f_direct_up_std": f_direct_std,
        "inv_f_direct_up": 1.0 / f_direct,
        "f_eta34": f_eta34,
        "f_eta34_std": f_eta34_std,
        "v_float_eta34": float(np.mean(v_float_34)),
        "rms_eta34_a": float(np.mean(rms_34)),
        "f_free": f_free,
        "f_free_std": f_free_std,
        "v_float_free": float(np.mean(v_float_free)),
        "exponent_free": float(np.mean(exponent_free)),
        "rms_free_a": float(np.mean(rms_free)),
        "f_linear": f_linear,
        "f_linear_std": f_linear_std,
        "linear_slope_a_per_v": float(np.mean(slope)),
        "rms_linear_a": float(np.mean(rms_linear)),
        "bracket_low": low,
        "bracket_high": high,
        "eta34_bracket_position": bracket_position(f_direct, f_linear, f_eta34),
        "n_cells_eta34_below_linear": int(np.sum(np.array(eta34) < np.array(linear))),
        "eta34_in_bracket": bracket_contains(f_direct, f_linear, f_eta34),
    }
    if pin is None:
        return row
    return row | _pinned_columns(pin, f_direct, per_cell, n_cells_pin_failed)


def _pinned_columns(
    pin: PinnedVFloat,
    f_direct: float,
    per_cell: dict,
    n_cells_pin_failed: int,
) -> dict:
    """The columns --pin-vf measured adds, and the only columns it adds.

    ``per_cell`` holds one entry per core cell whose pinned fit converged.  A
    pair on which none converged still returns every column, with the gate bin
    carrying the REFUSED verdict rather than a blank or a NaN.
    """
    nan = float("nan")
    converged = per_cell["ratio"]
    if converged:
        f_pinned = float(np.mean(converged))
        f_pinned_std = float(np.std(converged))
        f_pinned_p075 = float(np.mean(per_cell["ratio_p075"]))
        f_pinned_p075_std = float(np.std(per_cell["ratio_p075"]))
        exponent = float(np.mean(per_cell["exponent"]))
        rms_pin = float(np.mean(per_cell["rms"]))
        rms_pin_p075 = float(np.mean(per_cell["rms_p075"]))
        rms_ref = float(np.mean(per_cell["rms_free"]))
        rms_ratio = rms_pin / rms_ref if rms_ref > 0.0 else nan
        verdict = pinned_gate_verdict(f_direct, f_pinned)
    else:
        f_pinned = f_pinned_std = f_pinned_p075 = f_pinned_p075_std = nan
        exponent = rms_pin = rms_pin_p075 = rms_ref = rms_ratio = nan
        verdict = pinned_gate_verdict(f_direct, None)
    low, high = pinned_gate_band(f_direct)
    columns = {
        "vp_pin_estimator": pin.estimator,
        "vp_pin_v": pin.v_float,
        "vp_pin_cell_std_v": pin.cell_std,
        "vp_pin_n_cells": pin.n_cells,
        "vp_pin_n_time_bins": pin.n_time_bins,
    }
    for name in VP_ESTIMATORS_REPORTED:
        columns[f"vp_pin_{name}"] = pin.by_estimator.get(name, nan)
    columns.update(
        {
            "f_pinned": f_pinned,
            "f_pinned_std": f_pinned_std,
            "exponent_pinned": exponent,
            "rms_pinned_a": rms_pin,
            "rms_free_pinned_cells_a": rms_ref,
            "rms_pinned_over_rms_free": rms_ratio,
            "f_pinned_p075": f_pinned_p075,
            "f_pinned_p075_std": f_pinned_p075_std,
            "rms_pinned_p075_a": rms_pin_p075,
            "n_cells_pinned_fit": len(converged),
            "n_cells_pinned_fit_failed": int(n_cells_pin_failed),
            "pinned_gate_low": low,
            "pinned_gate_high": high,
            "pinned_in_gate": bool(converged) and low <= f_pinned <= high,
            "pinned_gate_verdict": verdict,
        }
    )
    return columns


def checked_output_path(path: Path) -> Path:
    """Refuse an output path that resolves inside the product directory."""
    resolved = path.expanduser().resolve()
    if FORBIDDEN_OUTPUT_DIR in resolved.parts:
        raise ValueError(
            f"--output {path} resolves to {resolved}, inside a {FORBIDDEN_OUTPUT_DIR}/ "
            "directory; this instrument writes measurements, not products, and must "
            "not write into the product chain"
        )
    return resolved


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "repo_root",
        type=Path,
        help="checkout holding config/may2026_run_manifest.toml and the raw runs",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="raw run directory (default: <repo-root>/data/may2026 per the manifest)",
    )
    parser.add_argument(
        "--fit-v-min",
        type=float,
        default=None,
        help="lower edge of the ES4 ion-branch fit window in volts "
        "(default: the deepest bias that ramp actually reaches)",
    )
    parser.add_argument(
        "--pin-vf",
        choices=("off", "measured"),
        default="off",
        help="pin the sheath-expansion base potential to the measured plasma "
        f"potential in {VP_PRODUCT_RELPATH} instead of fitting it (default: off, "
        "which reads no product and changes no output)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("es4_sweep_rest_bias_factor.csv"),
        help="CSV path; defaults to the working directory, and a path inside "
        f"{FORBIDDEN_OUTPUT_DIR}/ is refused",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = checked_output_path(args.output)
    manifest_path = args.repo_root / "config/may2026_run_manifest.toml"
    data_dir = args.data_dir or args.repo_root / "data/may2026"
    configs = load_run_manifest(manifest_path, data_dir=data_dir)
    pin_product = (
        args.repo_root / VP_PRODUCT_RELPATH if args.pin_vf == "measured" else None
    )

    print(f"manifest   {manifest_path}")
    print(f"raw runs   {data_dir}")
    print(f"plateau    {PLATEAU_T_MIN_MS}-{PLATEAU_T_MAX_MS} ms, whole cycles only")
    print(f"rest bias  mean v_sweep over each dead-time window past {REST_SETTLE_US:.0f} us of settling")
    print(f"core cells |I_i| >= {CORE_FRACTION:.2f} of profile peak on both runs of a pair")
    print(f"fit window [fit-v-min, {FIT_V_MAX:.1f}] V on the ES4 ion branch")
    print()
    print("The rest bias is an ANCHOR plus a MEASURED DEPARTURE: the zero-offset")
    print("convention pins the parked DC to the manifest's sweep voltage_start, and")
    print("only the departure from it is measured here.  Both are printed per run,")
    print("with the same rest level also given in the raw digitizer frame.")
    print()
    print("F = |I_i(ES3 rest bias)| / |I_i(ES4 rest bias)|  (>= 1)")
    print("1/F is the same factor with the ratio taken the other way (<= 1).")
    print()
    if pin_product is not None:
        print(f"pinned V_f {pin_product}")
        print(
            f"           estimator {VP_ESTIMATOR}, averaged over the same plateau and "
            "the same core cells"
        )
        print(
            "           the PLASMA potential is pinned and the amplitude and power are "
            "free; F_pinned(p=3/4) additionally holds the power at 3/4, which leaves no "
            "measured current in the ratio at all"
        )
        print(
            f"GATE(pinned): F_pinned inside [F_direct, "
            f"{1.0 + PINNED_GATE_TOLERANCE:.2f} x F_direct] supports the direct factor"
        )
        print()

    rows = []
    for port, rotation, run_deep, run_shallow in RUN_PAIRS:
        deep = read_run_sweep(configs[run_deep])
        shallow = read_run_sweep(configs[run_shallow])
        if deep.port != port or shallow.port != port:
            raise ValueError(f"manifest ports disagree with the pairing for port {port}")
        rows.append(analyse_pair(deep, shallow, args.fit_v_min, pin_product))
        row = rows[-1]
        print(
            f"port {port:>2} rot {rotation:>3}  runs {run_deep}/{run_shallow}  "
            f"cells {row['n_cells']:>2}  ramps {row['n_ramps_deep']}/{row['n_ramps_shallow']}  "
            f"dead {row['n_dead_windows_deep']}/{row['n_dead_windows_shallow']}  "
            f"shots/cell {row['n_shots_per_cell_deep']}/{row['n_shots_per_cell_shallow']}"
        )
        print(
            f"    ES3 bias    parked {row['v_parked_deep_v']:7.2f} V   offset applied "
            f"{row['v_offset_applied_deep_v']:7.2f} V   departure "
            f"{row['v_departure_deep_v']:+.3f} V   ->  rest {row['v_rest_deep_v']:8.3f} "
            f"+- {row['v_rest_deep_std_v']:.3f} V   (raw frame "
            f"{row['v_rest_deep_raw_frame_v']:.3f} V)"
        )
        print(
            f"    ES4 bias    parked {row['v_parked_shallow_v']:7.2f} V   offset applied "
            f"{row['v_offset_applied_shallow_v']:7.2f} V   departure "
            f"{row['v_departure_shallow_v']:+.3f} V   ->  rest {row['v_rest_shallow_v']:8.3f} "
            f"+- {row['v_rest_shallow_std_v']:.3f} V   (raw frame "
            f"{row['v_rest_shallow_raw_frame_v']:.3f} V)"
        )
        print(
            f"    ramp span   ES3 [{row['ramp_min_deep_v']:.2f}, {row['ramp_max_deep_v']:.2f}] V   "
            f"ES4 [{row['ramp_min_shallow_v']:.2f}, {row['ramp_max_shallow_v']:.2f}] V   "
            f"flyback {row['flyback_us_deep']:.1f}/{row['flyback_us_shallow']:.1f} us"
        )
        print(
            f"    fit window  [{row['fit_v_min']:.2f}, {row['fit_v_max']:.2f}] V, "
            f"{row['n_fit_samples']} ramp samples"
        )
        print(
            f"    F_direct  {row['f_direct_up']:.4f} +- {row['f_direct_up_std']:.4f}   "
            f"(1/F = {row['inv_f_direct_up']:.4f})"
        )
        print(
            f"    F_eta34   {row['f_eta34']:.4f} +- {row['f_eta34_std']:.4f}   "
            f"V_f {row['v_float_eta34']:9.2f} V   p 0.750   rms {row['rms_eta34_a']:.2e} A"
        )
        print(
            f"    F_free    {row['f_free']:.4f} +- {row['f_free_std']:.4f}   "
            f"V_f {row['v_float_free']:9.2f} V   p {row['exponent_free']:.3f}   "
            f"rms {row['rms_free_a']:.2e} A"
        )
        print(
            f"    F_linear  {row['f_linear']:.4f} +- {row['f_linear_std']:.4f}   "
            f"slope {row['linear_slope_a_per_v']:.3e} A/V   rms {row['rms_linear_a']:.2e} A"
        )
        print(
            f"    GATE  bracket [{row['bracket_low']:.4f}, {row['bracket_high']:.4f}] "
            f"vs F_eta34 {row['f_eta34']:.4f}  -> "
            f"{'INSIDE' if row['eta34_in_bracket'] else 'OUTSIDE'}   "
            f"(at {100 * row['eta34_bracket_position']:.0f}% of the way from direct to "
            f"linear; below F_linear in {row['n_cells_eta34_below_linear']}/{row['n_cells']} cells)"
        )
        if pin_product is not None:
            print(
                f"    V_f pinned  {row['vp_pin_v']:+.3f} +- {row['vp_pin_cell_std_v']:.3f} V "
                f"({row['vp_pin_estimator']}, {row['vp_pin_n_cells']} cells x "
                f"{row['vp_pin_n_time_bins']} plateau bins)   estimators: "
                + "  ".join(
                    f"{name} {row[f'vp_pin_{name}']:+.2f} V"
                    for name in VP_ESTIMATORS_REPORTED
                )
            )
            print(
                f"    F_pinned  {row['f_pinned']:.4f} +- {row['f_pinned_std']:.4f}   "
                f"V_f {row['vp_pin_v']:9.2f} V (pinned)   p {row['exponent_pinned']:.3f}   "
                f"rms {row['rms_pinned_a']:.2e} A   [free-fit control rms "
                f"{row['rms_free_pinned_cells_a']:.2e} A, ratio "
                f"{row['rms_pinned_over_rms_free']:.3f}]   "
                f"cells {row['n_cells_pinned_fit']}/{row['n_cells']} fitted"
            )
            print(
                f"    F_pinned(p=3/4)  {row['f_pinned_p075']:.4f} +- "
                f"{row['f_pinned_p075_std']:.4f}   rms {row['rms_pinned_p075_a']:.2e} A   "
                "(diagnostic: no measured current enters this ratio)"
            )
            print(
                f"    GATE(pinned)  band [{row['pinned_gate_low']:.4f}, "
                f"{row['pinned_gate_high']:.4f}] vs F_pinned {row['f_pinned']:.4f}  -> "
                f"{row['pinned_gate_verdict']}"
            )
        print()

    inside = sum(1 for row in rows if row["eta34_in_bracket"])
    positions = [100 * row["eta34_bracket_position"] for row in rows]
    # The two bracket ends only; the free exponent is a diagnostic, not an end.
    extrapolations = [
        value for row in rows for value in (row["f_eta34"], row["f_linear"])
    ]
    print(f"GATE: eta^(3/4) inside the bracket at {inside} of {len(rows)} port/face pairs")
    if inside == len(rows):
        print("GATE VERDICT: the factor is bracketed")
    else:
        print("GATE VERDICT: the bracket is the claim and no product is rescaled")
    print(
        f"THE GATE DOES NOT DISCRIMINATE: eta^(3/4) sits at {min(positions):.0f}-"
        f"{max(positions):.0f}% of the way from the direct end of the bracket to the "
        "linear end, so it tracks the linear extrapolation rather than testing it."
    )
    print(
        "The finding is that the direct route "
        f"({min(row['f_direct_up'] for row in rows):.2f}-"
        f"{max(row['f_direct_up'] for row in rows):.2f}) and both extrapolations "
        f"({min(extrapolations):.2f}-{max(extrapolations):.2f}) disagree by more than "
        "either one's spread.  The bracket is the claim."
    )
    if pin_product is not None:
        print()
        verdicts = [row["pinned_gate_verdict"] for row in rows]
        n_inside = sum(1 for verdict in verdicts if verdict.startswith("inside"))
        n_refused = sum(1 for verdict in verdicts if verdict.startswith("REFUSED"))
        print(
            f"GATE(pinned): inside at {n_inside} of {len(rows)} port/face pairs, "
            f"refused at {n_refused}"
        )
        ratios = [row["rms_pinned_over_rms_free"] for row in rows]
        finite = [value for value in ratios if np.isfinite(value)]
        if finite:
            print(
                f"Pinned-fit RMS residual is {min(finite):.3f}-{max(finite):.3f} times "
                "the free fit's over the same cells; the free fit carries one more free "
                "parameter and is the discrimination control, so a materially worse "
                "pinned residual is the finding and not a failure of the mode."
            )
        exponents = [row["exponent_pinned"] for row in rows if np.isfinite(row["exponent_pinned"])]
        if exponents:
            print(
                f"With the base pinned at the measured plasma potential the branch shows "
                f"p = {min(exponents):.3f}-{max(exponents):.3f}, against the 3/4 the "
                "sheath-expansion form assumes."
            )
    print("This script rescales no product either way; it writes numbers only.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
