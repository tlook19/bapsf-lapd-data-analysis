"""Channel-scale factor between the ES4 and ES1-3 sweep rest biases.

What this measures
------------------
Between voltage ramps the Langmuir sweep supply parks the probe at a fixed
negative rest bias, and it is that dead-time current -- not the ramp -- that
every ion-saturation product is built from.  The high-puff experiment set
sweeps a nominal +-20 V ramp on a 3 ohm sense resistor while the other three
sets sweep +-75 V on 1 ohm, so the two families park at DIFFERENT rest biases.
Ion current on a probe grows with the sheath, so a dead-time current measured
at the shallower bias is not on the same scale as one measured at the deeper
bias, and comparing the two families cell for cell needs a factor

  F = |I_i(V_rest, deep family)| / |I_i(V_rest, shallow family)|      (>= 1)

This script measures F.  The reciprocal 1/F = |I_i(shallow)| / |I_i(deep)|
(<= 1) is the same statement read the other way and is printed beside it; both
conventions appear in the table so a reader cannot mistake one for the other.

Both rest biases are MEASURED here, not assumed.  The script averages
``v_sweep`` over the settled part of each dead-time window inside the plateau
and reports the level it finds, per run.  That measured pair (V_deep,
V_shallow) is what both methods below are evaluated at, so the two are
comparable by construction.

Two independent routes to the same factor
-----------------------------------------
(i) DIRECT, from the deep-family ramp.  The +-75 V ramp sweeps through BOTH
    rest biases, so the ratio can simply be read off one measured ion branch
    with no functional form at all:

      F_direct = |I(V_deep)| / |I(V_shallow)|   interpolated on that ramp

    This is the least model-dependent number the data can give.

(ii) EXTRAPOLATED, from the shallow-family ramp.  The +-20 V ramp never
    reaches the deep bias, so its ion branch must be extended.  Three
    extensions are fitted over the same window and each is evaluated as a
    ratio between the same two biases:

      orbital-motion sheath expansion   |I| = A (V_f - V)^(3/4)
      the same form with the exponent free    |I| = A (V_f - V)^p
      a straight line                          |I| = a + b V

    The 3/4 form and the straight line are the two ends the bracket is drawn
    between; the free exponent says what power the branch actually shows.

What the fits can and cannot resolve
------------------------------------
The shallow family's ion branch spans only a few volts of sheath, so its
current changes by a few percent across the whole fit window.  Every one of the
three forms can be drawn through that with residuals of the same size, and the
printed per-model RMS says so directly.  The consequence is that ``V_f`` and a
free exponent are only weakly determined -- the 3/4 form buys its shape by
placing ``V_f`` wherever it must, including far above the swept range, which is
how a form with a 3/4 power ends up nearly straight over the window.  The
fitted ``V_f`` and exponent are reported for exactly that reason: they are the
diagnostic of how much of each extrapolation is the data and how much is the
form.  They are fitted shape parameters, not measurements of a floating
potential or of a sheath scaling.

The transfer is an ASSUMPTION
-----------------------------
Route (i) reads its ratio off a DIFFERENT plasma from the one the factor is
applied to: the deep-family ramp belongs to its own experiment set, at its own
bank voltage and puff.  Using it as the shallow family's factor assumes the
shape of the ion branch -- not its level -- transfers between the two plasmas.
Nothing in this data proves that.  It is stated here because the whole
comparison rests on it, and it is why route (ii), which stays inside the
shallow family's own ramp, is computed at all.

Pre-registered gate
-------------------
With the bracket drawn between the direct ratio and the linear extrapolation:

  the 3/4 ratio lands INSIDE the bracket at every port/face
      -> "the factor is bracketed"
  it lands OUTSIDE at any port/face
      -> "the bracket is the claim and no product is rescaled"

The script prints which side each pair lands on and the verdict sentence.  It
does not interpret them.  NO PRODUCT IS RESCALED BY THIS SCRIPT EITHER WAY --
it writes one CSV of numbers and nothing else, and its default output path is
deliberately outside ``processed/`` so it cannot enter the scoring chain.

Inputs
------
  config/may2026_run_manifest.toml    sweep schedule, channels, calibration
  data/may2026/*.hdf5                 raw runs, opened read-only

Both are read from the ``repo-root`` given on the command line, which is what
lets this run from a worktree against a checkout that holds the raw data.

Usage
-----
  python scripts/es4_sweep_rest_bias_factor.py <repo-root> [--output out.csv]
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from bapsf_lapd.density import inter_sweep_sample_slices
from bapsf_lapd.manifest import load_run_manifest
from bapsf_lapd.reader import LapdRun

# Port/face pairings: each shallow-family sweep run beside the deep-family run
# that sits at the same port and probe rotation.
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
) -> PowerLawFit:
    """Least-squares fit of A (V_f - V)**p to one ion branch.

    ``exponent`` fixes p and leaves (A, V_f) free; ``None`` frees p as well.
    V_f is bounded above the fit window so the base stays positive; it is a
    fitted shape parameter of the chosen form, not an independently measured
    floating potential, and a branch flatter than the form predicts will push
    it far outside the swept range.
    """
    v = np.asarray(v, dtype=float)
    i_abs = np.asarray(i_abs, dtype=float)
    v_top = float(v.max())
    v_float_lower = v_top + 0.5

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


def bracket_contains(direct: float, linear: float, candidate: float) -> bool:
    """Whether the candidate factor lies between the two bracket ends."""
    low, high = sorted((float(direct), float(linear)))
    return low <= float(candidate) <= high


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
    n_ramps: int
    n_dead_windows: int
    n_shots_per_position: int
    flyback_us: float


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

    # One contiguous read per channel spanning every window this run needs, with
    # the zero offset resolved once instead of per slice.
    span_start = min(ramp_slices[ramp_k[0]].start, dead_slices[dead_k[0]].start)
    span_stop = max(ramp_slices[ramp_k[-1]].stop, dead_slices[dead_k[-1]].stop)
    span = slice(span_start, span_stop)
    v_all = run.langmuir_traces(
        "v_sweep", span, zero_offset_v=run.default_zero_offset_v("v_sweep")
    )
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


def analyse_pair(deep: RunSweep, shallow: RunSweep, fit_v_min: float | None) -> dict:
    """Measure the factor both ways for one port/face pair."""
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

    def stat(values):
        return float(np.mean(values)), float(np.std(values))

    f_direct, f_direct_std = stat(direct)
    f_eta34, f_eta34_std = stat(eta34)
    f_free, f_free_std = stat(free)
    f_linear, f_linear_std = stat(linear)
    low, high = sorted((f_direct, f_linear))
    return {
        "port": shallow.port,
        "rotation_deg": int(shallow.rotation_deg),
        "run_deep": deep.run_id,
        "run_shallow": shallow.run_id,
        "experiment_set_deep": deep.experiment_set,
        "experiment_set_shallow": shallow.experiment_set,
        "v_rest_deep_v": v_deep,
        "v_rest_deep_std_v": deep.v_rest_std,
        "v_rest_shallow_v": v_shallow,
        "v_rest_shallow_std_v": shallow.v_rest_std,
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
        "eta34_in_bracket": bracket_contains(f_direct, f_linear, f_eta34),
    }


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
        help="lower edge of the shallow-family ion-branch fit window in volts "
        "(default: the deepest bias that ramp actually reaches)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("es4_sweep_rest_bias_factor.csv"),
        help="CSV path; defaults to the working directory, outside processed/",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest_path = args.repo_root / "config/may2026_run_manifest.toml"
    data_dir = args.data_dir or args.repo_root / "data/may2026"
    configs = load_run_manifest(manifest_path, data_dir=data_dir)

    print(f"manifest   {manifest_path}")
    print(f"raw runs   {data_dir}")
    print(f"plateau    {PLATEAU_T_MIN_MS}-{PLATEAU_T_MAX_MS} ms, whole cycles only")
    print(f"rest bias  mean v_sweep over each dead-time window past {REST_SETTLE_US:.0f} us of settling")
    print(f"core cells |I_i| >= {CORE_FRACTION:.2f} of profile peak on both runs of a pair")
    print(f"fit window [fit-v-min, {FIT_V_MAX:.1f}] V on the shallow-family ion branch")
    print()
    print("F = |I_i(deep rest bias)| / |I_i(shallow rest bias)|  (>= 1)")
    print("1/F is the same factor with the ratio taken the other way (<= 1).")
    print()

    rows = []
    for port, rotation, run_deep, run_shallow in RUN_PAIRS:
        deep = read_run_sweep(configs[run_deep])
        shallow = read_run_sweep(configs[run_shallow])
        if deep.port != port or shallow.port != port:
            raise ValueError(f"manifest ports disagree with the pairing for port {port}")
        rows.append(analyse_pair(deep, shallow, args.fit_v_min))
        row = rows[-1]
        print(
            f"port {port:>2} rot {rotation:>3}  runs {run_deep}/{run_shallow}  "
            f"cells {row['n_cells']:>2}  ramps {row['n_ramps_deep']}/{row['n_ramps_shallow']}  "
            f"dead {row['n_dead_windows_deep']}/{row['n_dead_windows_shallow']}  "
            f"shots/cell {row['n_shots_per_cell_deep']}/{row['n_shots_per_cell_shallow']}"
        )
        print(
            f"    rest bias   deep {row['v_rest_deep_v']:8.3f} +- {row['v_rest_deep_std_v']:.3f} V   "
            f"shallow {row['v_rest_shallow_v']:8.3f} +- {row['v_rest_shallow_std_v']:.3f} V"
        )
        print(
            f"    ramp span   deep [{row['ramp_min_deep_v']:.2f}, {row['ramp_max_deep_v']:.2f}] V   "
            f"shallow [{row['ramp_min_shallow_v']:.2f}, {row['ramp_max_shallow_v']:.2f}] V   "
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
            f"{'INSIDE' if row['eta34_in_bracket'] else 'OUTSIDE'}"
        )
        print()

    inside = sum(1 for row in rows if row["eta34_in_bracket"])
    print(f"GATE: eta^(3/4) inside the bracket at {inside} of {len(rows)} port/face pairs")
    if inside == len(rows):
        print("GATE VERDICT: the factor is bracketed")
    else:
        print("GATE VERDICT: the bracket is the claim and no product is rescaled")
    print("This script rescales no product either way; it writes numbers only.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
