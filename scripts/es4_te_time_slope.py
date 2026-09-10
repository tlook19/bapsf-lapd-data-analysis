"""ES4 T_e(t) plateau slope, both faces, at x = 0 and the core mean.

What this measures
-------------------
Every port swept in experiment set 4 is swept twice: a rotation-0 run and a
rotation-180 run (p21 is runs 42/43; p29 is runs 44/45).  ``--port`` selects
which one; the default (21) reproduces the original p21-only invocation
byte-for-byte.  For each of the two runs this instrument walks every plateau
cycle and fits T_e at the on-axis cell (x = 0) and the mean over the core
cells (|x| <= 10 cm), with two independent estimators built on the SAME
filtered raw sweep:

  (i)  the pipeline's own per-shot log-linear fit (the reference column
       ``refit_sweep_windows.refit_sweep`` returns), median over shots;
  (ii) the median, over ``refit_sweep``'s 5x5 retarding-window family, of
       that family's per-shot median -- the window-family estimate.

Both come from ``refit_sweep_windows.refit_sweep``, imported rather than
restated, so this instrument and the window-sensitivity product it draws on
can never drift apart on what "the window family" means.  Each cell/shot fit
also feeds an eligibility census: how many shots produced a finite default
fit (the >= 20 point and sanity-bound gates), and how many produced a
window-family grid that was not entirely refused (the ion-fit and >= 8
positive-point gates) -- printed per run so an empty or thin bin is visible
rather than silently averaged away.

Per estimator and face, an OLS line is fit to T_e(t) over the 12-19.5 ms
window of the stable plateau and reported as a percent-of-plateau-mean slope
with its standard error (``ols_slope_se``).  The plateau mean normalizing the
percent conversion is taken over a selectable window, ``--plateau-window-ms``,
defaulting to the full displayed plateau range (10-19.5 ms) -- matching the
read-only scratch instrument this ports and keeping the default output
unchanged.  Beside it, a second, fixed plateau mean is always printed over
``WINDOW_MATCHED_MS`` (15.0-19.5 ms): the window the ES4 sim1d-overlay prior
comparison uses (its T_e(t) table's last five 1 ms samples before the
overlay's own display cutoff), so a prior quoted at that window is compared
to a measurement taken over the same window rather than a wider one.

Pre-registered gate
--------------------
For every (run, estimator) series, with the OLS slope in percent of the
plateau mean per millisecond:

  slope in [-1.5, -0.5] %/ms  ->  "inside"
  slope outside that bracket  ->  "outside" with the value
  fewer than 3 finite cycles in the 12-19.5 ms window  ->  "REFUSED (n of N
      cycles)"

The gate does not interpret the slope beyond that bin; it is printed for
every run and estimator, never silently as a bare NaN.

Pre-registered excess bin
--------------------------
A second, independent bin compares a measured plateau mean against the
flow-cancelled chord-vs-probe excess bracket:

  value <= 0.6 eV             ->  "<= 0.6 eV"   (closes the excess bracket)
  value >= 1.0 eV              ->  ">= 1 eV"     (weighs the other way)
  0.6 eV < value < 1.0 eV      ->  "between (undetermined)"
  fewer than 3 finite cycles   ->  "REFUSED (n of N cycles)"

Printed for every estimator, window and face; never a bare NaN.

Inputs
------
  processed/langmuir_sweeps.hdf5   x_cm grid and the set-4 port/rotation
                                    attrs used to place and label the runs
  processed/es4_sim1d_overlay.npz  the ES4 overlay's per-port T_e(t) prior,
                                    read-only, for the window-matched prior
                                    comparison
  config/may2026_run_manifest.toml sweep schedule, channels, calibration
  data/may2026/*.hdf5              raw runs, opened read-only

Read-only.  No product is written by default; ``--output`` writes one CSV of
the per-cycle table and refuses a path inside ``processed/`` (this is a
one-shot measurement, not a member of the product chain).

Usage
-----
  PYTHONPATH=src python scripts/es4_te_time_slope.py [--port 21] [--output out.csv]
  PYTHONPATH=src python scripts/es4_te_time_slope.py --port 29
  PYTHONPATH=src python scripts/es4_te_time_slope.py --runs 44,45
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bapsf_lapd import ChannelKind, LapdDataset  # noqa: E402
from bapsf_lapd.filtering import butterworth_lowpass  # noqa: E402

import refit_sweep_windows as rsw  # noqa: E402  -- refit_sweep + the window family

MANIFEST = rsw.MANIFEST
SWEEPS_H5 = rsw.SWEEPS_H5
OVERLAY_NPZ = ROOT / "processed" / "es4_sim1d_overlay.npz"

#: The port measured when ``--port``/``--runs`` are not given -- the original
#: p21-only invocation, kept as the byte-identical default.
DEFAULT_PORT = 21

#: Displayed plateau range and the narrower window the OLS slope is fit over.
PLATEAU_DISPLAY_MS = (10.0, 19.5)
OLS_WINDOW_MS = (12.0, 19.5)

#: The fixed window the ES4 sim1d-overlay prior's T_e(t) table is read over
#: for the prior-vs-measured comparison (its last five 1 ms samples).  Always
#: reported beside the selectable ``--plateau-window-ms`` plateau mean.
WINDOW_MATCHED_MS = (15.0, 19.5)

#: Pre-registered excess-bin bracket (eV), independent of the OLS gate above.
EXCESS_BIN_LOW_EV = 0.6
EXCESS_BIN_HIGH_EV = 1.0

#: A cell counts as core if it sits within this radius of the axis; identical
#: to process_langmuir_sweeps.py's own EDGE_X_CM threshold, so every core
#: cell used here gets that pipeline's core (not edge) filter cutoff.
CORE_ABS_X_CM = 10.0

# Filter and ramp-clip parameters, matching process_langmuir_sweeps.py's own
# --cutoff-khz / --clip-us defaults (100 kHz, 10 us) -- not importable as
# named constants there, so restated here as call parameters only.
BASE_CUTOFF_HZ = 100.0e3
CLIP_S = 10.0e-6

# Sanity bounds discarding non-physical per-shot/per-window fits before
# either median, matching the read-only scratch instrument this ports.
TE_SANITY_MIN_EV = 0.05
TE_SANITY_MAX_EV = 30.0

#: Pre-registered gate bracket, in percent of the plateau mean per ms.
GATE_LOW_PCT_PER_MS = -1.5
GATE_HIGH_PCT_PER_MS = -0.5
MIN_FINITE_CYCLES = 3

ESTIMATORS = ("x0 default", "x0 family-med", "core default", "core family-med")

FORBIDDEN_OUTPUT_DIR = "processed"


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


def ols_slope_se(t, y):
    """OLS slope and its standard error over the finite (t, y) pairs.

    Returns ``(slope, se, intercept, n_finite)`` in y-units per unit of t.
    ``n_finite`` is the count of finite pairs used.  Fewer than 3 finite
    pairs, or fewer than 3 distinct t values, returns NaN slope/se.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    tf, yf = t[finite], y[finite]
    n = int(tf.size)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    design = np.vstack([tf, np.ones_like(tf)]).T
    coeffs, *_ = np.linalg.lstsq(design, yf, rcond=None)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    dof = n - 2
    ss_t = float(np.sum((tf - tf.mean()) ** 2))
    if dof <= 0 or ss_t <= 0.0:
        return slope, float("nan"), intercept, n
    resid = yf - (slope * tf + intercept)
    s2 = float(np.sum(resid**2)) / dof
    se = float(np.sqrt(s2 / ss_t))
    return slope, se, intercept, n


def pct_per_ms(slope, se, plateau_mean):
    """Convert an eV/ms slope and its SE into percent of *plateau_mean* per ms."""
    if not np.isfinite(plateau_mean) or plateau_mean == 0.0:
        return float("nan"), float("nan")
    return 100.0 * slope / plateau_mean, 100.0 * se / abs(plateau_mean)


def gate_verdict(
    slope_pct_per_ms: float,
    n_finite: int,
    n_total: int,
    *,
    low: float = GATE_LOW_PCT_PER_MS,
    high: float = GATE_HIGH_PCT_PER_MS,
    min_cycles: int = MIN_FINITE_CYCLES,
) -> str:
    """Pre-registered bin for one (run, estimator) plateau slope.

    Never returns a bare NaN: fewer than *min_cycles* finite cycles (or a
    non-finite slope even with enough cycles) reads as a REFUSED bin naming
    the cycle count, not as an unresolved value.
    """
    if n_finite < min_cycles or not np.isfinite(slope_pct_per_ms):
        return f"REFUSED ({n_finite} of {n_total} cycles)"
    if low <= slope_pct_per_ms <= high:
        return "inside"
    return f"outside ({slope_pct_per_ms:+.2f} %/ms)"


def excess_bin_verdict(
    value_ev: float,
    n_finite: int,
    n_total: int,
    *,
    low: float = EXCESS_BIN_LOW_EV,
    high: float = EXCESS_BIN_HIGH_EV,
    min_cycles: int = MIN_FINITE_CYCLES,
) -> str:
    """Pre-registered bin for one plateau mean against the excess bracket.

    Never returns a bare NaN: fewer than *min_cycles* finite cycles (or a
    non-finite value even with enough cycles) reads as a REFUSED bin naming
    the cycle count, not as an unresolved value.
    """
    if n_finite < min_cycles or not np.isfinite(value_ev):
        return f"REFUSED ({n_finite} of {n_total} cycles)"
    if value_ev <= low:
        return f"<= {low:.1f} eV"
    if value_ev >= high:
        return f">= {high:.0f} eV"
    return "between (undetermined)"


def _core_indices_and_x0(sweeps_h5_path: Path) -> tuple[np.ndarray, int]:
    with h5py.File(sweeps_h5_path, "r") as f:
        x_cm = f["x_cm"][:]
    ix0 = int(np.abs(x_cm).argmin())
    core_idx = np.flatnonzero(np.abs(x_cm) <= CORE_ABS_X_CM)
    return core_idx, ix0


def _set4_attrs(sweeps_h5_path: Path, run_id: str) -> tuple[int, int]:
    with h5py.File(sweeps_h5_path, "r") as f:
        g = f["experiment_sets"]["4"][run_id]
        return int(g.attrs["port"]), int(g.attrs["rotation_deg"])


def runs_for_port(
    sweeps_h5_path: Path, port: int, override_run_ids: tuple[str, ...] | None = None
) -> tuple[dict, ...]:
    """The (run_id, port, rotation_deg) specs for *port*, rot-0 before rot-180.

    Scanned off the processed file's own experiment-set-4 attrs rather than a
    hardcoded table, so a new port needs no code change here.  With
    *override_run_ids* given, those run ids are used verbatim (still labeled
    from the file's own attrs) instead of the port-based lookup -- for the
    rare case where the default rot-0/rot-180 pairing for a port isn't
    wanted.  Raises if *port* has no experiment-set-4 runs and
    *override_run_ids* is not given, or if an override run id is absent from
    the file.
    """
    with h5py.File(sweeps_h5_path, "r") as f:
        g = f["experiment_sets"]["4"]
        if override_run_ids is not None:
            run_ids = list(override_run_ids)
            missing = [rid for rid in run_ids if rid not in g]
            if missing:
                raise ValueError(
                    f"--runs names run id(s) {missing} not present in "
                    f"experiment set 4 of {sweeps_h5_path}"
                )
        else:
            run_ids = sorted(
                (rid for rid in g if int(g[rid].attrs["port"]) == port), key=int
            )
            if not run_ids:
                raise ValueError(
                    f"no experiment-set-4 runs found for port {port} in {sweeps_h5_path}"
                )
        specs = []
        for rid in run_ids:
            attrs = g[rid].attrs
            specs.append(
                {
                    "run_id": rid,
                    "port": int(attrs["port"]),
                    "rotation_deg": float(attrs["rotation_deg"]),
                }
            )
    return tuple(specs)


def overlay_prior_te_ev(overlay_npz_path: Path, port: int, window_ms: tuple[float, float]):
    """Mean prior T_e (eV) for *port* over *window_ms* from the ES4 overlay.

    Returns NaN if the overlay carries no row for *port*, or if no
    ``te_time_ms`` sample of that row falls in *window_ms*.  Read-only.
    """
    with np.load(overlay_npz_path, allow_pickle=True) as d:
        ports = np.asarray(d["port"])
        matches = np.flatnonzero(ports == port)
        if matches.size == 0:
            return float("nan")
        row = int(matches[0])
        time_ms = np.asarray(d["te_time_ms"], dtype=float)
        te_ev = np.asarray(d["te_mean_ev"], dtype=float)[row]
    in_window = (time_ms >= window_ms[0]) & (time_ms <= window_ms[1])
    if not np.any(in_window):
        return float("nan")
    return float(np.nanmean(te_ev[in_window]))


def cell_te_estimates(run, ramp, core_idx, core_cutoff_hz):
    """Per-core-cell (pipeline default, window-family median) T_e for one cycle.

    Returns two arrays shaped like *core_idx* (the median over shots of the
    pipeline's per-shot log-linear fit, and the median over the 5x5 window
    family, itself first medianed over shots -- both from
    ``refit_sweep_windows.refit_sweep``, with non-physical fits (outside
    ``(TE_SANITY_MIN_EV, TE_SANITY_MAX_EV]``) dropped before either median)
    and an eligibility dict counting, over every (cell, shot) in this cycle:
    ``n_shots`` attempted, ``n_default_admitted`` with a finite post-sanity
    default fit (the >= 20 point gate in ``refit_sweep`` and the sanity
    bound), and ``n_family_admitted`` with at least one finite post-sanity
    window-family entry (the ion-fit and >= 8 positive-point gates in
    ``refit_sweep``).
    """
    i_off = run.default_zero_offset_v(ChannelKind.I_SWEEP)
    v_off = run.default_zero_offset_v(ChannelKind.V_SWEEP)
    i_all = run.langmuir_traces(ChannelKind.I_SWEEP, ramp, zero_offset_v=i_off)
    v_all = run.langmuir_traces(ChannelKind.V_SWEEP, ramp, zero_offset_v=v_off)
    sample_rate_hz = run.config.acquisition.sample_rate_hz

    cell_default = np.full(len(core_idx), np.nan)
    cell_family_med = np.full(len(core_idx), np.nan)
    eligibility = {"n_shots": 0, "n_default_admitted": 0, "n_family_admitted": 0}
    for j, ic in enumerate(core_idx):
        i_c = butterworth_lowpass(
            i_all[ic], sample_rate_hz=sample_rate_hz, cutoff_hz=core_cutoff_hz, order=4
        )
        v_c = butterworth_lowpass(
            v_all[ic], sample_rate_hz=sample_rate_hz, cutoff_hz=core_cutoff_hz, order=4
        )
        default_per_shot = []
        grid_per_shot = []
        for s in range(i_c.shape[0]):
            te_default, grid = rsw.refit_sweep(v_c[s], i_c[s])
            default_per_shot.append(te_default)
            grid_per_shot.append(grid)
        default_per_shot = np.asarray(default_per_shot)
        grid_per_shot = np.asarray(grid_per_shot)
        bad_default = (default_per_shot <= TE_SANITY_MIN_EV) | (default_per_shot > TE_SANITY_MAX_EV)
        default_per_shot[bad_default] = np.nan
        bad_grid = (grid_per_shot <= TE_SANITY_MIN_EV) | (grid_per_shot > TE_SANITY_MAX_EV)
        grid_per_shot[bad_grid] = np.nan
        cell_default[j] = np.nanmedian(default_per_shot)
        cell_family_med[j] = np.nanmedian(np.nanmedian(grid_per_shot, axis=0))
        eligibility["n_shots"] += int(default_per_shot.size)
        eligibility["n_default_admitted"] += int(np.sum(np.isfinite(default_per_shot)))
        eligibility["n_family_admitted"] += int(
            np.sum(np.any(np.isfinite(grid_per_shot), axis=(1, 2)))
        )
    return cell_default, cell_family_med, eligibility


def run_plateau_series(dataset: LapdDataset, run_id: str, core_idx, ix0):
    """T_e(t) at x=0 and the core mean, both estimators, for one run's plateau.

    Returns ``(cycle_ms, series, eligibility)`` where *cycle_ms* covers
    ``PLATEAU_DISPLAY_MS``, *series* maps each label in ``ESTIMATORS`` to an
    array aligned with *cycle_ms*, and *eligibility* sums the per-cycle
    eligibility counts from ``cell_te_estimates`` over every displayed cycle.
    """
    run = dataset.run(run_id)
    cycle_ms = rsw.pls._cycle_start_times(run) * 1e3
    in_display = np.flatnonzero(
        (cycle_ms >= PLATEAU_DISPLAY_MS[0]) & (cycle_ms <= PLATEAU_DISPLAY_MS[1])
    )
    core_cutoff_hz, _edge_cutoff_hz = rsw.pls._cutoff_hz_for_run(run, BASE_CUTOFF_HZ)
    ramp_slices = run.sweep_ramp_sample_slices(clip_s=CLIP_S)

    j0 = int(np.flatnonzero(core_idx == ix0)[0])
    series = {label: np.full(len(in_display), np.nan) for label in ESTIMATORS}
    cycle_out = cycle_ms[in_display]
    eligibility = {"n_shots": 0, "n_default_admitted": 0, "n_family_admitted": 0}
    for row, k in enumerate(in_display):
        cell_default, cell_family_med, cycle_eligibility = cell_te_estimates(
            run, ramp_slices[k], core_idx, core_cutoff_hz
        )
        series["x0 default"][row] = cell_default[j0]
        series["x0 family-med"][row] = cell_family_med[j0]
        series["core default"][row] = float(np.nanmean(cell_default))
        series["core family-med"][row] = float(np.nanmean(cell_family_med))
        for key in eligibility:
            eligibility[key] += cycle_eligibility[key]
    return cycle_out, series, eligibility


def windowed_mean_ev(cycle_ms, values, window_ms):
    """Mean of *values* over cycles with ``cycle_ms`` in *window_ms* (inclusive).

    NaN if no cycle falls in the window.
    """
    in_window = (cycle_ms >= window_ms[0]) & (cycle_ms <= window_ms[1])
    if not np.any(in_window):
        return float("nan")
    return float(np.nanmean(values[in_window]))


def slope_report(cycle_ms, values, *, plateau_window_ms=PLATEAU_DISPLAY_MS):
    """Plateau means, OLS slope/SE in %/ms, and gate inputs for one series.

    ``plateau_mean_ev`` is the mean over *plateau_window_ms* (default: the
    full displayed range, matching the original unconditional mean).
    ``window_matched_mean_ev`` is always the mean over ``WINDOW_MATCHED_MS``,
    independent of *plateau_window_ms* -- the value compared against a prior
    quoted at that window.
    """
    plateau_mean = windowed_mean_ev(cycle_ms, values, plateau_window_ms)
    window_matched_mean = windowed_mean_ev(cycle_ms, values, WINDOW_MATCHED_MS)
    in_window = (cycle_ms >= OLS_WINDOW_MS[0]) & (cycle_ms <= OLS_WINDOW_MS[1])
    n_total = int(np.sum(in_window))
    slope, se, _intercept, n_finite = ols_slope_se(cycle_ms[in_window], values[in_window])
    slope_pct, se_pct = pct_per_ms(slope, se, plateau_mean)
    in_window_matched = (cycle_ms >= WINDOW_MATCHED_MS[0]) & (cycle_ms <= WINDOW_MATCHED_MS[1])
    n_finite_window_matched = int(np.sum(np.isfinite(values[in_window_matched])))
    n_total_window_matched = int(np.sum(in_window_matched))
    return {
        "plateau_mean_ev": plateau_mean,
        "window_matched_mean_ev": window_matched_mean,
        "n_finite_window_matched": n_finite_window_matched,
        "n_total_window_matched": n_total_window_matched,
        "slope_pct_per_ms": slope_pct,
        "se_pct_per_ms": se_pct,
        "n_finite": n_finite,
        "n_total": n_total,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"ES4 langmuir port to measure (default {DEFAULT_PORT}, reproducing "
        "the original p21-only invocation byte-for-byte)",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help="comma-separated experiment-set-4 run ids overriding the --port "
        "lookup (e.g. '44,45'); use only when the default rot-0/rot-180 "
        "pairing for a port is not wanted",
    )
    parser.add_argument(
        "--plateau-window-ms",
        nargs=2,
        type=float,
        default=PLATEAU_DISPLAY_MS,
        metavar=("LOW", "HIGH"),
        help="window (ms) the plateau mean and the pct-of-mean slope "
        f"normalization are computed over; default is the full displayed "
        f"range {PLATEAU_DISPLAY_MS}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional CSV path for the per-cycle table; a path inside "
        f"{FORBIDDEN_OUTPUT_DIR}/ is refused",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    output = checked_output_path(args.output) if args.output is not None else None
    plateau_window_ms = tuple(args.plateau_window_ms)
    override_run_ids = tuple(args.runs.split(",")) if args.runs else None

    dataset = LapdDataset.from_manifest(MANIFEST)
    core_idx, ix0 = _core_indices_and_x0(SWEEPS_H5)
    runs = runs_for_port(SWEEPS_H5, args.port, override_run_ids)

    print(f"manifest    {MANIFEST}")
    print(f"sweeps h5   {SWEEPS_H5}")
    print(f"port        {args.port}" + (" (runs overridden by --runs)" if override_run_ids else ""))
    print(f"core cells  |x| <= {CORE_ABS_X_CM:.1f} cm ({len(core_idx)} of the profile)")
    print(f"OLS window  {OLS_WINDOW_MS[0]:.1f}-{OLS_WINDOW_MS[1]:.1f} ms")
    print(f"plateau window   {plateau_window_ms[0]:.1f}-{plateau_window_ms[1]:.1f} ms")
    print(f"window-matched   {WINDOW_MATCHED_MS[0]:.1f}-{WINDOW_MATCHED_MS[1]:.1f} ms "
          "(fixed; the window the ES4 sim1d-overlay prior comparison uses)")
    print(f"gate        slope in [{GATE_LOW_PCT_PER_MS:+.1f}, {GATE_HIGH_PCT_PER_MS:+.1f}] %/ms -> inside; "
          f"< {MIN_FINITE_CYCLES} finite cycles -> REFUSED")
    print(f"excess bin  value <= {EXCESS_BIN_LOW_EV:.1f} eV -> closes the bracket; "
          f">= {EXCESS_BIN_HIGH_EV:.0f} eV -> weighs the other way; between -> undetermined; "
          f"< {MIN_FINITE_CYCLES} finite cycles -> REFUSED")
    print()

    overlay_prior_ev = overlay_prior_te_ev(OVERLAY_NPZ, args.port, WINDOW_MATCHED_MS)

    csv_rows = []
    for spec in runs:
        run_id, expected_port, expected_rot = (
            spec["run_id"],
            spec["port"],
            spec["rotation_deg"],
        )
        port, rot = _set4_attrs(SWEEPS_H5, run_id)
        if port != expected_port or rot != expected_rot:
            raise ValueError(
                f"run {run_id}: set-4 attrs report port {port} rot {rot}, "
                f"expected port {expected_port} rot {expected_rot}"
            )
        cycle_ms, series, eligibility = run_plateau_series(dataset, run_id, core_idx, ix0)

        print(f"=== run {run_id}  p{port} rot {rot}  T_e(t) ===")
        print(
            f"  eligibility  shots attempted={eligibility['n_shots']}  "
            f"default admitted={eligibility['n_default_admitted']}  "
            f"family admitted={eligibility['n_family_admitted']}  "
            "(summed over every cell and displayed cycle)"
        )
        header = "  t_ms  " + "  ".join(f"{label:>15s}" for label in ESTIMATORS)
        print(header)
        for row, t in enumerate(cycle_ms):
            vals = "  ".join(f"{series[label][row]:15.3f}" for label in ESTIMATORS)
            print(f"  {t:5.1f}  {vals}")
            csv_rows.append(
                {
                    "run_id": run_id,
                    "port": port,
                    "rotation_deg": rot,
                    "cycle_time_ms": t,
                    **{label.replace(" ", "_").replace("-", "_") + "_ev": series[label][row]
                       for label in ESTIMATORS},
                }
            )
        print()
        print(
            "  estimator          plateau mean (eV)   window-matched (eV)   "
            "OLS slope (%/ms)   SE (%/ms)   gate   excess bin"
        )
        for label in ESTIMATORS:
            report = slope_report(cycle_ms, series[label], plateau_window_ms=plateau_window_ms)
            verdict = gate_verdict(report["slope_pct_per_ms"], report["n_finite"], report["n_total"])
            excess_verdict = excess_bin_verdict(
                report["window_matched_mean_ev"],
                report["n_finite_window_matched"],
                report["n_total_window_matched"],
            )
            print(
                f"  {label:16s}   {report['plateau_mean_ev']:15.4f}   "
                f"{report['window_matched_mean_ev']:18.4f}   "
                f"{report['slope_pct_per_ms']:+15.3f}   {report['se_pct_per_ms']:8.3f}   "
                f"{verdict}   {excess_verdict}"
            )
        print()

        headline = slope_report(
            cycle_ms, series["core family-med"], plateau_window_ms=plateau_window_ms
        )
        measured_ev = headline["window_matched_mean_ev"]
        if not np.isfinite(overlay_prior_ev):
            print(f"  overlay prior  p{port}: no overlay row for this port")
        elif not np.isfinite(measured_ev) or measured_ev <= 0.0:
            print(
                f"  overlay prior  p{port} window-matched  {overlay_prior_ev:.4f} eV   "
                "measured (core family-med) unavailable over the window-matched window"
            )
        else:
            density_factor = math.sqrt(overlay_prior_ev / measured_ev)
            print(
                f"  overlay prior (window-matched)  {overlay_prior_ev:.4f} eV   "
                f"measured (core family-med, window-matched)  {measured_ev:.4f} eV   "
                f"density re-derivation factor sqrt(prior/measured) = {density_factor:.4f}"
            )
        print()

    if output is not None:
        fieldnames = list(csv_rows[0].keys()) if csv_rows else []
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Wrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
