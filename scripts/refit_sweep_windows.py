"""Varied-window re-fits of the raw Langmuir I-V sweeps.

Executes the pre-registered fit-window sensitivity test: for each
(experiment set, port, plateau cycle, shot) at x = 0, re-fit the SAME
raw I-V sweep over a 5 x 5 family of electron-retarding fit windows and
report the Te(window) sensitivity surface. The sets covered are the
EXPERIMENT_SETS constant below, and the list is recorded in the product's
experiment_sets attr. The decision rule compares the
window-family spread against the beta-hat-implied Te shifts from the transport-side
beta-collapse tables: window spread >= implied shift -> the sweep analysis
CAN carry the residual; window-stable Te with spread << implied shift ->
model Te error convicted.

Window family (PIPELINE-NATIVE parameterization -- a logged deviation from
the brief's V_float/knee wording, same physics: the pipeline's retarding
window is bounded below by an electron-current percentile and above by an
amplitude fraction of the electron-saturation level):

    lower bound  p_low  in {3, 8, 15, 25, 35}  (percentile of positive
                 electron current; the pipeline default is 8, with its
                 1.5%-of-max noise floor kept)
    upper bound  f_high in {0.05, 0.10, 0.15, 0.30, 0.50}  (fraction of
                 max electron current; the pipeline's amplitude fallback
                 uses 0.15)

Experiment set 4 is covered on the same window family and the same
protocol. Its rot-0 runs are 41 (port 11), 42 (port 21), 44 (port 29),
46 (port 41) and 48 (port 50); the rot-180 runs 43, 45 and 47 are excluded
by the rot-0 filter, as in every other set. Runs 42-48 belong to the
+/-20 V / 3 ohm circuit configuration whose CURRENT scale is being
reconciled separately. That reconciliation cannot move this table: the
tabulated quantity is the log-slope of the electron-retarding branch,
Te = 1 / (d ln(I_e) / dV), so multiplying the measured current by a
uniform scale shifts ln(I_e) by an additive constant and leaves the slope
-- and hence every Te(window) entry and the ln(max/min) spread built from
it -- unchanged. Run 41 (port 11), the set-4 row the Te record consumes,
is outside that circuit question in any case.

Set 4 has no rung in the transport-side beta-collapse tables, so it has no
ln(beta-hat) reference scale: its ln_beta_ref is NaN and its verdict reads
"no beta-hat reference". The decision rule is undefined without a
reference; the window spread itself is still measured and recorded, and it
is that spread -- not the verdict -- that the downstream x = 0 core control
consumes.

The ion-branch fit is held FIXED across window variants (the family
targets the retarding fit; ion-branch variation is the pre-registered
secondary and is NOT run here). The inverted density moves as
dln n = -1/2 dln Te by construction at fixed ion line, so only Te is
tabulated. No data corrections anywhere: outputs are sensitivity tables.

Usage:
    MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/refit_sweep_windows.py
        [--plateau-ms 10 19.5] [--output processed/sweep_window_refits.hdf5]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
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
    analyze_langmuir_sweep,
)

_spec = importlib.util.spec_from_file_location(
    "pls", ROOT / "scripts" / "process_langmuir_sweeps.py"
)
pls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pls)

MANIFEST = ROOT / "config" / "may2026_run_manifest.toml"
SWEEPS_H5 = ROOT / "processed" / "langmuir_sweeps.hdf5"

P_LOW = (3.0, 8.0, 15.0, 25.0, 35.0)
F_HIGH = (0.05, 0.10, 0.15, 0.30, 0.50)

#: The experiment sets re-fitted, in the order they are walked.  Recorded in
#: the product's experiment_sets attr so a consumer reads the coverage off the
#: file rather than inferring it from which groups happen to be present.
EXPERIMENT_SETS = (1, 2, 3, 4)

# beta-hat reference scales from the transport-side beta-collapse tables:
# per-rung centered mean ln(beta-hat) for the 2z reference family, and the es3
# far-port within-shot offsets. Hypothesis-test references, not data.
LN_BETA_RUNG = {1: 0.255, 2: 0.106, 3: -0.482}
LN_BETA_ES3_FAR = {41: 0.72, 50: 1.31}


def rot0_runs(sets=EXPERIMENT_SETS):
    """Yield (set_id, run_id, port) for rot-0 runs from the processed file.

    Walks every rot-0 run the processed source holds for each id in *sets*, in
    set then run order.  A set the source does not carry contributes nothing.
    """
    with h5py.File(SWEEPS_H5, "r") as f:
        for sid in sorted(f["experiment_sets"], key=int):
            if int(sid) not in sets:
                continue
            for rid in sorted(f["experiment_sets"][sid]):
                g = f["experiment_sets"][sid][rid]
                if int(g.attrs.get("rotation_deg", 0)) == 0:
                    yield int(sid), rid, int(g.attrs["port"])


def refit_sweep(voltage, current):
    """Return (te_default, te_grid[5,5]) for one filtered sweep."""
    finite = np.isfinite(voltage) & np.isfinite(current)
    v = np.asarray(voltage, float)[finite]
    i = np.asarray(current, float)[finite]
    if v.size < 20:
        return np.nan, np.full((len(P_LOW), len(F_HIGH)), np.nan)
    order = np.argsort(v)
    v, i = v[order], i[order]
    # pipeline default, for the reference column
    try:
        te_default = float(
            analyze_langmuir_sweep(v, i).log_linear_fit.electron_temperature_ev
        )
    except Exception:
        te_default = np.nan
    # fixed ion branch (pipeline defaults)
    try:
        ion_cut = np.nanquantile(v, 0.22)
        ion_fit = _robust_line_fit(v, i, v <= ion_cut, slope_min=0.0)
        ec = i - (ion_fit.slope * v + ion_fit.intercept)
    except Exception:
        return te_default, np.full((len(P_LOW), len(F_HIGH)), np.nan)
    positive = ec > 0
    if positive.sum() < 8:
        return te_default, np.full((len(P_LOW), len(F_HIGH)), np.nan)
    pv = ec[positive]
    i_max = float(np.nanmax(pv))
    grid = np.full((len(P_LOW), len(F_HIGH)), np.nan)
    for a, p_low in enumerate(P_LOW):
        low = max(np.nanpercentile(pv, p_low), i_max * 0.015)
        for b, f_high in enumerate(F_HIGH):
            high = i_max * f_high
            if high <= low:
                continue
            mask = positive & (ec >= low) & (ec <= high)
            mask = _contiguous_true_region(mask)
            if mask.sum() < 8:
                continue
            slope, _ = np.polyfit(v[mask], np.log(ec[mask]), 1)
            if slope > 0:
                grid[a, b] = 1.0 / slope
    return te_default, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plateau-ms", nargs=2, type=float, default=(10.0, 19.5))
    ap.add_argument("--clip-us", type=float, default=10.0)
    ap.add_argument("--cutoff-khz", type=float, default=None,
                    help="base filter cutoff; None = pipeline per-run policy")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument(
        "--output", type=Path,
        default=ROOT / "processed" / "sweep_window_refits.hdf5",
    )
    args = ap.parse_args()

    dataset = LapdDataset.from_manifest(MANIFEST)
    with h5py.File(SWEEPS_H5, "r") as f:
        x_cm = f["x_cm"][:]
    ix0 = int(np.abs(x_cm).argmin())

    results = {}
    for sid, rid, port in rot0_runs():
        run = dataset.run(rid)
        sw = run.config.sweep
        cycle_t = pls._cycle_start_times(run)
        in_plateau = np.flatnonzero(
            (cycle_t * 1e3 >= args.plateau_ms[0])
            & (cycle_t * 1e3 <= args.plateau_ms[1])
        )
        base = (args.cutoff_khz or 100.0) * 1e3
        core_cutoff_hz, _ = pls._cutoff_hz_for_run(run, base)
        sr = run.config.acquisition.sample_rate_hz
        i_off = run.default_zero_offset_v(ChannelKind.I_SWEEP)
        v_off = run.default_zero_offset_v(ChannelKind.V_SWEEP)
        ramp_slices = run.sweep_ramp_sample_slices(clip_s=args.clip_us * 1e-6)

        te_def_all = []
        te_grid_all = []
        for k in in_plateau:
            ramp = ramp_slices[k]
            i_all = run.langmuir_traces(
                ChannelKind.I_SWEEP, ramp, zero_offset_v=i_off
            )
            v_all = run.langmuir_traces(
                ChannelKind.V_SWEEP, ramp, zero_offset_v=v_off
            )
            i_x0 = butterworth_lowpass(
                i_all[ix0], sample_rate_hz=sr, cutoff_hz=core_cutoff_hz,
                order=args.order,
            )
            v_x0 = butterworth_lowpass(
                v_all[ix0], sample_rate_hz=sr, cutoff_hz=core_cutoff_hz,
                order=args.order,
            )
            for s in range(i_x0.shape[0]):
                te_d, grid = refit_sweep(v_x0[s], i_x0[s])
                te_def_all.append(te_d)
                te_grid_all.append(grid)
        te_def_all = np.asarray(te_def_all)
        te_grid_all = np.asarray(te_grid_all)  # (n_sweeps, 5, 5)
        # robust aggregation; physical bounds guard only (no corrections)
        te_def_all[(te_def_all <= 0.05) | (te_def_all > 30.0)] = np.nan
        te_grid_all[(te_grid_all <= 0.05) | (te_grid_all > 30.0)] = np.nan
        med_def = float(np.nanmedian(te_def_all))
        med_grid = np.nanmedian(te_grid_all, axis=0)
        n_used = int(np.isfinite(te_def_all).sum())
        results[(sid, port)] = {
            "te_default": med_def,
            "te_grid": med_grid,
            "n_sweeps": n_used,
            "run_id": rid,
        }

    # ---- report + decision rule
    print(f"window family: p_low {P_LOW} x f_high {F_HIGH}; plateau "
          f"{args.plateau_ms[0]}-{args.plateau_ms[1]} ms; x=0; ion branch "
          f"fixed at pipeline defaults")
    print(f"\n{'set':>3} {'port':>4} {'Te_def':>7} {'Te_min':>7} "
          f"{'Te_max':>7} {'dlnTe_win':>9} {'|lnB|ref':>8} {'verdict':>28}")
    out = {}
    for (sid, port), r in sorted(results.items()):
        g = r["te_grid"]
        te_min, te_max = np.nanmin(g), np.nanmax(g)
        dln_win = float(np.log(te_max / te_min)) if te_min > 0 else np.nan
        ln_ref = (
            abs(LN_BETA_RUNG[sid]) if sid in LN_BETA_RUNG else float("nan")
        )
        if sid == 3 and port in LN_BETA_ES3_FAR:
            ln_ref = LN_BETA_ES3_FAR[port]
        if not np.isfinite(dln_win):
            verdict = "insufficient windows"
        elif not np.isfinite(ln_ref):
            verdict = "no beta-hat reference"
        elif dln_win >= ln_ref:
            verdict = "sweep CAN carry residual"
        elif dln_win < 0.5 * ln_ref:
            verdict = "MODEL ERROR convicted"
        else:
            verdict = "partial attribution"
        print(f"{sid:>3} {port:>4} {r['te_default']:7.2f} {te_min:7.2f} "
              f"{te_max:7.2f} {dln_win:9.3f} {ln_ref:8.3f} {verdict:>28}")
        out[(sid, port)] = (r, dln_win, ln_ref, verdict)

    with h5py.File(args.output, "w") as f:
        f.attrs["p_low"] = P_LOW
        f.attrs["f_high"] = F_HIGH
        f.attrs["plateau_ms"] = args.plateau_ms
        f.attrs["experiment_sets"] = np.asarray(EXPERIMENT_SETS, dtype=np.int64)
        f.attrs["protocol"] = (
            "fit-window sensitivity: each raw I-V sweep at x = 0 re-fit over "
            "the 5 x 5 family of electron-retarding windows recorded in the "
            "p_low / f_high attrs, ion branch held at pipeline defaults; "
            "T_e(window) sensitivity tables only, no data corrections"
        )
        for (sid, port), (r, dln_win, ln_ref, verdict) in out.items():
            g = f.create_group(f"set{sid}/port{port}")
            g.attrs.update(
                run_id=r["run_id"], n_sweeps=r["n_sweeps"],
                te_default_ev=r["te_default"], dln_te_window=dln_win,
                ln_beta_ref=ln_ref, verdict=verdict,
            )
            g.create_dataset("te_window_ev", data=r["te_grid"])
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
