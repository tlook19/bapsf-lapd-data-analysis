"""Analyze all Langmuir sweeps and write position-averaged results to HDF5.

For each run in the May 2026 manifest this script:

  1. Loads I_SWEEP and V_SWEEP for every position/shot/cycle in one bulk
     read per cycle, filtering all 1020 shots at once.
  2. Fits each individual (position, shot, cycle) sweep with
     analyze_langmuir_sweep and flags it with evaluate_langmuir_quality.
  3. Averages Te and Vp over the 20 shots at each position, treating
     'bad'-severity shots as NaN so they are excluded from the mean.
  4. Writes mean, std (ddof=1), and SEM arrays—plus per-severity shot
     counts—to processed/langmuir_sweeps.hdf5 under a nested group
     structure: experiment_sets/{set_id}/{run_id}/.

HDF5 layout
-----------
/x_cm                           (51,) float64  scan positions in cm
/experiment_sets/
  {set_id}/                     attrs: label, v_bank_v, v_puff_v
    {run_id}/                   attrs: rotation_deg, port, z_cm
      cycle_time_s              (n_cycles,)  time at ramp start
      te_log_ev                 (51, n_cycles)  shot-averaged Te, log method
      te_log_ev_std             (51, n_cycles)
      te_log_ev_sem             (51, n_cycles)
      te_exp_ev                 (51, n_cycles)  shot-averaged Te, exp method
      te_exp_ev_std             (51, n_cycles)
      te_exp_ev_sem             (51, n_cycles)
      vp_derivative_v           (51, n_cycles)  plasma potential, derivative
      vp_derivative_v_std       (51, n_cycles)
      vp_derivative_v_sem       (51, n_cycles)
      vp_log_v                  (51, n_cycles)  plasma potential, log fit
      vp_log_v_std              (51, n_cycles)
      vp_log_v_sem              (51, n_cycles)
      vp_exp_v                  (51, n_cycles)  plasma potential, exp fit
      vp_exp_v_std              (51, n_cycles)
      vp_exp_v_sem              (51, n_cycles)
      n_ok                      (51, n_cycles)  int32, shots with severity ok
      n_warn                    (51, n_cycles)  int32
      n_bad                     (51, n_cycles)  int32  (includes n_arc, n_te_outlier)
      n_arc                     (51, n_cycles)  int32, subset of n_bad: pre-excluded by arc detector
      n_te_outlier              (51, n_cycles)  int32, subset of n_bad: rejected by per-cell Te sigma-clip
      te_log_ev_shots           (51, n_shots, n_cycles)  float32  per-shot Te (log); NaN for bad/arc shots, PRE-sigma-clip
      sev_shots                 (51, n_shots, n_cycles)  int8     shot quality pre-clip: 0=ok 1=warn 2=bad 3=arc

rotation_deg is stored as a run attribute so rot-0 and rot-180 runs can be
identified separately during plotting without parsing run IDs.

Sweep timing
------------
cycle_time_s[k] = sweep.t0_s + k * sweep.tau_cycle_s, i.e. the time at
the leading edge of the k-th voltage ramp relative to the SIS recording
window.  The --clip-us edge trim applied to avoid filter transients is NOT
included in this timestamp.

Usage
-----
  # One run (smoke test, needs local HDF5 data)
  python scripts/process_langmuir_sweeps.py --run-ids 46

  # Full dataset
  python scripts/process_langmuir_sweeps.py --run-ids all
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np

from bapsf_lapd import (
    ChannelKind,
    LapdDataset,
    LapdRun,
    analyze_langmuir_sweep,
    butterworth_lowpass,
    evaluate_langmuir_quality,
)


MANIFEST = Path("config/may2026_run_manifest.toml")
HDF5_OUTPUT = Path("processed/langmuir_sweeps.hdf5")
ARC_EXCLUSIONS_CSV = Path("processed/isweep_frontside_arc_shot_exclusions.csv")

# Probe scan positions: 51 points, 1 cm spacing, centered at x=0.
X_CM = np.linspace(-25.0, 25.0, 51)

QUANTITIES = ("te_log_ev", "te_exp_ev", "vp_derivative_v", "vp_log_v", "vp_exp_v")

# Integer codes for severity stored in per-shot accumulation arrays.
_SEV_OK = 0
_SEV_WARN = 1
_SEV_BAD = 2
_SEV_MAP = {"ok": _SEV_OK, "warn": _SEV_WARN, "bad": _SEV_BAD}

# Filter cutoff policy constants.
# Edge positions are |x| > EDGE_X_CM; they get a higher filter cutoff to
# resolve potentially sub-eV temperatures.
EDGE_X_CM = 10.0          # cm; probe positions outside this are "edge"

# Per-cell Te sigma-clip threshold.  After all shots in a (position, cycle)
# cell are fitted, any shot whose Te deviates more than K_SIGMA_CLIP times the
# MAD-based scale from the median is marked bad and excluded from the mean.
# Requires ≥ 3 finite shots in the cell; cells with fewer shots are not clipped.
K_SIGMA_CLIP = 3.0
TE_MIN_EV = 0.5           # eV; design target for cold-region resolution
HE_VP_VF_FACTOR = 4.45    # V_p − V_f ≈ 4.45 × T_e for He⁺
FILTER_SMEAR_FRACTION = 0.30  # allow at most this fraction of retarding window


def _load_arc_exclusions(csv_path: Path) -> dict[str, list[tuple[int, int, int]]]:
    """Read the arc-shot exclusion CSV and return {run_id: [(pos, shot, cyc), ...]}.

    Only rows with exclude_i_sweep == 1 are loaded.  The result covers all
    runs present in the file; runs with no exclusions are absent from the dict.
    """
    exclusions: dict[str, list[tuple[int, int, int]]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["exclude_i_sweep"]) != 1:
                continue
            rid = row["run_id"]
            if rid not in exclusions:
                exclusions[rid] = []
            exclusions[rid].append(
                (int(row["position_index"]), int(row["shot_index"]), int(row["cycle_index"]))
            )
    return exclusions


def _build_arc_mask(
    run_id: str,
    exclusions: dict[str, list[tuple[int, int, int]]],
    n_pos: int,
    n_shots: int,
    n_cycles: int,
) -> np.ndarray:
    """Return bool (n_pos, n_shots, n_cycles) mask; True = arc-excluded."""
    mask = np.zeros((n_pos, n_shots, n_cycles), dtype=bool)
    for pos, shot, cyc in exclusions.get(run_id, []):
        if pos < n_pos and shot < n_shots and cyc < n_cycles:
            mask[pos, shot, cyc] = True
    return mask


def _sigma_clip_te(
    per_shot: dict,
    sev_grid: np.ndarray,
    *,
    k_sigma: float = K_SIGMA_CLIP,
) -> np.ndarray:
    """In-place MAD sigma-clip of per-shot Te values within each (pos, cycle) cell.

    For each cell that has ≥ 3 finite Te values (ok or warn shots), compute the
    median and MAD-based scale.  Any shot whose Te deviates more than
    k_sigma × scale from the median is treated as a Te outlier:
      - Its entry in per_shot is set to NaN for ALL quantities (te_log, te_exp,
        vp_derivative, vp_log, vp_exp) so it cannot contaminate any average.
      - sev_grid is updated to _SEV_BAD for that shot.

    Scale floor: max(1.4826 × MAD, 0.10 × median).  This prevents zero-scale
    when all shots agree exactly (MAD = 0), giving a clip window of ±10 % × median
    as a floor.

    Returns
    -------
    te_outlier_grid : bool (n_pos, n_shots, n_cycles)
        True for each shot rejected by this step.  A subset of n_bad.
    """
    te_arr = per_shot["te_log_ev"]            # (n_pos, n_shots, n_cycles)
    n_pos, n_shots, n_cycles = te_arr.shape
    te_outlier_grid = np.zeros((n_pos, n_shots, n_cycles), dtype=bool)

    for pos_idx in range(n_pos):
        for cyc_idx in range(n_cycles):
            shots = te_arr[pos_idx, :, cyc_idx]   # (n_shots,)
            fin_idx = np.flatnonzero(np.isfinite(shots))
            if fin_idx.size < 3:
                continue                           # too few shots to clip reliably
            te_vals = shots[fin_idx]
            median  = float(np.median(te_vals))
            mad     = float(np.median(np.abs(te_vals - median)))
            scale   = max(1.4826 * mad, 0.10 * median)
            if scale <= 0:
                continue
            outlier = np.abs(te_vals - median) > k_sigma * scale
            if not outlier.any():
                continue
            for local_i, shot_i in enumerate(fin_idx):
                if outlier[local_i]:
                    for q in QUANTITIES:
                        per_shot[q][pos_idx, shot_i, cyc_idx] = np.nan
                    sev_grid[pos_idx, shot_i, cyc_idx] = _SEV_BAD
                    te_outlier_grid[pos_idx, shot_i, cyc_idx] = True

    return te_outlier_grid


def _sweep_rate_vs(sweep) -> float:
    """Total voltage excursion divided by ramp duration (V/s)."""
    return (sweep.voltage_end - sweep.voltage_start) / sweep.tau_ramp_s


def _min_cutoff_hz(sweep_rate_vs: float, te_min_ev: float = TE_MIN_EV) -> float:
    """Minimum filter cutoff to resolve T_e >= te_min_ev.

    Derived from δV_filter = sweep_rate / (2π f_c) < frac × (V_p − V_f),
    where V_p − V_f = HE_VP_VF_FACTOR × T_e for helium.
    """
    window_v = HE_VP_VF_FACTOR * te_min_ev
    return sweep_rate_vs / (2.0 * np.pi * FILTER_SMEAR_FRACTION * window_v)


def _cutoff_hz_for_run(run: LapdRun, base_cutoff_hz: float) -> tuple[float, float]:
    """Return (core_cutoff_hz, edge_cutoff_hz) for this run.

    Runs at p50 or p11 in ES4 (expected cold plasma throughout) get the
    higher cutoff applied to *all* positions.  All other runs use the base
    cutoff in the core and the higher cutoff at the plasma edge (|x| > EDGE_X_CM).
    """
    rate = _sweep_rate_vs(run.config.sweep)
    high = max(base_cutoff_hz, _min_cutoff_hz(rate))

    cfg = run.config
    is_p50 = cfg.probe.port == 50
    is_p11_es4 = cfg.probe.port == 11 and cfg.experiment_set.id == 4
    if is_p50 or is_p11_es4:
        return high, high
    return base_cutoff_hz, high


def _cycle_start_times(run: LapdRun) -> np.ndarray:
    """Return the time at the start of each ramp, before any clip offset."""
    sw = run.config.sweep
    return np.array([sw.t0_s + k * sw.tau_cycle_s for k in range(sw.n_cycles)])


def _process_run(
    run: LapdRun,
    *,
    clip_us: float,
    base_cutoff_hz: float,
    order: int,
    arc_mask: np.ndarray | None = None,
) -> dict:
    """Compute position-averaged Langmuir results for every cycle in one run.

    Loads data in bulk per cycle (one HDF5 read per channel per cycle).
    Applies position-group-dependent Butterworth cutoffs: a higher cutoff is
    used at plasma-edge positions (|x| > EDGE_X_CM) and for p50 / p11-ES4
    runs throughout, to resolve potentially sub-eV electron temperatures.

    Fitting is done per individual shot before averaging, so shot-to-shot
    V_plasma fluctuations do not smear the I-V transition in the fit.

    arc_mask : bool (n_pos, n_shots, n_cycles), optional
        Pre-computed arc-shot exclusion mask.  True entries are counted in
        n_arc and n_bad and are skipped by the I-V fitter.

    Returns a dict whose values are numpy arrays:
      cycle_time_s              (n_cycles,)
      {qty}, {qty}_std,         (n_positions, n_cycles) for each qty in
      {qty}_sem                 QUANTITIES
      n_ok, n_warn, n_bad       (n_positions, n_cycles)  int32
      n_arc                     (n_positions, n_cycles)  int32, subset of n_bad
      cutoff_core_khz           scalar — filter cutoff used at core positions
      cutoff_edge_khz           scalar — filter cutoff used at edge positions
    """
    n_pos = run.position_count()      # 51
    n_shots = run.shots_per_position()  # 20
    n_cycles = run.config.sweep.n_cycles
    clip_s = clip_us * 1e-6

    core_cutoff_hz, edge_cutoff_hz = _cutoff_hz_for_run(run, base_cutoff_hz)
    edge_mask = np.abs(X_CM) > EDGE_X_CM          # bool (n_pos,)
    core_idx = np.where(~edge_mask)[0]
    edge_idx = np.where(edge_mask)[0]

    # Pre-compute zero offsets once (two HDF5 reads total).
    i_offset = run.default_zero_offset_v(ChannelKind.I_SWEEP)
    v_offset = run.default_zero_offset_v(ChannelKind.V_SWEEP)

    ramp_slices = run.sweep_ramp_sample_slices(clip_s=clip_s)

    # Accumulators: (n_pos, n_shots, n_cycles).
    per_shot = {q: np.full((n_pos, n_shots, n_cycles), np.nan) for q in QUANTITIES}
    sev_grid = np.full((n_pos, n_shots, n_cycles), _SEV_BAD, dtype=np.int8)
    # Arc-excluded shots: pre-marked bad before fitting so the fitter never
    # sees contaminated sweeps.  Tracked separately for diagnostic output.
    arc_grid = np.zeros((n_pos, n_shots, n_cycles), dtype=bool)
    if arc_mask is not None:
        arc_grid[:] = arc_mask

    sr = run.sample_rate_hz()

    for cyc_idx, ramp_slice in enumerate(ramp_slices):
        # Load and calibrate the full (n_pos, n_shots, n_ramp_samples) block.
        # langmuir_traces applies the SIS Scale/Offset headers, the zero-offset
        # correction, and the channel calibration (resistor/gain/attenuation).
        i_all = run.langmuir_traces(ChannelKind.I_SWEEP, ramp_slice, zero_offset_v=i_offset)
        v_all = run.langmuir_traces(ChannelKind.V_SWEEP, ramp_slice, zero_offset_v=v_offset)

        # Apply position-dependent cutoffs.  When core and edge cutoffs are
        # equal (p50, p11-ES4, or ES3/ES4 where min-cutoff ≤ base), filter
        # the whole block at once; otherwise filter each group separately.
        if core_cutoff_hz == edge_cutoff_hz:
            i_filt = butterworth_lowpass(i_all, sample_rate_hz=sr, cutoff_hz=core_cutoff_hz,
                                         order=order, axis=-1)
            v_filt = butterworth_lowpass(v_all, sample_rate_hz=sr, cutoff_hz=core_cutoff_hz,
                                         order=order, axis=-1)
        else:
            i_filt = np.empty_like(i_all)
            v_filt = np.empty_like(v_all)
            i_filt[core_idx] = butterworth_lowpass(i_all[core_idx], sample_rate_hz=sr,
                                                    cutoff_hz=core_cutoff_hz, order=order, axis=-1)
            v_filt[core_idx] = butterworth_lowpass(v_all[core_idx], sample_rate_hz=sr,
                                                    cutoff_hz=core_cutoff_hz, order=order, axis=-1)
            i_filt[edge_idx] = butterworth_lowpass(i_all[edge_idx], sample_rate_hz=sr,
                                                    cutoff_hz=edge_cutoff_hz, order=order, axis=-1)
            v_filt[edge_idx] = butterworth_lowpass(v_all[edge_idx], sample_rate_hz=sr,
                                                    cutoff_hz=edge_cutoff_hz, order=order, axis=-1)

        for pos_idx in range(n_pos):
            peer = i_filt[pos_idx]  # (n_shots, n_ramp_samples) for arc detection
            for shot_idx in range(n_shots):
                if arc_grid[pos_idx, shot_idx, cyc_idx]:
                    # Shot flagged by pre-computed arc detector; skip fitting.
                    # sev_grid already initialised to _SEV_BAD.
                    continue
                voltage = v_filt[pos_idx, shot_idx]
                current = i_filt[pos_idx, shot_idx]
                try:
                    analysis = analyze_langmuir_sweep(voltage, current)
                    report = evaluate_langmuir_quality(
                        analysis,
                        current=current,
                        peer_current=peer,
                    )
                    sev_code = _SEV_MAP.get(report.severity, _SEV_BAD)
                    sev_grid[pos_idx, shot_idx, cyc_idx] = sev_code

                    if sev_code != _SEV_BAD:
                        per_shot["te_log_ev"][pos_idx, shot_idx, cyc_idx] = (
                            analysis.log_linear_fit.electron_temperature_ev
                        )
                        per_shot["te_exp_ev"][pos_idx, shot_idx, cyc_idx] = (
                            analysis.exponential_fit.electron_temperature_ev
                        )
                        per_shot["vp_derivative_v"][pos_idx, shot_idx, cyc_idx] = (
                            analysis.plasma_potential_derivative_v
                        )
                        vp_log = analysis.plasma_potential_log_intersection_v
                        per_shot["vp_log_v"][pos_idx, shot_idx, cyc_idx] = (
                            vp_log if vp_log is not None else np.nan
                        )
                        vp_exp = analysis.plasma_potential_exp_intersection_v
                        per_shot["vp_exp_v"][pos_idx, shot_idx, cyc_idx] = (
                            vp_exp if vp_exp is not None else np.nan
                        )
                except Exception:
                    sev_grid[pos_idx, shot_idx, cyc_idx] = _SEV_BAD

        pct = 100 * (cyc_idx + 1) / n_cycles
        sys.stdout.write(f"\r  cycle {cyc_idx + 1}/{n_cycles} ({pct:.0f}%)")
        sys.stdout.flush()

    sys.stdout.write("\n")

    # Snapshot raw per-shot Te BEFORE sigma-clip for HDF5 storage.
    # Stored as float32 to halve the footprint; NaN for bad/arc shots.
    te_log_shots_raw = per_shot["te_log_ev"].copy().astype(np.float32)

    # Shot-level quality array before clip: 0=ok, 1=warn, 2=bad, 3=arc.
    # Arc shots override whatever sev_grid stored (they never entered the fitter).
    sev_shots = sev_grid.astype(np.int8).copy()
    sev_shots[arc_grid] = 3

    # Sigma-clip per-cell Te outliers before averaging.
    # Any shot whose Te deviates more than K_SIGMA_CLIP × MAD-scale from the
    # cell median is nulled out and re-classified as bad.  This removes the
    # single-shot outliers that inflate the cell mean and std without raising
    # a quality flag (the shot itself may look like a valid I-V curve in
    # isolation, but is inconsistent with the other shots at that position).
    te_outlier_grid = _sigma_clip_te(per_shot, sev_grid)
    n_te_outlier_total = int(te_outlier_grid.sum())
    if n_te_outlier_total:
        sys.stdout.write(
            f"  Te sigma-clip: {n_te_outlier_total} shot(s) removed "
            f"(>{K_SIGMA_CLIP:.0f}×MAD from cell median)\n"
        )

    # Average over the shot axis (axis=1), excluding bad shots (NaN entries).
    result: dict = {"cycle_time_s": _cycle_start_times(run)}
    for q in QUANTITIES:
        arr = per_shot[q]  # (n_pos, n_shots, n_cycles)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(arr, axis=1)
            std = np.nanstd(arr, axis=1, ddof=1)
            n_finite = np.isfinite(arr).sum(axis=1)
            sem = np.where(n_finite > 1, std / np.sqrt(n_finite), np.nan)
        result[q] = mean
        result[f"{q}_std"] = std
        result[f"{q}_sem"] = sem

    result["n_ok"] = (sev_grid == _SEV_OK).sum(axis=1).astype(np.int32)
    result["n_warn"] = (sev_grid == _SEV_WARN).sum(axis=1).astype(np.int32)
    result["n_bad"] = (sev_grid == _SEV_BAD).sum(axis=1).astype(np.int32)
    # n_arc and n_te_outlier are subsets of n_bad (non-overlapping with each other).
    result["n_arc"] = arc_grid.sum(axis=1).astype(np.int32)
    result["n_te_outlier"] = te_outlier_grid.sum(axis=1).astype(np.int32)
    # Per-shot arrays — stored pre-sigma-clip so downstream code can apply
    # any filtering strategy (different k, median, etc.) without reprocessing.
    result["te_log_ev_shots"] = te_log_shots_raw  # (n_pos, n_shots, n_cycles)
    result["sev_shots"] = sev_shots                # (n_pos, n_shots, n_cycles)
    result["cutoff_core_khz"] = core_cutoff_hz / 1e3
    result["cutoff_edge_khz"] = edge_cutoff_hz / 1e3
    return result


_SCALAR_ATTRS = {"cutoff_core_khz", "cutoff_edge_khz"}


def _write_run_group(grp: h5py.Group, result: dict) -> None:
    for key, value in result.items():
        if key in _SCALAR_ATTRS:
            grp.attrs[key] = float(value)
        else:
            grp.create_dataset(key, data=np.asarray(value), compression="gzip", compression_opts=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-ids",
        default="all",
        help="Comma-separated run IDs like 01,46 or 'all' (default).",
    )
    parser.add_argument("--clip-us", type=float, default=10.0, help="Edge trim per ramp side (µs).")
    parser.add_argument(
        "--cutoff-khz", type=float, default=100.0,
        help=(
            "Base low-pass filter cutoff (kHz) applied to core positions at standard ports."
            "  Higher cutoffs are applied automatically to p50 runs, p11 in ES4, and all edge"
            " positions (|x| > %.0f cm) based on the sweep rate and a %.1f eV resolution target."
        ) % (EDGE_X_CM, TE_MIN_EV),
    )
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order.")
    parser.add_argument("--output", type=Path, default=HDF5_OUTPUT)
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=None,
        metavar="CSV",
        help=(
            "Arc-shot exclusion CSV produced by the arc detector.  Defaults to"
            f" {ARC_EXCLUSIONS_CSV} if that file exists, otherwise no exclusions"
            " are applied."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing HDF5 and skip runs that are already complete.",
    )
    args = parser.parse_args()

    # Resolve arc exclusion CSV.
    excl_path = args.exclusions
    if excl_path is None and ARC_EXCLUSIONS_CSV.exists():
        excl_path = ARC_EXCLUSIONS_CSV
    if excl_path is not None:
        sys.stdout.write(f"Loading arc exclusions from {excl_path}\n")
        arc_exclusions = _load_arc_exclusions(excl_path)
        n_excl_runs = len(arc_exclusions)
        n_excl_total = sum(len(v) for v in arc_exclusions.values())
        sys.stdout.write(f"  {n_excl_total} exclusions across {n_excl_runs} runs\n")
    else:
        arc_exclusions = {}
        sys.stdout.write("No arc exclusion file found; all shots will be fitted.\n")

    dataset = LapdDataset.from_manifest(MANIFEST)
    if args.run_ids == "all":
        run_ids = dataset.run_ids()
    else:
        run_ids = sorted(v.strip() for v in args.run_ids.split(",") if v.strip())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if (args.resume and args.output.exists()) else "w"

    with h5py.File(args.output, file_mode) as hf:
        if "x_cm" not in hf:
            ds = hf.create_dataset("x_cm", data=X_CM)
            ds.attrs["description"] = (
                "Probe scan positions: 51 points from -25 to +25 cm, 1 cm spacing, centered at x=0"
            )
        if "experiment_sets" not in hf:
            hf.create_group("experiment_sets")
        es_grp = hf["experiment_sets"]

        total_ok = total_warn = total_bad = 0
        for run_id in run_ids:
            cfg = dataset.config(run_id)
            exp_set = cfg.experiment_set
            es_id = str(exp_set.id)

            # Skip runs already fully written (presence of cycle_time_s is the
            # completion marker — it is the last dataset written per run).
            if (
                args.resume
                and es_id in es_grp
                and run_id in es_grp[es_id]
                and "cycle_time_s" in es_grp[es_id][run_id]
            ):
                sys.stdout.write(f"run {run_id}  — already complete, skipping\n")
                n_ok = int(es_grp[es_id][run_id]["n_ok"][:].sum())
                n_warn = int(es_grp[es_id][run_id]["n_warn"][:].sum())
                n_bad = int(es_grp[es_id][run_id]["n_bad"][:].sum())
                total_ok += n_ok
                total_warn += n_warn
                total_bad += n_bad
                continue

            # Create experiment-set group if this is the first run in the set.
            if es_id not in es_grp:
                g = es_grp.create_group(es_id)
                g.attrs["label"] = exp_set.label
                g.attrs["v_bank_v"] = exp_set.v_bank
                g.attrs["v_puff_v"] = exp_set.v_puff
                if exp_set.description:
                    g.attrs["description"] = exp_set.description

            # Remove a partially-written run group before reprocessing.
            if run_id in es_grp[es_id]:
                del es_grp[es_id][run_id]

            run = dataset.run(run_id)
            rot = cfg.probe.rotation_deg
            core_c, edge_c = _cutoff_hz_for_run(run, args.cutoff_khz * 1e3)
            sys.stdout.write(
                f"run {run_id}  exp_set={es_id}  port={cfg.probe.port}"
                f"  z={cfg.probe.z_cm:.1f} cm  rot={rot:.0f}°"
                f"  {run.config.sweep.n_cycles} cycles"
                f"  cutoff={core_c/1e3:.0f}/{edge_c/1e3:.0f} kHz (core/edge)\n"
            )

            n_pos = run.position_count()
            n_shots = run.shots_per_position()
            n_cycles = run.config.sweep.n_cycles
            arc_mask = _build_arc_mask(
                run_id, arc_exclusions, n_pos, n_shots, n_cycles
            )
            n_arc_run = int(arc_mask.sum())
            if n_arc_run:
                sys.stdout.write(f"  arc exclusions: {n_arc_run} shot-cycles pre-excluded\n")

            result = _process_run(
                run,
                clip_us=args.clip_us,
                base_cutoff_hz=args.cutoff_khz * 1e3,
                order=args.order,
                arc_mask=arc_mask,
            )

            run_grp = es_grp[es_id].create_group(run_id)
            run_grp.attrs["run_id"] = run_id
            run_grp.attrs["rotation_deg"] = float(rot)
            run_grp.attrs["port"] = int(cfg.probe.port)
            run_grp.attrs["z_cm"] = float(cfg.probe.z_cm)
            run_grp.attrs["experiment_set_id"] = int(exp_set.id)
            _write_run_group(run_grp, result)

            # Flush after each run so completed work survives a crash.
            hf.flush()

            n_ok = int(result["n_ok"].sum())
            n_warn = int(result["n_warn"].sum())
            n_bad = int(result["n_bad"].sum())
            n_teo = int(result["n_te_outlier"].sum())
            total_ok += n_ok
            total_warn += n_warn
            total_bad += n_bad
            sys.stdout.write(
                f"  → ok={n_ok}  warn={n_warn}  bad={n_bad}"
                + (f"  (te_outlier={n_teo})" if n_teo else "")
                + "\n"
            )

    sys.stdout.write(f"\nWrote {args.output}\n")
    sys.stdout.write(f"Grand total — ok={total_ok}  warn={total_warn}  bad={total_bad}\n")


if __name__ == "__main__":
    main()
