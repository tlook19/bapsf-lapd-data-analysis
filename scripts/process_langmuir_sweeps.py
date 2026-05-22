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
      n_bad                     (51, n_cycles)  int32

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
import sys
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

# Probe scan positions: 51 points, 1 cm spacing, centered at x=0.
X_CM = np.linspace(-25.0, 25.0, 51)

QUANTITIES = ("te_log_ev", "te_exp_ev", "vp_derivative_v", "vp_log_v", "vp_exp_v")

# Integer codes for severity stored in per-shot accumulation arrays.
_SEV_OK = 0
_SEV_WARN = 1
_SEV_BAD = 2
_SEV_MAP = {"ok": _SEV_OK, "warn": _SEV_WARN, "bad": _SEV_BAD}


def _cycle_start_times(run: LapdRun) -> np.ndarray:
    """Return the time at the start of each ramp, before any clip offset."""
    sw = run.config.sweep
    return np.array([sw.t0_s + k * sw.tau_cycle_s for k in range(sw.n_cycles)])


def _process_run(
    run: LapdRun,
    *,
    clip_us: float,
    cutoff_khz: float,
    order: int,
) -> dict:
    """Compute position-averaged Langmuir results for every cycle in one run.

    Loads data in bulk per cycle (one HDF5 read per channel per cycle) and
    applies the Butterworth filter across all 1020 shots at once to avoid
    redundant I/O and filter setup.

    Returns a dict whose values are numpy arrays:
      cycle_time_s              (n_cycles,)
      {qty}, {qty}_std,         (n_positions, n_cycles) for each qty in
      {qty}_sem                 QUANTITIES
      n_ok, n_warn, n_bad       (n_positions, n_cycles)  int32
    """
    n_pos = run.position_count()      # 51
    n_shots = run.shots_per_position()  # 20
    n_cycles = run.config.sweep.n_cycles
    clip_s = clip_us * 1e-6

    # Pre-compute zero offsets once (two HDF5 reads total).
    i_offset = run.default_zero_offset_v(ChannelKind.I_SWEEP)
    v_offset = run.default_zero_offset_v(ChannelKind.V_SWEEP)

    ramp_slices = run.sweep_ramp_sample_slices(clip_s=clip_s)

    # Accumulators: (n_pos, n_shots, n_cycles).
    per_shot = {q: np.full((n_pos, n_shots, n_cycles), np.nan) for q in QUANTITIES}
    sev_grid = np.full((n_pos, n_shots, n_cycles), _SEV_BAD, dtype=np.int8)

    for cyc_idx, ramp_slice in enumerate(ramp_slices):
        # Load and calibrate the full (n_pos, n_shots, n_ramp_samples) block.
        # langmuir_traces applies the SIS Scale/Offset headers, the zero-offset
        # correction, and the channel calibration (resistor/gain/attenuation).
        i_all = run.langmuir_traces(ChannelKind.I_SWEEP, ramp_slice, zero_offset_v=i_offset)
        v_all = run.langmuir_traces(ChannelKind.V_SWEEP, ramp_slice, zero_offset_v=v_offset)

        # Filter the entire block at once along the sample axis.
        sr = run.sample_rate_hz()
        cutoff = cutoff_khz * 1e3
        i_filt = butterworth_lowpass(i_all, sample_rate_hz=sr, cutoff_hz=cutoff, order=order, axis=-1)
        v_filt = butterworth_lowpass(v_all, sample_rate_hz=sr, cutoff_hz=cutoff, order=order, axis=-1)

        for pos_idx in range(n_pos):
            peer = i_filt[pos_idx]  # (n_shots, n_ramp_samples) for arc detection
            for shot_idx in range(n_shots):
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

    # Average over the shot axis (axis=1), excluding bad shots (NaN entries).
    result: dict = {"cycle_time_s": _cycle_start_times(run)}
    for q in QUANTITIES:
        arr = per_shot[q]  # (n_pos, n_shots, n_cycles)
        with np.errstate(all="ignore"):
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
    return result


def _write_run_group(grp: h5py.Group, result: dict) -> None:
    for key, value in result.items():
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
    parser.add_argument("--cutoff-khz", type=float, default=100.0, help="Low-pass filter cutoff (kHz).")
    parser.add_argument("--order", type=int, default=4, help="Butterworth filter order.")
    parser.add_argument("--output", type=Path, default=HDF5_OUTPUT)
    args = parser.parse_args()

    dataset = LapdDataset.from_manifest(MANIFEST)
    if args.run_ids == "all":
        run_ids = dataset.run_ids()
    else:
        run_ids = sorted(v.strip() for v in args.run_ids.split(",") if v.strip())

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output, "w") as hf:
        ds = hf.create_dataset("x_cm", data=X_CM)
        ds.attrs["description"] = (
            "Probe scan positions: 51 points from -25 to +25 cm, 1 cm spacing, centered at x=0"
        )

        es_grp = hf.create_group("experiment_sets")

        total_ok = total_warn = total_bad = 0
        for run_id in run_ids:
            cfg = dataset.config(run_id)
            run = dataset.run(run_id)
            exp_set = cfg.experiment_set
            es_id = str(exp_set.id)

            if es_id not in es_grp:
                g = es_grp.create_group(es_id)
                g.attrs["label"] = exp_set.label
                g.attrs["v_bank_v"] = exp_set.v_bank
                g.attrs["v_puff_v"] = exp_set.v_puff
                if exp_set.description:
                    g.attrs["description"] = exp_set.description

            rot = cfg.probe.rotation_deg
            sys.stdout.write(
                f"run {run_id}  exp_set={es_id}  port={cfg.probe.port}"
                f"  z={cfg.probe.z_cm:.1f} cm  rot={rot:.0f}°"
                f"  {run.config.sweep.n_cycles} cycles\n"
            )

            result = _process_run(run, clip_us=args.clip_us, cutoff_khz=args.cutoff_khz, order=args.order)

            run_grp = es_grp[es_id].create_group(run_id)
            run_grp.attrs["run_id"] = run_id
            run_grp.attrs["rotation_deg"] = float(rot)
            run_grp.attrs["port"] = int(cfg.probe.port)
            run_grp.attrs["z_cm"] = float(cfg.probe.z_cm)
            run_grp.attrs["experiment_set_id"] = int(exp_set.id)
            _write_run_group(run_grp, result)

            n_ok = int(result["n_ok"].sum())
            n_warn = int(result["n_warn"].sum())
            n_bad = int(result["n_bad"].sum())
            total_ok += n_ok
            total_warn += n_warn
            total_bad += n_bad
            sys.stdout.write(f"  → ok={n_ok}  warn={n_warn}  bad={n_bad}\n")

    sys.stdout.write(f"\nWrote {args.output}\n")
    sys.stdout.write(f"Grand total — ok={total_ok}  warn={total_warn}  bad={total_bad}\n")


if __name__ == "__main__":
    main()
