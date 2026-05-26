"""Compute electron density, Mach number, and plasma FWHM for all LAPD runs.

For each run the script reads the dead-time (inter-sweep) periods of the
Isat and Isweep channels, combines them with the shot-averaged T_e from
langmuir_sweeps.hdf5, and derives:

  n_e_R  — electron density from the Isat face (A_p_R, right/downstream)
  n_e_L  — electron density from the -Isweep face (A_p_L, left/upstream)
  mach   — Mach number M = ln(I_u / I_d) / K, area-normalised
  plasma_fwhm_cm — FWHM of the n_e_R radial profile at each cycle time

Physics
-------
Density:  n_e = I_sat / [exp(-1/2) * A_p * e * C_s]
          C_s = sqrt(k_B * T_e / m_i)  (ion sound speed)

Mach (rot=0, Isweep upstream):
  M = ln(n_e_L / n_e_R) / K

Mach (rot=180, Isweep downstream):
  M = ln(n_e_R / n_e_L) / K

Here n_e_L/n_e_R = (I_L/A_p_L) / (I_R/A_p_R), so the exp(-1/2)*e*C_s
factors cancel and M is simply the area-normalised current log-ratio.

Probe face convention (same as calibrate_probe_areas.py)
---------------------------------------------------------
A_p_L: left/upstream face (toward cathode at z=0), Isweep channel (negated).
A_p_R: right/downstream face, Isat channel.

rot=0 → Isweep (A_p_L) faces upstream; Isat (A_p_R) faces downstream.
rot=180 → Isweep (A_p_L) faces downstream; Isat (A_p_R) faces upstream.

Probe A (ports 11/50) has no interferometer; its area is estimated from
probe B.  Mach numbers are not computed for probe A.

Constants (edit at top of file)
--------------------------------
M_I_AMU = 4.003    He-4 ion mass
MACH_K  = 1.66     Chung et al.; M = ln(I_u/I_d) / MACH_K
# Shadow offset ≈ 0.244 stored in density.MACH_SHADOW_OFFSET; not applied here.

HDF5 output layout
------------------
/x_cm                              (51,) float64  scan positions in cm
/experiment_sets/
  {set_id}/                        attrs: label, v_bank_v, v_puff_v
    {run_id}/                      attrs: run_id, rotation_deg, port, z_cm,
                                          probe_id, ap_L_cm2, ap_R_cm2, ap_estimated
      inter_sweep_time_s           (n_cycles,) float64  dead-time midpoints
      n_e_R_m3                     (51, n_cycles) float64
      n_e_R_m3_std                 (51, n_cycles) float64  shot-to-shot std
      n_e_L_m3                     (51, n_cycles) float64
      n_e_L_m3_std                 (51, n_cycles) float64
      mach                         (51, n_cycles) float64  NaN for probe A
      mach_std                     (51, n_cycles) float64
      plasma_fwhm_cm               (n_cycles,) float64

Inputs
------
  config/may2026_run_manifest.toml
  processed/probe_area_calibration.toml  (from calibrate_probe_areas.py)
  processed/langmuir_sweeps.hdf5         (from process_langmuir_sweeps.py)
  data/may2026/*.hdf5

Output
------
  processed/density_mach.hdf5

Usage
-----
  python scripts/compute_density_mach.py
  python scripts/compute_density_mach.py --run-ids 02 03
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bapsf_lapd import ChannelKind, LapdDataset, LapdRun
from bapsf_lapd.density import (
    density_fwhm_cm,
    electron_density_m3,
    inter_sweep_sample_slices,
    ion_sound_speed_m_s,
)


MANIFEST = Path("config/may2026_run_manifest.toml")
CALIB_TOML = Path("processed/probe_area_calibration.toml")
SWEEPS_HDF5 = Path("processed/langmuir_sweeps.hdf5")
HDF5_OUTPUT = Path("processed/density_mach.hdf5")

# He-4 ion mass (amu).  Must match calibrate_probe_areas.py.
M_I_AMU = 4.003

# Mach probe calibration constant (Chung et al.).
# Formula: M = ln(I_upstream / I_downstream) / MACH_K
# Shadow offset ln(3/2)/1.66 ≈ 0.244 is stored in density.MACH_SHADOW_OFFSET but
# NOT applied here — record M_measured and apply post-hoc for high-flow cases.
# TODO: empirical shadow correction for 1.4 kG B-field (intermediate ρ_i regime).
MACH_K = 1.66

# Edge clip at both ends of each dead-time window to avoid ramp transients.
CLIP_S = 10e-6

# Probe scan positions: 51 points, 1 cm spacing, centred at x = 0.
X_CM = np.linspace(-25.0, 25.0, 51)

# Physical probe identity from the second digit of the two-digit run_id.
PROBE_FROM_DIGIT = {1: "A", 8: "A", 2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}


def _probe_id(run_id: str) -> str:
    return PROBE_FROM_DIGIT[int(run_id[1])]


def _load_calibration(path: Path) -> dict[str, dict[str, Any]]:
    """Load probe area calibration from TOML.  Returns keyed by "A"/"B"/"C"/"D"."""
    with open(path, "rb") as f:
        toml = tomllib.load(f)
    result = {}
    for probe_key in ("probe_A", "probe_B", "probe_C", "probe_D"):
        short = probe_key.replace("probe_", "")
        data = toml[probe_key]
        result[short] = {
            "ap_L_m2": data["ap_L_cm2"] * 1e-4,   # cm² → m²
            "ap_R_m2": data["ap_R_cm2"] * 1e-4,
            "ap_L_cm2": data["ap_L_cm2"],
            "ap_R_cm2": data["ap_R_cm2"],
            "estimated": bool(data.get("estimated", False)),
        }
    return result


def _inter_sweep_times_s(run: LapdRun) -> np.ndarray:
    """Midpoint time (s from trigger) of each dead-time period."""
    sw = run.config.sweep
    return np.array([
        sw.t0_s + k * sw.tau_cycle_s + 0.5 * (sw.tau_ramp_s + sw.tau_cycle_s)
        for k in range(sw.n_cycles)
    ])


def process_run(
    run: LapdRun,
    ap_R_m2: float,
    ap_L_m2: float,
    probe_id: str,
    te_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute density, Mach, and FWHM for one run.

    te_grid: (51, n_cycles) shot-averaged T_e in eV (NaN for bad-quality positions).

    Returns arrays keyed by output dataset name.
    """
    sw = run.config.sweep
    n_pos = run.config.acquisition.n_positions
    n_cycles = sw.n_cycles
    n_shots = run.config.acquisition.n_shots_per_position
    rotation_deg = run.config.probe.rotation_deg or 0

    dead_slices = inter_sweep_sample_slices(sw, run.config.acquisition, clip_s=CLIP_S)
    inter_times_s = _inter_sweep_times_s(run)

    isat_offset = run.default_zero_offset_v(ChannelKind.ISAT)
    isweep_offset = run.default_zero_offset_v(ChannelKind.I_SWEEP)

    n_e_R = np.full((n_pos, n_cycles), np.nan)
    n_e_R_std = np.full((n_pos, n_cycles), np.nan)
    n_e_L = np.full((n_pos, n_cycles), np.nan)
    n_e_L_std = np.full((n_pos, n_cycles), np.nan)
    mach = np.full((n_pos, n_cycles), np.nan)
    mach_std = np.full((n_pos, n_cycles), np.nan)
    fwhm = np.full(n_cycles, np.nan)

    for k in range(n_cycles):
        # Dead-time traces: (51, 20, n_dead_samples)
        traces_R = run.langmuir_traces(ChannelKind.ISAT, dead_slices[k], zero_offset_v=isat_offset)
        traces_L = run.langmuir_traces(ChannelKind.I_SWEEP, dead_slices[k], zero_offset_v=isweep_offset)

        # Per-shot dead-time mean: (51, 20)
        isat_R_shot = traces_R.mean(axis=2)
        isat_L_shot = -traces_L.mean(axis=2)  # negate Isweep polarity

        # T_e and sound speed at cycle k: (51,)
        te_k = te_grid[:, k] if k < te_grid.shape[1] else np.full(n_pos, np.nan)
        with np.errstate(invalid="ignore"):
            cs_k = ion_sound_speed_m_s(te_k, M_I_AMU)

        # Per-shot density: (51, 20) — cs_k broadcast via [:, None]
        with np.errstate(all="ignore"):
            n_e_R_shot = electron_density_m3(isat_R_shot, ap_R_m2, cs_k[:, None])
            n_e_L_shot = electron_density_m3(isat_L_shot, ap_L_m2, cs_k[:, None])

        # Shot-averaged density and std: (51,)
        with np.errstate(all="ignore"):
            n_e_R[:, k] = np.nanmean(n_e_R_shot, axis=1)
            n_e_L[:, k] = np.nanmean(n_e_L_shot, axis=1)
            if n_shots > 1:
                n_e_R_std[:, k] = np.nanstd(n_e_R_shot, axis=1, ddof=1)
                n_e_L_std[:, k] = np.nanstd(n_e_L_shot, axis=1, ddof=1)

        # Mach (probes B/C/D only)
        if probe_id != "A":
            with np.errstate(all="ignore"):
                # rot=0: Isweep/A_p_L faces upstream → n_e_L is upstream density
                # rot=180: Isat/A_p_R faces upstream → n_e_R is upstream density
                if rotation_deg == 0:
                    M_shot = np.log(n_e_L_shot / n_e_R_shot) / MACH_K
                else:
                    M_shot = np.log(n_e_R_shot / n_e_L_shot) / MACH_K
                mach[:, k] = np.nanmean(M_shot, axis=1)
                if n_shots > 1:
                    mach_std[:, k] = np.nanstd(M_shot, axis=1, ddof=1)

        # FWHM from n_e_R radial profile (the Isat face, available for all runs)
        fwhm[k] = density_fwhm_cm(n_e_R[:, k], X_CM)

    return {
        "inter_sweep_time_s": inter_times_s,
        "n_e_R_m3": n_e_R,
        "n_e_R_m3_std": n_e_R_std,
        "n_e_L_m3": n_e_L,
        "n_e_L_m3_std": n_e_L_std,
        "mach": mach,
        "mach_std": mach_std,
        "plasma_fwhm_cm": fwhm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-ids",
        nargs="+",
        default=["all"],
        metavar="ID",
        help="Two-digit run IDs to process (e.g. 02 03), or 'all'.",
    )
    args = parser.parse_args()
    target_ids: set[str] | None = None if "all" in args.run_ids else set(args.run_ids)

    if not CALIB_TOML.exists():
        sys.exit(f"Missing {CALIB_TOML} — run calibrate_probe_areas.py first.")
    if not SWEEPS_HDF5.exists():
        sys.exit(f"Missing {SWEEPS_HDF5} — run process_langmuir_sweeps.py first.")

    dataset = LapdDataset.from_manifest(MANIFEST)
    calibration = _load_calibration(CALIB_TOML)

    HDF5_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(HDF5_OUTPUT, "w") as out_hdf,
        h5py.File(SWEEPS_HDF5, "r") as sweeps_hdf,
    ):
        out_hdf.create_dataset("x_cm", data=X_CM)
        out_hdf["x_cm"].attrs["description"] = (
            "Probe scan positions in cm, centred at x = 0."
        )
        out_hdf["x_cm"].attrs["m_i_amu"] = M_I_AMU
        out_hdf["x_cm"].attrs["mach_K"] = MACH_K

        es_grp = out_hdf.create_group("experiment_sets")

        for set_id in dataset.experiment_set_ids():
            run_ids = dataset.experiment_set_run_ids(set_id)
            exp = dataset.config(run_ids[0]).experiment_set

            g_set = es_grp.create_group(str(set_id))
            g_set.attrs["label"] = exp.label
            g_set.attrs["v_bank_v"] = exp.v_bank
            g_set.attrs["v_puff_v"] = exp.v_puff

            for run_id in run_ids:
                if target_ids is not None and run_id not in target_ids:
                    continue

                hdf_key = f"experiment_sets/{set_id}/{run_id}"
                if hdf_key not in sweeps_hdf:
                    print(f"  [skip] run {run_id}: not found in {SWEEPS_HDF5}")
                    continue

                pid = _probe_id(run_id)
                calib = calibration[pid]
                run = dataset.run(run_id)
                cfg = run.config

                print(
                    f"  run {run_id}  probe={pid}  set={set_id}"
                    f"  rot={cfg.probe.rotation_deg}°  port={cfg.probe.port}"
                )

                te_grid = sweeps_hdf[f"{hdf_key}/te_log_ev"][()]  # (51, n_cycles)

                results = process_run(
                    run,
                    ap_R_m2=calib["ap_R_m2"],
                    ap_L_m2=calib["ap_L_m2"],
                    probe_id=pid,
                    te_grid=te_grid,
                )

                g_run = g_set.create_group(run_id)
                g_run.attrs["run_id"] = run_id
                g_run.attrs["rotation_deg"] = float(cfg.probe.rotation_deg or 0)
                g_run.attrs["port"] = int(cfg.probe.port or 0)
                g_run.attrs["z_cm"] = float(cfg.probe.z_cm or 0)
                g_run.attrs["probe_id"] = pid
                g_run.attrs["ap_L_cm2"] = calib["ap_L_cm2"]
                g_run.attrs["ap_R_cm2"] = calib["ap_R_cm2"]
                g_run.attrs["ap_estimated"] = calib["estimated"]

                for key, arr in results.items():
                    g_run.create_dataset(key, data=arr)

                peak = float(np.nanmax(results["n_e_R_m3"]))
                fwhm_mid = float(np.nanmedian(results["plasma_fwhm_cm"]))
                print(f"    peak n_e_R ≈ {peak:.2e} m⁻³   median FWHM ≈ {fwhm_mid:.1f} cm")

    print(f"\nWrote {HDF5_OUTPUT}")


if __name__ == "__main__":
    main()
