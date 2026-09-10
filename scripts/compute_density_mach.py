"""Compute electron density, Mach number, and plasma FWHM for all LAPD runs.

For each run the script reads the dead-time (inter-sweep) periods of the
Isat and Isweep channels, combines them with the filled T_e map from
te_filled.hdf5, and derives:

  n_e_R  — electron density from the Isat face (A_p_R, right/downstream)
  n_e_L  — electron density from the -Isweep face (A_p_L, left/upstream)
  mach   — Mach number M = ln(I_u / I_d) / K, area-normalised
  velocity_km_s — parallel flow speed M * C_s in km/s
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
MACH_K  = 1.66     Chung convention; M = ln(I_u/I_d) / MACH_K

The Mach error model (K bracket, face asymmetry, probe-wake bracket) is
declared in bapsf_lapd.density.  None of it is applied here: this script
records M_measured.

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
      cs_m_s                       (51, n_cycles) float64  from filled T_e
      velocity_km_s                (51, n_cycles) float64
      velocity_km_s_std            (51, n_cycles) float64
      plasma_fwhm_cm               (n_cycles,) float64

Inputs
------
  config/may2026_run_manifest.toml
  processed/probe_area_calibration.toml  (from calibrate_probe_areas.py)
  processed/te_filled.hdf5               (from fit_te_spatial.py)
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

from bapsf_lapd import (
    ChannelKind,
    LapdDataset,
    LapdRun,
    density_area_key_for_deadtime_source,
    effective_rotation_deg,
    electrical_connections_swapped,
)
from bapsf_lapd.density import (
    apply_probe_a_area_factor,
    density_fwhm_cm,
    electron_density_m3,
    inter_sweep_sample_slices,
    ion_sound_speed_m_s,
    load_probe_a_area_calibration,
)


MANIFEST = Path("config/may2026_run_manifest.toml")
CALIB_TOML = Path("processed/probe_area_calibration.toml")
PROBE_A_CALIB_TOML = Path("config/may2026_probe_a_area_calibration.toml")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
HDF5_OUTPUT = Path("processed/density_mach.hdf5")

# He-4 ion mass (amu).  Must match calibrate_probe_areas.py.
M_I_AMU = 4.003

# Mach probe calibration constant (Chung convention).
# Formula: M = ln(I_upstream / I_downstream) / MACH_K
# The value is a convention inside density.MACH_K_BRACKET, and the recorded
# Mach numbers carry it as the mach_K attribute.  The three systematics that
# accompany a reading — the K bracket, density.MACH_FACE_ASYMMETRY_M_RMS and
# the per-point density.MACH_SHADOW_BRACKET_M — are disclosed there and are
# NOT applied here; this script records M_measured.
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


def _interp_filled_te_to_deadtime(
    te_hdf: h5py.File,
    set_id: int,
    z_cm: float,
    dead_time_s: np.ndarray,
) -> np.ndarray:
    """Return filled T_e on the run's (x, dead-time) grid."""
    grp = te_hdf[f"experiment_sets/{set_id}"]
    x_cm = grp["x_cm"][()]
    if not np.allclose(x_cm, X_CM):
        raise ValueError(f"Filled T_e x grid differs from density grid for experiment set {set_id}")

    z_grid = grp["z_cm"][()]
    z_idx = int(np.argmin(np.abs(z_grid - z_cm)))
    te_time_ms = grp["cycle_time_ms"][()]
    dead_time_ms = dead_time_s * 1000.0
    te_z = grp["te_filled"][z_idx, :, :]
    return np.vstack([
        np.interp(dead_time_ms, te_time_ms, te_z[xi, :])
        for xi in range(te_z.shape[0])
    ])


def process_run(
    run: LapdRun,
    calib_m2: dict[str, float],
    probe_id: str,
    te_grid: np.ndarray,
    probe_a_calibration,
) -> dict[str, np.ndarray]:
    """Compute density, Mach, and FWHM for one run.

    te_grid: (51, n_cycles) filled T_e in eV, aligned to dead-time midpoints.

    ``calib_m2`` is the probe's calibration dict (``ap_L_m2``/``ap_R_m2``, and
    the ``_cm2`` twins) keyed by ELECTRODE, as loaded by ``_load_calibration``.
    The area applied to each channel's dead-time current follows the
    electrode that channel physically reached, per
    ``density_area_key_for_deadtime_source`` -- on a nominal run ISAT reads
    the right electrode and I_SWEEP the left one; on a run in
    ``ELECTRICAL_SWAP_RUN_IDS`` the two cables are crossed at the connector
    and the areas exchange with them.

    Returns arrays keyed by output dataset name.
    """
    sw = run.config.sweep
    n_pos = run.config.acquisition.n_positions
    n_cycles = sw.n_cycles
    n_shots = run.config.acquisition.n_shots_per_position
    run_id = run.config.run_id
    rotation_deg = effective_rotation_deg(run_id, run.config.probe.rotation_deg)
    connections_swapped = electrical_connections_swapped(run_id)
    area_isat_m2 = calib_m2[
        density_area_key_for_deadtime_source(run_id, ChannelKind.ISAT).replace("cm2", "m2")
    ]
    area_isweep_m2 = calib_m2[
        density_area_key_for_deadtime_source(run_id, ChannelKind.I_SWEEP).replace("cm2", "m2")
    ]

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
    cs_m_s = np.full((n_pos, n_cycles), np.nan)
    velocity_km_s = np.full((n_pos, n_cycles), np.nan)
    velocity_km_s_std = np.full((n_pos, n_cycles), np.nan)
    fwhm = np.full(n_cycles, np.nan)

    for k in range(n_cycles):
        # Dead-time traces: (51, 20, n_dead_samples)
        traces_R = run.langmuir_traces(ChannelKind.ISAT, dead_slices[k], zero_offset_v=isat_offset)
        traces_L = run.langmuir_traces(ChannelKind.I_SWEEP, dead_slices[k], zero_offset_v=isweep_offset)

        # Per-shot dead-time mean: (51, 20)
        isat_R_shot = traces_R.mean(axis=2)
        isat_L_shot = -traces_L.mean(axis=2)  # negate Isweep polarity
        isat_R_shot = apply_probe_a_area_factor(
            isat_R_shot, probe_id, probe_a_calibration
        )
        isat_L_shot = apply_probe_a_area_factor(
            isat_L_shot, probe_id, probe_a_calibration
        )

        # T_e and sound speed at cycle k: (51,)
        te_k = te_grid[:, k] if k < te_grid.shape[1] else np.full(n_pos, np.nan)
        with np.errstate(invalid="ignore"):
            cs_k = ion_sound_speed_m_s(te_k, M_I_AMU)
        cs_m_s[:, k] = cs_k

        # Per-shot density: (51, 20) — cs_k broadcast via [:, None]
        with np.errstate(all="ignore"):
            n_e_R_shot = electron_density_m3(isat_R_shot, area_isat_m2, cs_k[:, None])
            n_e_L_shot = electron_density_m3(isat_L_shot, area_isweep_m2, cs_k[:, None])

        # Shot-averaged density and std: (51,)
        with np.errstate(all="ignore"):
            n_e_R[:, k] = np.nanmean(n_e_R_shot, axis=1)
            n_e_L[:, k] = np.nanmean(n_e_L_shot, axis=1)
            if n_shots > 1:
                n_e_R_std[:, k] = np.nanstd(n_e_R_shot, axis=1, ddof=1)
                n_e_L_std[:, k] = np.nanstd(n_e_L_shot, axis=1, ddof=1)

        # Mach (probes B/C/D, plus known wiring-swap runs where ISAT appears to
        # be the upstream face).
        if probe_id != "A" or connections_swapped:
            with np.errstate(all="ignore"):
                # rot=0: Isweep/A_p_L faces upstream → n_e_L is upstream density
                # rot=180: Isat/A_p_R faces upstream → n_e_R is upstream density
                if connections_swapped:
                    M_shot = np.log(n_e_R_shot / n_e_L_shot) / MACH_K
                elif rotation_deg == 0:
                    M_shot = np.log(n_e_L_shot / n_e_R_shot) / MACH_K
                else:
                    M_shot = np.log(n_e_R_shot / n_e_L_shot) / MACH_K
                mach[:, k] = np.nanmean(M_shot, axis=1)
                if n_shots > 1:
                    mach_std[:, k] = np.nanstd(M_shot, axis=1, ddof=1)
                velocity_km_s[:, k] = mach[:, k] * cs_k / 1000.0
                velocity_km_s_std[:, k] = mach_std[:, k] * cs_k / 1000.0

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
        "cs_m_s": cs_m_s,
        "velocity_km_s": velocity_km_s,
        "velocity_km_s_std": velocity_km_s_std,
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
    parser.add_argument("--te-filled", type=Path, default=TE_FILLED_HDF5)
    parser.add_argument(
        "--probe-a-calibration",
        type=Path,
        default=PROBE_A_CALIB_TOML,
    )
    args = parser.parse_args()
    target_ids: set[str] | None = None if "all" in args.run_ids else set(args.run_ids)

    if not CALIB_TOML.exists():
        sys.exit(f"Missing {CALIB_TOML} — run calibrate_probe_areas.py first.")
    if not args.te_filled.exists():
        sys.exit(f"Missing {args.te_filled} — run fit_te_spatial.py first.")

    dataset = LapdDataset.from_manifest(MANIFEST)
    calibration = _load_calibration(CALIB_TOML)
    probe_a_calibration = load_probe_a_area_calibration(args.probe_a_calibration)

    HDF5_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(HDF5_OUTPUT, "w") as out_hdf,
        h5py.File(args.te_filled, "r") as te_hdf,
    ):
        out_hdf.attrs["source_te_hdf5"] = str(args.te_filled)
        out_hdf.attrs["source_probe_a_calibration_toml"] = str(
            args.probe_a_calibration
        )
        out_hdf.attrs["probe_a_factor"] = probe_a_calibration.factor
        out_hdf.attrs["probe_a_factor_lower_bound"] = (
            probe_a_calibration.lower_bound
        )
        out_hdf.attrs["probe_a_factor_upper_bound"] = (
            probe_a_calibration.upper_bound
        )
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

                if f"experiment_sets/{set_id}" not in te_hdf:
                    print(f"  [skip] run {run_id}: experiment set {set_id} not found in {args.te_filled}")
                    continue

                pid = _probe_id(run_id)
                calib = calibration[pid]
                run = dataset.run(run_id)
                cfg = run.config
                recorded_rot = float(cfg.probe.rotation_deg or 0)
                effective_rot = effective_rotation_deg(run_id, recorded_rot)

                print(
                    f"  run {run_id}  probe={pid}  set={set_id}"
                    f"  rot={effective_rot:.0f}°"
                    + (f" (recorded {recorded_rot:.0f}°)" if effective_rot != recorded_rot else "")
                    + f"  port={cfg.probe.port}"
                )

                inter_times_s = _inter_sweep_times_s(run)
                te_grid = _interp_filled_te_to_deadtime(
                    te_hdf,
                    set_id,
                    float(cfg.probe.z_cm or 0),
                    inter_times_s,
                )

                results = process_run(
                    run,
                    calib_m2=calib,
                    probe_id=pid,
                    te_grid=te_grid,
                    probe_a_calibration=probe_a_calibration,
                )

                g_run = g_set.create_group(run_id)
                g_run.attrs["run_id"] = run_id
                g_run.attrs["rotation_deg"] = float(effective_rot)
                g_run.attrs["rotation_deg_recorded"] = recorded_rot
                g_run.attrs["rotation_correction_applied"] = bool(effective_rot != recorded_rot)
                g_run.attrs["port"] = int(cfg.probe.port or 0)
                g_run.attrs["z_cm"] = float(cfg.probe.z_cm or 0)
                g_run.attrs["probe_id"] = pid
                g_run.attrs["ap_L_cm2"] = calib["ap_L_cm2"]
                g_run.attrs["ap_R_cm2"] = calib["ap_R_cm2"]
                g_run.attrs["electrical_swap_applied"] = bool(electrical_connections_swapped(run_id))
                g_run.attrs["ap_estimated"] = calib["estimated"]
                g_run.attrs["probe_a_area_factor_applied"] = (
                    probe_a_calibration.factor if pid == "A" else 1.0
                )

                for key, arr in results.items():
                    g_run.create_dataset(key, data=arr)

                peak = float(np.nanmax(results["n_e_R_m3"]))
                fwhm_mid = float(np.nanmedian(results["plasma_fwhm_cm"]))
                print(f"    peak n_e_R ≈ {peak:.2e} m⁻³   median FWHM ≈ {fwhm_mid:.1f} cm")

    print(f"\nWrote {HDF5_OUTPUT}")


if __name__ == "__main__":
    main()
