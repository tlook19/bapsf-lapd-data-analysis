"""Calibrate Langmuir probe face areas using interferometer line-integrated density.

For each physical probe that has a nearby interferometer chord (probes B, C, D
at ports 21, 29, 41), integrates the radial I_sat profile at each dead-time
cycle and matches the result to the interferometer line-integrated density to
solve for A_p_R (Isat channel face) and A_p_L (Isweep channel face, negated).

Probe face convention
---------------------
A_p_L: left face (upstream, toward cathode at z=0), always connected to the
        Isweep channel.  Between sweeps the Isweep channel collects ion
        saturation current at a fixed negative bias; the measured current must
        be negated because Isweep and Isat have opposite polarity.
A_p_R: right face (downstream), always connected to the Isat channel.

Calibration formula
-------------------
  A_p = trapz(I_sat(x) / [e * C_s(x) * exp(-0.5)], x) / ∫ n_e dl

where C_s = sqrt(k_B T_e / m_i) from the langmuir_sweeps.hdf5 T_e and the
interferometer provides ∫ n_e dl at each dead-time midpoint.

Cross-set variation
-------------------
Areas are computed independently for each experiment set (1–4) and reported
in a table.  Experiment set 1 ("nice plasma", 180 V bank) is used as the
primary calibration.  Probe A (ports 11/50) has no nearby interferometer;
its area is estimated from probe B with an ``estimated = true`` flag.

Interferometer port mapping (axial proximity)
---------------------------------------------
  Langmuir port 21 → interferometer port 20 (31.95 cm axial offset)
  Langmuir port 29 → interferometer port 29 (exact)
  Langmuir port 41 → interferometer port 40 (31.95 cm axial offset)

Inputs
------
  config/may2026_run_manifest.toml
  processed/langmuir_sweeps.hdf5  (T_e per position/cycle)
  processed/interferometer_experiment_set_stats.npz

Output
------
  processed/probe_area_calibration.toml

Usage
-----
  python scripts/calibrate_probe_areas.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bapsf_lapd import ChannelKind, LapdDataset, LapdRun
from bapsf_lapd.density import (
    calibrate_probe_area_m2,
    inter_sweep_sample_slices,
    ion_sound_speed_m_s,
)


MANIFEST = Path("config/may2026_run_manifest.toml")
SWEEPS_HDF5 = Path("processed/langmuir_sweeps.hdf5")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
OUTPUT_TOML = Path("processed/probe_area_calibration.toml")

# He-4 ion mass in amu.  Change to 1.008 for hydrogen, 39.948 for argon.
M_I_AMU = 4.003

# Stable plasma plateau for calibration (ms from SIS trigger).
CALIB_T_MIN_MS = 5.0
CALIB_T_MAX_MS = 15.0

# Edge clip at both ends of each dead-time window to avoid ramp transients.
CLIP_S = 10e-6

# Probe scan positions (51 points, 1 cm spacing, centred at x = 0).
X_CM = np.linspace(-25.0, 25.0, 51)
X_M = X_CM * 1e-2

# Physical probe identity from the second digit of the two-digit run_id.
PROBE_FROM_DIGIT = {1: "A", 8: "A", 2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}

# Nearest interferometer HDF5 port for each Langmuir probe port.
INTERF_PORT_FOR = {21: 20, 29: 29, 41: 40}

# Experiment set whose calibration is used as the primary reference.
REF_SET = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probe_id(run_id: str) -> str:
    return PROBE_FROM_DIGIT[int(run_id[1])]


def _load_interferometer(
    npz: dict[str, np.ndarray],
    experiment_set_id: int,
    interf_port: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_ms, line_integrated_m2) for a given experiment set and port."""
    prefix = f"set{experiment_set_id}_p{interf_port}"
    time_ms = npz[f"{prefix}_time_ms"]
    # NPZ stores line-integrated density in cm^-2; convert to m^-2.
    line_integrated_m2 = npz[f"{prefix}_line_integrated_mean_cm2"] * 1e4
    return time_ms, line_integrated_m2


def _dead_time_midpoints_ms(run: LapdRun) -> np.ndarray:
    """Midpoint time (ms) of each dead-time period relative to the SIS trigger."""
    sw = run.config.sweep
    return np.array([
        (sw.t0_s + k * sw.tau_cycle_s + sw.tau_ramp_s
         + sw.t0_s + (k + 1) * sw.tau_cycle_s) / 2 * 1000
        for k in range(sw.n_cycles)
    ])


def _shot_averaged_isat(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-position mean and std from (n_pos, n_shots, n_samples) traces.

    Averages over the time axis first, then over shots, giving a robust
    DC estimate of the ion saturation current at each probe position.
    """
    per_shot = traces.mean(axis=2)  # (n_pos, n_shots)
    mean = per_shot.mean(axis=1)    # (n_pos,)
    n_shots = traces.shape[1]
    std = per_shot.std(axis=1, ddof=1) if n_shots > 1 else np.zeros(traces.shape[0])
    return mean, std


# ---------------------------------------------------------------------------
# Per-run calibration
# ---------------------------------------------------------------------------

def calibrate_run(
    run: LapdRun,
    te_grid: np.ndarray,
    interf_time_ms: np.ndarray,
    interf_line_integrated_m2: np.ndarray,
) -> dict[str, list[float]]:
    """Calibrate A_p_R and A_p_L for one run over the stable plasma window.

    te_grid: (51, n_cycles) shot-averaged electron temperature in eV (NaN allowed).

    Returns {"ap_R_m2": [...], "ap_L_m2": [...]} — one value per calibration cycle.
    """
    sw = run.config.sweep
    dead_slices = inter_sweep_sample_slices(sw, run.config.acquisition, clip_s=CLIP_S)
    dead_mids_ms = _dead_time_midpoints_ms(run)
    in_window = (dead_mids_ms >= CALIB_T_MIN_MS) & (dead_mids_ms <= CALIB_T_MAX_MS)

    isat_offset = run.default_zero_offset_v(ChannelKind.ISAT)
    isweep_offset = run.default_zero_offset_v(ChannelKind.I_SWEEP)

    ap_R_list: list[float] = []
    ap_L_list: list[float] = []

    for k in range(sw.n_cycles):
        if not in_window[k]:
            continue

        # Load dead-time traces: (51, 20, n_dead_samples)
        traces_R = run.langmuir_traces(ChannelKind.ISAT, dead_slices[k], zero_offset_v=isat_offset)
        traces_L = run.langmuir_traces(ChannelKind.I_SWEEP, dead_slices[k], zero_offset_v=isweep_offset)

        isat_R, _ = _shot_averaged_isat(traces_R)
        isat_L, _ = _shot_averaged_isat(-traces_L)  # negate Isweep polarity

        # T_e at ramp k → sound speed at each position.
        te_k = te_grid[:, k] if k < te_grid.shape[1] else np.full(51, np.nan)
        with np.errstate(invalid="ignore"):
            cs_k = ion_sound_speed_m_s(te_k, M_I_AMU)

        # Interferometer value at the dead-time midpoint.
        interf_val = float(np.interp(dead_mids_ms[k], interf_time_ms, interf_line_integrated_m2))
        if not np.isfinite(interf_val) or interf_val <= 0:
            continue

        ap_R_k = calibrate_probe_area_m2(isat_R, cs_k, X_M, interf_val)
        ap_L_k = calibrate_probe_area_m2(isat_L, cs_k, X_M, interf_val)

        if np.isfinite(ap_R_k):
            ap_R_list.append(ap_R_k)
        if np.isfinite(ap_L_k):
            ap_L_list.append(ap_L_k)

    return {"ap_R_m2": ap_R_list, "ap_L_m2": ap_L_list}


# ---------------------------------------------------------------------------
# TOML writer
# ---------------------------------------------------------------------------

def _fmt(v: float, precision: int = 6) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.{precision}f}"


def _write_toml(summary: dict[str, dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Langmuir probe face area calibration — May 2026 LAPD experiment.\n",
        f"# Ion species: He-4  m_i = {M_I_AMU} amu\n",
        f"# Calibration time window: {CALIB_T_MIN_MS}–{CALIB_T_MAX_MS} ms (stable plasma plateau)\n",
        f"# Primary calibration: experiment set {REF_SET} ('nice plasma', 180 V bank)\n",
        "#\n",
        "# A_p_L: left (upstream) face area — Isweep channel\n",
        "# A_p_R: right (downstream) face area — Isat channel\n",
        "# Probe A: no nearby interferometer; area estimated from probe B\n",
        "\n",
    ]
    for probe_id in ("A", "B", "C", "D"):
        data = summary[probe_id]
        lines.append(f"[probe_{probe_id}]\n")
        lines.append(f"ap_L_cm2 = {_fmt(data['ap_L_cm2'])}\n")
        lines.append(f"ap_R_cm2 = {_fmt(data['ap_R_cm2'])}\n")
        lines.append(f"ap_L_std_cm2 = {_fmt(data['ap_L_std_cm2'])}\n")
        lines.append(f"ap_R_std_cm2 = {_fmt(data['ap_R_std_cm2'])}\n")
        if data.get("estimated"):
            lines.append("estimated = true  # no nearby interferometer; area copied from probe B\n")
        lines.append("\n")

        if "by_set" in data:
            lines.append(f"[probe_{probe_id}.by_set]\n")
            lines.append("# Per-set calibration to document cross-condition variation.\n")
            for set_id, vals in data["by_set"].items():
                lines.append(f"set{set_id}_ap_L_cm2     = {_fmt(vals['ap_L_cm2'])}\n")
                lines.append(f"set{set_id}_ap_L_std_cm2 = {_fmt(vals['ap_L_std_cm2'])}\n")
                lines.append(f"set{set_id}_ap_R_cm2     = {_fmt(vals['ap_R_cm2'])}\n")
                lines.append(f"set{set_id}_ap_R_std_cm2 = {_fmt(vals['ap_R_std_cm2'])}\n")
                lines.append(f"set{set_id}_n_calib_R    = {vals['n_R']}\n")
                lines.append(f"set{set_id}_n_calib_L    = {vals['n_L']}\n")
            lines.append("\n")

    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dataset = LapdDataset.from_manifest(MANIFEST)
    interf_npz = dict(np.load(INTERF_NPZ))

    # Accumulate calibration samples keyed by [probe_id][set_id]["ap_R/L_m2"]
    probe_ids = ("B", "C", "D")
    accum: dict[str, dict[int, dict[str, list[float]]]] = {
        pid: {sid: {"ap_R_m2": [], "ap_L_m2": []} for sid in range(1, 5)}
        for pid in probe_ids
    }

    with h5py.File(SWEEPS_HDF5, "r") as sweeps_hdf:
        for set_id in dataset.experiment_set_ids():
            for run_id in dataset.experiment_set_run_ids(set_id):
                pid = _probe_id(run_id)
                if pid not in probe_ids:
                    continue  # probe A has no interferometer

                run = dataset.run(run_id)
                port = run.config.probe.port
                if port not in INTERF_PORT_FOR:
                    print(f"  [skip] run {run_id}: port {port} not in interferometer map")
                    continue

                # Check that this run has been processed by process_langmuir_sweeps.py.
                hdf_key = f"experiment_sets/{set_id}/{run_id}"
                if hdf_key not in sweeps_hdf:
                    print(f"  [skip] run {run_id}: not found in {SWEEPS_HDF5}")
                    continue

                interf_port = INTERF_PORT_FOR[port]
                interf_time_ms, interf_m2 = _load_interferometer(interf_npz, set_id, interf_port)
                te_grid = sweeps_hdf[f"{hdf_key}/te_log_ev"][()]  # (51, n_cycles)

                print(f"  calibrating run {run_id}  probe={pid}  set={set_id}  port={port}")
                cal = calibrate_run(run, te_grid, interf_time_ms, interf_m2)

                accum[pid][set_id]["ap_R_m2"].extend(cal["ap_R_m2"])
                accum[pid][set_id]["ap_L_m2"].extend(cal["ap_L_m2"])

    # Summarise: mean ± std per probe per set; print cross-set table.
    print(
        "\n=== Probe Area Calibration ==="
        f"\n{'Probe':>6} {'Set':>4}  {'A_p_R (cm²)':>18}  {'A_p_L (cm²)':>18}  {'n_R':>5} {'n_L':>5}"
    )
    print("-" * 62)

    probe_summary: dict[str, dict[str, Any]] = {}
    for pid in probe_ids:
        by_set: dict[int, dict[str, Any]] = {}
        for set_id in range(1, 5):
            vals_R = np.array([v for v in accum[pid][set_id]["ap_R_m2"] if np.isfinite(v)]) * 1e4
            vals_L = np.array([v for v in accum[pid][set_id]["ap_L_m2"] if np.isfinite(v)]) * 1e4
            mean_R = float(np.mean(vals_R)) if len(vals_R) > 0 else np.nan
            mean_L = float(np.mean(vals_L)) if len(vals_L) > 0 else np.nan
            std_R = float(np.std(vals_R, ddof=1)) if len(vals_R) > 1 else np.nan
            std_L = float(np.std(vals_L, ddof=1)) if len(vals_L) > 1 else np.nan
            by_set[set_id] = {
                "ap_R_cm2": mean_R, "ap_L_cm2": mean_L,
                "ap_R_std_cm2": std_R, "ap_L_std_cm2": std_L,
                "n_R": len(vals_R), "n_L": len(vals_L),
            }
            print(
                f"{pid:>6} {set_id:>4}  "
                f"{mean_R:>8.4f} ± {std_R:<8.4f}  "
                f"{mean_L:>8.4f} ± {std_L:<8.4f}  "
                f"{len(vals_R):>5} {len(vals_L):>5}"
            )

        ref = by_set[REF_SET]
        probe_summary[pid] = {
            "ap_R_cm2": ref["ap_R_cm2"],
            "ap_L_cm2": ref["ap_L_cm2"],
            "ap_R_std_cm2": ref["ap_R_std_cm2"],
            "ap_L_std_cm2": ref["ap_L_std_cm2"],
            "by_set": by_set,
        }

    # Probe A: estimated from probe B (nearest port).
    probe_summary["A"] = {
        "ap_R_cm2": probe_summary["B"]["ap_R_cm2"],
        "ap_L_cm2": probe_summary["B"]["ap_L_cm2"],
        "ap_R_std_cm2": probe_summary["B"]["ap_R_std_cm2"],
        "ap_L_std_cm2": probe_summary["B"]["ap_L_std_cm2"],
        "estimated": True,
    }

    _write_toml(probe_summary, OUTPUT_TOML)
    print(f"\nWrote {OUTPUT_TOML}")


if __name__ == "__main__":
    main()
