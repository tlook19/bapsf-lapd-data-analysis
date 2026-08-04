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

where C_s = sqrt(k_B T_e / m_i) from the experiment-set-1 filled T_e grid and
the interferometer provides ∫ n_e dl at each dead-time midpoint.

Cross-set variation
-------------------
Areas are computed from experiment set 1 ("nice plasma", 180 V bank).  Probe A
(ports 11/50) has no nearby interferometer; its area is estimated from probe B
with an ``estimated = true`` flag.

Interferometer port mapping (axial proximity)
---------------------------------------------
  Langmuir port 21 → interferometer port 20 (31.95 cm axial offset)
  Langmuir port 29 → interferometer port 29 (exact)
  Langmuir port 41 → interferometer port 40 (31.95 cm axial offset)

Inputs
------
  processed/te_filled.hdf5  (experiment-set-1 filled T_e profile)
  processed/interferometer_experiment_set_stats.npz
  processed/isweep_deadtime_profiles.hdf5
  processed/isat_rot180_deadtime_profiles.hdf5

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

from bapsf_lapd.density import calibrate_probe_area_m2, ion_sound_speed_m_s


TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
ISWEEP_ROT0_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
ISAT_ROT180_HDF5 = Path("processed/isat_rot180_deadtime_profiles.hdf5")
OUTPUT_TOML = Path("processed/probe_area_calibration.toml")

# He-4 ion mass in amu.  Change to 1.008 for hydrogen, 39.948 for argon.
M_I_AMU = 4.003

# Stable plasma plateau for calibration (ms from SIS trigger).
CALIB_T_MIN_MS = 10.0
CALIB_T_MAX_MS = 19.0

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


def _load_te_filled_reference(path: Path) -> dict[str, np.ndarray]:
    """Load the experiment-set-1 filled T_e reference grid."""
    with h5py.File(path, "r") as hdf:
        grp = hdf["experiment_sets"][str(REF_SET)]
        return {
            "x_cm": grp["x_cm"][()],
            "z_cm": grp["z_cm"][()],
            "cycle_time_ms": grp["cycle_time_ms"][()],
            "te_filled": grp["te_filled"][()],
        }


def _te_grid_for_port(te_ref: dict[str, np.ndarray], port_z_cm: float) -> np.ndarray:
    """Return a filled T_e(x, cycle) grid at the nearest measured z row."""
    z_cm = te_ref["z_cm"]
    z_idx = int(np.argmin(np.abs(z_cm - port_z_cm)))
    te_grid = te_ref["te_filled"][z_idx, :, :]
    if not np.allclose(te_ref["x_cm"], X_CM):
        raise ValueError("te_filled x grid does not match the calibration x grid")
    return te_grid


# ---------------------------------------------------------------------------
# Profile calibration
# ---------------------------------------------------------------------------

def calibrate_profile(
    isat_grid: np.ndarray,
    profile_time_s: np.ndarray,
    te_grid: np.ndarray,
    te_time_ms: np.ndarray,
    interf_time_ms: np.ndarray,
    interf_line_integrated_m2: np.ndarray,
) -> dict[str, list[float]]:
    """Calibrate one probe face over the selected stable plasma window.

    isat_grid: (51, n_cycles) positive ion-saturation current in A.
    te_grid: (51, n_cycles) filled electron temperature in eV.

    Returns one area value per valid calibration cycle.
    """
    profile_time_ms = np.asarray(profile_time_s, dtype=np.float64) * 1000.0
    in_window = (profile_time_ms >= CALIB_T_MIN_MS) & (profile_time_ms <= CALIB_T_MAX_MS)

    ap_list: list[float] = []
    for k, time_ms in enumerate(profile_time_ms):
        if not in_window[k]:
            continue

        # Filled T_e at the dead-time midpoint → sound speed at each position.
        te_k = np.array([
            np.interp(time_ms, te_time_ms, te_grid[i, :])
            for i in range(te_grid.shape[0])
        ])
        with np.errstate(invalid="ignore"):
            cs_k = ion_sound_speed_m_s(te_k, M_I_AMU)

        # Interferometer value at the dead-time midpoint.
        interf_val = float(np.interp(time_ms, interf_time_ms, interf_line_integrated_m2))
        if not np.isfinite(interf_val) or interf_val <= 0:
            continue

        ap_k = calibrate_probe_area_m2(isat_grid[:, k], cs_k, X_M, interf_val)

        if np.isfinite(ap_k):
            ap_list.append(ap_k)

    return ap_list


def _accumulate_profile_file(
    hdf_path: Path,
    face_key: str,
    accum: dict[str, dict[str, list[float]]],
    te_ref: dict[str, np.ndarray],
    interf_npz: dict[str, np.ndarray],
) -> None:
    with h5py.File(hdf_path, "r") as hdf:
        if not np.allclose(hdf["x_cm"][()], X_CM):
            raise ValueError(f"{hdf_path} x grid does not match the calibration x grid")

        for run_id in sorted(hdf[f"experiment_sets/{REF_SET}"].keys()):
            pid = _probe_id(run_id)
            if pid not in accum:
                continue

            grp = hdf[f"experiment_sets/{REF_SET}/{run_id}"]
            port = int(grp.attrs["port"])
            if port not in INTERF_PORT_FOR:
                continue

            interf_port = INTERF_PORT_FOR[port]
            interf_time_ms, interf_m2 = _load_interferometer(interf_npz, REF_SET, interf_port)
            te_grid = _te_grid_for_port(te_ref, float(grp.attrs["z_cm"]))

            print(
                f"  calibrating run {run_id}  probe={pid}  set={REF_SET}"
                f"  port={port}  face={face_key}"
            )
            ap = calibrate_profile(
                grp["isat_a"][()],
                grp["inter_sweep_time_s"][()],
                te_grid,
                te_ref["cycle_time_ms"],
                interf_time_ms,
                interf_m2,
            )
            accum[pid][face_key].extend(ap)


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
        "# Probe A: base/reference area copied from probe B; apply the canonical\n",
        "# empirical factor in config/may2026_probe_a_area_calibration.toml once\n",
        "# to raw Probe A current in every downstream density/Mach workflow.\n",
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
            lines.append("estimated = true  # base/reference area copied from probe B\n")
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
    interf_npz = dict(np.load(INTERF_NPZ))
    te_ref = _load_te_filled_reference(TE_FILLED_HDF5)

    # Accumulate calibration samples keyed by [probe_id]["ap_R/L_m2"]
    probe_ids = ("B", "C", "D")
    accum: dict[str, dict[str, list[float]]] = {pid: {"ap_R_m2": [], "ap_L_m2": []} for pid in probe_ids}

    _accumulate_profile_file(ISAT_ROT180_HDF5, "ap_R_m2", accum, te_ref, interf_npz)
    _accumulate_profile_file(ISWEEP_ROT0_HDF5, "ap_L_m2", accum, te_ref, interf_npz)

    # Summarise: mean ± std per probe; print table.
    print(
        "\n=== Probe Area Calibration ==="
        f"\n{'Probe':>6}  {'A_p_R (cm²)':>18}  {'A_p_L (cm²)':>18}  {'n_R':>5} {'n_L':>5}"
    )
    print("-" * 56)

    probe_summary: dict[str, dict[str, Any]] = {}
    for pid in probe_ids:
        vals_R = np.array([v for v in accum[pid]["ap_R_m2"] if np.isfinite(v)]) * 1e4
        vals_L = np.array([v for v in accum[pid]["ap_L_m2"] if np.isfinite(v)]) * 1e4
        mean_R = float(np.mean(vals_R)) if len(vals_R) > 0 else np.nan
        mean_L = float(np.mean(vals_L)) if len(vals_L) > 0 else np.nan
        std_R = float(np.std(vals_R, ddof=1)) if len(vals_R) > 1 else np.nan
        std_L = float(np.std(vals_L, ddof=1)) if len(vals_L) > 1 else np.nan
        print(
            f"{pid:>6}  "
            f"{mean_R:>8.4f} ± {std_R:<8.4f}  "
            f"{mean_L:>8.4f} ± {std_L:<8.4f}  "
            f"{len(vals_R):>5} {len(vals_L):>5}"
        )
        probe_summary[pid] = {
            "ap_R_cm2": mean_R,
            "ap_L_cm2": mean_L,
            "ap_R_std_cm2": std_R,
            "ap_L_std_cm2": std_L,
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
