"""Compare ES line-integrated density from rot-0 Isweep and rot-180 Isat.

The rot-0 curve is loaded from ``processed/density_profiles_isweep.hdf5``.
The rot-180 curve is computed on the fly from
``processed/isat_rot180_deadtime_profiles.hdf5`` using filled T_e profiles and
the calibrated A_p_R face areas.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_line_integrated_density_vs_z_rot180.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_line_integrated_density_vs_z_rot180.py \\
      --experiment-set 3 --time-ms 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import h5py
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd.config import z_from_port
from bapsf_lapd.density import electron_density_m3, ion_sound_speed_m_s


ISWEEP_DENSITY_HDF5 = Path("processed/density_profiles_isweep.hdf5")
ISAT_ROT180_HDF5 = Path("processed/isat_rot180_deadtime_profiles.hdf5")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
CALIB_TOML = Path("processed/probe_area_calibration.toml")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
OUTPUT_DIR = Path("figures")

M_I_AMU = 4.003
INTERFEROMETER_PORTS = (20, 29, 40)
PROBE_FROM_DIGIT = {2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}


def _probe_id(run_id: str) -> str:
    return PROBE_FROM_DIGIT[int(run_id[1])]


def _line_integral_cm2(profile_m3: np.ndarray, x_m: np.ndarray) -> float:
    valid = np.isfinite(profile_m3) & (profile_m3 > 0)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapezoid(profile_m3[valid], x_m[valid]) / 1e4)


def _load_ap_r_m2(path: Path) -> dict[str, float]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {probe: data[f"probe_{probe}"]["ap_R_cm2"] * 1e-4 for probe in ("B", "C", "D")}


def _interp_te_to_times(te_x_cycle: np.ndarray, te_time_ms: np.ndarray, target_time_ms: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.interp(target_time_ms, te_time_ms, te_x_cycle[xi, :])
        for xi in range(te_x_cycle.shape[0])
    ])


def _isweep_line_integrals(
    density_hdf: h5py.File,
    experiment_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grp = density_hdf[f"experiment_sets/{experiment_set}"]
    x_m = density_hdf["x_cm"][()] * 1e-2
    z_cm = grp["z_cm"][()]
    time_ms = grp["inter_sweep_time_s"][()] * 1000.0
    density = grp["n_e_m3"][()]
    line = np.array([
        [_line_integral_cm2(density[zi, :, ti], x_m) for ti in range(density.shape[2])]
        for zi in range(density.shape[0])
    ])
    return z_cm, time_ms, line


def _rot180_line_integrals(
    isat_hdf: h5py.File,
    te_hdf: h5py.File,
    ap_r_m2: dict[str, float],
    experiment_set: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    isat_set = isat_hdf[f"experiment_sets/{experiment_set}"]
    te_set = te_hdf[f"experiment_sets/{experiment_set}"]
    x_cm = isat_hdf["x_cm"][()]
    x_m = x_cm * 1e-2
    if not np.allclose(x_cm, te_set["x_cm"][()]):
        raise ValueError("rot-180 Isat and filled-Te x grids do not match")

    te_z_cm = te_set["z_cm"][()]
    te_time_ms = te_set["cycle_time_ms"][()]
    te_filled = te_set["te_filled"][()]
    entries = []
    for run_id in sorted(isat_set.keys()):
        grp = isat_set[run_id]
        z_cm = float(grp.attrs["z_cm"])
        z_idx = int(np.argmin(np.abs(te_z_cm - z_cm)))
        time_s = grp["inter_sweep_time_s"][()]
        time_ms = time_s * 1000.0
        te_interp = _interp_te_to_times(te_filled[z_idx, :, :], te_time_ms, time_ms)
        cs = ion_sound_speed_m_s(te_interp, M_I_AMU)
        probe = _probe_id(run_id)
        density = electron_density_m3(grp["isat_a"][()], ap_r_m2[probe], cs)
        density = np.where((density > 0) & np.isfinite(density), density, np.nan)
        line = np.array([_line_integral_cm2(density[:, ti], x_m) for ti in range(density.shape[1])])
        entries.append((z_cm, time_ms, line, run_id))

    entries.sort(key=lambda item: item[0])
    z_cm = np.array([entry[0] for entry in entries])
    time_ms = entries[0][1]
    line = np.vstack([entry[2] for entry in entries])
    return z_cm, time_ms, line


def _interp_rows(values: np.ndarray, time_ms: np.ndarray, target_ms: float) -> np.ndarray:
    return np.array([np.interp(target_ms, time_ms, row) for row in values])


def make_plot(
    experiment_set: str,
    time_ms: float,
    output_dir: Path,
    *,
    isweep_density_path: Path,
    isat_rot180_path: Path,
    te_filled_path: Path,
    calibration_path: Path,
    interferometer_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ap_r_m2 = _load_ap_r_m2(calibration_path)

    with (
        h5py.File(isweep_density_path, "r") as density_hdf,
        h5py.File(isat_rot180_path, "r") as isat_hdf,
        h5py.File(te_filled_path, "r") as te_hdf,
        np.load(interferometer_path) as interf_npz,
    ):
        z_isweep, t_isweep, line_isweep = _isweep_line_integrals(density_hdf, experiment_set)
        z_rot180, t_rot180, line_rot180 = _rot180_line_integrals(isat_hdf, te_hdf, ap_r_m2, experiment_set)
        isweep_at_time = _interp_rows(line_isweep, t_isweep, time_ms)
        rot180_at_time = _interp_rows(line_rot180, t_rot180, time_ms)

        z_interf = np.array([z_from_port(port) for port in INTERFEROMETER_PORTS], dtype=np.float64)
        interf_at_time = np.array([
            float(np.interp(
                time_ms,
                interf_npz[f"set{experiment_set}_p{port}_time_ms"],
                interf_npz[f"set{experiment_set}_p{port}_line_integrated_mean_cm2"],
            ))
            for port in INTERFEROMETER_PORTS
        ])
        label = str(density_hdf[f"experiment_sets/{experiment_set}"].attrs.get("label", f"set {experiment_set}"))
        v_bank = float(density_hdf[f"experiment_sets/{experiment_set}"].attrs.get("v_bank_v", np.nan))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(z_isweep, isweep_at_time / 1e14, "o-", label="rot-0 Isweep / A_p_L", linewidth=1.8)
    ax.plot(z_rot180, rot180_at_time / 1e14, "^-", label="rot-180 Isat / A_p_R", linewidth=1.8)
    ax.plot(z_interf, interf_at_time / 1e14, "s--", label="interferometer", linewidth=1.8)

    for z, value in zip(z_isweep, isweep_at_time):
        ax.annotate(f"{value / 1e14:.2f}", (z, value / 1e14), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    for z, value in zip(z_rot180, rot180_at_time):
        ax.annotate(f"{value / 1e14:.2f}", (z, value / 1e14), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    for port, z, value in zip(INTERFEROMETER_PORTS, z_interf, interf_at_time):
        ax.annotate(f"p{port}\n{value / 1e14:.2f}", (z, value / 1e14), textcoords="offset points", xytext=(0, -34), ha="center", fontsize=8)

    title = f"Line-integrated density vs z\nES {experiment_set}: {label}, t = {time_ms:.1f} ms"
    if np.isfinite(v_bank):
        title += f" (V_bank = {v_bank:.0f} V)"
    ax.set_title(title)
    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"$\int n_e\,dl$ ($10^{14}$ cm$^{-2}$)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    time_tag = f"{time_ms:.1f}".replace(".", "p")
    out_path = output_dir / f"line_integrated_density_vs_z_rot180_compare_expset{experiment_set}_t{time_tag}ms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    print("Values at requested time, in 1e14 cm^-2:")
    for z, value in zip(z_isweep, isweep_at_time):
        print(f"  rot-0 Isweep z={z:.1f}: {value / 1e14:.3f}")
    for z, value in zip(z_rot180, rot180_at_time):
        print(f"  rot-180 Isat z={z:.1f}: {value / 1e14:.3f}")
    for port, z, value in zip(INTERFEROMETER_PORTS, z_interf, interf_at_time):
        print(f"  interferometer p{port} z={z:.1f}: {value / 1e14:.3f}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-set", default="3")
    parser.add_argument("--time-ms", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--isweep-density", type=Path, default=ISWEEP_DENSITY_HDF5)
    parser.add_argument("--isat-rot180", type=Path, default=ISAT_ROT180_HDF5)
    parser.add_argument("--te-filled", type=Path, default=TE_FILLED_HDF5)
    parser.add_argument("--calibration", type=Path, default=CALIB_TOML)
    parser.add_argument("--interferometer", type=Path, default=INTERF_NPZ)
    args = parser.parse_args()
    make_plot(
        str(args.experiment_set),
        args.time_ms,
        args.output_dir,
        isweep_density_path=args.isweep_density,
        isat_rot180_path=args.isat_rot180,
        te_filled_path=args.te_filled,
        calibration_path=args.calibration,
        interferometer_path=args.interferometer,
    )


if __name__ == "__main__":
    main()
