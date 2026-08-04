"""Plot line-integrated density versus axial position at one discharge time.

Compares probe-derived radial integrals from ``density_profiles_isweep.hdf5``
against the p20/p29/p40 interferometer line-integrated densities.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_line_integrated_density_vs_z.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_line_integrated_density_vs_z.py \\
      --experiment-set 3 --time-ms 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd.config import z_from_port


DENSITY_HDF5 = Path("processed/density_profiles_isweep.hdf5")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
OUTPUT_DIR = Path("figures")
INTERFEROMETER_PORTS = (20, 29, 40)


def _line_integral_cm2(profile_m3: np.ndarray, x_m: np.ndarray) -> float:
    valid = np.isfinite(profile_m3) & (profile_m3 > 0)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapezoid(profile_m3[valid], x_m[valid]) / 1e4)


def _interp_rows(values: np.ndarray, time_ms: np.ndarray, target_ms: float) -> np.ndarray:
    return np.array([
        np.interp(target_ms, time_ms, row)
        for row in values
    ])


def make_plot(
    density_path: Path,
    interferometer_path: Path,
    output_dir: Path,
    experiment_set: str,
    time_ms: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(density_path, "r") as density_hdf, np.load(interferometer_path) as interf_npz:
        set_grp = density_hdf[f"experiment_sets/{experiment_set}"]
        x_m = density_hdf["x_cm"][()] * 1e-2
        z_probe_cm = set_grp["z_cm"][()]
        density = set_grp["n_e_m3"][()]
        density_time_ms = set_grp["inter_sweep_time_s"][()] * 1000.0
        label = str(set_grp.attrs.get("label", f"set {experiment_set}"))
        v_bank = float(set_grp.attrs.get("v_bank_v", np.nan))

        probe_line_cm2 = np.array([
            [_line_integral_cm2(density[zi, :, ti], x_m) for ti in range(density.shape[2])]
            for zi in range(density.shape[0])
        ])
        probe_at_time = _interp_rows(probe_line_cm2, density_time_ms, time_ms)

        z_interf_cm = np.array([z_from_port(port) for port in INTERFEROMETER_PORTS], dtype=np.float64)
        interf_at_time = []
        for port in INTERFEROMETER_PORTS:
            prefix = f"set{experiment_set}_p{port}"
            interf_at_time.append(
                float(np.interp(
                    time_ms,
                    interf_npz[f"{prefix}_time_ms"],
                    interf_npz[f"{prefix}_line_integrated_mean_cm2"],
                ))
            )
        interf_at_time = np.array(interf_at_time)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(
        z_probe_cm,
        probe_at_time / 1e14,
        "o-",
        label="Isweep profile integral",
        linewidth=1.8,
        markersize=6,
    )
    ax.plot(
        z_interf_cm,
        interf_at_time / 1e14,
        "s--",
        label="interferometer",
        linewidth=1.8,
        markersize=6,
    )

    for z, value in zip(z_probe_cm, probe_at_time):
        ax.annotate(f"{value / 1e14:.2f}", (z, value / 1e14), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    for port, z, value in zip(INTERFEROMETER_PORTS, z_interf_cm, interf_at_time):
        ax.annotate(f"p{port}\n{value / 1e14:.2f}", (z, value / 1e14), textcoords="offset points", xytext=(0, -28), ha="center", fontsize=8)

    ax.set_xlabel("z (cm)")
    ax.set_ylabel(r"$\int n_e\,dl$ ($10^{14}$ cm$^{-2}$)")
    title = f"Line-integrated density vs z\nES {experiment_set}: {label}, t = {time_ms:.1f} ms"
    if np.isfinite(v_bank):
        title += f" (V_bank = {v_bank:.0f} V)"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    time_tag = f"{time_ms:.1f}".replace(".", "p")
    out_path = output_dir / f"line_integrated_density_vs_z_expset{experiment_set}_t{time_tag}ms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--density", type=Path, default=DENSITY_HDF5)
    parser.add_argument("--interferometer", type=Path, default=INTERF_NPZ)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment-set", default="3")
    parser.add_argument("--time-ms", type=float, default=15.0)
    args = parser.parse_args()
    make_plot(args.density, args.interferometer, args.output_dir, str(args.experiment_set), args.time_ms)


if __name__ == "__main__":
    main()
