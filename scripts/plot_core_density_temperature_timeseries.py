"""Plot core-averaged density and temperature versus time.

For each experiment set, this script averages density and filled T_e over the
radial core band -10 <= x <= 10 cm.  Each figure has two subplots:

  1. electron density versus time
  2. electron temperature versus time

Each line is one axial z position.  Separate outputs are written with radial
SEM and radial STD uncertainty bars.  Density SEM bars for Probe A additionally
include the propagated area-calibration uncertainty in quadrature.  Use
``--te-dataset te_masked`` to compare against the non-filled T_e grid.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_core_density_temperature_timeseries.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_core_density_temperature_timeseries.py --uncertainty sem
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np


DENSITY_HDF5 = Path("processed/density_profiles_isweep.hdf5")
TE_HDF5 = Path("processed/te_filled.hdf5")
OUTPUT_DIR = Path("figures")

X_MIN_CM = -10.0
X_MAX_CM = 10.0
DENSITY_SCALE_M3 = 1e19  # 10^13 cm^-3


@dataclass(frozen=True)
class CoreStats:
    time_ms: np.ndarray
    z_cm: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    sem: np.ndarray
    count: np.ndarray
    calibration_uncertainty: np.ndarray


def _nan_core_stats(values: np.ndarray, x_cm: np.ndarray, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, std, sem, count over x for values shaped (z, x, time)."""
    x_mask = (x_cm >= x_min) & (x_cm <= x_max)
    if not np.any(x_mask):
        raise ValueError(f"No x samples found in requested core band {x_min:g} to {x_max:g} cm")

    core = values[:, x_mask, :]
    count = np.sum(np.isfinite(core), axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(core, axis=1)
        std = np.nanstd(core, axis=1, ddof=1)
    sem = np.where(count > 1, std / np.sqrt(count), np.nan)
    std = np.where(count > 1, std, np.nan)
    return mean, std, sem, count


def _load_density_stats(hdf: h5py.File, es_id: str, x_min: float, x_max: float) -> CoreStats:
    grp = hdf[f"experiment_sets/{es_id}"]
    x_cm = hdf["x_cm"][()]
    density_scaled = grp["n_e_m3"][()] / DENSITY_SCALE_M3
    mean, std, sem, count = _nan_core_stats(density_scaled, x_cm, x_min, x_max)
    calibration_uncertainty = np.zeros_like(mean)
    runs_grp = grp.get("runs")
    if runs_grp is not None:
        for run_id, run_grp in runs_grp.items():
            if str(run_grp.attrs.get("probe_id", "")) != "A":
                continue
            z_cm = float(run_grp.attrs["z_cm"])
            z_idx = int(np.argmin(np.abs(grp["z_cm"][()] - z_cm)))
            relative = float(
                run_grp.attrs.get("probe_a_density_relative_uncertainty", 0.0)
            )
            calibration_uncertainty[z_idx] = np.abs(mean[z_idx]) * relative
    return CoreStats(
        time_ms=grp["inter_sweep_time_s"][()] * 1000.0,
        z_cm=grp["z_cm"][()],
        mean=mean,
        std=std,
        sem=sem,
        count=count,
        calibration_uncertainty=calibration_uncertainty,
    )


def _load_te_stats(
    hdf: h5py.File,
    es_id: str,
    x_min: float,
    x_max: float,
    *,
    dataset: str,
) -> CoreStats:
    grp = hdf[f"experiment_sets/{es_id}"]
    x_cm = grp["x_cm"][()]
    mean, std, sem, count = _nan_core_stats(grp[dataset][()], x_cm, x_min, x_max)
    return CoreStats(
        time_ms=grp["cycle_time_ms"][()],
        z_cm=grp["z_cm"][()],
        mean=mean,
        std=std,
        sem=sem,
        count=count,
        calibration_uncertainty=np.zeros_like(mean),
    )


def _plot_uncertainty(stats: CoreStats, uncertainty: str) -> np.ndarray:
    """Return plotted error, including Probe-A calibration only for SEM."""
    if uncertainty == "sem":
        return np.hypot(stats.sem, stats.calibration_uncertainty)
    return stats.std


def _experiment_label(density_hdf: h5py.File, te_hdf: h5py.File, es_id: str) -> tuple[str, float]:
    density_grp = density_hdf[f"experiment_sets/{es_id}"]
    te_grp = te_hdf[f"experiment_sets/{es_id}"]
    label = str(density_grp.attrs.get("label", te_grp.attrs.get("label", f"set {es_id}")))
    v_bank = float(density_grp.attrs.get("v_bank_v", te_grp.attrs.get("v_bank_v", np.nan)))
    return label, v_bank


def _plot_quantity(
    ax: plt.Axes,
    stats: CoreStats,
    uncertainty: str,
    *,
    ylabel: str,
    colors: list,
    ylim: tuple[float, float],
) -> None:
    yerr = _plot_uncertainty(stats, uncertainty)
    for zi, z_cm in enumerate(stats.z_cm):
        ax.errorbar(
            stats.time_ms,
            stats.mean[zi],
            yerr=yerr[zi],
            color=colors[zi % len(colors)],
            marker="o",
            markersize=2.4,
            linewidth=1.2,
            elinewidth=0.7,
            capsize=1.5,
            alpha=0.9,
            label=f"z = {z_cm:.1f} cm",
        )
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.25)


def _global_ylim(
    all_stats: list[CoreStats],
    uncertainty: str,
    *,
    pad_fraction: float = 0.08,
) -> tuple[float, float]:
    lo_values = []
    hi_values = []
    for stats in all_stats:
        yerr = _plot_uncertainty(stats, uncertainty)
        lo_values.append((stats.mean - yerr).ravel())
        hi_values.append((stats.mean + yerr).ravel())
    lo_arr = np.concatenate(lo_values)
    hi_arr = np.concatenate(hi_values)
    finite_lo = lo_arr[np.isfinite(lo_arr)]
    finite_hi = hi_arr[np.isfinite(hi_arr)]
    if finite_lo.size == 0 or finite_hi.size == 0:
        raise RuntimeError("Cannot determine global y limits; no finite plotted values found")

    ymin = float(np.nanmin(finite_lo))
    ymax = float(np.nanmax(finite_hi))
    if ymin == ymax:
        pad = 1.0 if ymin == 0 else abs(ymin) * pad_fraction
    else:
        pad = (ymax - ymin) * pad_fraction
    return ymin - pad, ymax + pad


def make_plot(
    density_stats: CoreStats,
    te_stats: CoreStats,
    *,
    es_id: str,
    label: str,
    v_bank: float,
    uncertainty: str,
    output_dir: Path,
    x_min: float,
    x_max: float,
    te_dataset: str,
    density_ylim: tuple[float, float],
    te_ylim: tuple[float, float],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = list(plt.get_cmap("tab10").colors)
    uncertainty_label = "SEM" if uncertainty == "sem" else "STD"

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.6), sharex=False, constrained_layout=True)
    _plot_quantity(
        axes[0],
        density_stats,
        uncertainty,
        ylabel=r"$n_e$ ($10^{13}$ cm$^{-3}$)",
        colors=colors,
        ylim=density_ylim,
    )
    _plot_quantity(
        axes[1],
        te_stats,
        uncertainty,
        ylabel=r"$T_e$ (eV)",
        colors=colors,
        ylim=te_ylim,
    )
    axes[1].set_xlabel("time (ms)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", title="Axial position", fontsize=8, title_fontsize=9)

    te_label = "filled $T_e$" if te_dataset == "te_filled" else "non-filled $T_e$"
    title = f"Core-averaged density and {te_label} - experiment set {es_id}: {label}"
    if np.isfinite(v_bank):
        title += f"  (V_bank = {v_bank:.0f} V)"
    uncertainty_detail = f"radial {uncertainty_label} error bars"
    if uncertainty == "sem" and np.any(density_stats.calibration_uncertainty > 0):
        uncertainty_detail += " (Probe A density includes area calibration in quadrature)"
    fig.suptitle(
        f"{title}\n{x_min:g} <= x <= {x_max:g} cm, {uncertainty_detail}",
        fontsize=11,
    )

    suffix = "filled_te" if te_dataset == "te_filled" else "masked_te"
    out_path = output_dir / f"core_density_temperature_timeseries_expset{es_id}_{suffix}_{uncertainty}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def make_all_plots(
    density_path: Path,
    te_path: Path,
    output_dir: Path,
    *,
    experiment_sets: set[str] | None,
    uncertainties: tuple[str, ...],
    x_min: float,
    x_max: float,
    te_dataset: str,
) -> list[Path]:
    outputs = []
    with h5py.File(density_path, "r") as density_hdf, h5py.File(te_path, "r") as te_hdf:
        density_ids = set(density_hdf["experiment_sets"].keys())
        te_ids = set(te_hdf["experiment_sets"].keys())
        loaded: list[tuple[str, CoreStats, CoreStats, str, float]] = []
        for es_id in sorted(density_ids & te_ids, key=int):
            if experiment_sets is not None and es_id not in experiment_sets:
                continue
            density_stats = _load_density_stats(density_hdf, es_id, x_min, x_max)
            te_stats = _load_te_stats(te_hdf, es_id, x_min, x_max, dataset=te_dataset)
            if not np.allclose(density_stats.z_cm, te_stats.z_cm):
                raise ValueError(f"Density and Te z grids differ for experiment set {es_id}")
            label, v_bank = _experiment_label(density_hdf, te_hdf, es_id)
            loaded.append((es_id, density_stats, te_stats, label, v_bank))

        if not loaded:
            raise RuntimeError("No matching experiment sets found in density and Te inputs")

        for uncertainty in uncertainties:
            density_ylim = _global_ylim([item[1] for item in loaded], uncertainty)
            te_ylim = _global_ylim([item[2] for item in loaded], uncertainty)
            print(
                f"{te_dataset} {uncertainty}: "
                f"density ylim=({density_ylim[0]:.3g}, {density_ylim[1]:.3g}) "
                f"Te ylim=({te_ylim[0]:.3g}, {te_ylim[1]:.3g})"
            )
            for es_id, density_stats, te_stats, label, v_bank in loaded:
                outputs.append(
                    make_plot(
                        density_stats,
                        te_stats,
                        es_id=es_id,
                        label=label,
                        v_bank=v_bank,
                        uncertainty=uncertainty,
                        output_dir=output_dir,
                        x_min=x_min,
                        x_max=x_max,
                        te_dataset=te_dataset,
                        density_ylim=density_ylim,
                        te_ylim=te_ylim,
                    )
                )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--density", type=Path, default=DENSITY_HDF5)
    parser.add_argument("--te-filled", type=Path, default=TE_HDF5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment-sets", nargs="+", default=None)
    parser.add_argument("--x-min", type=float, default=X_MIN_CM)
    parser.add_argument("--x-max", type=float, default=X_MAX_CM)
    parser.add_argument("--uncertainty", choices=["sem", "std", "both"], default="both")
    parser.add_argument(
        "--te-dataset",
        choices=["te_filled", "te_masked"],
        default="te_filled",
        help="T_e dataset to plot; te_masked is the non-filled comparison",
    )
    args = parser.parse_args()

    uncertainties = ("sem", "std") if args.uncertainty == "both" else (args.uncertainty,)
    target_sets = set(args.experiment_sets) if args.experiment_sets is not None else None
    make_all_plots(
        args.density,
        args.te_filled,
        args.output_dir,
        experiment_sets=target_sets,
        uncertainties=uncertainties,
        x_min=args.x_min,
        x_max=args.x_max,
        te_dataset=args.te_dataset,
    )


if __name__ == "__main__":
    main()
