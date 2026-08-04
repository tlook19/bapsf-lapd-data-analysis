"""Plot rot-0 Mach probe and flow-velocity dead-time contours.

This reads ``processed/mach_velocity.hdf5`` from ``compute_mach_velocity.py``.
Only effective rot-0 runs are used, so the upstream face is ``-I_SWEEP`` and
the downstream face is ``ISAT``.  Velocity is the Mach number multiplied by
the local ion sound speed from filled T_e, stored in km/s by the compute step.

Outputs per experiment set:
  - mach_contours_expset{N}.png
  - velocity_contours_expset{N}.png
  - mach_contours_expset{N}.gif, unless --no-animation is used
  - velocity_contours_expset{N}.gif, unless --no-animation is used
  - core_mach_velocity_timeseries_expset{N}.png

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_mach_velocity_contours.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_mach_velocity_contours.py --no-animation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import h5py
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


HDF5_INPUT = Path("processed/mach_velocity.hdf5")
OUTPUT_DIR = Path("figures")

SUBPLOT_NCOLS = 8
SUBPLOT_NROWS = 5
CMAP_SIGNED = "coolwarm"
X_MIN_CM = -10.0
X_MAX_CM = 10.0


def _pcolormesh_edges(centers: np.ndarray) -> np.ndarray:
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)
    mids = (centers[:-1] + centers[1:]) / 2
    lo = centers[0] - (centers[1] - centers[0]) / 2
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[lo], mids, [hi]])


def _interp_time_grid(values: np.ndarray, source_ms: np.ndarray, target_ms: np.ndarray) -> np.ndarray:
    if np.array_equal(source_ms, target_ms) or np.allclose(source_ms, target_ms):
        return values
    return np.vstack([
        np.interp(target_ms, source_ms, values[xi, :], left=np.nan, right=np.nan)
        for xi in range(values.shape[0])
    ])


def _load_dataset_or_velocity(run_grp: h5py.Group) -> np.ndarray:
    if "velocity_km_s" in run_grp:
        return run_grp["velocity_km_s"][()]
    return run_grp["mach"][()] * run_grp["cs_m_s"][()] / 1000.0


def _load_experiment_set(hf: h5py.File, es_id: str) -> dict | None:
    x_cm = hf["x_cm"][()]
    set_grp = hf[f"experiment_sets/{es_id}"]
    z_groups: dict[float, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    common_time_ms: np.ndarray | None = None

    for run_id in sorted(set_grp.keys()):
        run_grp = set_grp[run_id]
        if float(run_grp.attrs.get("rotation_deg", 0.0)) != 0.0:
            continue

        mach = run_grp["mach"][()]
        if not np.any(np.isfinite(mach)):
            continue

        velocity = _load_dataset_or_velocity(run_grp)
        time_ms = run_grp["inter_sweep_time_s"][()] * 1000.0
        if common_time_ms is None:
            common_time_ms = time_ms
        else:
            mach = _interp_time_grid(mach, time_ms, common_time_ms)
            velocity = _interp_time_grid(velocity, time_ms, common_time_ms)

        z_cm = float(run_grp.attrs["z_cm"])
        z_groups.setdefault(z_cm, []).append((mach, velocity, np.isfinite(mach).astype(float)))

    if not z_groups or common_time_ms is None:
        return None

    z_cm = np.array(sorted(z_groups), dtype=np.float64)
    mach_grid = np.full((len(z_cm), len(x_cm), len(common_time_ms)), np.nan)
    velocity_grid = np.full_like(mach_grid, np.nan)

    for zi, z in enumerate(z_cm):
        mach_sum = np.zeros_like(mach_grid[zi])
        velocity_sum = np.zeros_like(velocity_grid[zi])
        weight = np.zeros_like(mach_grid[zi])
        for mach, velocity, valid_weight in z_groups[float(z)]:
            good_mach = np.isfinite(mach)
            good_velocity = np.isfinite(velocity)
            mach_sum += np.where(good_mach, mach, 0.0)
            velocity_sum += np.where(good_velocity, velocity, 0.0)
            weight += valid_weight
        with np.errstate(invalid="ignore", divide="ignore"):
            mach_grid[zi] = np.where(weight > 0, mach_sum / weight, np.nan)
            velocity_grid[zi] = np.where(weight > 0, velocity_sum / weight, np.nan)

    return {
        "x_cm": x_cm,
        "z_cm": z_cm,
        "cycle_time_ms": common_time_ms,
        "mach": mach_grid,
        "velocity": velocity_grid,
        "es_id": es_id,
        "es_label": str(set_grp.attrs.get("label", f"set {es_id}")),
        "v_bank": float(set_grp.attrs.get("v_bank_v", np.nan)),
    }


def _symmetric_limit(values: np.ndarray, override: float | None) -> float:
    if override is not None:
        return abs(float(override))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError("No finite values available for color scale")
    limit = float(np.nanpercentile(np.abs(finite), 98))
    return limit if limit > 0 else 1.0


def _make_frame(
    ax: plt.Axes,
    values_2d: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    norm: mcolors.Normalize,
    title: str,
) -> None:
    ax.clear()
    ax.pcolormesh(x_edges, z_edges, values_2d, cmap=CMAP_SIGNED, norm=norm, rasterized=True)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel("x (cm)", fontsize=6)
    ax.set_ylabel("z (cm)", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_subplots(
    data: dict,
    output_dir: Path,
    *,
    field: str,
    label: str,
    filename_stem: str,
    limit: float,
) -> Path:
    values = data[field]
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    times = data["cycle_time_ms"]

    fig, axes = plt.subplots(
        SUBPLOT_NROWS,
        SUBPLOT_NCOLS,
        figsize=(SUBPLOT_NCOLS * 2.4, SUBPLOT_NROWS * 2.2),
        constrained_layout=True,
    )
    ax_flat = axes.flatten()
    for ci in range(values.shape[2]):
        _make_frame(
            ax_flat[ci],
            values[:, :, ci],
            x_edges,
            z_edges,
            norm,
            f"t = {times[ci]:.1f} ms",
        )
    for ax in ax_flat[values.shape[2]:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_SIGNED)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    title = f"{label} vs (x, z) - experiment set {data['es_id']}: {data['es_label']}"
    if np.isfinite(data["v_bank"]):
        title += f"  (V_bank = {data['v_bank']:.0f} V)"
    fig.suptitle(title, fontsize=11)

    out_path = output_dir / f"{filename_stem}_expset{data['es_id']}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_animation(
    data: dict,
    output_dir: Path,
    *,
    field: str,
    label: str,
    filename_stem: str,
    limit: float,
) -> Path:
    values = data[field]
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    times = data["cycle_time_ms"]

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, z_edges, values[:, :, 0], cmap=CMAP_SIGNED, norm=norm, rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(label, fontsize=10)
    ax.set_xlabel("x (cm)", fontsize=10)
    ax.set_ylabel("z (cm)", fontsize=10)
    title = ax.set_title(f"t = {times[0]:.1f} ms", fontsize=10)

    suptitle = f"{label} - ES {data['es_id']}: {data['es_label']}"
    if np.isfinite(data["v_bank"]):
        suptitle += f"  (V_bank = {data['v_bank']:.0f} V)"
    fig.suptitle(suptitle, fontsize=10)

    def update(ci: int):
        mesh.set_array(values[:, :, ci].ravel())
        title.set_text(f"t = {times[ci]:.1f} ms")
        return mesh, title

    anim = animation.FuncAnimation(fig, update, frames=values.shape[2], interval=300, blit=False)
    out_path = output_dir / f"{filename_stem}_expset{data['es_id']}.gif"
    anim.save(str(out_path), writer="pillow", dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _core_mean(values: np.ndarray, x_cm: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    x_mask = (x_cm >= x_min) & (x_cm <= x_max)
    if not np.any(x_mask):
        raise ValueError(f"No x samples found in requested core band {x_min:g} to {x_max:g} cm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values[:, x_mask, :], axis=1)


def plot_core_timeseries(
    data: dict,
    output_dir: Path,
    x_min: float,
    x_max: float,
    *,
    mach_ylim: tuple[float, float],
    velocity_ylim: tuple[float, float],
) -> Path:
    mach_core = _core_mean(data["mach"], data["x_cm"], x_min, x_max)
    velocity_core = _core_mean(data["velocity"], data["x_cm"], x_min, x_max)
    colors = list(plt.get_cmap("tab10").colors)

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.5), sharex=True, constrained_layout=True)
    for zi, z_cm in enumerate(data["z_cm"]):
        color = colors[zi % len(colors)]
        axes[0].plot(data["cycle_time_ms"], mach_core[zi], marker="o", markersize=2.6, linewidth=1.3, color=color, label=f"z = {z_cm:.1f} cm")
        axes[1].plot(data["cycle_time_ms"], velocity_core[zi], marker="o", markersize=2.6, linewidth=1.3, color=color)

    axes[0].axhline(0.0, color="0.25", linewidth=0.8)
    axes[1].axhline(0.0, color="0.25", linewidth=0.8)
    axes[0].set_ylabel("Mach number")
    axes[1].set_ylabel("velocity (km/s)")
    axes[0].set_ylim(*mach_ylim)
    axes[1].set_ylim(*velocity_ylim)
    axes[1].set_xlabel("dead-time midpoint (ms)")
    for ax in axes:
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right center", title="Axial position", fontsize=8, title_fontsize=9)

    title = f"Core-averaged Mach and velocity - experiment set {data['es_id']}: {data['es_label']}"
    if np.isfinite(data["v_bank"]):
        title += f"  (V_bank = {data['v_bank']:.0f} V)"
    fig.suptitle(f"{title}\n{x_min:g} <= x <= {x_max:g} cm", fontsize=11)

    out_path = output_dir / f"core_mach_velocity_timeseries_expset{data['es_id']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def _global_core_ylim(
    data_list: list[dict],
    field: str,
    x_min: float,
    x_max: float,
    *,
    pad_fraction: float = 0.08,
) -> tuple[float, float]:
    core_values = [
        _core_mean(data[field], data["x_cm"], x_min, x_max).ravel()
        for data in data_list
    ]
    finite = np.concatenate(core_values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError(f"No finite core {field} values found")

    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    if ymin == ymax:
        pad = 1.0 if ymin == 0 else abs(ymin) * pad_fraction
    else:
        pad = (ymax - ymin) * pad_fraction
    return ymin - pad, ymax + pad


def plot_all(
    input_path: Path,
    output_dir: Path,
    *,
    experiment_sets: set[str] | None,
    save_animation: bool,
    x_min: float,
    x_max: float,
    mach_limit: float | None,
    velocity_limit: float | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    with h5py.File(input_path, "r") as hf:
        data_list = []
        for es_id in sorted(hf["experiment_sets"].keys(), key=int):
            if experiment_sets is not None and es_id not in experiment_sets:
                continue
            data = _load_experiment_set(hf, es_id)
            if data is None:
                print(f"ES {es_id}: no finite rot-0 Mach data, skipping.")
                continue
            data_list.append(data)

    if not data_list:
        raise RuntimeError("No finite rot-0 Mach data found")

    all_mach = np.concatenate([data["mach"].ravel() for data in data_list])
    all_velocity = np.concatenate([data["velocity"].ravel() for data in data_list])
    mach_abs = _symmetric_limit(all_mach, mach_limit)
    velocity_abs = _symmetric_limit(all_velocity, velocity_limit)
    core_mach_ylim = _global_core_ylim(data_list, "mach", x_min, x_max)
    core_velocity_ylim = _global_core_ylim(data_list, "velocity", x_min, x_max)
    print(f"Mach color scale: +/- {mach_abs:.3g}")
    print(f"Velocity color scale: +/- {velocity_abs:.3g} km/s")
    print(f"Core Mach y-limit: {core_mach_ylim[0]:.3g} to {core_mach_ylim[1]:.3g}")
    print(f"Core velocity y-limit: {core_velocity_ylim[0]:.3g} to {core_velocity_ylim[1]:.3g} km/s")

    for data in data_list:
        outputs.append(plot_subplots(data, output_dir, field="mach", label="Mach number", filename_stem="mach_contours", limit=mach_abs))
        outputs.append(plot_subplots(data, output_dir, field="velocity", label="velocity (km/s)", filename_stem="velocity_contours", limit=velocity_abs))
        if save_animation:
            outputs.append(plot_animation(data, output_dir, field="mach", label="Mach number", filename_stem="mach_contours", limit=mach_abs))
            outputs.append(plot_animation(data, output_dir, field="velocity", label="velocity (km/s)", filename_stem="velocity_contours", limit=velocity_abs))
        outputs.append(
            plot_core_timeseries(
                data,
                output_dir,
                x_min,
                x_max,
                mach_ylim=core_mach_ylim,
                velocity_ylim=core_velocity_ylim,
            )
        )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=HDF5_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment-sets", nargs="+", default=None)
    parser.add_argument("--x-min", type=float, default=X_MIN_CM)
    parser.add_argument("--x-max", type=float, default=X_MAX_CM)
    parser.add_argument("--mach-limit", type=float, default=None, help="symmetric Mach color limit")
    parser.add_argument("--velocity-limit", type=float, default=None, help="symmetric velocity color limit in km/s")
    parser.add_argument("--no-animation", action="store_true")
    args = parser.parse_args()

    target_sets = set(args.experiment_sets) if args.experiment_sets is not None else None
    plot_all(
        args.input,
        args.output_dir,
        experiment_sets=target_sets,
        save_animation=not args.no_animation,
        x_min=args.x_min,
        x_max=args.x_max,
        mach_limit=args.mach_limit,
        velocity_limit=args.velocity_limit,
    )


if __name__ == "__main__":
    main()
