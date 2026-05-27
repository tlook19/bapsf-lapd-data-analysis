"""Plot T_e contour maps vs (x, z) for each discharge cycle.

For each experiment set, produces:
  - A multi-panel PNG with one subplot per cycle (8 × 5 grid, 40 panels total).
  - An animated GIF cycling through the same frames.

Axes
----
  x  : probe scan position in cm (−25 to +25)
  y  : axial position z in cm (port locations along the machine)
  color : T_e (eV) from the exponential fit, mean over ok/warn shots

Runs at the same z (rot=0 and rot=180 pairs) are averaged cell-by-cell.
Bad-dominated cells (n_bad > n_ok + n_warn) are masked.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_te_contours.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_te_contours.py \\
      --input processed/langmuir_sweeps.hdf5 --output-dir figures [--no-animation]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


HDF5_INPUT = Path("processed/langmuir_sweeps.hdf5")
OUTPUT_DIR = Path("figures")

CMAP = "plasma"
SUBPLOT_NCOLS = 8
SUBPLOT_NROWS = 5  # 8 × 5 = 40 panels for 40 cycles


def _pcolormesh_edges(centers: np.ndarray) -> np.ndarray:
    """Convert bin centers to edges via midpoint rule, extending the outer bins symmetrically."""
    mids = (centers[:-1] + centers[1:]) / 2
    lo = centers[0] - (centers[1] - centers[0]) / 2
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[lo], mids, [hi]])


def _load_experiment_set(hf: h5py.File, es_id: str) -> dict:
    """Load all data for one experiment set, averaging duplicate-z runs."""
    x_cm = hf["x_cm"][:]
    eg = hf["experiment_sets"][es_id]
    es_label = eg.attrs.get("label", f"set {es_id}")
    v_bank = eg.attrs.get("v_bank_v", "?")

    # Group runs by z position and collect te + quality counts.
    z_groups: dict[float, list] = {}
    cycle_time_s = None

    for run_id in sorted(eg.keys()):
        rg = eg[run_id]
        z = float(rg.attrs["z_cm"])
        if z not in z_groups:
            z_groups[z] = []
        te = rg["te_log_ev"][:]        # (n_pos, n_cycles)
        n_ok = rg["n_ok"][:]
        n_bad = rg["n_bad"][:]
        z_groups[z].append((te, n_ok, n_bad))
        if cycle_time_s is None:
            cycle_time_s = rg["cycle_time_s"][:]

    # Build sorted z array and averaged Te grid: (n_z, n_x, n_cycles).
    z_vals = sorted(z_groups.keys())
    n_z = len(z_vals)
    n_x = len(x_cm)
    n_cycles = len(cycle_time_s)

    te_grid = np.full((n_z, n_x, n_cycles), np.nan)

    for zi, z in enumerate(z_vals):
        entries = z_groups[z]
        te_sum = np.zeros((n_x, n_cycles))
        weight = np.zeros((n_x, n_cycles))
        mask_bad = np.zeros((n_x, n_cycles), dtype=bool)

        for te, n_ok, n_bad in entries:
            # Mask cells where bad shots dominate.
            cell_bad = n_bad > n_ok
            mask_bad |= cell_bad
            valid = ~cell_bad
            te_sum += np.where(valid, te, 0.0)
            weight += valid.astype(float)

        with np.errstate(invalid="ignore"):
            te_avg = np.where(weight > 0, te_sum / weight, np.nan)
        # Re-apply the bad mask (all contributing runs flagged bad).
        all_bad = mask_bad & (weight == 0)
        te_avg[all_bad] = np.nan

        te_grid[zi] = te_avg

    return {
        "te": te_grid,           # (n_z, n_x, n_cycles)
        "x_cm": x_cm,
        "z_cm": np.array(z_vals),
        "cycle_time_ms": cycle_time_s * 1e3,
        "es_label": es_label,
        "v_bank": v_bank,
        "es_id": es_id,
    }


def _make_frame(ax: plt.Axes, te_2d: np.ndarray, x_edges: np.ndarray,
                z_edges: np.ndarray, norm, title: str) -> None:
    """Draw a single pcolormesh frame on ax."""
    ax.clear()
    ax.pcolormesh(x_edges, z_edges, te_2d, cmap=CMAP, norm=norm,
                  rasterized=True)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel("x (cm)", fontsize=6)
    ax.set_ylabel("z (cm)", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_subplots(data: dict, output_dir: Path, vmin: float, vmax: float) -> Path:
    """Save a static figure with one subplot per cycle."""
    import matplotlib.colors as mcolors

    te = data["te"]                    # (n_z, n_x, n_cycles)
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    times = data["cycle_time_ms"]
    n_cycles = te.shape[2]
    es_id = data["es_id"]
    es_label = data["es_label"]
    v_bank = data["v_bank"]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        SUBPLOT_NROWS, SUBPLOT_NCOLS,
        figsize=(SUBPLOT_NCOLS * 2.4, SUBPLOT_NROWS * 2.2),
        constrained_layout=True,
    )
    ax_flat = axes.flatten()

    for ci in range(n_cycles):
        ax = ax_flat[ci]
        te_frame = te[:, :, ci]     # (n_z, n_x)
        t_ms = times[ci]
        _make_frame(ax, te_frame, x_edges, z_edges, norm, f"t = {t_ms:.1f} ms")

    # Hide unused panels.
    for ax in ax_flat[n_cycles:]:
        ax.set_visible(False)

    # Shared colorbar.
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label("$T_e$ (eV)  [log-linear fit]", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"$T_e$ (log-linear) vs (x, z) — experiment set {es_id}: {es_label}"
        f"  (V_bank = {v_bank} V)",
        fontsize=11,
    )

    out_path = output_dir / f"te_contours_expset{es_id}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_animation(data: dict, output_dir: Path, vmin: float, vmax: float) -> Path:
    """Save an animated GIF cycling through discharge cycles."""
    import matplotlib.colors as mcolors

    te = data["te"]                    # (n_z, n_x, n_cycles)
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    times = data["cycle_time_ms"]
    n_cycles = te.shape[2]
    es_id = data["es_id"]
    es_label = data["es_label"]
    v_bank = data["v_bank"]

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

    mesh = ax.pcolormesh(x_edges, z_edges, te[:, :, 0], cmap=CMAP, norm=norm,
                         rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("$T_e$ (eV)  [log-linear]", fontsize=10)

    ax.set_xlabel("x (cm)", fontsize=10)
    ax.set_ylabel("z (cm)", fontsize=10)
    title = ax.set_title(f"t = {times[0]:.1f} ms", fontsize=10)

    fig.suptitle(
        f"$T_e$ (log-linear) — ES {es_id}: {es_label}  (V_bank = {v_bank} V)",
        fontsize=10,
    )

    def update(ci: int):
        mesh.set_array(te[:, :, ci].ravel())
        title.set_text(f"t = {times[ci]:.1f} ms")
        return mesh, title

    anim = animation.FuncAnimation(
        fig, update, frames=n_cycles, interval=300, blit=False
    )

    out_path = output_dir / f"te_contours_expset{es_id}.gif"
    anim.save(str(out_path), writer="pillow", dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_all(hdf5_path: Path, output_dir: Path, save_animation: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as hf:
        es_ids = sorted(hf["experiment_sets"].keys(), key=int)

        for es_id in es_ids:
            data = _load_experiment_set(hf, es_id)
            te = data["te"]

            # Compute a common color scale across all cycles for this ES.
            finite = te[np.isfinite(te)]
            if finite.size == 0:
                print(f"ES {es_id}: no finite T_e data, skipping.")
                continue
            vmin = max(0.0, float(np.percentile(finite, 2)))
            vmax = float(np.percentile(finite, 98))

            plot_subplots(data, output_dir, vmin, vmax)
            if save_animation:
                plot_animation(data, output_dir, vmin, vmax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=HDF5_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-animation", action="store_true",
                        help="skip saving animated GIFs")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found — run process_langmuir_sweeps.py first."
        )

    plot_all(args.input, args.output_dir, save_animation=not args.no_animation)


if __name__ == "__main__":
    main()
