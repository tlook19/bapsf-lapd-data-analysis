"""Plot T_e contour maps vs (x, z) for each discharge cycle.

For each experiment set, produces:
  - A multi-panel PNG with one subplot per cycle (8 × 5 grid, 40 panels total).
  - An animated GIF cycling through the same frames.

Axes
----
  x  : probe scan position in cm (−25 to +25)
  y  : axial position z in cm (port locations along the machine)
  color : T_e (eV) from the best-of-log-or-exp fit, averaged over valid shots

Masking
-------
Runs at the same z are averaged cell-by-cell, weighted by the number of valid
shots (n_ok + n_warn) that contributed to each cell's te_best_ev estimate.
Cells are hidden only when the total number of valid shots across all runs at
that z falls below --min-shots (default: 2).  This preserves cells with a
handful of good shots rather than hiding them solely because bad shots
outnumber good ones — the best-fit selection already discards unreliable fits
at the per-shot level.

Only rot=0 runs are used; rot=180 runs are excluded because the downstream
probe face sits in the probe's own magnetic shadow.  The rotation_deg attribute
reflects any known swap corrections (e.g. ES3 p21 runs 32/33).

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_te_contours.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_te_contours.py \\
      --input processed/langmuir_sweeps.hdf5 --output-dir figures \\
      [--min-shots 2] [--no-animation]
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

# Default minimum valid-shot count to show a cell.  Cells with fewer
# contributing shots (n_ok + n_warn) across all runs at the same z are NaN.
DEFAULT_MIN_SHOTS = 2


def _pcolormesh_edges(centers: np.ndarray) -> np.ndarray:
    """Convert bin centers to edges via midpoint rule, extending the outer bins symmetrically."""
    mids = (centers[:-1] + centers[1:]) / 2
    lo = centers[0] - (centers[1] - centers[0]) / 2
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[lo], mids, [hi]])


def _load_experiment_set(hf: h5py.File, es_id: str, min_shots: int = DEFAULT_MIN_SHOTS) -> dict:
    """Load and average te_best_ev for one experiment set (rot=0 runs only).

    Multiple runs at the same z are averaged cell-by-cell, weighted by the
    number of valid shots (n_ok + n_warn) that contributed to each cell's
    te_best_ev estimate.  A cell is included in the average only when its
    valid-shot count meets or exceeds ``min_shots``; cells where no run at
    that z reaches the threshold are left as NaN.
    """
    x_cm = hf["x_cm"][:]
    eg = hf["experiment_sets"][es_id]
    es_label = eg.attrs.get("label", f"set {es_id}")
    v_bank = eg.attrs.get("v_bank_v", "?")

    z_groups: dict[float, list] = {}
    cycle_time_s = None

    for run_id in sorted(eg.keys()):
        rg = eg[run_id]
        if float(rg.attrs.get("rotation_deg", 0)) != 0.0:
            continue                              # skip rot=180 runs
        z = float(rg.attrs["z_cm"])
        if z not in z_groups:
            z_groups[z] = []
        te      = rg["te_best_ev"][:]            # (n_pos, n_cycles)
        n_valid = rg["n_ok"][:] + rg["n_warn"][:]  # shots that went into te_best_ev
        z_groups[z].append((te, n_valid))
        if cycle_time_s is None:
            cycle_time_s = rg["cycle_time_s"][:]

    z_vals   = sorted(z_groups.keys())
    n_x      = len(x_cm)
    n_cycles = len(cycle_time_s)

    te_grid = np.full((len(z_vals), n_x, n_cycles), np.nan)

    for zi, z in enumerate(z_vals):
        te_sum = np.zeros((n_x, n_cycles))
        weight = np.zeros((n_x, n_cycles))

        for te, n_valid in z_groups[z]:
            # A cell contributes only when it has enough valid shots and a
            # finite Te value.  Weight by valid-shot count so runs with more
            # good shots have proportionally more influence on the average.
            good = (n_valid >= min_shots) & np.isfinite(te)
            te_sum += np.where(good, te * n_valid, 0.0)
            weight += np.where(good, n_valid, 0.0)

        with np.errstate(invalid="ignore"):
            te_grid[zi] = np.where(weight > 0, te_sum / weight, np.nan)

    return {
        "te":            te_grid,       # (n_z, n_x, n_cycles)
        "x_cm":          x_cm,
        "z_cm":          np.array(z_vals),
        "cycle_time_ms": cycle_time_s * 1e3,
        "es_label":      es_label,
        "v_bank":        v_bank,
        "es_id":         es_id,
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


def plot_subplots(data: dict, output_dir: Path, vmin: float, vmax: float,
                  suffix: str = "") -> Path:
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
    cbar.set_label("$T_e$ (eV)  [best-fit (log or exp)]", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"$T_e$ (best-fit) vs (x, z) — experiment set {es_id}: {es_label}"
        f"  (V_bank = {v_bank} V)",
        fontsize=11,
    )

    out_path = output_dir / f"te_contours_expset{es_id}{suffix}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_animation(data: dict, output_dir: Path, vmin: float, vmax: float,
                   suffix: str = "") -> Path:
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
    cbar.set_label("$T_e$ (eV)  [best-fit]", fontsize=10)

    ax.set_xlabel("x (cm)", fontsize=10)
    ax.set_ylabel("z (cm)", fontsize=10)
    title = ax.set_title(f"t = {times[0]:.1f} ms", fontsize=10)

    fig.suptitle(
        f"$T_e$ (best-fit) — ES {es_id}: {es_label}  (V_bank = {v_bank} V)",
        fontsize=10,
    )

    def update(ci: int):
        mesh.set_array(te[:, :, ci].ravel())
        title.set_text(f"t = {times[ci]:.1f} ms")
        return mesh, title

    anim = animation.FuncAnimation(
        fig, update, frames=n_cycles, interval=300, blit=False
    )

    out_path = output_dir / f"te_contours_expset{es_id}{suffix}.gif"
    anim.save(str(out_path), writer="pillow", dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_all(hdf5_path: Path, output_dir: Path, save_animation: bool,
             min_shots: int = DEFAULT_MIN_SHOTS,
             vmin_override: float | None = None,
             vmax_override: float | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as hf:
        es_ids = sorted(hf["experiment_sets"].keys(), key=int)

        # --- Pass 1: load all data and compute a GLOBAL colour scale ----------
        data_list: list[dict] = []
        all_finite: list[np.ndarray] = []
        for es_id in es_ids:
            d = _load_experiment_set(hf, es_id, min_shots=min_shots)
            finite = d["te"][np.isfinite(d["te"])]
            if finite.size == 0:
                print(f"ES {es_id}: no finite T_e data, skipping.")
                continue
            data_list.append(d)
            all_finite.append(finite)

        if not all_finite:
            print("No finite T_e data found in any experiment set.")
            return

        combined = np.concatenate(all_finite)
        vmin = vmin_override if vmin_override is not None else max(0.0, float(np.percentile(combined, 2)))
        vmax = vmax_override if vmax_override is not None else float(np.percentile(combined, 98))
        src = "manual override" if (vmin_override or vmax_override) else "2nd–98th percentile across all ESs"
        print(f"Global colour scale: vmin={vmin:.2f} eV  vmax={vmax:.2f} eV  ({src})")
        print(f"Min-shots threshold: {min_shots}")

        # --- Pass 2: plot each ES -------------------------------------------
        for d in data_list:
            plot_subplots(d, output_dir, vmin, vmax)
            if save_animation:
                plot_animation(d, output_dir, vmin, vmax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=HDF5_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-animation", action="store_true",
                        help="skip saving animated GIFs")
    parser.add_argument(
        "--min-shots", type=int, default=DEFAULT_MIN_SHOTS,
        metavar="N",
        help=(
            f"Minimum number of valid shots (n_ok + n_warn) required to show a "
            f"cell.  Cells below this threshold are masked.  Default: {DEFAULT_MIN_SHOTS}. "
            f"Set to 1 to show any cell with at least one good shot."
        ),
    )
    parser.add_argument("--vmin", type=float, default=None,
                        help="Override colour scale minimum (eV). "
                             "Default: 2nd percentile across all experiment sets.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Override colour scale maximum (eV). "
                             "Default: 98th percentile across all experiment sets.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found — run process_langmuir_sweeps.py first."
        )

    plot_all(args.input, args.output_dir,
             save_animation=not args.no_animation,
             min_shots=args.min_shots,
             vmin_override=args.vmin, vmax_override=args.vmax)


if __name__ == "__main__":
    main()
