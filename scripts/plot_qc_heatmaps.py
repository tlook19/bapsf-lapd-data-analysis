"""Plot QC bad-shot heatmaps for every run in the processed HDF5.

Produces one PNG per experiment set (4 files).  Each figure has 8 subplots
arranged in a 2 × 4 grid — one per run in manifest order.  The heatmap shows
how many of the 20 shots at each (x-position, sweep-cycle) cell were flagged
as 'bad' by the Langmuir QC pipeline.

Axes
----
  x  : probe scan position in cm (−25 to +25)
  y  : time at ramp start in ms (increasing upward)
  color : n_bad, 0–20 shots, colormap 'Reds'

Runs with rotation_deg == 180 are marked with an asterisk (*) in their
subplot title so rot-0/rot-180 pairs at the same port can be distinguished.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_qc_heatmaps.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_qc_heatmaps.py \\
      --input processed/langmuir_sweeps.hdf5 --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


HDF5_INPUT = Path("processed/langmuir_sweeps.hdf5")
OUTPUT_DIR = Path("figures")

CMAP = "Reds"
VMAX = 20  # maximum possible bad shots per cell (= shots per position)
NCOLS = 4
NROWS = 2


def _plot_experiment_set(
    ax_grid: list,
    run_ids: list[str],
    run_groups: dict,
    x_cm: np.ndarray,
    es_label: str,
    es_id: str,
) -> plt.cm.ScalarMappable:
    """Fill one figure's subplot grid with heatmaps; return the ScalarMappable."""
    norm = mcolors.Normalize(vmin=0, vmax=VMAX)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)

    for ax, run_id in zip(ax_grid, run_ids):
        grp = run_groups[run_id]
        n_bad = grp["n_bad"][:]          # (51, n_cycles)
        time_ms = grp["cycle_time_s"][:] * 1e3
        rot = float(grp.attrs["rotation_deg"])
        port = int(grp.attrs["port"])

        # pcolormesh expects edges, not centres — build half-step-padded edges.
        dx = (x_cm[1] - x_cm[0]) / 2
        x_edges = np.concatenate([[x_cm[0] - dx], x_cm + dx])
        dt = (time_ms[1] - time_ms[0]) / 2 if len(time_ms) > 1 else 0.5
        t_edges = np.concatenate([[time_ms[0] - dt], time_ms + dt])

        # n_bad is (n_pos, n_cycles); transpose to (n_cycles, n_pos) for plot.
        ax.pcolormesh(x_edges, t_edges, n_bad.T, cmap=CMAP, norm=norm,
                      rasterized=True)

        rot_tag = " *" if rot == 180.0 else ""
        ax.set_title(f"run {run_id} | p{port}{rot_tag}", fontsize=8)
        ax.set_xlabel("x (cm)", fontsize=7)
        ax.set_ylabel("t (ms)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.axvline(0, color="white", lw=0.5, alpha=0.6)  # mark x=0

    return mappable


def plot_all(hdf5_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as hf:
        x_cm = hf["x_cm"][:]
        es_root = hf["experiment_sets"]

        for es_id in sorted(es_root.keys(), key=int):
            es_grp = es_root[es_id]
            es_label = es_grp.attrs.get("label", f"set {es_id}")
            v_bank = es_grp.attrs.get("v_bank_v", "?")
            run_ids = sorted(es_grp.keys())

            fig, axes = plt.subplots(
                NROWS, NCOLS,
                figsize=(14, 6),
                constrained_layout=True,
            )
            ax_flat = axes.flatten().tolist()

            # Hide any unused subplots (in case a set has fewer than 8 runs).
            for ax in ax_flat[len(run_ids):]:
                ax.set_visible(False)

            mappable = _plot_experiment_set(
                ax_flat[:len(run_ids)],
                run_ids,
                {rid: es_grp[rid] for rid in run_ids},
                x_cm,
                es_label,
                es_id,
            )

            cbar = fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.02)
            cbar.set_label("bad fits (out of 20 shots)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

            fig.suptitle(
                f"QC bad-shot map — experiment set {es_id}: {es_label}  "
                f"(V_bank = {v_bank} V)\n"
                "* = probe rotation 180°",
                fontsize=10,
            )

            out_path = output_dir / f"qc_heatmap_expset{es_id}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=HDF5_INPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found — run process_langmuir_sweeps.py first."
        )

    plot_all(args.input, args.output_dir)


if __name__ == "__main__":
    main()
