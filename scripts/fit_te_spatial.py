"""Fit T_e spatial profiles with machine-boundary conditions.

Loads the strict-masked T_e(x, z, cycle) grid from langmuir_sweeps.hdf5,
adds boundary sentinel points (T_e = TE_BOUNDARY_EV) at the machine walls
in x and at the cathode/anode ends in z, then fills all NaN cells using a
2-D thin-plate-spline RBF interpolant.

The resulting te_filled array is intended for probe-area calibration and
density analysis where a continuous T_e map is required.

Boundary conditions
-------------------
  |x| = X_WALL_CM   →  T_e = TE_BOUNDARY_EV  (drift-tube wall)
  z  = Z_CATHODE_CM →  T_e = TE_BOUNDARY_EV  (cathode end plate)
  z  = Z_ANODE_CM   →  T_e = TE_BOUNDARY_EV  (anode end plate)

Coordinates are normalised to [0, 1] before the RBF fit so the
~25-fold difference in x vs z span does not make the kernel anisotropic.

Outputs
-------
processed/te_filled.hdf5
  /experiment_sets/{es_id}/
    attrs : label, v_bank_v
    x_cm            (n_x,)
    z_cm            (n_z,)
    cycle_time_ms   (n_cycles,)
    te_masked       (n_z, n_x, n_cycles)   strict-masked data (NaN where hidden)
    te_filled       (n_z, n_x, n_cycles)   RBF-interpolated, boundary-filled

figures/te_filled_expset{es_id}.png
  Masked vs filled T_e at four representative cycles.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/fit_te_spatial.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/fit_te_spatial.py \\
      --input processed/langmuir_sweeps.hdf5 --output processed/te_filled.hdf5 \\
      --x-wall 35 --z-cathode 100 --z-anode 1900 --te-boundary 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
HDF5_INPUT  = Path("processed/langmuir_sweeps.hdf5")
HDF5_OUTPUT = Path("processed/te_filled.hdf5")
OUTPUT_DIR  = Path("figures")

PORT_SPACING_CM = 31.95   # from config.py
PORT_2_Z_CM     = 182.5   # from config.py

TE_BOUNDARY_EV  = 0.1     # T_e at machine boundaries
X_WALL_CM       = 35.0    # x boundary (outside ±25 cm scan)
Z_CATHODE_CM    = PORT_2_Z_CM - 2 * PORT_SPACING_CM   # ≈ 119 cm (before port 1)
Z_ANODE_CM      = PORT_2_Z_CM + 64 * PORT_SPACING_CM  # ≈ 2227 cm (past last port)

RBF_SMOOTHING   = 0.3     # allows small deviations from data to prevent ringing
CMAP            = "plasma"


# ---------------------------------------------------------------------------
# Data loading (strict mask: n_bad > n_ok)
# ---------------------------------------------------------------------------
def _load_experiment_set(hf: h5py.File, es_id: str) -> dict:
    """Load strict-masked T_e(n_z, n_x, n_cycles) for one experiment set."""
    x_cm = hf["x_cm"][:]
    eg   = hf["experiment_sets"][es_id]

    z_groups: dict[float, list] = {}
    cycle_time_s = None

    for run_id in sorted(eg.keys()):
        rg = eg[run_id]
        z  = float(rg.attrs["z_cm"])
        if z not in z_groups:
            z_groups[z] = []
        te    = rg["te_log_ev"][:]
        n_ok  = rg["n_ok"][:]
        n_bad = rg["n_bad"][:]
        z_groups[z].append((te, n_ok, n_bad))
        if cycle_time_s is None:
            cycle_time_s = rg["cycle_time_s"][:]

    z_vals   = sorted(z_groups.keys())
    n_z      = len(z_vals)
    n_x      = len(x_cm)
    n_cycles = len(cycle_time_s)
    te_grid  = np.full((n_z, n_x, n_cycles), np.nan)

    for zi, z in enumerate(z_vals):
        te_sum = np.zeros((n_x, n_cycles))
        weight = np.zeros((n_x, n_cycles))
        mask_bad = np.zeros((n_x, n_cycles), dtype=bool)

        for te, n_ok, n_bad in z_groups[z]:
            cell_bad = n_bad > n_ok          # strict mask
            mask_bad |= cell_bad
            valid     = ~cell_bad
            te_sum   += np.where(valid, te, 0.0)
            weight   += valid.astype(float)

        with np.errstate(invalid="ignore"):
            te_avg = np.where(weight > 0, te_sum / weight, np.nan)
        all_bad = mask_bad & (weight == 0)
        te_avg[all_bad] = np.nan
        te_grid[zi] = te_avg

    return {
        "te":             te_grid,
        "x_cm":           x_cm,
        "z_cm":           np.array(z_vals),
        "cycle_time_ms":  cycle_time_s * 1e3,
        "es_label":       eg.attrs.get("label", f"set {es_id}"),
        "v_bank":         eg.attrs.get("v_bank_v", "?"),
        "es_id":          es_id,
    }


# ---------------------------------------------------------------------------
# 2-D RBF fill
# ---------------------------------------------------------------------------
def _build_sentinel_points(
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    *,
    x_wall: float,
    z_lo: float,
    z_hi: float,
    te_boundary: float,
    n_x_wall: int = 30,
    n_z_end: int  = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sentinel_coords, sentinel_values) arrays.

    Sentinel points are placed:
      - Along x = ±x_wall from z_lo to z_hi  (drift-tube wall)
      - Along z = z_lo and z = z_hi from -x_wall to +x_wall (end plates)
    All carry T_e = te_boundary.
    """
    z_wall = np.linspace(z_lo, z_hi, n_x_wall)
    x_ends = np.linspace(-x_wall, x_wall, n_z_end)

    coords = []
    # x walls
    for zw in z_wall:
        coords.append((-x_wall, zw))
        coords.append(( x_wall, zw))
    # z end plates
    for xe in x_ends:
        coords.append((xe, z_lo))
        coords.append((xe, z_hi))

    coords  = np.array(coords)          # (N, 2)  columns: [x, z]
    values  = np.full(len(coords), te_boundary)
    return coords, values


def _normalise(x: np.ndarray, z: np.ndarray,
               x_wall: float, z_lo: float, z_hi: float
               ) -> np.ndarray:
    """Stack and normalise (x, z) to the unit square."""
    x_n = (x + x_wall) / (2 * x_wall)
    z_n = (z - z_lo)   / (z_hi - z_lo)
    return np.column_stack([x_n, z_n])


def fill_te_cycle(
    te_2d: np.ndarray,          # (n_z, n_x)  NaN where masked
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    *,
    x_wall:      float = X_WALL_CM,
    z_lo:        float = Z_CATHODE_CM,
    z_hi:        float = Z_ANODE_CM,
    te_boundary: float = TE_BOUNDARY_EV,
    smoothing:   float = RBF_SMOOTHING,
) -> np.ndarray:
    """Return filled (n_z, n_x) T_e array with no NaN cells.

    Uses a thin-plate-spline RBF fit through the valid data points and the
    machine-boundary sentinels, evaluated on the full (x, z) grid.
    """
    zz, xx = np.meshgrid(z_cm, x_cm, indexing="ij")  # (n_z, n_x) each
    valid = np.isfinite(te_2d)

    # Data coordinates and values
    data_coords = np.column_stack([xx[valid], zz[valid]])   # [x, z]
    data_values = te_2d[valid]

    if data_values.size < 4:
        # Too few points; return boundary value everywhere.
        return np.full_like(te_2d, te_boundary)

    # Sentinel coordinates and values
    sent_coords, sent_values = _build_sentinel_points(
        x_cm, z_cm, x_wall=x_wall, z_lo=z_lo, z_hi=z_hi, te_boundary=te_boundary
    )

    all_coords = np.vstack([data_coords, sent_coords])
    all_values = np.concatenate([data_values, sent_values])

    # Normalise to unit square (avoids ~25× z/x aspect ratio skewing kernel)
    all_norm = _normalise(all_coords[:, 0], all_coords[:, 1], x_wall, z_lo, z_hi)

    rbf = RBFInterpolator(
        all_norm, all_values,
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )

    # Evaluate on full grid
    grid_norm = _normalise(xx.ravel(), zz.ravel(), x_wall, z_lo, z_hi)
    te_out = rbf(grid_norm).reshape(te_2d.shape)
    return np.clip(te_out, te_boundary, None)


def fill_te_grid(
    te_grid: np.ndarray,        # (n_z, n_x, n_cycles)
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Fill every cycle in the grid; prints progress."""
    n_cycles = te_grid.shape[2]
    filled = np.empty_like(te_grid)
    for ci in range(n_cycles):
        filled[:, :, ci] = fill_te_cycle(te_grid[:, :, ci], x_cm, z_cm, **kwargs)
        pct = 100 * (ci + 1) / n_cycles
        sys.stdout.write(f"\r    cycle {ci + 1}/{n_cycles} ({pct:.0f}%)")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return filled


# ---------------------------------------------------------------------------
# Comparison figure
# ---------------------------------------------------------------------------
def _plot_comparison(data: dict, te_filled: np.ndarray, output_dir: Path) -> None:
    """Four-cycle comparison: masked T_e (top) vs filled T_e (bottom)."""
    te_masked = data["te"]          # (n_z, n_x, n_cycles)
    x_cm      = data["x_cm"]
    z_cm      = data["z_cm"]
    times     = data["cycle_time_ms"]
    es_id     = data["es_id"]
    es_label  = data["es_label"]
    v_bank    = data["v_bank"]
    n_cycles  = te_masked.shape[2]

    # Representative cycle indices
    cyc_indices = [
        0,
        n_cycles // 3,
        2 * n_cycles // 3,
        n_cycles - 1,
    ]

    # Shared colour scale from filled data (has no NaN)
    vmin = max(0.0, float(np.percentile(te_filled, 2)))
    vmax = float(np.percentile(te_filled, 98))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def edges(arr):
        mid = (arr[:-1] + arr[1:]) / 2
        lo  = arr[0]  - (arr[1]  - arr[0])  / 2
        hi  = arr[-1] + (arr[-1] - arr[-2]) / 2
        return np.concatenate([[lo], mid, [hi]])

    x_edges = edges(x_cm)
    z_edges = edges(z_cm)

    fig, axes = plt.subplots(
        2, len(cyc_indices),
        figsize=(4.5 * len(cyc_indices), 7),
        constrained_layout=True,
    )

    row_labels = ["masked", "filled"]
    grids      = [te_masked, te_filled]

    for row, (label, grid) in enumerate(zip(row_labels, grids)):
        for col, ci in enumerate(cyc_indices):
            ax = axes[row, col]
            frame = grid[:, :, ci]
            ax.pcolormesh(x_edges, z_edges, frame,
                          cmap=CMAP, norm=norm, rasterized=True)
            ax.set_title(f"t = {times[ci]:.1f} ms", fontsize=8)
            ax.set_xlabel("x (cm)", fontsize=7)
            ax.set_ylabel("z (cm)", fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(f"{label}\nz (cm)", fontsize=7)

    sm   = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("$T_e$ (eV)  [log-linear]", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"$T_e$ spatial fill — ES {es_id}: {es_label}  (V_bank = {v_bank} V)\n"
        f"Top: strict-masked data   Bottom: RBF-filled (boundary = {TE_BOUNDARY_EV} eV)",
        fontsize=10,
    )

    out = output_dir / f"te_filled_expset{es_id}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",       type=Path,  default=HDF5_INPUT)
    parser.add_argument("--output",      type=Path,  default=HDF5_OUTPUT)
    parser.add_argument("--output-dir",  type=Path,  default=OUTPUT_DIR)
    parser.add_argument("--x-wall",      type=float, default=X_WALL_CM,
                        help="x position of drift-tube wall boundary (cm)")
    parser.add_argument("--z-cathode",   type=float, default=Z_CATHODE_CM,
                        help="z position of cathode end-plate boundary (cm)")
    parser.add_argument("--z-anode",     type=float, default=Z_ANODE_CM,
                        help="z position of anode end-plate boundary (cm)")
    parser.add_argument("--te-boundary", type=float, default=TE_BOUNDARY_EV,
                        help="T_e boundary value (eV)")
    parser.add_argument("--smoothing",   type=float, default=RBF_SMOOTHING,
                        help="RBF smoothing parameter (0 = exact interpolation)")
    parser.add_argument("--no-plots",    action="store_true",
                        help="skip comparison figures")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found — run process_langmuir_sweeps.py first."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fill_kwargs = dict(
        x_wall      = args.x_wall,
        z_lo        = args.z_cathode,
        z_hi        = args.z_anode,
        te_boundary = args.te_boundary,
        smoothing   = args.smoothing,
    )

    with h5py.File(args.input, "r") as hf_in, \
         h5py.File(args.output, "w") as hf_out:

        es_ids = sorted(hf_in["experiment_sets"].keys(), key=int)
        hf_out.create_group("experiment_sets")

        for es_id in es_ids:
            print(f"\nES {es_id}:")
            data = _load_experiment_set(hf_in, es_id)

            te_masked = data["te"]
            x_cm      = data["x_cm"]
            z_cm      = data["z_cm"]

            n_finite = int(np.isfinite(te_masked).sum())
            n_total  = int(te_masked.size)
            print(f"  {n_finite}/{n_total} cells finite ({100*n_finite/n_total:.1f}%) before fill")

            print(f"  Fitting {te_masked.shape[2]} cycles …")
            te_filled = fill_te_grid(te_masked, x_cm, z_cm, **fill_kwargs)

            # Write to HDF5
            grp = hf_out["experiment_sets"].create_group(es_id)
            grp.attrs["label"]    = data["es_label"]
            grp.attrs["v_bank_v"] = float(data["v_bank"])
            grp.create_dataset("x_cm",           data=x_cm,                compression="gzip")
            grp.create_dataset("z_cm",           data=z_cm,                compression="gzip")
            grp.create_dataset("cycle_time_ms",  data=data["cycle_time_ms"], compression="gzip")
            grp.create_dataset("te_masked",      data=te_masked,            compression="gzip", compression_opts=4)
            grp.create_dataset("te_filled",      data=te_filled,            compression="gzip", compression_opts=4)
            hf_out.flush()

            if not args.no_plots:
                _plot_comparison(data, te_filled, args.output_dir)

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
