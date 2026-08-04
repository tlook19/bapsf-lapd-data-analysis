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
ISWEEP_DEADTIME_INPUT = Path("processed/isweep_deadtime_profiles.hdf5")
OUTPUT_DIR  = Path("figures")

PORT_SPACING_CM = 31.95   # from config.py
PORT_2_Z_CM     = 182.5   # from config.py

TE_BOUNDARY_EV  = 0.1     # T_e at machine boundaries
X_WALL_CM       = 35.0    # x boundary (outside ±25 cm scan)
Z_CATHODE_CM    = PORT_2_Z_CM - 2 * PORT_SPACING_CM   # ≈ 119 cm (before port 1)
Z_ANODE_CM      = PORT_2_Z_CM + 64 * PORT_SPACING_CM  # ≈ 2227 cm (past last port)

# Per-zone RBF smoothing: lower → surface fits that point more exactly.
# Core points (|x| ≤ X_CORE_CM) are trusted most; edge points (|x| ≥ X_EDGE_CM)
# are allowed to deviate more because the signal is noisier there.
# Values between X_CORE_CM and X_EDGE_CM get linearly interpolated smoothing.
X_CORE_CM          = 10.0   # inner boundary of transition zone (cm)
X_EDGE_CM          = 15.0   # outer boundary of transition zone (cm)
SMOOTHING_CORE     = 0.05   # tight fit in plasma core
SMOOTHING_EDGE     = 2.0    # loose fit at plasma edge / scrape-off layer
SMOOTHING_SENTINEL = 1e-4   # near-exact enforcement of wall boundary conditions

# Edge prior: measurements near the scan edge are fluctuation-prone and can
# create unrealistically hot shoulders in the filled profile.  Add low-Te anchor
# points in the SOL so the RBF is pinned in the core but pulled down near edges.
EDGE_ANCHOR_CM        = 22.0
TE_EDGE_ANCHOR_EV     = 0.5
SMOOTHING_EDGE_ANCHOR = 0.02
N_EDGE_ANCHOR_Z       = 40

# Z-row coverage threshold: any z-position whose overall fraction of finite cells
# (summed across all x-positions and cycles) falls below this value is excluded
# from the RBF data entirely.  The RBF then smoothly extrapolates from the last
# reliable z-row toward the anode boundary.  ES4's last port has only ~0.3 %
# coverage (3 cells out of 1020); without this filter those outlier cells corrupt
# the fit.  10 % keeps ES4's penultimate port (32 %) while dropping the last one.
MIN_Z_COVERAGE  = 0.10

# Later low-bank sets reach the lower T_e resolution of the swept analysis at
# downstream ports. Keep only the axial rows previously judged reliable; the
# remaining rows are reconstructed from these anchors and the end boundaries.
TRUSTED_TE_PORTS_BY_ES = {
    "3": (11, 29),
    "4": (11,),
}

# Monotonic-z upper bound.  T_e is expected to decrease monotonically with
# axial distance z (away from the cathode).  Any downstream cell whose Te
# exceeds the minimum finite upstream Te by more than this fractional padding
# is masked before the RBF fill.  25 % gives room for shot-to-shot
# fluctuations while still capping grossly inflated downstream estimates.
MONO_Z_PADDING  = 0.25

# Core hot spots can be real when a high T_e cell coincides with an upstream
# -I_SWEEP dead-time current spike.  Preserve those core cells from the
# monotonic-z outlier mask; edge spikes remain masked because they are usually
# edge instability/fluctuation artifacts.
CORE_HOTSPOT_CURRENT_SIGMA = 3.0
CORE_HOTSPOT_CURRENT_RATIO = 1.25

CMAP            = "plasma"


# ---------------------------------------------------------------------------
# Data loading (strict mask: n_bad > n_ok)
# ---------------------------------------------------------------------------
def _load_experiment_set(hf: h5py.File, es_id: str) -> dict:
    """Load strict-masked T_e(n_z, n_x, n_cycles) for one experiment set.

    Only rot=0 runs are used.  Probe shadowing makes rot=180 measurements
    unreliable for T_e (the downstream face sits in the probe's own shadow).
    The rotation_deg attribute reflects any known swap corrections (e.g. ES3
    p21 runs 32/33).
    """
    x_cm = hf["x_cm"][:]
    eg   = hf["experiment_sets"][es_id]

    z_groups: dict[float, list] = {}
    port_by_z: dict[float, int] = {}
    cycle_time_s = None

    for run_id in sorted(eg.keys()):
        rg  = eg[run_id]
        rot = float(rg.attrs.get("rotation_deg", 0))
        if rot != 0.0:
            continue                              # skip rot=180 runs
        z  = float(rg.attrs["z_cm"])
        port = int(rg.attrs["port"])
        if z in port_by_z and port_by_z[z] != port:
            raise ValueError(f"Inconsistent port metadata at z={z:g} cm")
        port_by_z[z] = port
        if z not in z_groups:
            z_groups[z] = []
        te    = rg["te_best_ev"][:]
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
        "port":           np.array([port_by_z[z] for z in z_vals], dtype=np.int16),
        "cycle_time_ms":  cycle_time_s * 1e3,
        "es_label":       eg.attrs.get("label", f"set {es_id}"),
        "v_bank":         eg.attrs.get("v_bank_v", "?"),
        "es_id":          es_id,
    }


def _load_filled_source_experiment_set(hf: h5py.File, es_id: str) -> dict:
    """Load te_masked from a previous te_filled.hdf5-style product."""
    grp = hf["experiment_sets"][es_id]
    z_cm = grp["z_cm"][()]
    ports = (
        grp["port"][()]
        if "port" in grp
        else np.rint((z_cm - PORT_2_Z_CM) / PORT_SPACING_CM + 2).astype(np.int16)
    )
    return {
        "te":             grp["te_masked"][()],
        "x_cm":           grp["x_cm"][()],
        "z_cm":           z_cm,
        "port":           ports,
        "cycle_time_ms":  grp["cycle_time_ms"][()],
        "es_label":       grp.attrs.get("label", f"set {es_id}"),
        "v_bank":         grp.attrs.get("v_bank_v", "?"),
        "es_id":          es_id,
    }


def _load_input_experiment_set(hf: h5py.File, es_id: str, input_mode: str) -> dict:
    if input_mode == "filled":
        return _load_filled_source_experiment_set(hf, es_id)
    if input_mode == "langmuir":
        return _load_experiment_set(hf, es_id)
    grp = hf["experiment_sets"][es_id]
    if "te_masked" in grp:
        return _load_filled_source_experiment_set(hf, es_id)
    return _load_experiment_set(hf, es_id)


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


def _build_edge_anchor_points(
    *,
    x_anchor: float,
    z_cm: np.ndarray,
    z_lo: float,
    z_hi: float,
    te_edge: float,
    n_z: int = N_EDGE_ANCHOR_Z,
) -> tuple[np.ndarray, np.ndarray]:
    """Return low-Te SOL anchor points inside the measured scan envelope."""
    z_min = max(float(np.nanmin(z_cm)), z_lo)
    z_max = min(float(np.nanmax(z_cm)), z_hi)
    z_vals = np.linspace(z_min, z_max, n_z)
    coords = []
    for z in z_vals:
        coords.append((-x_anchor, z))
        coords.append((x_anchor, z))
    return np.array(coords), np.full(len(coords), te_edge)


def _normalise(x: np.ndarray, z: np.ndarray,
               x_wall: float, z_lo: float, z_hi: float
               ) -> np.ndarray:
    """Stack and normalise (x, z) to the unit square."""
    x_n = (x + x_wall) / (2 * x_wall)
    z_n = (z - z_lo)   / (z_hi - z_lo)
    return np.column_stack([x_n, z_n])


def _point_smoothing(
    x_pts: np.ndarray,
    *,
    x_core:          float = X_CORE_CM,
    x_edge:          float = X_EDGE_CM,
    smoothing_core:  float = SMOOTHING_CORE,
    smoothing_edge:  float = SMOOTHING_EDGE,
) -> np.ndarray:
    """Per-point smoothing that increases from core to edge.

    |x| ≤ x_core            → smoothing_core  (tight: surface must fit closely)
    x_core < |x| < x_edge   → linearly interpolated
    |x| ≥ x_edge            → smoothing_edge  (loose: noisier edge data)
    """
    t = np.clip((np.abs(x_pts) - x_core) / (x_edge - x_core), 0.0, 1.0)
    return smoothing_core + t * (smoothing_edge - smoothing_core)


def fill_te_cycle(
    te_2d: np.ndarray,          # (n_z, n_x)  NaN where masked
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    *,
    x_wall:          float = X_WALL_CM,
    z_lo:            float = Z_CATHODE_CM,
    z_hi:            float = Z_ANODE_CM,
    te_boundary:     float = TE_BOUNDARY_EV,
    x_core:          float = X_CORE_CM,
    x_edge:          float = X_EDGE_CM,
    smoothing_core:  float = SMOOTHING_CORE,
    smoothing_edge:  float = SMOOTHING_EDGE,
    edge_anchor:     float = EDGE_ANCHOR_CM,
    te_edge_anchor:  float = TE_EDGE_ANCHOR_EV,
    smoothing_edge_anchor: float = SMOOTHING_EDGE_ANCHOR,
) -> np.ndarray:
    """Return filled (n_z, n_x) T_e array with no NaN cells.

    Uses a thin-plate-spline RBF with position-dependent smoothing:
      |x| ≤ x_core  → smoothing_core (fit closely; trusted core data)
      |x| ≥ x_edge  → smoothing_edge (fit loosely; noisy edge data)
    Sentinel boundary points use smoothing=0 (exact enforcement).
    """
    zz, xx = np.meshgrid(z_cm, x_cm, indexing="ij")  # (n_z, n_x) each
    valid = np.isfinite(te_2d)

    # Data coordinates and values
    data_coords = np.column_stack([xx[valid], zz[valid]])   # [x, z]
    data_values = te_2d[valid]

    if data_values.size < 4:
        return np.full_like(te_2d, te_boundary)

    # Per-point smoothing for data points
    data_smoothing = _point_smoothing(
        data_coords[:, 0],
        x_core=x_core, x_edge=x_edge,
        smoothing_core=smoothing_core, smoothing_edge=smoothing_edge,
    )

    # Sentinel coordinates, values, and smoothing (exact = 0)
    sent_coords, sent_values = _build_sentinel_points(
        x_cm, z_cm, x_wall=x_wall, z_lo=z_lo, z_hi=z_hi, te_boundary=te_boundary
    )
    sent_smoothing = np.full(len(sent_values), SMOOTHING_SENTINEL)

    edge_coords, edge_values = _build_edge_anchor_points(
        x_anchor=edge_anchor,
        z_cm=z_cm,
        z_lo=z_lo,
        z_hi=z_hi,
        te_edge=te_edge_anchor,
    )
    edge_smoothing = np.full(len(edge_values), smoothing_edge_anchor)

    all_coords    = np.vstack([data_coords,    edge_coords,    sent_coords])
    all_values    = np.concatenate([data_values,    edge_values,    sent_values])
    all_smoothing = np.concatenate([data_smoothing, edge_smoothing, sent_smoothing])

    # Normalise to unit square (avoids ~25× z/x aspect ratio skewing kernel)
    all_norm = _normalise(all_coords[:, 0], all_coords[:, 1], x_wall, z_lo, z_hi)

    rbf = RBFInterpolator(
        all_norm, all_values,
        kernel="thin_plate_spline",
        smoothing=all_smoothing,
    )

    # Evaluate on full grid
    grid_norm = _normalise(xx.ravel(), zz.ravel(), x_wall, z_lo, z_hi)
    te_out = rbf(grid_norm).reshape(te_2d.shape)
    return np.clip(te_out, te_boundary, None)


def fill_te_grid(
    te_grid: np.ndarray,        # (n_z, n_x, n_cycles)
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    *,
    min_z_coverage: float = MIN_Z_COVERAGE,
    **kwargs,                   # forwarded to fill_te_cycle
) -> np.ndarray:
    """Fill every cycle in the grid; prints progress.

    Before fitting, any z-row whose fraction of finite cells (pooled over all
    x-positions and cycles) is below *min_z_coverage* is blanked to NaN in
    every cycle.  This prevents sparse, noisy outlier cells at poorly-sampled
    ports from corrupting the RBF interpolant.  The blank rows are then filled
    by smooth extrapolation from neighbouring valid z-rows and the boundary
    sentinel points (T_e = te_boundary at the cathode/anode end plates).
    """
    n_z = te_grid.shape[0]
    z_coverage = np.array([np.isfinite(te_grid[zi]).mean() for zi in range(n_z)])
    sparse = z_coverage < min_z_coverage
    if sparse.any():
        te_grid = te_grid.copy()
        te_grid[sparse] = np.nan
        for zi in np.flatnonzero(sparse):
            print(
                f"    z = {z_cm[zi]:.1f} cm excluded: coverage "
                f"{100 * z_coverage[zi]:.1f}% < {100 * min_z_coverage:.0f}% threshold"
            )

    n_cycles = te_grid.shape[2]
    filled = np.empty_like(te_grid)
    for ci in range(n_cycles):
        filled[:, :, ci] = fill_te_cycle(te_grid[:, :, ci], x_cm, z_cm, **kwargs)
        pct = 100 * (ci + 1) / n_cycles
        sys.stdout.write(f"\r    cycle {ci + 1}/{n_cycles} ({pct:.0f}%)")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return filled


def _mask_untrusted_te_ports(
    te_grid: np.ndarray,
    ports: np.ndarray,
    experiment_set_id: str,
    *,
    trusted_ports_by_es: dict[str, tuple[int, ...]] = TRUSTED_TE_PORTS_BY_ES,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Blank later-set port rows whose swept T_e is not diagnostically reliable."""
    trusted = trusted_ports_by_es.get(str(experiment_set_id))
    if trusted is None:
        return te_grid.copy(), ()
    ports = np.asarray(ports, dtype=int)
    untrusted = tuple(int(port) for port in ports if int(port) not in trusted)
    out = te_grid.copy()
    out[~np.isin(ports, trusted)] = np.nan
    return out, untrusted


def _enforce_core_mean_monotonic_z(
    te_grid: np.ndarray,
    x_cm: np.ndarray,
    *,
    core_x_cm: float = X_CORE_CM,
    te_floor: float = TE_BOUNDARY_EV,
) -> tuple[np.ndarray, int, float]:
    """Scale downstream rows so core-mean T_e is non-increasing with z.

    Each row keeps its radial shape. Scaling is performed about ``te_floor``
    so the imposed wall/end temperature is not pushed below its boundary.
    """
    out = np.asarray(te_grid, dtype=np.float64).copy()
    core = np.abs(np.asarray(x_cm)) <= core_x_cm
    adjusted = 0
    maximum_fractional_reduction = 0.0
    if not np.any(core):
        raise ValueError("No x positions fall inside the monotonic core region")

    for cycle_idx in range(out.shape[2]):
        previous_mean = np.inf
        for z_idx in range(out.shape[0]):
            row = out[z_idx, :, cycle_idx]
            mean = float(np.nanmean(row[core]))
            if not np.isfinite(mean):
                continue
            if mean > previous_mean:
                if mean <= te_floor or previous_mean <= te_floor:
                    scale = 0.0
                else:
                    scale = (previous_mean - te_floor) / (mean - te_floor)
                scale = float(np.clip(scale, 0.0, 1.0))
                out[z_idx, :, cycle_idx] = te_floor + (row - te_floor) * scale
                adjusted += 1
                maximum_fractional_reduction = max(
                    maximum_fractional_reduction,
                    1.0 - scale,
                )
                mean = previous_mean
            previous_mean = min(previous_mean, mean)
    return out, adjusted, maximum_fractional_reduction


# ---------------------------------------------------------------------------
# Core hot-spot preservation from -I_SWEEP dead-time current
# ---------------------------------------------------------------------------
def _robust_current_spike_mask(
    current_x_cycle: np.ndarray,
    x_cm: np.ndarray,
    *,
    core_x_cm: float = X_CORE_CM,
    sigma: float = CORE_HOTSPOT_CURRENT_SIGMA,
    ratio: float = CORE_HOTSPOT_CURRENT_RATIO,
) -> np.ndarray:
    """Return mask of core current spikes, robustly thresholded per cycle."""
    current = np.asarray(current_x_cycle, dtype=np.float64)
    mask = np.zeros(current.shape, dtype=bool)
    core = np.abs(x_cm) <= core_x_cm
    if not core.any():
        return mask

    core_current = current[core]
    finite_core = np.isfinite(core_current)
    with np.errstate(all="ignore"):
        median = np.nanmedian(core_current, axis=0)
        mad = np.nanmedian(np.abs(core_current - median[np.newaxis, :]), axis=0)
    robust_sigma = 1.4826 * mad

    finite_counts = finite_core.sum(axis=0)
    has_baseline = (finite_counts >= 3) & np.isfinite(median)
    if not has_baseline.any():
        return mask

    sigma_threshold = median + sigma * robust_sigma
    ratio_threshold = np.where(
        median > 0.0,
        median * ratio,
        median + np.abs(median) * (ratio - 1.0),
    )
    threshold = np.maximum(sigma_threshold, ratio_threshold)
    core_spikes = (
        np.isfinite(core_current)
        & has_baseline[np.newaxis, :]
        & (core_current > threshold[np.newaxis, :])
    )
    mask[np.flatnonzero(core), :] = core_spikes
    return mask


def _interp_current_to_te_cycles(
    current_x_cycle: np.ndarray,
    current_time_ms: np.ndarray,
    te_time_ms: np.ndarray,
) -> np.ndarray:
    if (
        current_x_cycle.shape[1] == te_time_ms.size
        and current_time_ms.shape == te_time_ms.shape
        and np.allclose(current_time_ms, te_time_ms)
    ):
        return current_x_cycle
    return np.vstack([
        np.interp(te_time_ms, current_time_ms, current_x_cycle[xi, :])
        for xi in range(current_x_cycle.shape[0])
    ])


def _load_current_hotspot_mask(
    hf: h5py.File,
    es_id: str,
    *,
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    cycle_time_ms: np.ndarray,
    core_x_cm: float,
    sigma: float,
    ratio: float,
) -> np.ndarray:
    """Build a (n_z, n_x, n_cycles) mask of corroborated core hot spots."""
    mask = np.zeros((len(z_cm), len(x_cm), len(cycle_time_ms)), dtype=bool)
    if "x_cm" not in hf or "experiment_sets" not in hf or es_id not in hf["experiment_sets"]:
        return mask

    current_x_cm = hf["x_cm"][()]
    if not np.allclose(x_cm, current_x_cm):
        raise ValueError(f"-I_SWEEP dead-time x grid does not match T_e grid for ES {es_id}")

    eg = hf[f"experiment_sets/{es_id}"]
    for run_id in sorted(eg.keys()):
        rg = eg[run_id]
        source_channel = str(rg.attrs.get("deadtime_source_channel", ""))
        source_invert = bool(rg.attrs.get("deadtime_source_invert_polarity", False))
        if source_channel != "i_sweep" or not source_invert:
            continue

        z_idx = int(np.argmin(np.abs(z_cm - float(rg.attrs["z_cm"]))))
        if not np.isclose(z_cm[z_idx], float(rg.attrs["z_cm"]), atol=1e-3):
            continue

        current_time_ms = rg["inter_sweep_time_s"][()] * 1000.0
        current = _interp_current_to_te_cycles(
            rg["isat_a"][()],
            current_time_ms,
            cycle_time_ms,
        )
        mask[z_idx] |= _robust_current_spike_mask(
            current,
            x_cm,
            core_x_cm=core_x_cm,
            sigma=sigma,
            ratio=ratio,
        )
    return mask


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

    # Colour scale from the masked finite values — matches plot_te_contours.py.
    # (Using te_filled would compress the scale because boundary-fill adds many
    # 0.1 eV points, making the masked top row look darker than in te_contours.)
    finite = te_masked[np.isfinite(te_masked)]
    if finite.size == 0:
        finite = te_filled.ravel()
    vmin = max(0.0, float(np.percentile(finite, 2)))
    vmax = float(np.percentile(finite, 98))
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
    cbar.set_label("$T_e$ (eV)  [best-fit]", fontsize=9)
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
# Monotonic-z upper bound
# ---------------------------------------------------------------------------
def _monotonic_z_bound(
    te_grid: np.ndarray,        # (n_z, n_x, n_cycles)
    z_cm: np.ndarray,
    *,
    padding: float = MONO_Z_PADDING,
    preserve_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Mask downstream cells that violate the expected z-monotonicity of T_e.

    For each z-row z_i (i > 0), the upper bound is a **per-cycle global
    maximum** over all upstream z-rows and all x positions:

        bound(cycle) = max( T_e[z_j, x, cycle]  for j < i, all x, finite )
                       × (1 + padding)

    Using the spatial maximum (rather than a per-x value) correctly handles
    radial heat spreading: if a hot spot at x=0 upstream spreads outward by
    the time it reaches z_i, a position at x=6 can legitimately exceed its
    own upstream value as long as it stays within the global upstream maximum.
    The bound is only applied where at least one upstream finite value exists.
    The most-upstream z-row (i=0) is never touched.

    Parameters
    ----------
    padding : float
        Fractional allowance above the upstream peak.  0.25 → 25 % above
        the highest reliable upstream measurement anywhere in the profile.

    Returns
    -------
    te_bounded : (n_z, n_x, n_cycles) — copy of te_grid with outliers masked
    n_masked   : total number of cells set to NaN by this step
    n_preserved: total number of outlier cells preserved by preserve_mask
    """
    te_out   = te_grid.copy()
    n_masked = 0
    n_preserved = 0
    if preserve_mask is None:
        preserve_mask = np.zeros(te_grid.shape, dtype=bool)
    elif preserve_mask.shape != te_grid.shape:
        raise ValueError(
            f"preserve_mask shape {preserve_mask.shape} does not match T_e grid {te_grid.shape}"
        )

    for zi in range(1, len(z_cm)):
        # Global maximum T_e across all upstream z-rows and all x, per cycle.
        # Shape: (n_cycles,)  — one bound value per time point.
        with np.errstate(all="ignore"):   # nanmax of all-NaN slice → NaN, handled below
            upstream_max = np.nanmax(
                te_grid[:zi].reshape(-1, te_grid.shape[2]), axis=0
            )
        has_upstream = np.isfinite(upstream_max)          # (n_cycles,)
        upper_bound  = upstream_max * (1.0 + padding)     # (n_cycles,)

        # Broadcast bound from (n_cycles,) → (n_x, n_cycles) for comparison.
        exceeds = (
            np.isfinite(te_grid[zi])
            & has_upstream[np.newaxis, :]
            & (te_grid[zi] > upper_bound[np.newaxis, :])
        )
        preserved = exceeds & preserve_mask[zi]
        mask_out = exceeds & ~preserve_mask[zi]
        if exceeds.any():
            te_out[zi][mask_out] = np.nan
            n_masked += int(mask_out.sum())
            n_preserved += int(preserved.sum())
            n_ref = int(
                (np.isfinite(te_grid[zi]) & has_upstream[np.newaxis, :]).sum()
            )
            print(
                f"    z = {z_cm[zi]:.1f} cm: {mask_out.sum()} / {n_ref} cells "
                f"capped, {preserved.sum()} preserved by -I_SWEEP hot spot "
                f"(bound = upstream_max × {1+padding:.2f})"
            )

    return te_out, n_masked, n_preserved


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
    parser.add_argument("--input-mode", choices=["auto", "langmuir", "filled"], default="auto",
                        help="Input layout: process_langmuir_sweeps output, previous te_filled output, or auto-detect.")
    parser.add_argument("--x-wall",      type=float, default=X_WALL_CM,
                        help="x position of drift-tube wall boundary (cm)")
    parser.add_argument("--z-cathode",   type=float, default=Z_CATHODE_CM,
                        help="z position of cathode end-plate boundary (cm)")
    parser.add_argument("--z-anode",     type=float, default=Z_ANODE_CM,
                        help="z position of anode end-plate boundary (cm)")
    parser.add_argument("--te-boundary",    type=float, default=TE_BOUNDARY_EV,
                        help="T_e boundary value at machine walls (eV)")
    parser.add_argument("--x-core",         type=float, default=X_CORE_CM,
                        help="Inner edge of weighting transition zone (cm). "
                             "Points with |x| ≤ x-core get smoothing-core.")
    parser.add_argument("--x-edge",         type=float, default=X_EDGE_CM,
                        help="Outer edge of weighting transition zone (cm). "
                             "Points with |x| ≥ x-edge get smoothing-edge.")
    parser.add_argument("--smoothing-core", type=float, default=SMOOTHING_CORE,
                        help="RBF smoothing for core region (low = exact fit).")
    parser.add_argument("--smoothing-edge", type=float, default=SMOOTHING_EDGE,
                        help="RBF smoothing for edge region (high = loose fit).")
    parser.add_argument("--edge-anchor", type=float, default=EDGE_ANCHOR_CM,
                        help="|x| position for low-Te SOL anchor points (cm).")
    parser.add_argument("--te-edge-anchor", type=float, default=TE_EDGE_ANCHOR_EV,
                        help="T_e value assigned to SOL edge anchor points (eV).")
    parser.add_argument("--smoothing-edge-anchor", type=float, default=SMOOTHING_EDGE_ANCHOR,
                        help="RBF smoothing for SOL edge anchor points.")
    parser.add_argument("--mono-z-padding", type=float, default=MONO_Z_PADDING,
                        help="Fractional allowance above the upstream T_e envelope for the "
                             "monotonic-z upper bound.  Set to a large value (e.g. 1e6) to "
                             f"disable.  Default: {MONO_Z_PADDING}.")
    parser.add_argument("--isweep-deadtime", type=Path, default=ISWEEP_DEADTIME_INPUT,
                        help="Optional -I_SWEEP dead-time profile HDF5 used to preserve "
                             "core T_e hot spots that coincide with current spikes.")
    parser.add_argument("--no-current-hotspot-preservation", action="store_true",
                        help="Do not preserve monotonic-z outliers corroborated by core "
                             "-I_SWEEP dead-time current spikes.")
    parser.add_argument("--core-hotspot-current-sigma", type=float,
                        default=CORE_HOTSPOT_CURRENT_SIGMA,
                        help="Robust per-cycle sigma threshold for core current spikes.")
    parser.add_argument("--core-hotspot-current-ratio", type=float,
                        default=CORE_HOTSPOT_CURRENT_RATIO,
                        help="Minimum current/median ratio for core current spikes.")
    parser.add_argument("--min-z-coverage", type=float, default=MIN_Z_COVERAGE,
                        help="Minimum fraction of finite cells (across all x and cycles) "
                             "for a z-row to be included in the RBF fit. Rows below this "
                             "threshold are blanked and filled by boundary extrapolation. "
                             f"Default: {MIN_Z_COVERAGE}.")
    parser.add_argument("--no-plots",       action="store_true",
                        help="skip comparison figures")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"{args.input} not found — run process_langmuir_sweeps.py first."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fill_kwargs = dict(
        x_wall          = args.x_wall,
        z_lo            = args.z_cathode,
        z_hi            = args.z_anode,
        te_boundary     = args.te_boundary,
        x_core          = args.x_core,
        x_edge          = args.x_edge,
        smoothing_core  = args.smoothing_core,
        smoothing_edge  = args.smoothing_edge,
        edge_anchor     = args.edge_anchor,
        te_edge_anchor  = args.te_edge_anchor,
        smoothing_edge_anchor = args.smoothing_edge_anchor,
        min_z_coverage  = args.min_z_coverage,
    )

    current_hf = (
        None
        if args.no_current_hotspot_preservation or not args.isweep_deadtime.exists()
        else h5py.File(args.isweep_deadtime, "r")
    )

    try:
        with h5py.File(args.input, "r") as hf_in, \
             h5py.File(args.output, "w") as hf_out:

            es_ids = sorted(hf_in["experiment_sets"].keys(), key=int)
            hf_out.create_group("experiment_sets")

            for es_id in es_ids:
                print(f"\nES {es_id}:")
                data = _load_input_experiment_set(hf_in, es_id, args.input_mode)

                te_masked = data["te"]
                x_cm      = data["x_cm"]
                z_cm      = data["z_cm"]
                ports     = data["port"]

                n_finite = int(np.isfinite(te_masked).sum())
                n_total  = int(te_masked.size)
                print(f"  {n_finite}/{n_total} cells finite ({100*n_finite/n_total:.1f}%) before fill")

                te_masked, untrusted_ports = _mask_untrusted_te_ports(
                    te_masked,
                    ports,
                    es_id,
                )
                if untrusted_ports:
                    print(
                        "  Low-T_e/unreliable downstream rows excluded before fill: "
                        + ", ".join(f"p{port}" for port in untrusted_ports)
                    )

                preserve_mask = None
                if current_hf is not None:
                    preserve_mask = _load_current_hotspot_mask(
                        current_hf,
                        es_id,
                        x_cm=x_cm,
                        z_cm=z_cm,
                        cycle_time_ms=data["cycle_time_ms"],
                        core_x_cm=args.x_core,
                        sigma=args.core_hotspot_current_sigma,
                        ratio=args.core_hotspot_current_ratio,
                    )
                    print(
                        f"  Core -I_SWEEP hot-spot preservation: "
                        f"{int(preserve_mask.sum())} candidate cells"
                    )

                # Apply monotonic-z upper bound: downstream cells whose Te exceeds
                # the upstream Te envelope are masked before the RBF, unless a
                # core -I_SWEEP dead-time spike corroborates a real hot spot.
                print(f"  Monotonic-z bound (padding={args.mono_z_padding:.0%}) …")
                te_masked, n_mono, n_preserved = _monotonic_z_bound(
                    te_masked, z_cm, padding=args.mono_z_padding, preserve_mask=preserve_mask
                )
                if n_mono:
                    print(f"    {n_mono} cells masked by monotonicity bound")
                else:
                    print(f"    No cells exceeded the upstream bound")
                if n_preserved:
                    print(f"    {n_preserved} core hot-spot cells preserved")

                print(f"  Fitting {te_masked.shape[2]} cycles …")
                te_filled = fill_te_grid(te_masked, x_cm, z_cm, **fill_kwargs)
                te_filled, n_monotonic_rows, max_monotonic_reduction = (
                    _enforce_core_mean_monotonic_z(
                        te_filled,
                        x_cm,
                        core_x_cm=args.x_core,
                        te_floor=args.te_boundary,
                    )
                )
                print(
                    "  Core-mean monotonic-z enforcement: "
                    f"{n_monotonic_rows} row-cycles adjusted, "
                    f"maximum reduction {max_monotonic_reduction:.1%}"
                )

                # Write to HDF5
                grp = hf_out["experiment_sets"].create_group(es_id)
                grp.attrs["label"]    = data["es_label"]
                grp.attrs["v_bank_v"] = float(data["v_bank"])
                grp.attrs["current_hotspot_preservation_enabled"] = bool(current_hf is not None)
                grp.attrs["current_hotspot_preserved_cells"] = int(n_preserved)
                grp.attrs["current_hotspot_current_sigma"] = float(args.core_hotspot_current_sigma)
                grp.attrs["current_hotspot_current_ratio"] = float(args.core_hotspot_current_ratio)
                grp.attrs["trusted_te_ports"] = ",".join(
                    str(port) for port in TRUSTED_TE_PORTS_BY_ES.get(es_id, tuple(ports))
                )
                grp.attrs["excluded_unreliable_te_ports"] = ",".join(
                    str(port) for port in untrusted_ports
                )
                grp.attrs["core_mean_te_monotonic_z_enforced"] = True
                grp.attrs["core_mean_te_monotonic_adjusted_row_cycles"] = int(
                    n_monotonic_rows
                )
                grp.attrs["core_mean_te_monotonic_max_fractional_reduction"] = float(
                    max_monotonic_reduction
                )
                grp.create_dataset("x_cm",           data=x_cm,                compression="gzip")
                grp.create_dataset("z_cm",           data=z_cm,                compression="gzip")
                grp.create_dataset("port",           data=ports,               compression="gzip")
                grp.create_dataset("cycle_time_ms",  data=data["cycle_time_ms"], compression="gzip")
                grp.create_dataset("te_masked",      data=te_masked,            compression="gzip", compression_opts=4)
                grp.create_dataset("te_filled",      data=te_filled,            compression="gzip", compression_opts=4)
                hf_out.flush()

                if not args.no_plots:
                    plot_data = dict(data)
                    plot_data["te"] = te_masked
                    _plot_comparison(plot_data, te_filled, args.output_dir)
    finally:
        if current_hf is not None:
            current_hf.close()

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
