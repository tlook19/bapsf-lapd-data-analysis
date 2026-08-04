"""Compute per-cell manual-review flags and write them into langmuir_sweeps.hdf5.

For each (pos_idx, cycle_idx) cell in every run group the script evaluates
five independent flag criteria and packs them into a uint8 bitmask dataset.
A separate bool dataset marks any cell that has at least one flag set.

Flag bits
---------
  bit 0  QUALITY   — ≥ bad_frac_thresh of shots are bad
  bit 1  DISAGREE  — |Te_log − Te_exp| / Te_best ≥ disagree_frac
  bit 2  SPREAD    — Te_best_std / Te_best ≥ cv_thresh   (inter-shot CV)
  bit 3  SPATIAL   — |Te_best − spatial local median| > k_sigma × MAD
  bit 4  TEMPORAL  — |Te_best − temporal local median| > k_sigma × MAD

Default thresholds
------------------
  --bad-frac     0.50   flag if ≥ 50 % of shots are bad
  --disagree     0.30   flag if log/exp Te differ by ≥ 30 % of Te_best
  --cv           0.30   flag if inter-shot std ≥ 30 % of Te_best
  --k-sigma      3.0    MAD multiplier for spatial and temporal outlier tests
  --spatial-win  5      half-window (positions) for local median
  --temporal-win 7      half-window (cycles) for local median

New datasets per run group
--------------------------
  review_flags      (51, n_cycles)  uint8   bitmask (bits 0–4 above)
  review_flag_any   (51, n_cycles)  bool    True if any bit is set

  Group attributes added:
    review_flag_bits    JSON string describing each bit
    review_thresholds   JSON string with the threshold values used

Usage
-----
  # Flag all runs in the default HDF5
  ./.venv/bin/python scripts/flag_te_review.py

  # Flag specific runs with custom thresholds
  ./.venv/bin/python scripts/flag_te_review.py --run-ids 32,46 --disagree 0.25

  # Re-flag (overwrites existing flag datasets)
  ./.venv/bin/python scripts/flag_te_review.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter

HDF5_PATH = Path("processed/langmuir_sweeps.hdf5")

FLAG_QUALITY  = np.uint8(0x01)
FLAG_DISAGREE = np.uint8(0x02)
FLAG_SPREAD   = np.uint8(0x04)
FLAG_SPATIAL  = np.uint8(0x08)
FLAG_TEMPORAL = np.uint8(0x10)

FLAG_BIT_DESCRIPTIONS = {
    "bit0_QUALITY":  "bad_frac >= bad_frac_thresh: majority of shots are bad",
    "bit1_DISAGREE": "|Te_log - Te_exp| / Te_best >= disagree_frac",
    "bit2_SPREAD":   "Te_best_std / Te_best >= cv_thresh (inter-shot CV)",
    "bit3_SPATIAL":  "|Te_best - local_spatial_median| > k_sigma * MAD",
    "bit4_TEMPORAL": "|Te_best - local_temporal_median| > k_sigma * MAD",
}


# ---------------------------------------------------------------------------
# Individual flag computations
# ---------------------------------------------------------------------------

def _flag_quality(
    n_ok: np.ndarray,
    n_warn: np.ndarray,
    n_bad: np.ndarray,
    *,
    bad_frac_thresh: float,
) -> np.ndarray:
    """bit 0: bad-shot fraction ≥ threshold."""
    n_total = (n_ok + n_warn + n_bad).astype(float)
    with np.errstate(invalid="ignore"):
        frac = np.where(n_total > 0, n_bad.astype(float) / n_total, 1.0)
    return frac >= bad_frac_thresh


def _flag_disagree(
    te_log: np.ndarray,
    te_exp: np.ndarray,
    te_best: np.ndarray,
    *,
    disagree_frac: float,
    te_min: float = 0.2,
) -> np.ndarray:
    """bit 1: relative log/exp disagreement ≥ threshold."""
    valid = (
        np.isfinite(te_log) & np.isfinite(te_exp)
        & np.isfinite(te_best) & (te_best > te_min)
    )
    with np.errstate(invalid="ignore"):
        rel = np.where(valid, np.abs(te_log - te_exp) / te_best, 0.0)
    return valid & (rel >= disagree_frac)


def _flag_spread(
    te_best: np.ndarray,
    te_best_std: np.ndarray,
    *,
    cv_thresh: float,
    te_min: float = 0.2,
) -> np.ndarray:
    """bit 2: inter-shot coefficient of variation ≥ threshold."""
    valid = np.isfinite(te_best) & np.isfinite(te_best_std) & (te_best > te_min)
    with np.errstate(invalid="ignore"):
        cv = np.where(valid, te_best_std / te_best, 0.0)
    return valid & (cv >= cv_thresh)


def _local_median_mad(
    te_best: np.ndarray,
    *,
    window: int,
    axis: int,
    te_min: float = 0.2,
) -> tuple[np.ndarray, float]:
    """Return (local_median, global_MAD_scale) for outlier detection.

    NaN cells are filled with the global nanmedian before filtering so they
    don't propagate into neighbours.  The returned local_median retains NaN
    at those positions.
    """
    finite = np.isfinite(te_best) & (te_best > te_min)
    global_med = float(np.nanmedian(te_best[finite])) if finite.any() else 0.0

    filled = np.where(finite, te_best, global_med)
    size   = (window, 1) if axis == 0 else (1, window)
    local_med = median_filter(filled, size=size, mode="nearest")
    # Restore NaN at originally non-finite positions
    local_med = np.where(finite, local_med, np.nan)

    residuals = np.abs(te_best - local_med)
    mad       = float(np.nanmedian(residuals[finite])) if finite.any() else 0.0
    # Scale floor: 5 % of global median to avoid zero-scale when plasma is uniform
    scale = max(1.4826 * mad, 0.05 * global_med) if global_med > 0 else max(1.4826 * mad, 1e-3)
    return local_med, scale


def _flag_spatial(
    te_best: np.ndarray,
    *,
    window: int,
    k_sigma: float,
) -> np.ndarray:
    """bit 3: spatial outlier along the position axis (axis 0)."""
    finite = np.isfinite(te_best) & (te_best > 0.2)
    local_med, scale = _local_median_mad(te_best, window=window, axis=0)
    residuals = np.abs(te_best - local_med)
    return finite & (residuals > k_sigma * scale)


def _flag_temporal(
    te_best: np.ndarray,
    *,
    window: int,
    k_sigma: float,
) -> np.ndarray:
    """bit 4: temporal outlier along the cycle axis (axis 1)."""
    finite = np.isfinite(te_best) & (te_best > 0.2)
    local_med, scale = _local_median_mad(te_best, window=window, axis=1)
    residuals = np.abs(te_best - local_med)
    return finite & (residuals > k_sigma * scale)


# ---------------------------------------------------------------------------
# Per-run flagging
# ---------------------------------------------------------------------------

def flag_run_group(
    grp: h5py.Group,
    *,
    bad_frac_thresh: float,
    disagree_frac: float,
    cv_thresh: float,
    k_sigma: float,
    spatial_win: int,
    temporal_win: int,
    overwrite: bool,
) -> dict[str, int]:
    """Compute and write review_flags / review_flag_any into *grp*.

    Returns a dict of flag counts: {name: n_flagged_cells}.
    """
    run_id = grp.attrs.get("run_id", grp.name.split("/")[-1])

    # Skip if flags already written and not overwriting
    if "review_flags" in grp:
        if not overwrite:
            sys.stdout.write(f"  run {run_id}: flags already present, skipping "
                             f"(use --overwrite to recompute)\n")
            return {}
        del grp["review_flags"]
        del grp["review_flag_any"]

    # Load needed arrays — all (51, n_cycles)
    te_log   = grp["te_log_ev"][:]
    te_exp   = grp["te_exp_ev"][:]
    te_best  = grp["te_best_ev"][:]
    te_std   = grp["te_best_ev_std"][:]
    n_ok     = grp["n_ok"][:].astype(int)
    n_warn   = grp["n_warn"][:].astype(int)
    n_bad    = grp["n_bad"][:].astype(int)

    f_quality  = _flag_quality(n_ok, n_warn, n_bad, bad_frac_thresh=bad_frac_thresh)
    f_disagree = _flag_disagree(te_log, te_exp, te_best, disagree_frac=disagree_frac)
    f_spread   = _flag_spread(te_best, te_std, cv_thresh=cv_thresh)
    f_spatial  = _flag_spatial(te_best, window=spatial_win, k_sigma=k_sigma)
    f_temporal = _flag_temporal(te_best, window=temporal_win, k_sigma=k_sigma)

    flags = (
          f_quality.astype(np.uint8)  * FLAG_QUALITY
        | f_disagree.astype(np.uint8) * FLAG_DISAGREE
        | f_spread.astype(np.uint8)   * FLAG_SPREAD
        | f_spatial.astype(np.uint8)  * FLAG_SPATIAL
        | f_temporal.astype(np.uint8) * FLAG_TEMPORAL
    )
    flag_any = flags != 0

    ds = grp.create_dataset("review_flags", data=flags, compression="gzip", compression_opts=4)
    ds.attrs["description"] = (
        "uint8 bitmask: bit0=QUALITY, bit1=DISAGREE, bit2=SPREAD, "
        "bit3=SPATIAL, bit4=TEMPORAL"
    )
    ds2 = grp.create_dataset("review_flag_any", data=flag_any, compression="gzip", compression_opts=4)
    ds2.attrs["description"] = "True if any review flag is set for this (pos, cycle) cell"

    thresholds = dict(
        bad_frac_thresh=bad_frac_thresh,
        disagree_frac=disagree_frac,
        cv_thresh=cv_thresh,
        k_sigma=k_sigma,
        spatial_win=spatial_win,
        temporal_win=temporal_win,
    )
    grp.attrs["review_flag_bits"]       = json.dumps(FLAG_BIT_DESCRIPTIONS)
    grp.attrs["review_thresholds"]      = json.dumps(thresholds)

    counts = {
        "QUALITY":  int(f_quality.sum()),
        "DISAGREE": int(f_disagree.sum()),
        "SPREAD":   int(f_spread.sum()),
        "SPATIAL":  int(f_spatial.sum()),
        "TEMPORAL": int(f_temporal.sum()),
        "ANY":      int(flag_any.sum()),
    }
    n_cells = flags.size
    pct = 100 * counts["ANY"] / n_cells if n_cells else 0
    sys.stdout.write(
        f"  run {run_id}: {counts['ANY']}/{n_cells} cells flagged ({pct:.1f}%)  "
        + "  ".join(f"{k}={v}" for k, v in counts.items() if k != "ANY")
        + "\n"
    )
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hdf5", type=Path, default=HDF5_PATH,
        help=f"Path to langmuir_sweeps.hdf5.  Default: {HDF5_PATH}",
    )
    parser.add_argument(
        "--run-ids", default="all",
        help="Comma-separated run IDs (e.g. 32,46) or 'all'.",
    )
    parser.add_argument(
        "--bad-frac", type=float, default=0.50, metavar="F",
        help="QUALITY flag: flag if bad-shot fraction >= F.  Default: 0.50",
    )
    parser.add_argument(
        "--disagree", type=float, default=0.30, metavar="F",
        help="DISAGREE flag: |Te_log-Te_exp|/Te_best >= F.  Default: 0.30",
    )
    parser.add_argument(
        "--cv", type=float, default=0.30, metavar="F",
        help="SPREAD flag: Te_best_std/Te_best >= F.  Default: 0.30",
    )
    parser.add_argument(
        "--k-sigma", type=float, default=3.0, metavar="K",
        help="SPATIAL/TEMPORAL flags: MAD multiplier.  Default: 3.0",
    )
    parser.add_argument(
        "--spatial-win", type=int, default=5, metavar="N",
        help="SPATIAL flag: median filter window (positions).  Default: 5",
    )
    parser.add_argument(
        "--temporal-win", type=int, default=7, metavar="N",
        help="TEMPORAL flag: median filter window (cycles).  Default: 7",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute and overwrite flags even if already present.",
    )
    args = parser.parse_args()

    if not args.hdf5.exists():
        sys.exit(f"HDF5 file not found: {args.hdf5}")

    kwargs = dict(
        bad_frac_thresh=args.bad_frac,
        disagree_frac=args.disagree,
        cv_thresh=args.cv,
        k_sigma=args.k_sigma,
        spatial_win=args.spatial_win,
        temporal_win=args.temporal_win,
        overwrite=args.overwrite,
    )

    target_ids: set[str] | None = (
        None if args.run_ids == "all"
        else {v.strip() for v in args.run_ids.split(",") if v.strip()}
    )

    totals: dict[str, int] = {}
    n_runs = 0

    with h5py.File(args.hdf5, "a") as hf:
        es_grp = hf.get("experiment_sets", {})
        for es_id in sorted(es_grp.keys()):
            for run_id in sorted(es_grp[es_id].keys()):
                if target_ids is not None and run_id not in target_ids:
                    continue
                sys.stdout.write(f"run {run_id}  (es={es_id})\n")
                counts = flag_run_group(es_grp[es_id][run_id], **kwargs)
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
                if counts:
                    n_runs += 1

    if n_runs:
        sys.stdout.write(
            f"\nFlagged {n_runs} run(s).  Grand totals: "
            + "  ".join(f"{k}={v}" for k, v in totals.items())
            + "\n"
        )
    else:
        sys.stdout.write("No runs flagged (all may already have flags; use --overwrite).\n")


if __name__ == "__main__":
    main()
