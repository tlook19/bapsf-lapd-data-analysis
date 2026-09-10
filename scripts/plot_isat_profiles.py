"""Process and plot dead-time Isat contour maps for rot-0 runs.

For each experiment set, this script uses only the inter-sweep dead-time
windows of one current channel, averages over shots and dead-time samples,
applies one probe-A area factor to runs ending in 1/8, and plots current vs
(x, z) for each discharge cycle.

The default probe-A factor is calibrated from experiment set 1 using
FWHM-core-averaged density over the stable 10--19 ms plateau.  One-sided
high-current shot outliers are removed before averaging.  The factor is chosen
so p11 density is no greater than p21 while p50 density is no less than p41,
consistent with the simulated downstream density increase.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_isat_profiles.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_isat_profiles.py --no-animation
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_isat_profiles.py --use-existing
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd import (
    ChannelKind,
    LapdDataset,
    LapdRun,
    density_area_key_for_deadtime_source,
    effective_deadtime_source,
    effective_rotation_deg,
)
from bapsf_lapd.density import (
    electron_density_m3,
    inter_sweep_sample_slices,
    ion_sound_speed_m_s,
    load_probe_a_area_calibration,
)


MANIFEST = Path("config/may2026_run_manifest.toml")
HDF5_OUTPUT = Path("processed/isat_profiles.hdf5")
OUTPUT_DIR = Path("figures")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
AREA_CALIB_TOML = Path("processed/probe_area_calibration.toml")
PROBE_A_CALIB_TOML = Path("config/may2026_probe_a_area_calibration.toml")

CLIP_S = 10e-6
X_CM = np.linspace(-25.0, 25.0, 51)
CMAP = "viridis"
SUBPLOT_NCOLS = 8
SUBPLOT_NROWS = 5
DEFAULT_HIGH_SHOT_SIGMA = 3.0
DEFAULT_HIGH_SHOT_RATIO = 1.5
DEFAULT_MIN_SHOTS_USED = 10
DEFAULT_CALIB_T_MIN_MS = 10.0
DEFAULT_CALIB_T_MAX_MS = 19.0
M_I_AMU = 4.003

PROBE_FROM_DIGIT = {1: "A", 8: "A", 2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}


def _pcolormesh_edges(centers: np.ndarray) -> np.ndarray:
    mids = (centers[:-1] + centers[1:]) / 2
    lo = centers[0] - (centers[1] - centers[0]) / 2
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[lo], mids, [hi]])


def _inter_sweep_times_s(run: LapdRun) -> np.ndarray:
    sw = run.config.sweep
    return np.array([
        sw.t0_s + k * sw.tau_cycle_s + 0.5 * (sw.tau_ramp_s + sw.tau_cycle_s)
        for k in range(sw.n_cycles)
    ])


def _middle_cycle_mask(n_cycles: int, middle_fraction: float) -> np.ndarray:
    if not 0.0 < middle_fraction <= 1.0:
        raise ValueError("middle_fraction must be in (0, 1]")
    edge = (1.0 - middle_fraction) / 2.0
    lo = int(np.floor(edge * n_cycles))
    hi = int(np.ceil((1.0 - edge) * n_cycles))
    mask = np.zeros(n_cycles, dtype=bool)
    mask[lo:hi] = True
    return mask


def _fwhm_bounds_cm(profile: np.ndarray, x_cm: np.ndarray) -> tuple[float, float]:
    """Return exterior half-max bounds, allowing hollow/two-peaked profiles.

    If multiple significant lobes are present, the bounds are the left exterior
    half-height crossing of the left lobe and the right exterior half-height
    crossing of the right lobe.  This keeps a central depletion from splitting
    the core, and keeps an asymmetric tall lobe from hiding a lower companion
    lobe that is still physically part of the plasma cross-section.
    """
    y = np.asarray(profile, dtype=np.float64)
    x = np.asarray(x_cm, dtype=np.float64)
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan

    peak = float(np.nanmax(y))
    if not np.isfinite(peak) or peak <= 0:
        return np.nan, np.nan
    peak_indices = []
    min_peak = 0.25 * peak
    for i in range(1, len(y) - 1):
        if not valid[i] or y[i] < min_peak:
            continue
        if y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            peak_indices.append(i)

    if len(peak_indices) >= 2:
        left_peak = int(peak_indices[0])
        right_peak = int(peak_indices[-1])
        left_level = y[left_peak] / 2.0
        right_level = y[right_peak] / 2.0
    else:
        left_level = peak / 2.0
        right_level = peak / 2.0
        above = np.flatnonzero(valid & (y >= left_level))
        if above.size == 0:
            return np.nan, np.nan
        left_peak = int(above[0])
        right_peak = int(above[-1])

    def interp(i_below: int, i_above: int, level: float) -> float:
        y0 = y[i_below]
        y1 = y[i_above]
        if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
            return float(x[i_above])
        frac = (level - y0) / (y1 - y0)
        return float(x[i_below] + frac * (x[i_above] - x[i_below]))

    left = float(x[left_peak])
    for i in range(left_peak - 1, -1, -1):
        if not valid[i]:
            continue
        if y[i] <= left_level:
            left = interp(i, left_peak, left_level)
            break
        left_peak = i

    right = float(x[right_peak])
    for i in range(right_peak + 1, len(y)):
        if not valid[i]:
            continue
        if y[i] <= right_level:
            right = interp(i, right_peak, right_level)
            break
        right_peak = i

    if right < left:
        return np.nan, np.nan
    return left, right


def _core_mean_isat(isat_a: np.ndarray, middle_mask: np.ndarray) -> tuple[float, float, float]:
    mid_profile = np.nanmedian(isat_a[:, middle_mask], axis=1)
    left, right = _fwhm_bounds_cm(mid_profile, X_CM)
    if not np.isfinite(left) or not np.isfinite(right):
        core_mask = np.isfinite(mid_profile)
    else:
        core_mask = (X_CM >= left) & (X_CM <= right)
    mean = float(np.nanmean(isat_a[core_mask][:, middle_mask]))
    return mean, left, right


def _probe_id(run_id: str) -> str:
    return PROBE_FROM_DIGIT[int(run_id[1])]


def _load_probe_areas_m2(path: Path) -> dict[str, dict[str, float]]:
    with open(path, "rb") as stream:
        data = tomllib.load(stream)
    return {
        probe: {
            "ap_L_cm2": float(data[f"probe_{probe}"]["ap_L_cm2"]) * 1e-4,
            "ap_R_cm2": float(data[f"probe_{probe}"]["ap_R_cm2"]) * 1e-4,
        }
        for probe in ("A", "B", "C", "D")
    }


def _interp_te_to_profile_times(
    te_x_cycle: np.ndarray,
    te_time_ms: np.ndarray,
    profile_time_ms: np.ndarray,
) -> np.ndarray:
    return np.vstack([
        np.interp(profile_time_ms, te_time_ms, te_x_cycle[x_idx, :])
        for x_idx in range(te_x_cycle.shape[0])
    ])


def _calibrate_probe_a_density_factor(
    core_density_raw_m3: dict[str, float],
) -> tuple[float, dict[str, float | str]]:
    """Bracket one probe-A factor using the simulated axial density trend.

    Probe-A densities are unscaled.  Interior-probe densities already use
    their interferometer-calibrated areas.  The constraints are
    ``f * n11 <= n21`` and ``f * n50 >= n41``.
    """
    required = {"01", "02", "06", "08"}
    missing = sorted(required - set(core_density_raw_m3))
    if missing:
        raise ValueError(
            "Cannot calibrate probe-A density factor; missing runs: "
            + ", ".join(missing)
        )

    n11 = float(core_density_raw_m3["01"])
    n21 = float(core_density_raw_m3["02"])
    n41 = float(core_density_raw_m3["06"])
    n50 = float(core_density_raw_m3["08"])
    values = np.array([n11, n21, n41, n50], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Cannot calibrate probe-A factor from non-positive core densities")

    lower = n41 / n50
    upper = n21 / n11
    factor = float(np.sqrt(lower * upper))
    exact_bracket = float(lower <= upper)
    return factor, {
        "density_constraint": "f*n_p11 <= n_p21 and f*n_p50 >= n_p41",
        "density_metric": "FWHM-core mean electron density",
        "run01_core_density_raw_m3": n11,
        "run02_core_density_m3": n21,
        "run06_core_density_m3": n41,
        "run08_core_density_raw_m3": n50,
        "factor_lower_bound": float(lower),
        "factor_upper_bound": float(upper),
        "exact_bracket": exact_bracket,
    }


def _fixed_probe_a_factor(path: Path) -> tuple[float, dict[str, float | str]]:
    """Load the analysis-wide Probe A factor and its bracket metadata."""
    calibration = load_probe_a_area_calibration(path)
    return calibration.factor, {
        "factor_lower_bound": calibration.lower_bound,
        "factor_upper_bound": calibration.upper_bound,
        "exact_bracket": 1.0,
        "factor_source": str(path),
        "factor_policy": "canonical Probe A factor applied to both faces and all sets",
    }


def _density_bracket_from_profiles(
    profile_hdf: h5py.File,
    te_path: Path,
    area_path: Path,
    *,
    time_min_ms: float,
    time_max_ms: float,
) -> tuple[float, dict[str, float | str]]:
    """Compute the ES1 Probe-A factor from outlier-filtered raw profiles."""
    if not time_min_ms < time_max_ms:
        raise ValueError("calibration time_min_ms must be less than time_max_ms")

    areas_m2 = _load_probe_areas_m2(area_path)
    core_density: dict[str, float] = {}
    core_bounds: dict[str, tuple[float, float]] = {}

    with h5py.File(te_path, "r") as te_hdf:
        te_set = te_hdf["experiment_sets/1"]
        te_x_cm = te_set["x_cm"][()]
        if not np.allclose(te_x_cm, profile_hdf["x_cm"][()]):
            raise ValueError("Profile and filled-Te x grids do not match")
        te_z_cm = te_set["z_cm"][()]
        te_time_ms = te_set["cycle_time_ms"][()]
        te_filled = te_set["te_filled"][()]

        for run_id in ("01", "02", "06", "08"):
            grp = profile_hdf[f"experiment_sets/1/{run_id}"]
            profile_time_ms = grp["inter_sweep_time_s"][()] * 1000.0
            time_mask = (
                (profile_time_ms >= time_min_ms)
                & (profile_time_ms <= time_max_ms)
            )
            if not np.any(time_mask):
                raise ValueError(
                    f"Run {run_id} has no profile cycles in "
                    f"{time_min_ms:g}--{time_max_ms:g} ms"
                )

            z_idx = int(np.argmin(np.abs(te_z_cm - float(grp.attrs["z_cm"]))))
            te_interp = _interp_te_to_profile_times(
                te_filled[z_idx, :, :],
                te_time_ms,
                profile_time_ms,
            )
            cs = ion_sound_speed_m_s(te_interp, M_I_AMU)
            probe = _probe_id(run_id)
            area_key = str(grp.attrs.get("density_area_key", "ap_L_cm2"))
            density = electron_density_m3(
                grp["isat_a_raw"][()],
                areas_m2[probe][area_key],
                cs,
            )
            density = np.where((density > 0) & np.isfinite(density), density, np.nan)
            mean_density, left, right = _core_mean_isat(density, time_mask)
            core_density[run_id] = mean_density
            core_bounds[run_id] = (left, right)

    factor, metadata = _calibrate_probe_a_density_factor(core_density)
    metadata.update({
        "time_min_ms": float(time_min_ms),
        "time_max_ms": float(time_max_ms),
        "te_hdf5": str(te_path),
        "area_calibration_toml": str(area_path),
        "high_current_outliers_excluded": 1.0,
    })
    for run_id, (left, right) in core_bounds.items():
        metadata[f"run{run_id}_core_fwhm_left_cm"] = float(left)
        metadata[f"run{run_id}_core_fwhm_right_cm"] = float(right)
    return factor, metadata


def _calibrate_probe_a_factor(
    set1_scalars: dict[str, float],
) -> tuple[float, dict[str, float]]:
    required = {"01", "02", "04", "06", "08"}
    missing = sorted(required - set(set1_scalars))
    if missing:
        raise ValueError(f"Cannot calibrate probe-A factor; missing runs: {', '.join(missing)}")

    i01 = set1_scalars["01"]
    i08 = set1_scalars["08"]
    interior = np.array([set1_scalars[r] for r in ("02", "04", "06")], dtype=np.float64)
    if i01 <= 0 or i08 <= 0 or not np.all(interior > 0):
        raise ValueError("Cannot calibrate probe-A factor from non-positive Isat means")

    # Require f * I01 >= all interior ports and f * I08 <= all interior ports.
    lower = float(np.nanmax(interior / i01))
    upper = float(np.nanmin(interior / i08))
    if lower <= upper:
        factor = float(np.sqrt(lower * upper))
        exact_bracket = 1.0
    else:
        # No exact bracket exists; choose the least-bad factor in log space.
        factor = float(np.sqrt(lower * upper))
        exact_bracket = 0.0

    return factor, {
        "run01_core_mean_a": float(i01),
        "run08_core_mean_a": float(i08),
        "interior_min_core_mean_a": float(np.nanmin(interior)),
        "interior_max_core_mean_a": float(np.nanmax(interior)),
        "factor_lower_bound": lower,
        "factor_upper_bound": upper,
        "exact_bracket": exact_bracket,
    }


def _calibrate_paired_factor(
    set1_scalars: dict[str, float],
    reference_path: Path,
) -> tuple[float, dict[str, float]]:
    """Calibrate rot-180 backside factor against paired rot-0 frontside runs."""
    pairs = {"03": "02", "05": "04", "07": "06"}
    missing = sorted(set(pairs) - set(set1_scalars))
    if missing:
        raise ValueError(f"Cannot calibrate paired factor; missing runs: {', '.join(missing)}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing paired calibration reference: {reference_path}")

    source = []
    target = []
    with h5py.File(reference_path, "r") as ref_h5:
        for source_run, target_run in pairs.items():
            source.append(float(set1_scalars[source_run]))
            target.append(float(ref_h5[f"experiment_sets/1/{target_run}"].attrs["calibration_core_mean_raw_a"]))

    source_arr = np.array(source, dtype=np.float64)
    target_arr = np.array(target, dtype=np.float64)
    if np.any(source_arr <= 0) or np.any(target_arr <= 0):
        raise ValueError("Cannot calibrate paired factor from non-positive core means")

    ratios = target_arr / source_arr
    factor = float(np.exp(np.nanmean(np.log(ratios))))
    return factor, {
        "paired_source_runs": ",".join(pairs.keys()),
        "paired_target_runs": ",".join(pairs.values()),
        "paired_reference_hdf5": str(reference_path),
        "paired_ratio_geomean": factor,
        "paired_ratio_min": float(np.nanmin(ratios)),
        "paired_ratio_max": float(np.nanmax(ratios)),
        "paired_ratio_run03_to_run02": float(ratios[0]),
        "paired_ratio_run05_to_run04": float(ratios[1]),
        "paired_ratio_run07_to_run06": float(ratios[2]),
        "exact_bracket": 1.0,
    }


def _high_shot_outlier_mask(
    shot_means: np.ndarray,
    *,
    sigma: float,
    ratio: float,
    min_shots_used: int,
) -> np.ndarray:
    """Mask one-sided high-current shot outliers per position/cycle."""
    median = np.nanmedian(shot_means, axis=1, keepdims=True)
    abs_dev = np.abs(shot_means - median)
    mad = np.nanmedian(abs_dev, axis=1, keepdims=True)
    robust_sigma = 1.4826 * mad
    threshold_sigma = median + sigma * robust_sigma
    threshold_ratio = median * ratio
    threshold = np.where(median > 0, np.minimum(threshold_sigma, threshold_ratio), threshold_sigma)
    mask = np.isfinite(shot_means) & (shot_means > threshold)

    n_pos, n_shots, n_cycles = shot_means.shape
    min_keep = max(1, min(int(min_shots_used), n_shots))
    for pos_idx in range(n_pos):
        for cycle_idx in range(n_cycles):
            values = shot_means[pos_idx, :, cycle_idx]
            finite = np.isfinite(values)
            if finite.sum() <= min_keep:
                mask[pos_idx, :, cycle_idx] = False
                continue
            if np.sum(finite & ~mask[pos_idx, :, cycle_idx]) >= min_keep:
                continue
            keep_candidates = np.argsort(np.where(finite, values, np.inf))[:min_keep]
            mask[pos_idx, keep_candidates, cycle_idx] = False
    return mask


def _deadtime_shot_means(
    run: LapdRun,
    *,
    channel_kind: ChannelKind,
    invert_polarity: bool,
) -> np.ndarray:
    sw = run.config.sweep
    n_pos = run.config.acquisition.n_positions
    n_cycles = sw.n_cycles
    n_shots = run.config.acquisition.n_shots_per_position
    dead_slices = inter_sweep_sample_slices(sw, run.config.acquisition, clip_s=CLIP_S)
    zero_offset = run.default_zero_offset_v(channel_kind)
    channel = run.config.channel(channel_kind)

    # HDF5 chunks are full traces per flattened shot.  Reading one row once and
    # averaging every dead-time window from that row avoids decompressing the
    # same shot once per cycle.
    shot_means = np.full((n_pos, n_shots, n_cycles), np.nan)
    with run.open() as h5:
        data = h5[channel.hdf5_path]
        headers = h5[f"{channel.hdf5_path} headers"][:]
        for flat_shot in range(run.expected_flat_shot_count()):
            pos_idx = flat_shot // n_shots
            shot_idx = flat_shot % n_shots
            raw = data[flat_shot, :]
            scale = float(headers["Scale"][flat_shot])
            offset = float(headers["Offset"][flat_shot])
            for k, dead_slice in enumerate(dead_slices):
                mean_voltage = float(raw[dead_slice].mean()) * scale + offset - zero_offset
                value = channel.apply_calibration(mean_voltage)
                shot_means[pos_idx, shot_idx, k] = -value if invert_polarity else value
    return shot_means


def process_run(
    run: LapdRun,
    *,
    channel_kind: ChannelKind,
    invert_polarity: bool,
    reject_high_shots: bool,
    rejection_channels: str,
    high_shot_sigma: float,
    high_shot_ratio: float,
    min_shots_used: int,
) -> dict[str, np.ndarray]:
    n_shots = run.config.acquisition.n_shots_per_position
    shot_means = _deadtime_shot_means(
        run,
        channel_kind=channel_kind,
        invert_polarity=invert_polarity,
    )

    if reject_high_shots:
        masks = []
        if rejection_channels in {"source", "both"} or channel_kind == ChannelKind.I_SWEEP:
            masks.append(
                _high_shot_outlier_mask(
                    shot_means,
                    sigma=high_shot_sigma,
                    ratio=high_shot_ratio,
                    min_shots_used=min_shots_used,
                )
            )
        detector_specs = []
        if rejection_channels == "both":
            detector_specs = [(ChannelKind.I_SWEEP, True), (ChannelKind.ISAT, False)]
        elif rejection_channels == "i_sweep":
            detector_specs = [(ChannelKind.I_SWEEP, True)]
        for other_kind, other_invert in detector_specs:
            if other_kind == channel_kind:
                continue
            other_means = _deadtime_shot_means(
                run,
                channel_kind=other_kind,
                invert_polarity=other_invert,
            )
            masks.append(
                _high_shot_outlier_mask(
                    other_means,
                    sigma=high_shot_sigma,
                    ratio=high_shot_ratio,
                    min_shots_used=min_shots_used,
                )
            )
        if not masks:
            high_outlier = np.zeros_like(shot_means, dtype=bool)
        else:
            high_outlier = np.logical_or.reduce(masks)

        # Re-apply the minimum-retained-shots guard after combining masks from
        # both faces.  Arc-like shots are shared, but the reduced mean should
        # never be based on only a tiny tail of the shot ensemble.
        n_pos, _, n_cycles = high_outlier.shape
        min_keep = max(1, min(int(min_shots_used), n_shots))
        for pos_idx in range(n_pos):
            for cycle_idx in range(n_cycles):
                values = shot_means[pos_idx, :, cycle_idx]
                finite = np.isfinite(values)
                if finite.sum() <= min_keep:
                    high_outlier[pos_idx, :, cycle_idx] = False
                    continue
                if np.sum(finite & ~high_outlier[pos_idx, :, cycle_idx]) >= min_keep:
                    continue
                keep = np.argsort(np.where(finite, values, np.inf))[:min_keep]
                high_outlier[pos_idx, keep, cycle_idx] = False
    else:
        high_outlier = np.zeros_like(shot_means, dtype=bool)
    filtered = np.where(high_outlier, np.nan, shot_means)

    isat = np.nanmean(filtered, axis=1)
    isat_std = np.nanstd(filtered, axis=1, ddof=1) if n_shots > 1 else np.zeros_like(isat)
    n_rejected = np.sum(high_outlier, axis=1).astype(np.int16)
    n_used = np.sum(np.isfinite(filtered), axis=1).astype(np.int16)

    return {
        "inter_sweep_time_s": _inter_sweep_times_s(run),
        "isat_a_raw": isat,
        "isat_a_raw_std": isat_std,
        "n_shots_used": n_used,
        "n_high_shots_rejected": n_rejected,
    }


def build_hdf5(
    output_path: Path,
    *,
    middle_fraction: float,
    channel_kind: ChannelKind,
    invert_polarity: bool,
    rotation_deg: float,
    reject_high_shots: bool,
    rejection_channels: str,
    calibration_mode: str,
    factor_attr_name: str,
    front_reference: Path,
    te_path: Path,
    area_calibration_path: Path,
    probe_a_calibration_path: Path,
    calibration_time_min_ms: float,
    calibration_time_max_ms: float,
    high_shot_sigma: float,
    high_shot_ratio: float,
    min_shots_used: int,
) -> float:
    dataset = LapdDataset.from_manifest(MANIFEST)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    set1_scalars: dict[str, float] = {}

    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("x_cm", data=X_CM)
        hf.attrs["area_factor_attr_name"] = factor_attr_name
        hf.attrs["probe_a_factor_applies_to_run_suffixes"] = "1,8"
        hf.attrs["calibration_experiment_set"] = "1"
        hf.attrs["calibration_mode"] = calibration_mode
        hf.attrs["calibration_middle_fraction"] = middle_fraction
        hf.attrs["rotation_filter_deg"] = float(rotation_deg)
        hf.attrs["clip_s"] = CLIP_S
        hf.attrs["deadtime_source_channel"] = channel_kind.value
        hf.attrs["deadtime_source_invert_polarity"] = bool(invert_polarity)
        hf.attrs["high_shot_rejection_enabled"] = bool(reject_high_shots)
        hf.attrs["high_shot_rejection_channels"] = rejection_channels
        hf.attrs["high_shot_rejection_sigma"] = high_shot_sigma
        hf.attrs["high_shot_rejection_ratio"] = high_shot_ratio
        hf.attrs["high_shot_rejection_min_shots_used"] = min_shots_used

        if calibration_mode == "probe_a_density_bracket" and not reject_high_shots:
            raise ValueError(
                "probe_a_density_bracket requires high-current shot rejection; "
                "remove --no-shot-rejection"
            )

        es_grp = hf.create_group("experiment_sets")
        for set_id in dataset.experiment_set_ids():
            exp = dataset.config(dataset.experiment_set_run_ids(set_id)[0]).experiment_set
            g_set = es_grp.create_group(str(set_id))
            g_set.attrs["label"] = exp.label
            g_set.attrs["v_bank_v"] = exp.v_bank
            g_set.attrs["v_puff_v"] = exp.v_puff

        for set_id in dataset.experiment_set_ids():
            for run_id in dataset.experiment_set_run_ids(set_id):
                run = dataset.run(run_id)
                cfg = run.config
                recorded_rot = float(cfg.probe.rotation_deg or 0)
                effective_rot = effective_rotation_deg(run_id, recorded_rot)
                if effective_rot != float(rotation_deg):
                    continue
                source_kind, source_invert, source_overridden = effective_deadtime_source(
                    run_id,
                    cfg.probe.port,
                    channel_kind,
                    invert_polarity,
                )

                rot_note = f" (recorded {recorded_rot:.0f})" if effective_rot != recorded_rot else ""
                source_note = (
                    f" source={source_kind.value}" if source_overridden else ""
                )
                print(
                    f"  run {run_id}  set={set_id}  rot={rotation_deg:.0f}{rot_note}"
                    f"  port={cfg.probe.port}  z={cfg.probe.z_cm:.1f} cm{source_note}",
                    flush=True,
                )
                result = process_run(
                    run,
                    channel_kind=source_kind,
                    invert_polarity=source_invert,
                    reject_high_shots=reject_high_shots,
                    rejection_channels=rejection_channels,
                    high_shot_sigma=high_shot_sigma,
                    high_shot_ratio=high_shot_ratio,
                    min_shots_used=min_shots_used,
                )
                mid_mask = _middle_cycle_mask(result["isat_a_raw"].shape[1], middle_fraction)
                core_mean, fwhm_left, fwhm_right = _core_mean_isat(result["isat_a_raw"], mid_mask)
                if str(set_id) == "1":
                    if source_overridden:
                        calibration_result = process_run(
                            run,
                            channel_kind=channel_kind,
                            invert_polarity=invert_polarity,
                            reject_high_shots=reject_high_shots,
                            rejection_channels=rejection_channels,
                            high_shot_sigma=high_shot_sigma,
                            high_shot_ratio=high_shot_ratio,
                            min_shots_used=min_shots_used,
                        )
                        calibration_mid = _middle_cycle_mask(
                            calibration_result["isat_a_raw"].shape[1],
                            middle_fraction,
                        )
                        calibration_core_mean, _, _ = _core_mean_isat(
                            calibration_result["isat_a_raw"],
                            calibration_mid,
                        )
                        set1_scalars[run_id] = calibration_core_mean
                    else:
                        set1_scalars[run_id] = core_mean

                g_run = hf[f"experiment_sets/{set_id}"].create_group(run_id)
                g_run.attrs["run_id"] = run_id
                g_run.attrs["rotation_deg"] = float(effective_rot)
                g_run.attrs["rotation_deg_recorded"] = recorded_rot
                g_run.attrs["rotation_correction_applied"] = bool(effective_rot != recorded_rot)
                g_run.attrs["port"] = int(cfg.probe.port or 0)
                g_run.attrs["z_cm"] = float(cfg.probe.z_cm or 0)
                g_run.attrs["requested_deadtime_source_channel"] = channel_kind.value
                g_run.attrs["requested_deadtime_source_invert_polarity"] = bool(invert_polarity)
                g_run.attrs["deadtime_source_channel"] = source_kind.value
                g_run.attrs["deadtime_source_invert_polarity"] = bool(source_invert)
                g_run.attrs["deadtime_source_overridden"] = bool(source_overridden)
                g_run.attrs["density_area_key"] = density_area_key_for_deadtime_source(
                    run_id,
                    source_kind,
                )
                g_run.attrs["calibration_core_mean_raw_a"] = core_mean
                g_run.attrs["calibration_fwhm_left_cm"] = fwhm_left
                g_run.attrs["calibration_fwhm_right_cm"] = fwhm_right
                g_run.create_dataset("inter_sweep_time_s", data=result["inter_sweep_time_s"])
                g_run.create_dataset("isat_a_raw", data=result["isat_a_raw"])
                g_run.create_dataset("isat_a_raw_std", data=result["isat_a_raw_std"])
                g_run.create_dataset("n_shots_used", data=result["n_shots_used"])
                g_run.create_dataset("n_high_shots_rejected", data=result["n_high_shots_rejected"])
                hf.flush()

                rejected_total = int(np.sum(result["n_high_shots_rejected"]))
                total_cells = int(np.prod(result["n_high_shots_rejected"].shape))
                max_rej = int(np.max(result["n_high_shots_rejected"]))
                print(
                    f"    middle-core mean Isat = {core_mean * 1e6:.2f} uA"
                    f"  FWHM core=({fwhm_left:.1f}, {fwhm_right:.1f}) cm",
                    flush=True,
                )
                print(
                    f"    rejected {rejected_total} high shots across {total_cells} x-cycle cells"
                    f"  (max {max_rej}/20 in one cell)",
                    flush=True,
                )

        if calibration_mode == "fixed_probe_a":
            factor, calib = _fixed_probe_a_factor(probe_a_calibration_path)
        elif calibration_mode == "probe_a_density_bracket":
            factor, calib = _density_bracket_from_profiles(
                hf,
                te_path,
                area_calibration_path,
                time_min_ms=calibration_time_min_ms,
                time_max_ms=calibration_time_max_ms,
            )
        elif calibration_mode == "probe_a_bracket":
            factor, calib = _calibrate_probe_a_factor(set1_scalars)
        elif calibration_mode == "paired_front":
            factor, calib = _calibrate_paired_factor(set1_scalars, front_reference)
        else:
            raise ValueError(f"Unsupported calibration_mode: {calibration_mode!r}")

        hf.attrs["probe_a_isat_area_factor"] = factor
        hf.attrs[factor_attr_name] = factor
        hf.attrs["calibration_time_min_ms"] = calibration_time_min_ms
        hf.attrs["calibration_time_max_ms"] = calibration_time_max_ms
        for key, value in calib.items():
            hf.attrs[f"calibration_{key}"] = value

        print(
            f"\n{factor_attr_name} from experiment set 1:"
            f" {factor:.4g}  (exact bracket: {bool(calib['exact_bracket'])})",
            flush=True,
        )
        if "factor_lower_bound" in calib:
            print(
                "  bracket interval:"
                f" [{calib['factor_lower_bound']:.4g}, {calib['factor_upper_bound']:.4g}]",
                flush=True,
            )
        if "paired_ratio_min" in calib:
            print(
                "  paired ratios:"
                f" min={calib['paired_ratio_min']:.4g}"
                f" max={calib['paired_ratio_max']:.4g}",
                flush=True,
            )

        for set_id in sorted(hf["experiment_sets"].keys(), key=int):
            for run_id in sorted(hf[f"experiment_sets/{set_id}"].keys()):
                g_run = hf[f"experiment_sets/{set_id}/{run_id}"]
                scale = factor if run_id[1] in {"1", "8"} else 1.0
                g_run.attrs["probe_a_area_factor_applied"] = scale
                g_run.create_dataset("isat_a", data=g_run["isat_a_raw"][:] * scale)
                g_run.create_dataset("isat_a_std", data=g_run["isat_a_raw_std"][:] * scale)
        hf.flush()

    print(f"\nWrote {output_path}", flush=True)
    return factor


def recalibrate_existing_hdf5(
    output_path: Path,
    *,
    te_path: Path,
    area_calibration_path: Path,
    probe_a_calibration_path: Path,
    calibration_mode: str,
    calibration_time_min_ms: float,
    calibration_time_max_ms: float,
) -> float:
    """Recalibrate stored outlier-filtered profiles without rereading raw runs."""
    with h5py.File(output_path, "r+") as hf:
        if not bool(hf.attrs.get("high_shot_rejection_enabled", False)):
            raise ValueError(
                f"{output_path} was built without high-current shot rejection"
            )
        if calibration_mode == "fixed_probe_a":
            factor, calib = _fixed_probe_a_factor(probe_a_calibration_path)
        elif calibration_mode == "probe_a_density_bracket":
            factor, calib = _density_bracket_from_profiles(
                hf,
                te_path,
                area_calibration_path,
                time_min_ms=calibration_time_min_ms,
                time_max_ms=calibration_time_max_ms,
            )
        else:
            raise ValueError(
                "Existing profiles may only be recalibrated with fixed_probe_a "
                "or probe_a_density_bracket"
            )

        for key in list(hf.attrs):
            if key.startswith("calibration_"):
                del hf.attrs[key]
        hf.attrs["calibration_experiment_set"] = "1"
        hf.attrs["calibration_mode"] = calibration_mode
        hf.attrs["calibration_time_min_ms"] = calibration_time_min_ms
        hf.attrs["calibration_time_max_ms"] = calibration_time_max_ms
        hf.attrs["probe_a_isat_area_factor"] = factor
        factor_attr_name = str(hf.attrs.get("area_factor_attr_name", "front_area_factor"))
        hf.attrs[factor_attr_name] = factor
        for key, value in calib.items():
            hf.attrs[f"calibration_{key}"] = value

        for set_id in sorted(hf["experiment_sets"].keys(), key=int):
            for run_id in sorted(hf[f"experiment_sets/{set_id}"].keys()):
                grp = hf[f"experiment_sets/{set_id}/{run_id}"]
                scale = factor if run_id[1] in {"1", "8"} else 1.0
                grp.attrs["probe_a_area_factor_applied"] = scale
                scaled = grp["isat_a_raw"][...] * scale
                scaled_std = grp["isat_a_raw_std"][...] * scale
                if "isat_a" in grp:
                    grp["isat_a"][...] = scaled
                else:
                    grp.create_dataset("isat_a", data=scaled)
                if "isat_a_std" in grp:
                    grp["isat_a_std"][...] = scaled_std
                else:
                    grp.create_dataset("isat_a_std", data=scaled_std)
        hf.flush()

    print(
        f"Recalibrated {output_path}: probe-A factor={factor:.6g} "
        f"using {calibration_mode}",
        flush=True,
    )
    return factor


def _load_experiment_set(hf: h5py.File, es_id: str) -> dict:
    x_cm = hf["x_cm"][:]
    eg = hf["experiment_sets"][es_id]
    es_label = eg.attrs.get("label", f"set {es_id}")
    v_bank = eg.attrs.get("v_bank_v", "?")
    rotation_filter = float(hf.attrs.get("rotation_filter_deg", 0.0))

    entries = []
    cycle_time_s = None
    for run_id in sorted(eg.keys()):
        rg = eg[run_id]
        if float(rg.attrs["rotation_deg"]) != rotation_filter:
            continue
        entries.append((float(rg.attrs["z_cm"]), run_id, rg["isat_a"][:]))
        if cycle_time_s is None:
            cycle_time_s = rg["inter_sweep_time_s"][:]

    if not entries:
        return {}

    entries.sort(key=lambda item: item[0])
    z_cm = np.array([item[0] for item in entries], dtype=np.float64)
    run_ids = [item[1] for item in entries]
    isat = np.stack([item[2] for item in entries], axis=0)

    return {
        "isat": isat,
        "x_cm": x_cm,
        "z_cm": z_cm,
        "run_ids": run_ids,
        "cycle_time_ms": cycle_time_s * 1e3,
        "es_label": es_label,
        "v_bank": v_bank,
        "es_id": es_id,
        "factor": float(hf.attrs["probe_a_isat_area_factor"]),
    }


def _make_frame(
    ax: plt.Axes,
    isat_2d_ua: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    norm,
    title: str,
) -> None:
    ax.clear()
    ax.pcolormesh(x_edges, z_edges, isat_2d_ua, cmap=CMAP, norm=norm, rasterized=True)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel("x (cm)", fontsize=6)
    ax.set_ylabel("z (cm)", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_subplots(data: dict, output_dir: Path, vmin: float, vmax: float) -> Path:
    import matplotlib.colors as mcolors

    isat_ua = data["isat"] * 1e6
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    times = data["cycle_time_ms"]
    n_cycles = isat_ua.shape[2]
    es_id = data["es_id"]
    es_label = data["es_label"]
    v_bank = data["v_bank"]
    factor = data["factor"]
    plot_prefix = data["plot_prefix"]
    channel_label = data["channel_label"]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        SUBPLOT_NROWS,
        SUBPLOT_NCOLS,
        figsize=(SUBPLOT_NCOLS * 2.4, SUBPLOT_NROWS * 2.2),
        constrained_layout=True,
    )
    ax_flat = axes.flatten()

    for ci in range(n_cycles):
        _make_frame(
            ax_flat[ci],
            isat_ua[:, :, ci],
            x_edges,
            z_edges,
            norm,
            f"t = {times[ci]:.1f} ms",
        )
    for ax in ax_flat[n_cycles:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label(f"dead-time {channel_label} (uA)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Dead-time {channel_label} vs (x, z) - experiment set {es_id}: {es_label}"
        f"  (V_bank = {v_bank} V, probe-A factor = {factor:.3g})",
        fontsize=11,
    )

    out_path = output_dir / f"{plot_prefix}_expset{es_id}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_animation(data: dict, output_dir: Path, vmin: float, vmax: float) -> Path:
    import matplotlib.colors as mcolors

    isat_ua = data["isat"] * 1e6
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    times = data["cycle_time_ms"]
    n_cycles = isat_ua.shape[2]
    es_id = data["es_id"]
    es_label = data["es_label"]
    v_bank = data["v_bank"]
    factor = data["factor"]
    plot_prefix = data["plot_prefix"]
    channel_label = data["channel_label"]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, z_edges, isat_ua[:, :, 0], cmap=CMAP, norm=norm, rasterized=True)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(f"dead-time {channel_label} (uA)", fontsize=10)
    ax.set_xlabel("x (cm)", fontsize=10)
    ax.set_ylabel("z (cm)", fontsize=10)
    title = ax.set_title(f"t = {times[0]:.1f} ms", fontsize=10)
    fig.suptitle(
        f"Dead-time {channel_label} - ES {es_id}: {es_label}"
        f"  (V_bank = {v_bank} V, A factor = {factor:.3g})",
        fontsize=10,
    )

    def update(ci: int):
        mesh.set_array(isat_ua[:, :, ci].ravel())
        title.set_text(f"t = {times[ci]:.1f} ms")
        return mesh, title

    anim = animation.FuncAnimation(fig, update, frames=n_cycles, interval=300, blit=False)
    out_path = output_dir / f"{plot_prefix}_expset{es_id}.gif"
    anim.save(str(out_path), writer="pillow", dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_all(
    hdf5_path: Path,
    output_dir: Path,
    save_animation: bool,
    *,
    plot_prefix: str,
    channel_label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as hf:
        for es_id in sorted(hf["experiment_sets"].keys(), key=int):
            data = _load_experiment_set(hf, es_id)
            if not data:
                print(f"ES {es_id}: no runs match rotation filter, skipping.")
                continue
            data["plot_prefix"] = plot_prefix
            data["channel_label"] = channel_label
            finite_ua = (data["isat"] * 1e6)[np.isfinite(data["isat"])]
            finite_ua = finite_ua[finite_ua > 0]
            if finite_ua.size == 0:
                print(f"ES {es_id}: no finite positive Isat data, skipping.")
                continue
            vmin = max(0.0, float(np.percentile(finite_ua, 2)))
            vmax = float(np.percentile(finite_ua, 98))
            plot_subplots(data, output_dir, vmin, vmax)
            if save_animation:
                plot_animation(data, output_dir, vmin, vmax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=HDF5_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--plot-prefix",
        default="isat_contours",
        help="filename prefix for generated contour PNG/GIF outputs",
    )
    parser.add_argument(
        "--source-channel",
        choices=["isat", "i_sweep"],
        default="i_sweep",
        help="dead-time current channel to process; i_sweep is polarity-inverted",
    )
    parser.add_argument(
        "--rotation",
        choices=["0", "180"],
        default="0",
        help="probe rotation to include in the contour product",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["fixed_probe_a", "probe_a_density_bracket", "probe_a_bracket", "paired_front"],
        default="fixed_probe_a",
        help="Probe A factor policy; fixed_probe_a is the analysis-wide default",
    )
    parser.add_argument(
        "--factor-attr-name",
        default="front_area_factor",
        help="HDF5 attribute name used for the calibrated area factor",
    )
    parser.add_argument(
        "--front-reference",
        type=Path,
        default=Path("processed/isweep_deadtime_profiles.hdf5"),
        help="rot-0 frontside HDF5 used by paired_front calibration",
    )
    parser.add_argument("--te-filled", type=Path, default=TE_FILLED_HDF5)
    parser.add_argument("--area-calibration", type=Path, default=AREA_CALIB_TOML)
    parser.add_argument(
        "--probe-a-calibration",
        type=Path,
        default=PROBE_A_CALIB_TOML,
        help="canonical Probe A factor applied by fixed_probe_a mode",
    )
    parser.add_argument(
        "--calibration-time-min-ms",
        type=float,
        default=DEFAULT_CALIB_T_MIN_MS,
        help="start of density-bracket calibration plateau",
    )
    parser.add_argument(
        "--calibration-time-max-ms",
        type=float,
        default=DEFAULT_CALIB_T_MAX_MS,
        help="end of density-bracket calibration plateau",
    )
    parser.add_argument("--use-existing", action="store_true", help="plot from existing processed HDF5")
    parser.add_argument(
        "--recalibrate-existing",
        action="store_true",
        help="recompute the density bracket from stored outlier-filtered raw profiles, then plot",
    )
    parser.add_argument("--no-animation", action="store_true", help="skip saving animated GIFs")
    parser.add_argument(
        "--no-shot-rejection",
        action="store_true",
        help="disable one-sided high-shot rejection",
    )
    parser.add_argument(
        "--rejection-channels",
        choices=["source", "both", "i_sweep"],
        default="source",
        help="detect high-shot outliers on the source channel only or on both probe faces",
    )
    parser.add_argument(
        "--middle-fraction",
        type=float,
        default=0.5,
        help="central fraction used by the legacy current-bracket calibration",
    )
    parser.add_argument(
        "--high-shot-sigma",
        type=float,
        default=DEFAULT_HIGH_SHOT_SIGMA,
        help="reject Isat shots above median + this many robust sigmas per x/cycle",
    )
    parser.add_argument(
        "--high-shot-ratio",
        type=float,
        default=DEFAULT_HIGH_SHOT_RATIO,
        help="also reject Isat shots above this multiple of the median per x/cycle",
    )
    parser.add_argument(
        "--min-shots-used",
        type=int,
        default=DEFAULT_MIN_SHOTS_USED,
        help="minimum finite shots retained per x/cycle after high-shot rejection",
    )
    args = parser.parse_args()

    if args.use_existing and args.recalibrate_existing:
        parser.error("--use-existing and --recalibrate-existing are mutually exclusive")

    if args.recalibrate_existing:
        if not args.output.exists():
            sys.exit(f"Missing {args.output}; run without --recalibrate-existing first.")
        recalibrate_existing_hdf5(
            args.output,
            te_path=args.te_filled,
            area_calibration_path=args.area_calibration,
            probe_a_calibration_path=args.probe_a_calibration,
            calibration_mode=args.calibration_mode,
            calibration_time_min_ms=args.calibration_time_min_ms,
            calibration_time_max_ms=args.calibration_time_max_ms,
        )
        with h5py.File(args.output, "r") as hf:
            source_channel = str(hf.attrs.get("deadtime_source_channel", args.source_channel))
            invert_polarity = bool(
                hf.attrs.get("deadtime_source_invert_polarity", source_channel == "i_sweep")
            )
    elif args.use_existing:
        if not args.output.exists():
            sys.exit(f"Missing {args.output}; run without --use-existing first.")
        with h5py.File(args.output, "r") as hf:
            source_channel = str(hf.attrs.get("deadtime_source_channel", args.source_channel))
            invert_polarity = bool(hf.attrs.get("deadtime_source_invert_polarity", source_channel == "i_sweep"))
    else:
        source_channel = args.source_channel
        invert_polarity = source_channel == ChannelKind.I_SWEEP.value
        build_hdf5(
            args.output,
            middle_fraction=args.middle_fraction,
            channel_kind=ChannelKind(source_channel),
            invert_polarity=invert_polarity,
            rotation_deg=float(args.rotation),
            reject_high_shots=not args.no_shot_rejection,
            rejection_channels=args.rejection_channels,
            calibration_mode=args.calibration_mode,
            factor_attr_name=args.factor_attr_name,
            front_reference=args.front_reference,
            te_path=args.te_filled,
            area_calibration_path=args.area_calibration,
            probe_a_calibration_path=args.probe_a_calibration,
            calibration_time_min_ms=args.calibration_time_min_ms,
            calibration_time_max_ms=args.calibration_time_max_ms,
            high_shot_sigma=args.high_shot_sigma,
            high_shot_ratio=args.high_shot_ratio,
            min_shots_used=args.min_shots_used,
        )

    channel_label = "-I_SWEEP" if invert_polarity else source_channel.upper()
    plot_all(
        args.output,
        args.output_dir,
        save_animation=not args.no_animation,
        plot_prefix=args.plot_prefix,
        channel_label=channel_label,
    )


if __name__ == "__main__":
    main()
