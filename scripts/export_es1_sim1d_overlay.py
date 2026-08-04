"""Export one experiment set for the bapsf-transport sim1d notebook.

The NPZ product is self-contained and uses simulation-facing units:

* core density and total SEM in cm^-3;
* core electron temperature and radial SEM in eV;
* offset-corrected upstream ion-saturation current at x=0 in A;
* offset-corrected discharge current in A and cathode-anode voltage in V;
* per-port fractional spread of the fit-window T_e re-fits, dimensionless;
* all time axes in ms relative to the experimental SIS trigger.

Probe-A density SEM includes the propagated area-calibration uncertainty.
The ion-current decay uses the same one-sided high-current detector as the
profile pipeline.  A shot is excluded from the continuous decay trace if it is
flagged in any pre-afterglow inter-sweep cell at x=0.  The retained traces are
100 kHz low-pass filtered and reduced to approximately 10 us time bins before
the shot mean and SEM are computed.
Each run's discharge current has its own additive channel zero offset
subtracted before smoothing; the per-run values are exported alongside the
traces.  Discharge traces are smoothed with the same 9-sample moving average
used by ``process_discharge_experiment_summary.py`` and averaged over both
stored traces from all runs in the selected experiment set.  Raw cathode-anode
voltage is negative, so the exported overlay voltage is multiplied by -1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

from bapsf_lapd import ChannelKind, LapdDataset
from bapsf_lapd.filtering import butterworth_lowpass
from plot_core_density_temperature_timeseries import (
    DENSITY_SCALE_M3,
    _load_density_stats,
    _load_te_stats,
    _plot_uncertainty,
)
from plot_decay_rates_x0 import (
    _load_zero_offsets,
    _padded_sample_window,
    _zero_offset_v,
)
from plot_isat_profiles import _deadtime_shot_means, _high_shot_outlier_mask


MANIFEST = Path("config/may2026_run_manifest.toml")
DENSITY_HDF5 = Path("processed/density_profiles_isweep.hdf5")
TE_HDF5 = Path("processed/te_filled.hdf5")
ISAT_PROFILE_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
ZERO_OFFSETS = Path("processed/trace_zero_offsets.toml")
WINDOW_REFITS_HDF5 = Path("processed/sweep_window_refits.hdf5")
PORTS = np.array([11, 21, 29, 41, 50], dtype=np.int16)
X_MIN_CM = -10.0
X_MAX_CM = 10.0
DISCHARGE_SMOOTHING_SAMPLES = 9
DENSITY_SCALE_CM3 = DENSITY_SCALE_M3 * 1.0e-6
ISAT_DECAY_STOP_S = 47.5e-3
ISAT_DECAY_FILTER_PAD_S = 0.1e-3
ISAT_DECAY_CUTOFF_HZ = 100.0e3
ISAT_DECAY_BIN_S = 10.0e-6


def _isat_decay_stats(
    dataset: LapdDataset,
    profile_path: Path,
    zero_offsets_path: Path,
    experiment_set_id: int,
) -> dict[str, np.ndarray | float | int | str]:
    """Return offset-corrected x=0 upstream Isat mean and shot SEM."""
    zero_offsets = _load_zero_offsets(zero_offsets_path)
    means = []
    sems = []
    ports = []
    run_ids = []
    n_used = []
    n_rejected = []
    zero_offset_v = []
    source_channels = []
    source_inverted = []
    afterglow_start_ms = []
    reference_time_ms = None
    source_by_run: dict[str, tuple[ChannelKind, bool]] = {}

    with h5py.File(profile_path, "r") as profile_hdf:
        if not bool(profile_hdf.attrs.get("high_shot_rejection_enabled", False)):
            raise ValueError(f"{profile_path} was built without high-current rejection")
        sigma = float(profile_hdf.attrs["high_shot_rejection_sigma"])
        ratio = float(profile_hdf.attrs["high_shot_rejection_ratio"])
        min_shots = int(profile_hdf.attrs["high_shot_rejection_min_shots_used"])
        x_cm = profile_hdf["x_cm"][()]
        x_idx = int(np.argmin(np.abs(x_cm)))
        if not np.isclose(x_cm[x_idx], 0.0):
            raise ValueError("x=0 is not present in the Isat profile grid")
        set_group = profile_hdf[f"experiment_sets/{experiment_set_id}"]
        selected_run_ids = sorted(set_group.keys())
        for run_id in selected_run_ids:
            run_group = set_group[run_id]
            source_by_run[run_id] = (
                ChannelKind(str(run_group.attrs["deadtime_source_channel"])),
                bool(run_group.attrs["deadtime_source_invert_polarity"]),
            )

    for run_id in selected_run_ids:
        run = dataset.run(run_id)
        channel, invert_polarity = source_by_run[run_id]

        # Recreate the profile pipeline's per-cycle outlier flags, then use a
        # fixed shot ensemble for the continuous decay so its SEM is coherent.
        deadtime_means = _deadtime_shot_means(
            run,
            channel_kind=channel,
            invert_polarity=invert_polarity,
        )
        flagged = _high_shot_outlier_mask(
            deadtime_means,
            sigma=sigma,
            ratio=ratio,
            min_shots_used=min_shots,
        )[x_idx]
        reject_shot = np.any(flagged, axis=1)
        if np.sum(~reject_shot) < min_shots:
            flag_counts = np.sum(flagged, axis=1)
            keep = np.argsort(flag_counts)[:min_shots]
            reject_shot[:] = True
            reject_shot[keep] = False

        sweep = run.config.sweep
        decay_start_s = sweep.t0_s + sweep.n_cycles * sweep.tau_cycle_s
        sample, crop = _padded_sample_window(
            run,
            decay_start_s,
            ISAT_DECAY_STOP_S,
            ISAT_DECAY_FILTER_PAD_S,
        )
        channel_config = run.config.channel(channel)
        shot_start = run.flat_shot_index(x_idx, 0)
        shot_stop = shot_start + run.shots_per_position()
        with run.open() as raw_hdf:
            raw = raw_hdf[channel_config.hdf5_path][shot_start:shot_stop, sample].astype(
                np.float64
            )
            headers = raw_hdf[f"{channel_config.hdf5_path} headers"][shot_start:shot_stop]

        offset_v = _zero_offset_v(run, channel, zero_offsets)
        voltage = raw * headers["Scale"][:, None] + headers["Offset"][:, None]
        voltage -= offset_v
        current = channel_config.apply_calibration(voltage)
        if invert_polarity:
            current = -current
        current = butterworth_lowpass(
            current,
            sample_rate_hz=run.sample_rate_hz(),
            cutoff_hz=ISAT_DECAY_CUTOFF_HZ,
            axis=-1,
        )[:, crop]

        samples_per_bin = max(1, int(round(ISAT_DECAY_BIN_S / run.sample_dt_s())))
        n_bins = current.shape[1] // samples_per_bin
        current = current[:, : n_bins * samples_per_bin]
        current = current.reshape(current.shape[0], n_bins, samples_per_bin).mean(axis=2)
        start_index = sample.start + crop.start
        time_s = run.time_axis(n_bins * samples_per_bin, start_index=start_index)
        time_ms = (
            time_s[: n_bins * samples_per_bin]
            .reshape(n_bins, samples_per_bin)
            .mean(axis=1)
            * 1000.0
        )

        retained = current[~reject_shot]
        if reference_time_ms is None:
            reference_time_ms = time_ms
        elif not np.allclose(time_ms, reference_time_ms, rtol=0.0, atol=1e-9):
            raise ValueError(f"Isat decay time grid differs for run {run_id}")
        mean = np.mean(retained, axis=0)
        std = np.std(retained, axis=0, ddof=1)
        means.append(mean)
        sems.append(std / np.sqrt(retained.shape[0]))
        ports.append(int(run.config.probe.port or 0))
        run_ids.append(run_id)
        n_used.append(retained.shape[0])
        n_rejected.append(int(np.sum(reject_shot)))
        zero_offset_v.append(offset_v)
        source_channels.append(channel.value)
        source_inverted.append(invert_polarity)
        afterglow_start_ms.append(decay_start_s * 1000.0)

    if reference_time_ms is None:
        raise RuntimeError(
            f"No experiment-set-{experiment_set_id} upstream Isat decay traces found"
        )
    order = np.argsort(ports)
    return {
        "time_ms": reference_time_ms,
        "mean_a": np.asarray(means)[order],
        "sem_a": np.asarray(sems)[order],
        "port": np.asarray(ports, dtype=np.int16)[order],
        "run_id": np.asarray(run_ids)[order],
        "n_shots_used": np.asarray(n_used, dtype=np.int16)[order],
        "n_shots_rejected": np.asarray(n_rejected, dtype=np.int16)[order],
        "zero_offset_v": np.asarray(zero_offset_v)[order],
        "source_channel": np.asarray(source_channels)[order],
        "source_inverted": np.asarray(source_inverted, dtype=np.bool_)[order],
        "afterglow_start_ms": np.asarray(afterglow_start_ms)[order],
        "cutoff_hz": ISAT_DECAY_CUTOFF_HZ,
        "bin_s": ISAT_DECAY_BIN_S,
        "outlier_sigma": sigma,
        "outlier_ratio": ratio,
        "outlier_min_shots": min_shots,
        "outlier_method": "exclude shot if flagged in any inter-sweep cycle at x=0",
    }


def _te_window_spread_frac(
    refits_path: Path,
    experiment_set_id: int,
    ports: np.ndarray,
) -> np.ndarray:
    """Return the per-port fractional spread of the fit-window ``T_e`` re-fits.

    For each port, ``ptp(te_window_ev) / mean(te_window_ev)`` over the
    ``p_low`` x ``f_high`` fit-window grid stored in
    ``processed/sweep_window_refits.hdf5``.  The result is dimensionless, is
    aligned element-wise with ``ports`` (and hence with the overlay's ``port``
    and ``z_cm`` axes), and measures how far a port's ``T_e`` moves when the
    retarding-region fit window is varied rather than how uncertain its mean is.

    The value is ``NaN`` where the experiment set or the port has no re-fit
    group, and ``NaN`` where a port's grid contains a failed re-fit, since a
    single missing window makes the full spread undetermined rather than
    smaller.
    """
    spread = np.full(len(ports), np.nan, dtype=np.float64)
    with h5py.File(refits_path, "r") as refits_hdf:
        set_group = refits_hdf.get(f"set{experiment_set_id}")
        if set_group is None:
            return spread
        for index, port in enumerate(ports):
            port_group = set_group.get(f"port{int(port)}")
            if port_group is None:
                continue
            grid = np.asarray(port_group["te_window_ev"][()], dtype=np.float64)
            spread[index] = np.ptp(grid) / np.mean(grid)
    return spread


def _discharge_stats(
    dataset: LapdDataset,
    experiment_set_id: int,
) -> dict[str, np.ndarray | int]:
    """Return the offset-corrected experiment-set discharge current and voltage.

    Each run's discharge current has its own additive channel zero offset
    (``LapdRun.discharge_zero_offset_stats``) subtracted before smoothing, so
    the exported trace reads zero where no current flows.
    """
    currents = []
    voltages_raw = []
    zero_offsets_a = []
    run_ids = []
    reference_time_s = None

    for run_id in dataset.experiment_set_run_ids(experiment_set_id):
        run = dataset.run(run_id)
        current, voltage, time_s = run.discharge_traces()
        zero_offset_a = run.discharge_zero_offset_stats().offset_a
        current = current - zero_offset_a
        if reference_time_s is None:
            reference_time_s = time_s
        elif not np.allclose(time_s, reference_time_s, rtol=0.0, atol=1e-12):
            raise ValueError(f"Discharge time grid differs for run {run_id}")
        zero_offsets_a.append(zero_offset_a)
        run_ids.append(run_id)
        currents.append(
            uniform_filter1d(
                current,
                size=DISCHARGE_SMOOTHING_SAMPLES,
                axis=1,
                mode="nearest",
            )
        )
        voltages_raw.append(
            uniform_filter1d(
                voltage,
                size=DISCHARGE_SMOOTHING_SAMPLES,
                axis=1,
                mode="nearest",
            )
        )

    if reference_time_s is None:
        raise RuntimeError(f"No experiment-set-{experiment_set_id} discharge traces found")
    current_all = np.concatenate(currents, axis=0)
    voltage_all_raw = np.concatenate(voltages_raw, axis=0)
    n_traces = current_all.shape[0]
    current_std = np.std(current_all, axis=0, ddof=1)
    voltage_std = np.std(voltage_all_raw, axis=0, ddof=1)
    return {
        "time_ms": reference_time_s * 1000.0,
        "current_mean_a": np.mean(current_all, axis=0),
        "current_sem_a": current_std / np.sqrt(n_traces),
        "voltage_positive_mean_v": -np.mean(voltage_all_raw, axis=0),
        "voltage_sem_v": voltage_std / np.sqrt(n_traces),
        "n_traces": n_traces,
        "zero_offset_a": np.asarray(zero_offsets_a, dtype=np.float64),
        "zero_offset_run_id": np.asarray(run_ids),
    }


def export_overlay(
    density_path: Path,
    te_path: Path,
    isat_profile_path: Path,
    zero_offsets_path: Path,
    manifest_path: Path,
    output_path: Path,
    experiment_set_id: int = 1,
    window_refits_path: Path = WINDOW_REFITS_HDF5,
) -> Path:
    experiment_set_key = str(experiment_set_id)
    with h5py.File(density_path, "r") as density_hdf, h5py.File(te_path, "r") as te_hdf:
        density = _load_density_stats(
            density_hdf,
            experiment_set_key,
            X_MIN_CM,
            X_MAX_CM,
        )
        te = _load_te_stats(
            te_hdf,
            experiment_set_key,
            X_MIN_CM,
            X_MAX_CM,
            dataset="te_filled",
        )
        if not np.allclose(density.z_cm, te.z_cm):
            raise ValueError(
                f"ES{experiment_set_id} density and temperature z grids differ"
            )
        if len(density.z_cm) != len(PORTS):
            raise ValueError(
                f"Expected {len(PORTS)} ES{experiment_set_id} port rows, "
                f"found {len(density.z_cm)}"
            )
        experiment_label = str(
            density_hdf[f"experiment_sets/{experiment_set_key}"].attrs["label"]
        )
        factor = float(density_hdf.attrs["probe_a_factor"])
        factor_lower = float(density_hdf.attrs["probe_a_factor_lower_bound"])
        factor_upper = float(density_hdf.attrs["probe_a_factor_upper_bound"])
    dataset = LapdDataset.from_manifest(manifest_path)
    isat_decay = _isat_decay_stats(
        dataset,
        isat_profile_path,
        zero_offsets_path,
        experiment_set_id,
    )
    if not np.array_equal(isat_decay["port"], PORTS):
        raise ValueError(
            f"ES{experiment_set_id} Isat decay ports {isat_decay['port']} "
            f"do not match expected {PORTS}"
        )
    discharge = _discharge_stats(dataset, experiment_set_id)
    te_window_spread = _te_window_spread_frac(
        window_refits_path,
        experiment_set_id,
        PORTS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.array(5, dtype=np.int16),
        experiment_set_id=np.array(experiment_set_id, dtype=np.int16),
        experiment_label=np.array(experiment_label),
        port=PORTS,
        z_cm=density.z_cm,
        density_time_ms=density.time_ms,
        density_mean_cm3=density.mean * DENSITY_SCALE_CM3,
        density_total_sem_cm3=_plot_uncertainty(density, "sem") * DENSITY_SCALE_CM3,
        density_radial_sem_cm3=density.sem * DENSITY_SCALE_CM3,
        density_core_count=density.count,
        te_time_ms=te.time_ms,
        te_mean_ev=te.mean,
        te_sem_ev=te.sem,
        te_core_count=te.count,
        te_window_spread_frac=te_window_spread,
        core_x_min_cm=np.array(X_MIN_CM),
        core_x_max_cm=np.array(X_MAX_CM),
        probe_a_factor=np.array(factor),
        probe_a_factor_lower_bound=np.array(factor_lower),
        probe_a_factor_upper_bound=np.array(factor_upper),
        isat_decay_time_ms=isat_decay["time_ms"],
        isat_decay_mean_a=isat_decay["mean_a"],
        isat_decay_sem_a=isat_decay["sem_a"],
        isat_decay_port=isat_decay["port"],
        isat_decay_run_id=isat_decay["run_id"],
        isat_decay_n_shots_used=isat_decay["n_shots_used"],
        isat_decay_n_shots_rejected=isat_decay["n_shots_rejected"],
        isat_decay_zero_offset_v=isat_decay["zero_offset_v"],
        isat_decay_source_channel=isat_decay["source_channel"],
        isat_decay_source_inverted=isat_decay["source_inverted"],
        isat_decay_afterglow_start_ms=isat_decay["afterglow_start_ms"],
        isat_decay_cutoff_hz=np.array(isat_decay["cutoff_hz"]),
        isat_decay_bin_s=np.array(isat_decay["bin_s"]),
        isat_decay_outlier_sigma=np.array(isat_decay["outlier_sigma"]),
        isat_decay_outlier_ratio=np.array(isat_decay["outlier_ratio"]),
        isat_decay_outlier_min_shots=np.array(
            isat_decay["outlier_min_shots"],
            dtype=np.int16,
        ),
        isat_decay_outlier_method=np.array(isat_decay["outlier_method"]),
        isat_decay_current_correction=np.array(
            "digitizer scale/offset, measured zero-offset subtraction, channel calibration"
        ),
        isat_decay_face=np.array(
            "upstream-facing electrical channel at x=0; see per-port source metadata"
        ),
        discharge_time_ms=discharge["time_ms"],
        discharge_current_mean_a=discharge["current_mean_a"],
        discharge_current_sem_a=discharge["current_sem_a"],
        discharge_voltage_positive_mean_v=discharge["voltage_positive_mean_v"],
        discharge_voltage_sem_v=discharge["voltage_sem_v"],
        discharge_n_traces=np.array(discharge["n_traces"], dtype=np.int16),
        discharge_zero_offset_a=discharge["zero_offset_a"],
        discharge_zero_offset_run_id=discharge["zero_offset_run_id"],
        discharge_current_correction=np.array(
            "per-run additive zero offset from the post-connect pre-avalanche "
            "window subtracted before smoothing"
        ),
        discharge_smoothing_samples=np.array(
            DISCHARGE_SMOOTHING_SAMPLES,
            dtype=np.int16,
        ),
        discharge_voltage_sign=np.array(
            "positive overlay is -1 times raw cathode-anode voltage"
        ),
        density_uncertainty=np.array(
            "radial SEM plus Probe-A area calibration in quadrature"
        ),
    )
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=Path, default=DENSITY_HDF5)
    parser.add_argument("--te-filled", type=Path, default=TE_HDF5)
    parser.add_argument("--isat-profiles", type=Path, default=ISAT_PROFILE_HDF5)
    parser.add_argument("--zero-offsets", type=Path, default=ZERO_OFFSETS)
    parser.add_argument("--window-refits", type=Path, default=WINDOW_REFITS_HDF5)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--experiment-set", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        f"processed/es{args.experiment_set}_sim1d_overlay.npz"
    )
    export_overlay(
        args.density,
        args.te_filled,
        args.isat_profiles,
        args.zero_offsets,
        args.manifest,
        output,
        args.experiment_set,
        args.window_refits,
    )


if __name__ == "__main__":
    main()
