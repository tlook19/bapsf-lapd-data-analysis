"""Plot moving photodiode averages grouped by experiment set and z position."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd import ChannelKind, LapdDataset
from bapsf_lapd.config import z_from_port
from plot_export import configure_latex_friendly_matplotlib, parse_formats, save_figure
from plot_labels import experiment_title


MANIFEST = Path("config/may2026_run_manifest.toml")
OUTPUT_SEM = Path("figures/moving_diode_by_experiment_set_sem.png")
OUTPUT_STD = Path("figures/moving_diode_by_experiment_set_std.png")
N_PLOT_SAMPLES = 1600
ERRORBAR_EVERY = 80


@dataclass
class RunningTraceStats:
    count: int
    mean: np.ndarray
    m2: np.ndarray
    time_ms: np.ndarray

    @classmethod
    def empty(cls, n_points: int, time_ms: np.ndarray) -> RunningTraceStats:
        return cls(
            count=0,
            mean=np.zeros(n_points, dtype=np.float64),
            m2=np.zeros(n_points, dtype=np.float64),
            time_ms=time_ms,
        )

    def update_batch(self, values: np.ndarray) -> None:
        batch_count = values.shape[0]
        if batch_count == 0:
            return

        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * batch_count / new_count
        self.m2 = self.m2 + batch_m2 + delta**2 * self.count * batch_count / new_count
        self.count = new_count

    @property
    def std(self) -> np.ndarray:
        if self.count <= 1:
            return np.zeros_like(self.mean)
        return np.sqrt(self.m2 / (self.count - 1))

    @property
    def stderr(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros_like(self.mean)
        return self.std / np.sqrt(self.count)


def _sample_indices(nsamples: int) -> np.ndarray:
    return np.unique(
        np.linspace(0, nsamples - 1, min(N_PLOT_SAMPLES, nsamples), dtype=np.int64)
    )


def build_grouped_stats(
    dataset: LapdDataset,
) -> tuple[dict[tuple[int, int], RunningTraceStats], dict[tuple[int, int], int], float]:
    """Return moving-diode stats keyed by (experiment_set_id, rounded_z_cm)."""
    groups: dict[tuple[int, int], RunningTraceStats] = {}
    ports: dict[tuple[int, int], int] = {}
    normalization = 0.0

    first_run = dataset.run(dataset.run_ids()[0])
    first_channel = first_run.config.channel(ChannelKind.MOVING_PHOTODIODE)
    nsamples = first_run.dataset_info(first_channel.hdf5_path)["shape"][1]
    sample_indices = _sample_indices(nsamples)
    time_ms = sample_indices * first_run.sample_dt_s() * 1e3

    for run_id in dataset.run_ids():
        run = dataset.run(run_id)
        channel = run.config.channel(ChannelKind.MOVING_PHOTODIODE)
        if channel.port is None:
            raise ValueError(f"Run {run_id} has no moving photodiode port configured")

        experiment_set_id = run.config.experiment_set.id
        z_label = round(z_from_port(channel.port))
        key = (experiment_set_id, z_label)
        if key not in groups:
            groups[key] = RunningTraceStats.empty(len(sample_indices), time_ms)
            ports[key] = channel.port
        elif ports[key] != channel.port:
            raise ValueError(
                f"Experiment set {experiment_set_id}, z={z_label} cm maps to "
                f"multiple moving photodiode ports: {ports[key]} and {channel.port}"
            )

        with h5py.File(run.path, "r") as h5:
            data = h5[channel.hdf5_path][()]
            headers = h5[f"{channel.hdf5_path} headers"][:]

        scales = headers["Scale"].astype(np.float64)
        offsets = headers["Offset"].astype(np.float64)

        row_peaks = data.max(axis=1).astype(np.float64) * scales + offsets
        normalization = max(normalization, float(row_peaks.max()))

        raw_plot = data[:, sample_indices].astype(np.float64)
        voltages = raw_plot * scales[:, None] + offsets[:, None]
        groups[key].update_batch(voltages)

    if normalization <= 0:
        raise ValueError("Moving photodiode normalization peak must be positive")
    return groups, ports, normalization


def make_plot(
    dataset: LapdDataset,
    groups: dict[tuple[int, int], RunningTraceStats],
    ports: dict[tuple[int, int], int],
    normalization: float,
    output: Path,
    *,
    uncertainty: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    z_values = sorted({z_cm for _, z_cm in groups})
    cmap = plt.get_cmap("tab10")
    colors = {z_cm: cmap(index % 10) for index, z_cm in enumerate(z_values)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    axes_by_set = dict(zip(dataset.experiment_set_ids(), axes.flat, strict=True))

    for experiment_set_id, ax in axes_by_set.items():
        run_ids = dataset.experiment_set_run_ids(experiment_set_id)
        experiment = dataset.config(run_ids[0]).experiment_set
        z_for_set = sorted(z_cm for set_id, z_cm in groups if set_id == experiment_set_id)

        for z_cm in z_for_set:
            key = (experiment_set_id, z_cm)
            stats = groups[key]
            yerr = stats.stderr if uncertainty == "sem" else stats.std
            label = f"z={z_cm} cm (p{ports[key]}, n={stats.count})"
            ax.plot(
                stats.time_ms,
                stats.mean / normalization,
                color=colors[z_cm],
                lw=1.5,
                label=label,
            )
            ax.fill_between(
                stats.time_ms,
                (stats.mean - yerr) / normalization,
                (stats.mean + yerr) / normalization,
                color=colors[z_cm],
                alpha=0.14,
                linewidth=0,
            )
            errorbar_slice = slice(None, None, ERRORBAR_EVERY)
            ax.errorbar(
                stats.time_ms[errorbar_slice],
                stats.mean[errorbar_slice] / normalization,
                yerr=yerr[errorbar_slice] / normalization,
                fmt="none",
                ecolor=colors[z_cm],
                elinewidth=0.75,
                capsize=1.8,
                capthick=0.75,
                alpha=0.85,
            )

        ax.set_title(experiment_title(experiment_set_id, experiment))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False, fontsize=8)

    uncertainty_label = "standard error" if uncertainty == "sem" else "standard deviation"
    for ax in axes[:, 0]:
        ax.set_ylabel("Moving photodiode / global peak")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (ms)")
    fig.suptitle(
        f"Moving Photodiode Averages by Experiment Set and z (+/-1 {uncertainty_label})",
        fontsize=15,
    )
    save_figure(fig, output, formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats: png, pdf, svg, latex, vector, or all.",
    )
    parser.add_argument(
        "--uncertainty",
        choices=["sem", "std", "both"],
        default="both",
        help="Which uncertainty plot to export.",
    )
    args = parser.parse_args()
    formats = parse_formats(args.formats)

    configure_latex_friendly_matplotlib()
    dataset = LapdDataset.from_manifest(MANIFEST)
    groups, ports, normalization = build_grouped_stats(dataset)
    if args.uncertainty in {"sem", "both"}:
        make_plot(dataset, groups, ports, normalization, OUTPUT_SEM, uncertainty="sem", formats=formats)
    if args.uncertainty in {"std", "both"}:
        make_plot(dataset, groups, ports, normalization, OUTPUT_STD, uncertainty="std", formats=formats)


if __name__ == "__main__":
    main()
