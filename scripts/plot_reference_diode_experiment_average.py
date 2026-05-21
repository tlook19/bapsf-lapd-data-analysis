"""Plot reference photodiode experiment-set averages with uncertainty bands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd import ChannelKind, LapdDataset
from plot_export import configure_latex_friendly_matplotlib, parse_formats, save_figure
from plot_labels import experiment_title


MANIFEST = Path("config/may2026_run_manifest.toml")
OUTPUT = Path("figures/reference_diode_experiment_average.png")
N_PLOT_SAMPLES = 1600
ERRORBAR_EVERY = 80


@dataclass(frozen=True)
class BinnedTraceStats:
    time_ms: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    stderr: np.ndarray
    n: np.ndarray


def _update_point(values: np.ndarray, count: int, mean: float, m2: float) -> tuple[int, float, float]:
    """Update scalar Welford stats with one vector of observations."""
    for value in values:
        count += 1
        delta = float(value) - mean
        mean += delta / count
        m2 += delta * (float(value) - mean)
    return count, mean, m2


def _merge_batch(
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge a 2D batch shaped (observation, time) into per-time Welford stats."""
    batch_count = values.shape[0]
    if batch_count == 0:
        return counts, means, m2

    batch_mean = values.mean(axis=0)
    batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)

    new_counts = counts + batch_count
    delta = batch_mean - means
    means = means + delta * batch_count / new_counts
    m2 = m2 + batch_m2 + delta**2 * counts * batch_count / new_counts
    return new_counts, means, m2


def reference_experiment_average(
    dataset: LapdDataset,
    experiment_set_id: int,
    *,
    n_plot_samples: int = N_PLOT_SAMPLES,
) -> BinnedTraceStats:
    """Average reference diode traces for one experiment set at strided times."""
    run_ids = dataset.experiment_set_run_ids(experiment_set_id)
    if not run_ids:
        raise ValueError(f"No runs found for experiment set {experiment_set_id}")

    first_run = dataset.run(run_ids[0])
    channel = first_run.config.channel(ChannelKind.REFERENCE_PHOTODIODE)
    nsamples = first_run.dataset_info(channel.hdf5_path)["shape"][1]
    sample_indices = np.unique(
        np.linspace(0, nsamples - 1, min(n_plot_samples, nsamples), dtype=np.int64)
    )
    n_points = len(sample_indices)

    counts = np.zeros(n_points, dtype=np.int64)
    means = np.zeros(n_points, dtype=np.float64)
    m2 = np.zeros(n_points, dtype=np.float64)

    for run_id in run_ids:
        run = dataset.run(run_id)
        channel = run.config.channel(ChannelKind.REFERENCE_PHOTODIODE)
        header_path = f"{channel.hdf5_path} headers"
        with h5py.File(run.path, "r") as h5:
            data = h5[channel.hdf5_path]
            headers = h5[header_path][:]
            scales = headers["Scale"].astype(np.float64)
            offsets = headers["Offset"].astype(np.float64)

            # The data are gzip-compressed in chunks shaped (1, nsamples), so
            # column selection still has to decompress every full trace chunk.
            # Reading the full run array once is faster and keeps memory bounded
            # because we process one run at a time.
            raw_values = data[()][:, sample_indices].astype(np.float64)
            voltages = raw_values * scales[:, None] + offsets[:, None]
            counts, means, m2 = _merge_batch(counts, means, m2, voltages)

    std = np.zeros_like(means)
    mask = counts > 1
    std[mask] = np.sqrt(m2[mask] / (counts[mask] - 1))
    stderr = np.zeros_like(means)
    stderr[mask] = std[mask] / np.sqrt(counts[mask])

    time_ms = sample_indices * first_run.sample_dt_s() * 1e3
    return BinnedTraceStats(time_ms=time_ms, mean=means, std=std, stderr=stderr, n=counts)


def make_plot(output: Path, *, uncertainty: str = "stderr", formats: tuple[str, ...] = ("png",)) -> None:
    dataset = LapdDataset.from_manifest(MANIFEST)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
        4: "tab:red",
    }

    for experiment_set_id in dataset.experiment_set_ids():
        stats = reference_experiment_average(dataset, experiment_set_id)
        experiment = dataset.config(dataset.experiment_set_run_ids(experiment_set_id)[0]).experiment_set
        label = experiment_title(experiment_set_id, experiment)
        color = colors[experiment_set_id]
        yerr = stats.stderr if uncertainty == "stderr" else stats.std
        ax.plot(stats.time_ms, stats.mean, lw=1.6, color=color, label=label)
        ax.fill_between(
            stats.time_ms,
            stats.mean - yerr,
            stats.mean + yerr,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        errorbar_slice = slice(None, None, ERRORBAR_EVERY)
        ax.errorbar(
            stats.time_ms[errorbar_slice],
            stats.mean[errorbar_slice],
            yerr=yerr[errorbar_slice],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            capthick=0.8,
            alpha=0.9,
        )

    ax.set_title("Reference Photodiode Experiment-Set Averages")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Reference photodiode signal (V)")
    uncertainty_label = (
        "+/-1 standard error of the mean"
        if uncertainty == "stderr"
        else "+/-1 standard deviation"
    )
    ax.text(
        0.01,
        0.02,
        f"Shaded bands and capped bars show {uncertainty_label}.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.75,
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

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
        choices=["stderr", "std"],
        default="stderr",
        help="Uncertainty band/errorbar type.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    configure_latex_friendly_matplotlib()
    make_plot(args.output, uncertainty=args.uncertainty, formats=parse_formats(args.formats))


if __name__ == "__main__":
    main()
