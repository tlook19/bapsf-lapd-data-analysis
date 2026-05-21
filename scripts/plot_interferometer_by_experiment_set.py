"""Plot interferometer experiment-set averages with uncertainty bands."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd import LapdDataset
from bapsf_lapd.config import z_from_port
from plot_export import configure_latex_friendly_matplotlib, parse_formats, save_figure
from plot_labels import experiment_title


MANIFEST = Path("config/may2026_run_manifest.toml")
OUTPUT_SEM = Path("figures/interferometer_by_experiment_set_sem.png")
OUTPUT_STD = Path("figures/interferometer_by_experiment_set_std.png")
STATS_OUTPUT = Path("processed/interferometer_experiment_set_stats.npz")
INTERFEROMETER_PORTS = [20, 29, 40]
ERRORBAR_EVERY = 100
PLASMA_DIAMETER_CM = 40.0
M3_TO_CM3 = 1e-6
PLOT_SCALE_CM3 = 1e13


@dataclass
class RunningTraceStats:
    count: int
    mean: np.ndarray
    m2: np.ndarray
    time_ms: np.ndarray

    @classmethod
    def empty(cls, time_ms: np.ndarray) -> RunningTraceStats:
        return cls(
            count=0,
            mean=np.zeros_like(time_ms, dtype=np.float64),
            m2=np.zeros_like(time_ms, dtype=np.float64),
            time_ms=time_ms,
        )

    def update(self, values: np.ndarray) -> None:
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)

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


def _phase_group(port: int) -> str:
    return f"diagnostics/interferometer/phase_p{port}"


def _time_group(port: int) -> str:
    return "diagnostics/interferometer/time_array_p40" if port == 40 else "diagnostics/interferometer/time_array"


def _reference_time_grid(dataset: LapdDataset, experiment_set_id: int, port: int) -> np.ndarray:
    for run_id in dataset.experiment_set_run_ids(experiment_set_id):
        run = dataset.run(run_id)
        with h5py.File(run.path, "r") as h5:
            time_group = _time_group(port)
            keys = sorted(h5[time_group].keys(), key=lambda item: int(item))
            if keys:
                return h5[f"{time_group}/{keys[0]}"][()].astype(np.float64)
    raise ValueError(f"No interferometer time grid found for set {experiment_set_id}, port {port}")


def build_grouped_stats(
    dataset: LapdDataset,
) -> dict[tuple[int, int], RunningTraceStats]:
    """Return calibrated interferometer stats keyed by (experiment_set_id, port)."""
    groups: dict[tuple[int, int], RunningTraceStats] = {}

    for experiment_set_id in dataset.experiment_set_ids():
        for port in INTERFEROMETER_PORTS:
            common_time = _reference_time_grid(dataset, experiment_set_id, port)
            stats = RunningTraceStats.empty(common_time)
            phase_group = _phase_group(port)
            time_group = _time_group(port)

            for run_id in dataset.experiment_set_run_ids(experiment_set_id):
                run = dataset.run(run_id)
                with h5py.File(run.path, "r") as h5:
                    calibration = float(h5[phase_group].attrs["calibration factor (m^-3/rad)"])
                    keys = sorted(h5[phase_group].keys(), key=lambda item: int(item))
                    for key in keys:
                        phase_ds = h5[f"{phase_group}/{key}"]
                        if bool(phase_ds.attrs.get("rigol_missing", False)):
                            continue
                        phase = phase_ds[()].astype(np.float64)
                        time_ms = h5[f"{time_group}/{key}"][()].astype(np.float64)
                        line_average_density_cm3 = phase * calibration * M3_TO_CM3
                        if np.array_equal(time_ms, common_time):
                            aligned = line_average_density_cm3
                        else:
                            # LeCroy time arrays for p20/p29 differ by tiny
                            # offsets at the endpoints. Endpoint filling keeps
                            # those traces in the experiment average without
                            # changing the visible timebase.
                            aligned = np.interp(common_time, time_ms, line_average_density_cm3)
                        stats.update(aligned)
            groups[(experiment_set_id, port)] = stats
    return groups


def save_stats(groups: dict[tuple[int, int], RunningTraceStats], output: Path) -> None:
    """Save line-average and line-integrated interferometer stats for reuse."""
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for (experiment_set_id, port), stats in groups.items():
        prefix = f"set{experiment_set_id}_p{port}"
        arrays[f"{prefix}_time_ms"] = stats.time_ms
        arrays[f"{prefix}_n"] = np.array(stats.count, dtype=np.int64)
        arrays[f"{prefix}_line_average_mean_cm3"] = stats.mean
        arrays[f"{prefix}_line_average_std_cm3"] = stats.std
        arrays[f"{prefix}_line_average_stderr_cm3"] = stats.stderr
        arrays[f"{prefix}_line_integrated_mean_cm2"] = stats.mean * PLASMA_DIAMETER_CM
        arrays[f"{prefix}_line_integrated_std_cm2"] = stats.std * PLASMA_DIAMETER_CM
        arrays[f"{prefix}_line_integrated_stderr_cm2"] = stats.stderr * PLASMA_DIAMETER_CM
    np.savez_compressed(output, **arrays)
    print(output)


def make_plot(
    dataset: LapdDataset,
    groups: dict[tuple[int, int], RunningTraceStats],
    output: Path,
    *,
    uncertainty: str,
    formats: tuple[str, ...] = ("png",),
) -> None:
    cmap = plt.get_cmap("tab10")
    z_values = [round(z_from_port(port)) for port in INTERFEROMETER_PORTS]
    colors = {z_cm: cmap(index % 10) for index, z_cm in enumerate(sorted(z_values))}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False, sharey=True)
    axes_by_set = dict(zip(dataset.experiment_set_ids(), axes.flat, strict=True))

    for experiment_set_id, ax in axes_by_set.items():
        run_ids = dataset.experiment_set_run_ids(experiment_set_id)
        experiment = dataset.config(run_ids[0]).experiment_set

        for port in INTERFEROMETER_PORTS:
            stats = groups[(experiment_set_id, port)]
            z_cm = round(z_from_port(port))
            yerr = stats.stderr if uncertainty == "sem" else stats.std
            label = f"z={z_cm} cm (p{port}, n={stats.count})"
            ax.plot(
                stats.time_ms,
                stats.mean / PLOT_SCALE_CM3,
                color=colors[z_cm],
                lw=1.5,
                label=label,
            )
            ax.fill_between(
                stats.time_ms,
                (stats.mean - yerr) / PLOT_SCALE_CM3,
                (stats.mean + yerr) / PLOT_SCALE_CM3,
                color=colors[z_cm],
                alpha=0.14,
                linewidth=0,
            )
            errorbar_slice = slice(None, None, ERRORBAR_EVERY)
            ax.errorbar(
                stats.time_ms[errorbar_slice],
                stats.mean[errorbar_slice] / PLOT_SCALE_CM3,
                yerr=yerr[errorbar_slice] / PLOT_SCALE_CM3,
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
        ax.set_ylabel(r"Line-average density ($10^{13}$ cm$^{-3}$)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (ms)")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.text(
        0.5,
        0.025,
        "Calibration assumes a 40 cm plasma diameter. Line-integrated density = 40 cm x plotted line-average density.",
        ha="center",
        fontsize=9,
        alpha=0.8,
    )
    fig.suptitle(
        f"Interferometer Averages by Experiment Set (+/-1 {uncertainty_label})",
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
    groups = build_grouped_stats(dataset)
    save_stats(groups, STATS_OUTPUT)
    if args.uncertainty in {"sem", "both"}:
        make_plot(dataset, groups, OUTPUT_SEM, uncertainty="sem", formats=formats)
    if args.uncertainty in {"std", "both"}:
        make_plot(dataset, groups, OUTPUT_STD, uncertainty="std", formats=formats)


if __name__ == "__main__":
    main()
