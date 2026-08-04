"""Plot normalized x=0 upstream-face decay traces by experiment set."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd import ChannelKind, LapdDataset, effective_rotation_deg, electrical_connections_swapped
from bapsf_lapd.filtering import butterworth_lowpass
from plot_export import configure_latex_friendly_matplotlib, parse_formats, save_figure
from plot_labels import experiment_title


MANIFEST = Path("config/may2026_run_manifest.toml")
ZERO_OFFSETS = Path("processed/trace_zero_offsets.toml")
OUTPUT = Path("figures/x0_upstream_decay_by_es_100khz.png")

X_CM = np.linspace(-25.0, 25.0, 51)
X_TARGET_CM = 0.0
T0_S = 20.0e-3
T1_S = 23.0e-3
FILTER_PAD_S = 0.1e-3
YLIM = (0.0, 1.3)

CHANNEL_STYLES = {
    ChannelKind.I_SWEEP: {
        "label": r"$-I_{\mathrm{SWEEP}}$",
        "linestyle": ":",
        "polarity": -1.0,
    },
    ChannelKind.ISAT: {
        "label": r"$I_{\mathrm{SAT}}$",
        "linestyle": "-",
        "polarity": 1.0,
    },
}


@dataclass(frozen=True)
class DecayTrace:
    run_id: str
    port: int
    effective_rotation_deg: float
    channel: ChannelKind
    upstream: bool
    time_ms: np.ndarray
    normalized_current: np.ndarray


def _x_index_for_target(x_cm: np.ndarray, target_cm: float) -> int:
    """Return the nearest scan index for a requested x location."""
    idx = int(np.argmin(np.abs(x_cm - target_cm)))
    if not np.isclose(x_cm[idx], target_cm):
        raise ValueError(f"x={target_cm:g} cm is not present in the scan grid")
    return idx


def _sample_window(run, start_s: float, stop_s: float) -> slice:
    """Return a sample slice including both requested endpoint times."""
    dt_s = run.sample_dt_s()
    start = int(round(start_s / dt_s))
    stop = int(round(stop_s / dt_s)) + 1
    return slice(start, stop)


def _padded_sample_window(run, start_s: float, stop_s: float, pad_s: float) -> tuple[slice, slice]:
    """Return a padded read slice plus the plotted sub-slice inside it."""
    sample = _sample_window(run, start_s, stop_s)
    pad_samples = int(round(pad_s / run.sample_dt_s()))
    padded_start = max(0, sample.start - pad_samples)
    padded_stop = sample.stop + pad_samples
    crop = slice(sample.start - padded_start, sample.stop - padded_start)
    return slice(padded_start, padded_stop), crop


def _is_upstream_channel(
    run_id: str,
    port: int | None,
    effective_rotation: float,
    channel: ChannelKind,
) -> bool:
    """Return whether a plotted electrical channel is the upstream-facing probe face."""
    del port  # reserved for future geometry-specific wiring exceptions
    if electrical_connections_swapped(run_id):
        return channel == ChannelKind.ISAT
    if effective_rotation == 0:
        return channel == ChannelKind.I_SWEEP
    return channel == ChannelKind.ISAT


def _load_zero_offsets(path: Path) -> dict[str, dict[str, float]]:
    """Load cached pre-calibration channel offsets, keyed by run ID and channel value."""
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        data = tomllib.load(f)
    offsets: dict[str, dict[str, float]] = {}
    for run_id, channel_data in data.get("runs", {}).items():
        offsets[run_id] = {
            channel: float(values["offset_v"])
            for channel, values in channel_data.items()
            if isinstance(values, dict) and "offset_v" in values
        }
    return offsets


def _zero_offset_v(run, channel: ChannelKind, zero_offsets: dict[str, dict[str, float]]) -> float:
    """Return a cached zero offset when present, otherwise compute from the HDF5 trace tail."""
    cached = zero_offsets.get(run.config.run_id, {}).get(channel.value)
    if cached is not None:
        return cached
    return run.default_zero_offset_v(channel)


def _normalized_trace(
    run,
    channel: ChannelKind,
    sample: slice,
    x_index: int,
    zero_offsets: dict[str, dict[str, float]],
    *,
    crop: slice | None = None,
    cutoff_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return time in ms and the x=0 shot-average trace normalized at t=20 ms."""
    style = CHANNEL_STYLES[channel]
    channel_config = run.config.channel(channel)
    shot_start = run.flat_shot_index(x_index, 0)
    shot_stop = shot_start + run.shots_per_position()
    with run.open() as h5:
        raw = h5[channel_config.hdf5_path][shot_start:shot_stop, sample].astype(np.float64)
        headers = h5[f"{channel_config.hdf5_path} headers"][shot_start:shot_stop]

    voltage = raw * headers["Scale"][:, None] + headers["Offset"][:, None]
    zero_offset_v = _zero_offset_v(run, channel, zero_offsets)
    if zero_offset_v:
        voltage = voltage - zero_offset_v
    current = channel_config.apply_calibration(voltage)
    if cutoff_hz is not None:
        current = butterworth_lowpass(
            current,
            sample_rate_hz=run.sample_rate_hz(),
            cutoff_hz=cutoff_hz,
            axis=-1,
        )
    if crop is not None:
        current = current[:, crop]
    shot_average = style["polarity"] * current.mean(axis=0)
    norm = float(shot_average[0])
    if not np.isfinite(norm) or np.isclose(norm, 0.0):
        raise ValueError(f"Cannot normalize {channel.value} for run {run.config.run_id}: value at 20 ms is {norm}")
    start_index = sample.start if crop is None else sample.start + crop.start
    time_ms = run.time_axis(shot_average.size, start_index=start_index) * 1e3
    return time_ms, shot_average / norm


def collect_decay_traces(
    dataset: LapdDataset,
    experiment_set_id: int,
    zero_offsets: dict[str, dict[str, float]],
    *,
    cutoff_hz: float | None,
) -> list[DecayTrace]:
    """Collect normalized x=0 upstream traces for one experiment set."""
    x_index = _x_index_for_target(X_CM, X_TARGET_CM)
    traces: list[DecayTrace] = []

    for run_id in dataset.experiment_set_run_ids(experiment_set_id):
        run = dataset.run(run_id)
        if cutoff_hz is None:
            sample = _sample_window(run, T0_S, T1_S)
            crop = None
        else:
            sample, crop = _padded_sample_window(run, T0_S, T1_S, FILTER_PAD_S)
        recorded_rotation = float(run.config.probe.rotation_deg or 0.0)
        effective_rotation = effective_rotation_deg(run_id, recorded_rotation)
        port = run.config.probe.port

        print(
            f"  ES{experiment_set_id} run {run_id}: port {int(port or 0)} "
            f"rot={effective_rotation:.0f} x=0 upstream decay"
        )
        for channel in (ChannelKind.I_SWEEP, ChannelKind.ISAT):
            upstream = _is_upstream_channel(run_id, port, effective_rotation, channel)
            if not upstream:
                continue
            time_ms, normalized = _normalized_trace(
                run,
                channel,
                sample,
                x_index,
                zero_offsets,
                crop=crop,
                cutoff_hz=cutoff_hz,
            )
            traces.append(
                DecayTrace(
                    run_id=run_id,
                    port=int(port or 0),
                    effective_rotation_deg=effective_rotation,
                    channel=channel,
                    upstream=upstream,
                    time_ms=time_ms,
                    normalized_current=normalized,
                )
            )

    return traces


def _output_for_experiment_set(output: Path, experiment_set_id: int) -> Path:
    """Return a per-experiment-set output path based on a common base path."""
    return output.with_name(f"{output.stem}_es{experiment_set_id}{output.suffix}")


def _port_colors(ports: list[int]) -> dict[int, tuple[float, float, float, float]]:
    """Assign stable Matplotlib tab colors to the ports in one experiment set."""
    cmap = plt.get_cmap("tab10")
    return {port: cmap(i % cmap.N) for i, port in enumerate(sorted(set(ports)))}


def make_plots(
    output: Path,
    *,
    formats: tuple[str, ...] = ("png",),
    cutoff_hz: float | None = 100e3,
) -> None:
    """Create one upstream-face decay plot per experiment set."""
    dataset = LapdDataset.from_manifest(MANIFEST)
    zero_offsets = _load_zero_offsets(ZERO_OFFSETS)

    for experiment_set_id in dataset.experiment_set_ids():
        fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
        run_ids = dataset.experiment_set_run_ids(experiment_set_id)
        experiment = dataset.config(run_ids[0]).experiment_set
        traces = collect_decay_traces(dataset, experiment_set_id, zero_offsets, cutoff_hz=cutoff_hz)
        colors = _port_colors([trace.port for trace in traces])

        for trace in traces:
            style = CHANNEL_STYLES[trace.channel]
            ax.plot(
                trace.time_ms,
                trace.normalized_current,
                color=colors[trace.port],
                linestyle=style["linestyle"],
                linewidth=1.35,
                alpha=0.82,
            )

        ax.set_title(experiment_title(experiment_set_id, experiment), fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.axhline(1.0, color="0.35", linewidth=0.8, alpha=0.45)
        ax.set_xlim(T0_S * 1e3, T1_S * 1e3)
        ax.set_ylim(*YLIM)
        ax.set_ylabel("Upstream current / upstream current at 20 ms")
        ax.set_xlabel("Time (ms)")

        handles = [
            mlines.Line2D([], [], color=colors[port], lw=1.8, label=f"p{port}")
            for port in sorted(colors)
        ]
        handles.extend(
            [
                mlines.Line2D([], [], color="0.25", lw=1.8, ls=CHANNEL_STYLES[ChannelKind.I_SWEEP]["linestyle"], label=CHANNEL_STYLES[ChannelKind.I_SWEEP]["label"]),
                mlines.Line2D([], [], color="0.25", lw=1.8, ls=CHANNEL_STYLES[ChannelKind.ISAT]["linestyle"], label=CHANNEL_STYLES[ChannelKind.ISAT]["label"]),
            ]
        )
        ax.legend(handles=handles, title="Port / Channel", loc="best", frameon=False)
        title = f"x=0 ES{experiment_set_id} Upstream-Face Ion-Current Decay"
        if cutoff_hz is not None:
            title += f", {cutoff_hz / 1e3:g} kHz low-pass"
        fig.suptitle(title, fontsize=13)

        save_figure(
            fig,
            _output_for_experiment_set(output, experiment_set_id),
            formats,
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats: png, pdf, svg, latex, vector, or all.",
    )
    parser.add_argument(
        "--cutoff-khz",
        type=float,
        default=100.0,
        help="Optional low-pass cutoff in kHz. Use 0 to disable filtering.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    configure_latex_friendly_matplotlib()
    cutoff_hz = None if args.cutoff_khz <= 0 else args.cutoff_khz * 1e3
    make_plots(args.output, formats=parse_formats(args.formats), cutoff_hz=cutoff_hz)


if __name__ == "__main__":
    main()
