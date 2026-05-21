"""Compare raw and low-pass filtered Langmuir sweep slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd import ChannelKind, LapdDataset, butterworth_lowpass
from plot_export import configure_latex_friendly_matplotlib, parse_formats, save_figure


MANIFEST = Path("config/may2026_run_manifest.toml")
OUTPUT = Path("figures/run46_filter_cutoff_diagnostic.png")


def _parse_csv_numbers(value: str, cast):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _load_ramp_cycles(
    run,
    channel: ChannelKind,
    flat_shot: int,
    *,
    clip_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load configured ramp windows as (cycle, sample) for one flattened shot."""
    ramp_slices = run.sweep_ramp_sample_slices(clip_s=clip_s)
    offsets = {kind: run.default_zero_offset_v(kind) for kind in (ChannelKind.I_SWEEP, ChannelKind.V_SWEEP)}
    cycles = [
        run.trace(channel, flat_shot, ramp_slice, zero_offset_v=offsets[channel]).astype(np.float64)
        for ramp_slice in ramp_slices
    ]
    time_us = run.time_axis(cycles[0].shape[-1]) * 1e6
    return np.stack(cycles, axis=0), time_us


def _mean_cycle(values: np.ndarray, cycle_indices: tuple[int, ...]) -> np.ndarray:
    return values[list(cycle_indices)].mean(axis=0)


def _plot_time_comparison(
    ax,
    time_us: np.ndarray,
    raw: np.ndarray,
    filtered: dict[tuple[int, float], np.ndarray],
) -> None:
    ax.plot(time_us, raw, color="black", lw=1.0, alpha=0.8, label="unfiltered")
    for (order, cutoff_hz), values in filtered.items():
        ax.plot(time_us, values, lw=1.2, label=f"order {order}, {cutoff_hz / 1e3:g} kHz")
    ax.set_xlabel("Ramp time (us)")
    ax.set_ylabel(r"$I_\mathrm{sweep}$ (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def _plot_iv_comparison(
    ax,
    voltage: np.ndarray,
    current: np.ndarray,
    filtered_current: dict[tuple[int, float], np.ndarray],
    filtered_voltage: dict[tuple[int, float], np.ndarray],
) -> None:
    ax.plot(voltage, current, color="black", lw=1.0, alpha=0.75, label="unfiltered")
    for key, current_values in filtered_current.items():
        order, cutoff_hz = key
        ax.plot(
            filtered_voltage[key],
            current_values,
            lw=1.2,
            label=f"order {order}, {cutoff_hz / 1e3:g} kHz",
        )
    ax.set_xlabel(r"$V_\mathrm{sweep}$ (V)")
    ax.set_ylabel(r"$I_\mathrm{sweep}$ (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)


def make_plot(
    *,
    run_id: str,
    position_index: int,
    shot_index: int,
    cycle_indices: tuple[int, ...],
    cutoff_hz_values: tuple[float, ...],
    orders: tuple[int, ...],
    clip_us: float,
    output: Path,
    formats: tuple[str, ...],
) -> None:
    dataset = LapdDataset.from_manifest(MANIFEST)
    run = dataset.run(run_id)
    flat_shot = run.flat_shot_index(position_index, shot_index)

    clip_s = clip_us * 1e-6
    i_cycles, time_us = _load_ramp_cycles(run, ChannelKind.I_SWEEP, flat_shot, clip_s=clip_s)
    v_cycles, _ = _load_ramp_cycles(run, ChannelKind.V_SWEEP, flat_shot, clip_s=clip_s)
    if any(cycle < 0 or cycle >= i_cycles.shape[0] for cycle in cycle_indices):
        raise IndexError(f"cycle indices must be in [0, {i_cycles.shape[0]})")

    raw_current = _mean_cycle(i_cycles, cycle_indices)
    raw_voltage = _mean_cycle(v_cycles, cycle_indices)
    sample_rate_hz = run.sample_rate_hz()

    filtered_current = {}
    filtered_voltage = {}
    for order in orders:
        for cutoff_hz in cutoff_hz_values:
            key = (order, cutoff_hz)
            current_filtered_cycles = butterworth_lowpass(
                i_cycles[list(cycle_indices)],
                sample_rate_hz=sample_rate_hz,
                cutoff_hz=cutoff_hz,
                order=order,
                axis=-1,
            )
            voltage_filtered_cycles = butterworth_lowpass(
                v_cycles[list(cycle_indices)],
                sample_rate_hz=sample_rate_hz,
                cutoff_hz=cutoff_hz,
                order=order,
                axis=-1,
            )
            filtered_current[key] = current_filtered_cycles.mean(axis=0)
            filtered_voltage[key] = voltage_filtered_cycles.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    _plot_time_comparison(axes[0], time_us, raw_current, filtered_current)
    _plot_iv_comparison(axes[1], raw_voltage, raw_current, filtered_current, filtered_voltage)

    cycle_text = ", ".join(str(cycle) for cycle in cycle_indices)
    fig.suptitle(
        (
            f"Run {run_id} Filter Diagnostic: position {position_index}, shot {shot_index}, "
            f"cycles {cycle_text}, clip {clip_us:g} us"
        ),
        fontsize=14,
    )
    save_figure(fig, output, formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="46")
    parser.add_argument("--position-index", type=int, default=25)
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--cycles", default="2,3,4", help="Comma-separated ramp cycle indices to average.")
    parser.add_argument("--clip-us", type=float, default=0.0, help="Clip this many microseconds from each ramp edge.")
    parser.add_argument(
        "--cutoffs-khz",
        default="100,200,500,1000",
        help="Comma-separated cutoff frequencies in kHz.",
    )
    parser.add_argument("--orders", default="2,4", help="Comma-separated Butterworth filter orders.")
    parser.add_argument(
        "--formats",
        default="png",
        help="Comma-separated output formats: png, pdf, svg, latex, vector, or all.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    configure_latex_friendly_matplotlib()
    make_plot(
        run_id=args.run_id,
        position_index=args.position_index,
        shot_index=args.shot_index,
        cycle_indices=_parse_csv_numbers(args.cycles, int),
        cutoff_hz_values=tuple(value * 1e3 for value in _parse_csv_numbers(args.cutoffs_khz, float)),
        orders=_parse_csv_numbers(args.orders, int),
        clip_us=args.clip_us,
        output=args.output,
        formats=parse_formats(args.formats),
    )


if __name__ == "__main__":
    main()
