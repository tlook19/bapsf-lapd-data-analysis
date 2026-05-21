"""Fit one Langmuir sweep and plot extraction diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd import (
    ChannelKind,
    LapdDataset,
    analyze_langmuir_sweep,
    butterworth_lowpass,
)
from plot_export import configure_latex_friendly_matplotlib, parse_formats, save_figure


MANIFEST = Path("config/may2026_run_manifest.toml")
OUTPUT = Path("figures/run46_flatshot510_cycle18_langmuir_fit_diagnostic.png")


def _load_cycle(run, channel: ChannelKind, flat_shot: int, cycle: int, clip_s: float) -> np.ndarray:
    ramp_slice = run.sweep_ramp_sample_slices(clip_s=clip_s)[cycle]
    return run.trace(channel, flat_shot, ramp_slice).astype(np.float64)


def _load_filtered_sweep(
    *,
    run_id: str,
    flat_shot: int,
    cycle: int,
    clip_us: float,
    cutoff_khz: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = LapdDataset.from_manifest(MANIFEST)
    run = dataset.run(run_id)
    clip_s = clip_us * 1e-6
    current = _load_cycle(run, ChannelKind.I_SWEEP, flat_shot, cycle, clip_s)
    voltage = _load_cycle(run, ChannelKind.V_SWEEP, flat_shot, cycle, clip_s)
    current = butterworth_lowpass(
        current,
        sample_rate_hz=run.sample_rate_hz(),
        cutoff_hz=cutoff_khz * 1e3,
        order=order,
    )
    voltage = butterworth_lowpass(
        voltage,
        sample_rate_hz=run.sample_rate_hz(),
        cutoff_hz=cutoff_khz * 1e3,
        order=order,
    )
    return voltage, current


def _plot_regions(ax, analysis) -> None:
    v = analysis.voltage
    i = analysis.current
    ax.scatter(v[analysis.ion_fit.mask], i[analysis.ion_fit.mask], s=8, alpha=0.6, label="ion fit region")
    ax.scatter(
        v[analysis.log_linear_fit.mask],
        i[analysis.log_linear_fit.mask],
        s=8,
        alpha=0.6,
        label="retarding fit region",
    )
    ax.scatter(
        v[analysis.saturation_fit.mask],
        i[analysis.saturation_fit.mask],
        s=8,
        alpha=0.6,
        label="saturation fit region",
    )


def make_plot(
    *,
    run_id: str,
    flat_shot: int,
    cycle: int,
    clip_us: float,
    cutoff_khz: float,
    order: int,
    output: Path,
    formats: tuple[str, ...],
) -> None:
    voltage, current = _load_filtered_sweep(
        run_id=run_id,
        flat_shot=flat_shot,
        cycle=cycle,
        clip_us=clip_us,
        cutoff_khz=cutoff_khz,
        order=order,
    )
    analysis = analyze_langmuir_sweep(voltage, current)
    v = analysis.voltage
    i = analysis.current
    ion = analysis.ion_fit.evaluate(v)
    sat = analysis.saturation_fit.evaluate(v)
    fit_mask = analysis.log_linear_fit.mask
    v_fit = v[fit_mask]
    i_fit = i[fit_mask]
    log_model_fit = analysis.ion_fit.evaluate(v_fit) + analysis.log_linear_fit.electron_current(v_fit)
    exp_model_fit = analysis.exponential_fit.evaluate(v_fit)

    fit_padding = max(0.75, 0.35 * (float(v_fit.max()) - float(v_fit.min())))
    zoom_mask = (v >= v_fit.min() - fit_padding) & (v <= v_fit.max() + fit_padding)
    v_zoom = v[zoom_mask]
    i_zoom = i[zoom_mask]
    log_model_zoom = analysis.ion_fit.evaluate(v_zoom) + analysis.log_linear_fit.electron_current(v_zoom)
    exp_model_zoom = analysis.exponential_fit.evaluate(v_zoom)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(v, i, color="black", lw=1.0, label="filtered I-V")
    _plot_regions(ax, analysis)
    ax.plot(v, ion, color="tab:blue", lw=1.3, label="ion line")
    ax.plot(v, sat, color="tab:green", lw=1.3, label="saturation line")
    ax.axvspan(v_fit.min(), v_fit.max(), color="tab:orange", alpha=0.12, label="fit window")
    ax.axvline(analysis.plasma_potential_derivative_v, color="0.3", lw=1.0, ls=":", label=r"$V_p$ deriv.")
    if analysis.plasma_potential_log_intersection_v is not None:
        ax.axvline(
            analysis.plasma_potential_log_intersection_v,
            color="tab:orange",
            lw=1.0,
            ls=":",
            label=r"$V_p$ log int.",
        )
    if analysis.plasma_potential_exp_intersection_v is not None:
        ax.axvline(
            analysis.plasma_potential_exp_intersection_v,
            color="tab:red",
            lw=1.0,
            ls=":",
            label=r"$V_p$ exp int.",
        )
    ax.set_xlabel(r"$V_\mathrm{sweep}$ (V)")
    ax.set_ylabel(r"$I_\mathrm{sweep}$ (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[0, 1]
    ax.plot(v_zoom, i_zoom, color="black", lw=1.2, label="filtered I-V")
    ax.scatter(v_fit, i_fit, s=18, color="tab:orange", zorder=3, label="fit samples")
    ax.plot(v_zoom, log_model_zoom, color="tab:orange", lw=1.8, label="log-linear model")
    ax.plot(v_zoom, exp_model_zoom, color="tab:red", lw=1.5, ls="--", label="direct exponential")
    ax.axvline(analysis.plasma_potential_derivative_v, color="0.3", lw=1.0, ls=":", label=r"$V_p$ deriv.")
    if analysis.plasma_potential_log_intersection_v is not None:
        ax.axvline(analysis.plasma_potential_log_intersection_v, color="tab:orange", lw=1.0, ls=":")
    if analysis.plasma_potential_exp_intersection_v is not None:
        ax.axvline(analysis.plasma_potential_exp_intersection_v, color="tab:red", lw=1.0, ls=":")
    ax.set_title("Retarding-Region Zoom")
    ax.set_xlabel(r"$V_\mathrm{sweep}$ (V)")
    ax.set_ylabel(r"$I_\mathrm{sweep}$ (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    electron_current = i - ion
    positive = electron_current > 0
    ax.plot(v[positive], np.log(electron_current[positive]), color="0.5", lw=1.0, label=r"$\ln(I_e)$")
    ax.scatter(
        v[fit_mask],
        np.log(electron_current[fit_mask]),
        s=12,
        color="tab:orange",
        label="fit region",
    )
    ax.plot(
        v_fit,
        analysis.log_linear_fit.intercept + analysis.log_linear_fit.slope * v_fit,
        color="tab:red",
        lw=1.5,
        label="robust line",
    )
    ax.set_xlabel(r"$V_\mathrm{sweep}$ (V)")
    ax.set_ylabel(r"$\ln(I_e)$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.plot(v_fit, i_fit - log_model_fit, color="tab:orange", lw=1.2, label="log-linear residual")
    ax.plot(v_fit, i_fit - exp_model_fit, color="tab:red", lw=1.2, label="direct exp residual")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel(r"$V_\mathrm{sweep}$ (V)")
    ax.set_ylabel("Residual (A)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    vp_log = "none" if analysis.plasma_potential_log_intersection_v is None else f"{analysis.plasma_potential_log_intersection_v:.2f} V"
    vp_exp = "none" if analysis.plasma_potential_exp_intersection_v is None else f"{analysis.plasma_potential_exp_intersection_v:.2f} V"
    fig.suptitle(
        (
            f"Run {run_id}, flat shot {flat_shot}, cycle {cycle}: "
            f"Te log={analysis.log_linear_fit.electron_temperature_ev:.2f} eV, "
            f"Te exp={analysis.exponential_fit.electron_temperature_ev:.2f} eV, "
            f"Vp deriv={analysis.plasma_potential_derivative_v:.2f} V, "
            f"Vp log={vp_log}, Vp exp={vp_exp}"
        ),
        fontsize=12,
    )
    save_figure(fig, output, formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="46")
    parser.add_argument("--flat-shot", type=int, default=510)
    parser.add_argument("--cycle", type=int, default=18)
    parser.add_argument("--clip-us", type=float, default=10.0)
    parser.add_argument("--cutoff-khz", type=float, default=100.0)
    parser.add_argument("--order", type=int, default=4)
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
        flat_shot=args.flat_shot,
        cycle=args.cycle,
        clip_us=args.clip_us,
        cutoff_khz=args.cutoff_khz,
        order=args.order,
        output=args.output,
        formats=parse_formats(args.formats),
    )


if __name__ == "__main__":
    main()
