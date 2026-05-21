"""Generate Langmuir sweep quality-control reports.

Iterates over one or more runs from the May 2026 manifest and writes one CSV
row per (run, flat_shot, cycle) to the output file.  Pass --run-ids all to
process every run in the manifest in a single invocation; the CSV is written
incrementally so a full-dataset run does not hold all rows in memory at once.

Usage examples
--------------
# One run, default sampled shots and one cycle (fast smoke-test)
python scripts/process_langmuir_quality_report.py --run-ids 46

# All runs, five evenly-spaced shots, two cycles
python scripts/process_langmuir_quality_report.py \\
    --run-ids all --flat-shots "0,255,510,765,1019" --cycles "0,10"

# Full analysis -- all runs, all positions, all shots, all cycles (long-running)
python scripts/process_langmuir_quality_report.py \\
    --run-ids all --flat-shots all --cycles all
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bapsf_lapd import (
    ChannelKind,
    LapdDataset,
    LapdRun,
    analyze_langmuir_sweep,
    butterworth_lowpass,
    evaluate_langmuir_quality,
)


MANIFEST = Path("config/may2026_run_manifest.toml")
CSV_OUTPUT = Path("processed/langmuir_quality_report.csv")


@dataclass(frozen=True)
class QualityRow:
    run_id: str
    flat_shot: int
    position_index: int
    shot_index: int
    cycle: int
    severity: str
    flag_codes: str
    te_log_ev: float | None
    te_exp_ev: float | None
    vp_derivative_v: float | None
    vp_log_v: float | None
    vp_exp_v: float | None
    fit_window_width_v: float | None
    fit_sample_count: int | None
    rms_residual_a: float | None
    max_abs_residual_a: float | None
    max_jump_a: float | None
    arc_like_sample_count: int | None
    error: str


def _parse_indices(value: str, *, max_value: int | None = None) -> list[int]:
    if value == "all":
        if max_value is None:
            raise ValueError("max_value is required for 'all'")
        return list(range(max_value))
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            start, stop, *rest = item.split(":")
            step = int(rest[0]) if rest else 1
            indices.extend(range(int(start), int(stop), step))
        else:
            indices.append(int(item))
    return sorted(dict.fromkeys(indices))


def _parse_run_ids(value: str, dataset: LapdDataset) -> list[str]:
    """Return a sorted list of run IDs from a comma-separated string or 'all'."""
    if value == "all":
        return dataset.run_ids()
    return sorted(dict.fromkeys(v.strip() for v in value.split(",") if v.strip()))


def _load_cycle(run: LapdRun, channel: ChannelKind, flat_shot: int, cycle: int, clip_s: float) -> np.ndarray:
    ramp_slice = run.sweep_ramp_sample_slices(clip_s=clip_s)[cycle]
    return run.trace(channel, flat_shot, ramp_slice).astype(np.float64)


def _filtered_cycle(
    run: LapdRun, flat_shot: int, cycle: int, clip_us: float, cutoff_khz: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
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


def _peer_currents(
    run: LapdRun, flat_shot: int, cycle: int, clip_us: float, cutoff_khz: float, order: int
) -> np.ndarray:
    position_index = flat_shot // run.shots_per_position()
    clip_s = clip_us * 1e-6
    ramp_slice = run.sweep_ramp_sample_slices(clip_s=clip_s)[cycle]
    traces = []
    for shot_index in range(run.shots_per_position()):
        peer_flat_shot = run.flat_shot_index(position_index, shot_index)
        traces.append(run.trace(ChannelKind.I_SWEEP, peer_flat_shot, ramp_slice).astype(np.float64))
    traces = np.stack(traces, axis=0)
    return butterworth_lowpass(
        traces,
        sample_rate_hz=run.sample_rate_hz(),
        cutoff_hz=cutoff_khz * 1e3,
        order=order,
        axis=-1,
    )


def build_rows(
    *,
    run: LapdRun,
    run_id: str,
    flat_shots: list[int],
    cycles: list[int],
    clip_us: float,
    cutoff_khz: float,
    order: int,
) -> list[QualityRow]:
    """Analyze every requested (flat_shot, cycle) pair for one run.

    Accepts a pre-loaded LapdRun so that callers iterating over multiple runs
    can manage dataset loading themselves without redundant HDF5 opens.
    """
    rows: list[QualityRow] = []
    for flat_shot in flat_shots:
        position_index = flat_shot // run.shots_per_position()
        shot_index = flat_shot % run.shots_per_position()
        for cycle in cycles:
            try:
                voltage, current = _filtered_cycle(run, flat_shot, cycle, clip_us, cutoff_khz, order)
                peer_current = _peer_currents(run, flat_shot, cycle, clip_us, cutoff_khz, order)
                analysis = analyze_langmuir_sweep(voltage, current)
                report = evaluate_langmuir_quality(
                    analysis,
                    current=current,
                    peer_current=peer_current,
                )
                rows.append(
                    QualityRow(
                        run_id=run_id,
                        flat_shot=flat_shot,
                        position_index=position_index,
                        shot_index=shot_index,
                        cycle=cycle,
                        severity=report.severity,
                        flag_codes=";".join(flag.code for flag in report.flags),
                        te_log_ev=analysis.log_linear_fit.electron_temperature_ev,
                        te_exp_ev=analysis.exponential_fit.electron_temperature_ev,
                        vp_derivative_v=analysis.plasma_potential_derivative_v,
                        vp_log_v=analysis.plasma_potential_log_intersection_v,
                        vp_exp_v=analysis.plasma_potential_exp_intersection_v,
                        fit_window_width_v=report.fit_window_width_v,
                        fit_sample_count=report.fit_sample_count,
                        rms_residual_a=report.rms_residual_a,
                        max_abs_residual_a=report.max_abs_residual_a,
                        max_jump_a=report.max_jump_a,
                        arc_like_sample_count=report.arc_like_sample_count,
                        error="",
                    )
                )
            except Exception as exc:
                rows.append(
                    QualityRow(
                        run_id=run_id,
                        flat_shot=flat_shot,
                        position_index=position_index,
                        shot_index=shot_index,
                        cycle=cycle,
                        severity="bad",
                        flag_codes="analysis_failed",
                        te_log_ev=None,
                        te_exp_ev=None,
                        vp_derivative_v=None,
                        vp_log_v=None,
                        vp_exp_v=None,
                        fit_window_width_v=None,
                        fit_sample_count=None,
                        rms_residual_a=None,
                        max_abs_residual_a=None,
                        max_jump_a=None,
                        arc_like_sample_count=None,
                        error=str(exc),
                    )
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run-ids",
        default="46",
        help="Comma-separated run IDs like 01,46 or 'all' for every run in the manifest.",
    )
    parser.add_argument(
        "--flat-shots",
        default="0,255,510,765,1019",
        help="Comma list/ranges like 0,510 or 0:1020:20, or 'all'.",
    )
    parser.add_argument("--cycles", default="18", help="Comma list/ranges like 0,18 or 'all'.")
    parser.add_argument("--clip-us", type=float, default=10.0)
    parser.add_argument("--cutoff-khz", type=float, default=100.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--output", type=Path, default=CSV_OUTPUT)
    args = parser.parse_args()

    dataset = LapdDataset.from_manifest(MANIFEST)
    run_ids = _parse_run_ids(args.run_ids, dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_counts: dict[str, int] = {"ok": 0, "warn": 0, "bad": 0}
    writer: csv.DictWriter | None = None

    with args.output.open("w", newline="") as f:
        for run_id in run_ids:
            run = dataset.run(run_id)
            flat_shots = _parse_indices(args.flat_shots, max_value=run.expected_flat_shot_count())
            cycles = _parse_indices(args.cycles, max_value=run.config.sweep.n_cycles)
            rows = build_rows(
                run=run,
                run_id=run_id,
                flat_shots=flat_shots,
                cycles=cycles,
                clip_us=args.clip_us,
                cutoff_khz=args.cutoff_khz,
                order=args.order,
            )
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(QualityRow.__dataclass_fields__))
                writer.writeheader()
            if writer is not None:
                for row in rows:
                    writer.writerow(row.__dict__)
            f.flush()
            counts = {s: sum(r.severity == s for r in rows) for s in ("ok", "warn", "bad")}
            for s in counts:
                total_counts[s] += counts[s]
            sys.stdout.write(f"run {run_id}: {counts}\n")

    sys.stdout.write(f"\nWrote {args.output}  ({sum(total_counts.values())} rows)\n")
    sys.stdout.write(f"Total severity counts: {total_counts}\n")


if __name__ == "__main__":
    main()
