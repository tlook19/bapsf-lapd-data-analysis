"""Consecutive-shot step screen: level steps inside a run, at fixed position.

WHY
    A Langmuir current channel can change STATE part-way through a run -- its
    recorded level steps and stays stepped -- while the plasma, the other
    channel and the reference photodiode do not move.  The level either side is
    plausible, so nothing downstream notices; the digitizer rail only shows up
    if the stepped level happens to overrun the converter.  This screen looks
    for the step itself, over every run of the manifest and both Langmuir
    current channels, so that a state is found where it exists rather than only
    where it railed.  The registrations it feeds are in
    scripts/annotate_state_mask.py.

THE STATISTIC
    Per shot: the plateau-window signal, i.e. the mean over the CLIP_S-trimmed
    inter-sweep dead-time samples of the PLATEAU_MS cycles, in volts through the
    shot's own SIS Scale/Offset, minus the shot's own tail DC level (the last
    TAIL_S of the record, after the plasma is gone).  Per consecutive pair of
    shots AT THE SAME POSITION -- the probe does not move within a position, so
    the profile is held fixed and the ratio is a pure level ratio -- the screen
    reports ``ln(s[j+1] / s[j])`` and flags ``|ln ratio| > ln(LN_RATIO_FLAG)``.

WHAT THE BARE THRESHOLD DOES AND DOES NOT SEPARATE
    The threshold is a level criterion, not a significance criterion, and the
    shot-to-shot ratio is noise-dominated wherever the plateau signal approaches
    the record's noise floor: the outermost positions of every run, and the
    I_SWEEP channel over most of its record, where the dead-time current is a
    few millivolts against a tail noise of comparable size.  It therefore flags
    pairs on every run and both channels, and the per-pair flag count is a
    census, not a verdict.  Reading it as a verdict requires either a
    significance criterion or a restriction to positions carrying signal, and
    this screen imposes neither: it banks the full table.

    A step that falls exactly on a position boundary is INVISIBLE here by
    construction, because no pair spans two positions.  Run 43's ISAT step up at
    shot 240 is such a step (the boundary from x = -14 cm to x = -13 cm); its
    step back down at shot 696 falls inside a position and is seen.  The
    boundary-crossing ratio is reported alongside, as a separate column, and is
    NOT flagged: it carries the profile move between the two positions as well
    as any level step, and the two are not separable from one channel alone.

Writes a CSV of every position-internal pair whose ratio is flagged, a CSV of
the per-(run, channel) summary, and a JSON summary.  Reads every manifest run on
both channels, so it is IO-bound and takes minutes.  Needs h5py and numpy: the
environment.yml env, not the dead ./.venv.

Usage:
    python screen_consecutive_shot_steps.py <repo-root> <out-prefix>

writes ``<out-prefix>_flags.csv``, ``<out-prefix>_summary.csv`` and
``<out-prefix>_summary.json``.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd import LapdDataset, ChannelKind          # noqa: E402
from bapsf_lapd.density import inter_sweep_sample_slices  # noqa: E402

CLIP_S = 10e-6            # same trim as scripts/plot_isat_profiles.py
PLATEAU_MS = (14.0, 19.0)
TAIL_S = 200e-6
LN_RATIO_FLAG = 1.2

RULE = (
    "consecutive shots at a fixed position; flag |ln(s[j+1]/s[j])| > ln "
    f"{LN_RATIO_FLAG:g}, where s is the plateau-window dead-time signal in volts "
    "minus the shot's own tail level"
)


def plateau_signal_per_shot(run, kind):
    """Tail-removed plateau-window signal of every shot, shaped (position, shot)."""
    cfg = run.config
    ch = cfg.channel(kind)
    sw, acq = cfg.sweep, cfg.acquisition
    dead = inter_sweep_sample_slices(sw, acq, clip_s=CLIP_S)
    t_ms = np.array([
        (sw.t0_s + k * sw.tau_cycle_s + 0.5 * (sw.tau_ramp_s + sw.tau_cycle_s)) * 1e3
        for k in range(sw.n_cycles)
    ])
    plateau = [k for k, t in enumerate(t_ms) if PLATEAU_MS[0] <= t <= PLATEAU_MS[1]]
    lo = min(dead[k].start for k in plateau)
    hi = max(dead[k].stop for k in plateau)
    idx = np.concatenate(
        [np.arange(dead[k].start, dead[k].stop) for k in plateau]
    ) - lo
    n_tail = int(round(TAIL_S / acq.sample_dt_s))

    with run.open() as h5:
        data = h5[ch.hdf5_path]
        headers = h5[ch.hdf5_path + " headers"]
        scale = headers["Scale"][()].astype(np.float64)
        offset = headers["Offset"][()].astype(np.float64)
        block = data[:, lo:hi].astype(np.float64)
        tail = data[:, -n_tail:].astype(np.float64)

    signal = block[:, idx].mean(axis=1) * scale + offset
    tail_level = tail.mean(axis=1) * scale + offset
    n_shots = run.shots_per_position()
    return (signal - tail_level).reshape(run.position_count(), n_shots)


def screen_run_channel(signal):
    """Position-internal and boundary-crossing consecutive-shot ln ratios."""
    magnitude = np.abs(signal)
    with np.errstate(all="ignore"):
        within = np.log(magnitude[:, 1:] / magnitude[:, :-1])
        across = np.log(magnitude[1:, 0] / magnitude[:-1, -1])
    flagged = np.isfinite(within) & (np.abs(within) > np.log(LN_RATIO_FLAG))
    return within, across, flagged


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="repository root holding config/may2026_run_manifest.toml "
             "and the raw run directory it names",
    )
    parser.add_argument(
        "out_prefix",
        type=Path,
        help="path prefix for the flag CSV, the summary CSV and the summary JSON",
    )
    args = parser.parse_args(argv)

    ds = LapdDataset.from_manifest(args.repo_root / "config/may2026_run_manifest.toml")
    run_ids = list(ds.run_ids())

    flag_rows = []
    summary_rows = []
    for run_id in run_ids:
        run = ds.run(run_id)
        for kind in (ChannelKind.ISAT, ChannelKind.I_SWEEP):
            signal = plateau_signal_per_shot(run, kind)
            within, across, flagged = screen_run_channel(signal)
            n_shots = signal.shape[1]
            for position, shot in zip(*np.nonzero(flagged)):
                flag_rows.append({
                    "run_id": run_id,
                    "channel": kind.value,
                    "position": int(position),
                    "shot": int(position) * n_shots + int(shot),
                    "signal_before_v": float(signal[position, shot]),
                    "signal_after_v": float(signal[position, shot + 1]),
                    "ln_ratio": float(within[position, shot]),
                })
            finite_within = within[np.isfinite(within)]
            finite_across = across[np.isfinite(across)]
            summary_rows.append({
                "run_id": run_id,
                "channel": kind.value,
                "median_abs_signal_v": float(np.median(np.abs(signal))),
                "pairs_screened": int(np.isfinite(within).sum()),
                "pairs_flagged": int(flagged.sum()),
                "max_abs_ln_ratio": float(np.abs(finite_within).max()),
                "max_abs_ln_ratio_boundary_unflagged": float(
                    np.abs(finite_across).max()
                ),
            })
            print(f"screened run {run_id} ({kind.value}): "
                  f"{int(flagged.sum())} of {int(np.isfinite(within).sum())} "
                  f"position-internal pairs flagged, "
                  f"max |ln ratio| {np.abs(finite_within).max():.3f}", flush=True)

    flags_csv = args.out_prefix.with_name(args.out_prefix.name + "_flags.csv")
    summary_csv = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    summary_json = args.out_prefix.with_name(args.out_prefix.name + "_summary.json")
    flags_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(flags_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flag_rows[0]))
        writer.writeheader()
        writer.writerows(flag_rows)
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_json.write_text(json.dumps(
        {"rule": RULE,
         "clip_s": CLIP_S,
         "plateau_ms": list(PLATEAU_MS),
         "tail_s": TAIL_S,
         "ln_ratio_flag": LN_RATIO_FLAG,
         "runs": run_ids,
         "summary": summary_rows},
        indent=2, default=str,
    ))
    print(f"\nwrote {flags_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_json}")


if __name__ == "__main__":
    main()
