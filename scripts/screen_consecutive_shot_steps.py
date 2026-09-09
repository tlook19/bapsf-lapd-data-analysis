"""Step screen: level steps inside a run that the plasma does not share.

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

THE PER-SHOT SIGNAL AND THE NOISE IT IS MEASURED AGAINST
    Per shot, the plateau-window signal is the mean over the CLIP_S-trimmed
    inter-sweep dead-time samples of the PLATEAU_MS cycles, in volts through the
    shot's own SIS Scale/Offset, minus the shot's own tail level (the last
    TAIL_S of the record, after the plasma is gone).  The run's noise level on
    that channel is the median over shots of the tail sample standard deviation,
    in the same volts -- the record's own noise floor, which differs by a factor
    of two between the padded and unpadded channel settings and is therefore
    taken per (run, channel) rather than assumed.

    A sample is ELIGIBLE when its |signal| exceeds SIGNIFICANCE_FACTOR times
    that noise level.  Every comparison below is restricted to eligible
    samples: without that restriction the ratio of two near-zero numbers is a
    ratio of noise, and a bare level threshold flags the outermost positions of
    every run and most of the I_SWEEP record.

LEG 1 -- FIXED POSITION
    Consecutive shots at the SAME position: the probe does not move within a
    position, so the profile is held fixed and the ratio is a pure level ratio.
    Flags ``|ln(s[j+1] / s[j])| > ln(LN_RATIO_FLAG)`` on eligible pairs.

    It does not separate a channel-level change from a plasma-level one: a
    plasma that changes between two consecutive shots moves the recorded signal
    exactly as a channel that changes does, and one channel's own record holds
    nothing that tells them apart.

    A step that falls exactly on a position boundary is invisible to this leg by
    construction, because no pair spans two positions.  Leg 2 is what reaches it.

LEG 2 -- POSITION BOUNDARY, TWO-CHANNEL
    Across a position boundary the profile moves, and a single-channel ratio
    cannot tell a profile move from a level step.  The TWO-CHANNEL ratio removes
    the part of that move COMMON to the two channels: both are sampled at the
    same position at the same time, so a change that scales both alike cancels
    in ISAT/I_SWEEP while a level step on one channel does not.  Per boundary,
    the screen takes block means of the last BLOCK_SHOTS shots of the lower
    position and the first BLOCK_SHOTS of the upper, forms
    ``R = |ISAT| / |I_SWEEP|`` on each side, and flags
    ``|ln(R_upper / R_lower)| > ln(LN_RATIO_FLAG)``.

    Only the common part cancels.  The two channels are the probe's upstream and
    downstream faces -- the pair whose ratio IS the Mach signal -- so their
    DIFFERENTIAL response to x survives: at a boundary where the two faces'
    profiles differ in slope, R steps without either channel having changed
    state.  That is what a boundary flagged at the same x in several runs is,
    and it is not separable here from a per-run channel state.

    The step is ATTRIBUTED to the channel whose own single-channel boundary
    ratio departs the further from unity: with a profile move ``p`` common to
    both channels and a level step ``f`` on one of them, that channel's ratio is
    ``p*f`` and the other's is ``p``, so the larger |ln| names the channel that
    moved.  Both single-channel ratios are reported beside the flag, so the
    attribution can be checked rather than trusted.

Writes a CSV of every flagged pair for each leg, a per-(run, channel) summary
CSV and a JSON summary.  Reads every manifest run on both channels, so it is
IO-bound and takes minutes.  Needs h5py and numpy: the environment.yml env, not
the dead ./.venv.

Usage:
    python screen_consecutive_shot_steps.py <repo-root> <out-prefix>

writes ``<out-prefix>_fixed_position_flags.csv``,
``<out-prefix>_boundary_flags.csv``, ``<out-prefix>_summary.csv`` and
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
SIGNIFICANCE_FACTOR = 5.0
BLOCK_SHOTS = 4

FIXED_POSITION_RULE = (
    "consecutive shots at a fixed position, both eligible; flag "
    f"|ln(s[j+1]/s[j])| > ln {LN_RATIO_FLAG:g}, where s is the plateau-window "
    "dead-time signal in volts minus the shot's own tail level"
)
BOUNDARY_RULE = (
    f"across a position boundary, block means of the last {BLOCK_SHOTS} shots "
    f"below and the first {BLOCK_SHOTS} above, all four eligible; flag "
    f"|ln(R_upper/R_lower)| > ln {LN_RATIO_FLAG:g} on the two-channel ratio "
    "R = |ISAT|/|I_SWEEP|, in which the profile move cancels; attributed to the "
    "channel whose own boundary ratio departs the further from unity"
)
ELIGIBILITY_RULE = (
    f"|signal| > {SIGNIFICANCE_FACTOR:g} x the run's own tail noise level on "
    "that channel (median over shots of the tail sample standard deviation, in "
    "volts)"
)


def plateau_signal_and_noise(run, kind):
    """Tail-removed plateau signal per (position, shot), and the noise level.

    The signal is in volts; the noise level is the median over shots of the tail
    sample standard deviation in the same volts.
    """
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
    noise = float(np.median(tail.std(axis=1) * scale))
    shape = (run.position_count(), run.shots_per_position())
    return (signal - tail_level).reshape(shape), noise


def eligible_samples(signal, noise):
    """True where a sample's magnitude clears the significance floor."""
    return np.abs(signal) > SIGNIFICANCE_FACTOR * noise


def fixed_position_leg(signal, noise):
    """Position-internal consecutive-shot ln ratios, and which are flagged."""
    eligible = eligible_samples(signal, noise)
    magnitude = np.abs(signal)
    with np.errstate(all="ignore"):
        ratios = np.log(magnitude[:, 1:] / magnitude[:, :-1])
    pair_eligible = eligible[:, 1:] & eligible[:, :-1]
    flagged = (
        pair_eligible
        & np.isfinite(ratios)
        & (np.abs(ratios) > np.log(LN_RATIO_FLAG))
    )
    return ratios, pair_eligible, flagged


def _block_means(signal):
    """Block means below and above every position boundary."""
    lower = np.abs(signal[:-1, -BLOCK_SHOTS:]).mean(axis=1)
    upper = np.abs(signal[1:, :BLOCK_SHOTS]).mean(axis=1)
    return lower, upper


def boundary_leg(isat_signal, isat_noise, isweep_signal, isweep_noise):
    """Two-channel ratio step at every position boundary, and its attribution."""
    isat_lower, isat_upper = _block_means(isat_signal)
    isweep_lower, isweep_upper = _block_means(isweep_signal)

    floor_isat = SIGNIFICANCE_FACTOR * isat_noise
    floor_isweep = SIGNIFICANCE_FACTOR * isweep_noise
    eligible = (
        (isat_lower > floor_isat) & (isat_upper > floor_isat)
        & (isweep_lower > floor_isweep) & (isweep_upper > floor_isweep)
    )

    with np.errstate(all="ignore"):
        isat_ratio = np.log(isat_upper / isat_lower)
        isweep_ratio = np.log(isweep_upper / isweep_lower)
        two_channel = isat_ratio - isweep_ratio
    flagged = (
        eligible
        & np.isfinite(two_channel)
        & (np.abs(two_channel) > np.log(LN_RATIO_FLAG))
    )
    attributed = np.where(
        np.abs(isat_ratio) >= np.abs(isweep_ratio),
        ChannelKind.ISAT.value,
        ChannelKind.I_SWEEP.value,
    )
    return two_channel, isat_ratio, isweep_ratio, eligible, flagged, attributed


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
        help="path prefix for the two flag CSVs, the summary CSV and the JSON",
    )
    args = parser.parse_args(argv)

    ds = LapdDataset.from_manifest(args.repo_root / "config/may2026_run_manifest.toml")
    run_ids = list(ds.run_ids())

    fixed_rows = []
    boundary_rows = []
    summary_rows = []
    for run_id in run_ids:
        run = ds.run(run_id)
        per_channel = {}
        for kind in (ChannelKind.ISAT, ChannelKind.I_SWEEP):
            signal, noise = plateau_signal_and_noise(run, kind)
            per_channel[kind] = (signal, noise)
            ratios, pair_eligible, flagged = fixed_position_leg(signal, noise)
            n_shots = signal.shape[1]
            for position, shot in zip(*np.nonzero(flagged)):
                fixed_rows.append({
                    "run_id": run_id,
                    "channel": kind.value,
                    "position": int(position),
                    "shot": int(position) * n_shots + int(shot),
                    "signal_before_v": float(signal[position, shot]),
                    "signal_after_v": float(signal[position, shot + 1]),
                    "ln_ratio": float(ratios[position, shot]),
                })
            summary_rows.append({
                "run_id": run_id,
                "channel": kind.value,
                "tail_noise_v": noise,
                "eligibility_floor_v": SIGNIFICANCE_FACTOR * noise,
                "median_abs_signal_v": float(np.median(np.abs(signal))),
                "samples_eligible": int(eligible_samples(signal, noise).sum()),
                "samples_total": int(signal.size),
                "fixed_position_pairs_eligible": int(pair_eligible.sum()),
                "fixed_position_pairs_flagged": int(flagged.sum()),
            })
            print(f"screened run {run_id} ({kind.value}): noise {noise:.5f} V, "
                  f"{int(eligible_samples(signal, noise).sum())}/{signal.size} "
                  f"samples eligible, "
                  f"{int(flagged.sum())}/{int(pair_eligible.sum())} eligible "
                  f"fixed-position pairs flagged", flush=True)

        isat_signal, isat_noise = per_channel[ChannelKind.ISAT]
        isweep_signal, isweep_noise = per_channel[ChannelKind.I_SWEEP]
        two_channel, isat_ratio, isweep_ratio, eligible, flagged, attributed = (
            boundary_leg(isat_signal, isat_noise, isweep_signal, isweep_noise)
        )
        n_shots = isat_signal.shape[1]
        for boundary in np.flatnonzero(flagged):
            boundary_rows.append({
                "run_id": run_id,
                "position_below": int(boundary),
                "position_above": int(boundary) + 1,
                "shot_boundary": f"{int(boundary) * n_shots + n_shots - 1}|"
                                 f"{(int(boundary) + 1) * n_shots}",
                "ln_two_channel_step": float(two_channel[boundary]),
                "ln_isat_ratio": float(isat_ratio[boundary]),
                "ln_isweep_ratio": float(isweep_ratio[boundary]),
                "attributed_channel": str(attributed[boundary]),
            })
        print(f"screened run {run_id} (boundary): "
              f"{int(flagged.sum())}/{int(eligible.sum())} eligible boundaries "
              f"flagged", flush=True)
        for row in summary_rows[-2:]:
            row["boundaries_eligible"] = int(eligible.sum())
            row["boundaries_flagged_attributed_here"] = int(
                (flagged & (attributed == row["channel"])).sum()
            )

    fixed_csv = args.out_prefix.with_name(
        args.out_prefix.name + "_fixed_position_flags.csv")
    boundary_csv = args.out_prefix.with_name(
        args.out_prefix.name + "_boundary_flags.csv")
    summary_csv = args.out_prefix.with_name(args.out_prefix.name + "_summary.csv")
    summary_json = args.out_prefix.with_name(args.out_prefix.name + "_summary.json")
    fixed_csv.parent.mkdir(parents=True, exist_ok=True)

    fixed_fields = ["run_id", "channel", "position", "shot", "signal_before_v",
                    "signal_after_v", "ln_ratio"]
    boundary_fields = ["run_id", "position_below", "position_above",
                       "shot_boundary", "ln_two_channel_step", "ln_isat_ratio",
                       "ln_isweep_ratio", "attributed_channel"]
    with open(fixed_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed_fields)
        writer.writeheader()
        writer.writerows(fixed_rows)
    with open(boundary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=boundary_fields)
        writer.writeheader()
        writer.writerows(boundary_rows)
    summary_json.write_text(json.dumps(
        {"fixed_position_rule": FIXED_POSITION_RULE,
         "boundary_rule": BOUNDARY_RULE,
         "eligibility_rule": ELIGIBILITY_RULE,
         "clip_s": CLIP_S,
         "plateau_ms": list(PLATEAU_MS),
         "tail_s": TAIL_S,
         "ln_ratio_flag": LN_RATIO_FLAG,
         "significance_factor": SIGNIFICANCE_FACTOR,
         "block_shots": BLOCK_SHOTS,
         "runs": run_ids,
         "fixed_position_flags": fixed_rows,
         "boundary_flags": boundary_rows,
         "summary": summary_rows},
        indent=2, default=str,
    ))
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nfixed-position leg: {len(fixed_rows)} flag(s)")
    for row in fixed_rows:
        print(f"  run {row['run_id']} {row['channel']} position "
              f"{row['position']} shot {row['shot']}|{row['shot'] + 1} "
              f"ln ratio {row['ln_ratio']:+.3f}")
    print(f"boundary leg: {len(boundary_rows)} flag(s)")
    for row in boundary_rows:
        print(f"  run {row['run_id']} boundary {row['shot_boundary']} "
              f"(positions {row['position_below']}|{row['position_above']}) "
              f"ln step {row['ln_two_channel_step']:+.3f} "
              f"(isat {row['ln_isat_ratio']:+.3f}, "
              f"isweep {row['ln_isweep_ratio']:+.3f}) "
              f"-> {row['attributed_channel']}")
    print(f"\nwrote {fixed_csv}")
    print(f"wrote {boundary_csv}")
    print(f"wrote {summary_csv}")
    print(f"wrote {summary_json}")


if __name__ == "__main__":
    main()
