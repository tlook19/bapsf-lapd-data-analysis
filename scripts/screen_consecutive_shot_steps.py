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

LEG 3 -- PERSISTENCE (``--persistence N``, off unless asked for)
    Both legs above fire on transients as readily as on states: shot-to-shot
    spread at the outer positions, and two-face profile structure at the
    boundaries common to many runs, flag dozens of (run, channel) rows.  A
    state is a step that STAYS stepped, so this leg keeps only the flagged
    steps that come back.

    REGISTERED 2026-09-09 at N = 200 shots: a channel state that persists for
    at least 200 consecutive shots is a state, not a burst -- five times the
    longest transient excursion on record (run 22's clean 40-shot excursion)
    and under half of run 43's matched step separation (456 shots).

    A flagged step is placed at the shot index of the first shot on the NEW
    level -- ``position * shots_per_position + shot + 1`` for leg 1, the first
    shot of the upper position for leg 2 -- and carries a SIGNED level ratio:
    leg 1's own ``ln`` ratio, and for leg 2 the two-channel step, whose sign is
    inverted when the boundary is attributed to I_SWEEP so that both legs report
    the ratio of the attributed channel's own level.

    MATCH.  A step at shot ``j`` with ratio ``r_j`` is matched by the EARLIEST
    later step at shot ``k`` with the opposite sign and a comparable magnitude:
    ``|r_j + r_k| <= MATCH_TOLERANCE_LN``, i.e. the two ratios' product within
    ln(LN_RATIO_FLAG) of zero -- the same resolution the flag threshold itself
    is set at, so a return the screen cannot distinguish from the step is a
    match.  The separation is ``k - j``, the number of shots held at the
    stepped level.

    A matched step is CONSUMED by its pair: every flagged step is either the
    onset of one span or the return of one, never both.  Without that, the
    return half of an excursion is itself an unmatched step and is reported as
    a state running to the end of the run -- which is what a clean excursion
    back to the run's own baseline is not.

    LEVEL TEST.  The step is persistent only if the signal does not come back
    between ``j`` and ``k``.  The level observable is the flagged channel's own
    TWO-CHANNEL ratio ``q = ln(|s_flagged| / |s_other|)``, per shot -- leg 2's
    profile-cancelling observable, used here because the probe moves over the
    shots between ``j`` and ``k`` and a raw single-channel level would read the
    profile's own excursions as returns.  The pre-step level ``q_pre`` is the
    mean of ``q`` over the last BLOCK_SHOTS shots before ``j`` that are eligible
    on BOTH channels; a shot ``i`` with ``j <= i < k``, eligible on both
    channels, is a RETURN when ``|q[i] - q_pre| < ln(LN_RATIO_FLAG)``, and one
    return anywhere in that span breaks the step.  Only the part of the profile
    move common to the two channels cancels, so the two faces' differential
    response to x survives here exactly as it does in leg 2: a differential
    excursion of more than a factor LN_RATIO_FLAG toward the pre-step level
    reads as a return, which breaks persistence rather than granting it.

    OPEN-ENDED.  A step with no matched later step and no return before the run
    ends is reported separately as open-ended with the number of shots to the
    end of the run; it is a persistent state only if that number is at least N.
    A step whose span carries a return is neither persistent nor open-ended.

Writes a CSV of every flagged pair for each leg, a per-(run, channel) summary
CSV and a JSON summary.  Reads every manifest run on both channels, so it is
IO-bound and takes minutes.  Needs h5py and numpy: the environment.yml env, not
the dead ./.venv.

Usage:
    python screen_consecutive_shot_steps.py <repo-root> <out-prefix>
                                            [--persistence N]

writes ``<out-prefix>_fixed_position_flags.csv``,
``<out-prefix>_boundary_flags.csv``, ``<out-prefix>_summary.csv`` and
``<out-prefix>_summary.json``.  ``--persistence`` adds a report section to
stdout after those and changes none of the four files.
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
MATCH_TOLERANCE_LN = float(np.log(LN_RATIO_FLAG))
RETURN_TOLERANCE_LN = float(np.log(LN_RATIO_FLAG))
SENSITIVITY_N = (100, 400)
GATE_RUN_ID = "43"
GATE_CHANNEL = "isat"

FIXED_POSITION_RULE = (
    "consecutive shots at a fixed position, both eligible; flag "
    f"|ln(s[j+1]/s[j])| > ln {LN_RATIO_FLAG:g}, where s is the plateau-window "
    "dead-time signal in volts minus the shot's own tail level"
)
BOUNDARY_RULE = (
    f"across a position boundary, block means of the last {BLOCK_SHOTS} shots "
    f"below and the first {BLOCK_SHOTS} above, all four eligible; flag "
    f"|ln(R_upper/R_lower)| > ln {LN_RATIO_FLAG:g} on the two-channel ratio "
    "R = |ISAT|/|I_SWEEP|, in which only the part of the profile move common "
    "to the two channels cancels -- their differential response to x "
    "survives; attributed to the channel whose own boundary ratio departs "
    "the further from unity"
)
ELIGIBILITY_RULE = (
    f"|signal| > {SIGNIFICANCE_FACTOR:g} x the run's own tail noise level on "
    "that channel (median over shots of the tail sample standard deviation, in "
    "volts)"
)
PERSISTENCE_RULE = (
    "a flagged step at shot j is persistent when the earliest later flagged "
    "step k of the opposite sign has the two ratios' product within "
    f"ln {LN_RATIO_FLAG:g} of zero, no shot in [j, k) eligible on both "
    "channels comes back to within "
    f"ln {LN_RATIO_FLAG:g} of the pre-step two-channel level "
    f"q = ln(|s_flagged|/|s_other|) (its mean over the last {BLOCK_SHOTS} "
    "both-eligible shots before j), and k - j is at least N"
)
PERSISTENCE_REGISTRATION = (
    "a channel state that persists for at least 200 consecutive shots is a "
    "state, not a burst -- five times the longest transient excursion on "
    "record (run 22's clean 40-shot excursion) and under half of run 43's "
    "matched step separation (456 shots)"
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


def flagged_steps(fixed_ratios, fixed_flagged, boundary_step, boundary_flagged,
                  attributed, channel):
    """Both legs' flags for one channel as ``(shot, signed ln ratio)`` pairs.

    The shot index is the first shot on the NEW level, in whole-run numbering.
    Leg 2's two-channel step is the attributed channel's own level ratio, with
    the sign inverted when the boundary is attributed to the denominator
    channel.
    """
    n_shots = fixed_ratios.shape[1] + 1
    steps = []
    for position, shot in zip(*np.nonzero(fixed_flagged)):
        steps.append((int(position) * n_shots + int(shot) + 1,
                      float(fixed_ratios[position, shot])))
    sign = 1.0 if channel == ChannelKind.ISAT.value else -1.0
    for boundary in np.flatnonzero(boundary_flagged & (attributed == channel)):
        steps.append(((int(boundary) + 1) * n_shots,
                      sign * float(boundary_step[boundary])))
    steps.sort()
    return steps


def level_observable(signal, other_signal):
    """Per-shot two-channel level ``ln(|signal| / |other_signal|)``, flattened.

    The profile move common to the two channels cancels; their differential
    response to position does not.
    """
    with np.errstate(all="ignore"):
        return np.log(np.abs(signal) / np.abs(other_signal)).reshape(-1)


def both_eligible_samples(signal, noise, other_signal, other_noise):
    """True per shot where both channels clear the significance floor."""
    return (eligible_samples(signal, noise)
            & eligible_samples(other_signal, other_noise)).reshape(-1)


def _pre_step_level(level, eligible, shot):
    """Mean two-channel level over the eligible shots just before ``shot``.

    ``None`` when no eligible shot precedes it, which leaves the step untestable
    rather than persistent.
    """
    before = np.flatnonzero(eligible[:shot] & np.isfinite(level[:shot]))
    if before.size == 0:
        return None
    return float(level[before[-BLOCK_SHOTS:]].mean())


def _returns_to_level(level, eligible, pre_level, start, stop):
    """First shot in ``[start, stop)`` that comes back to the pre-step level."""
    for shot in range(start, min(stop, level.size)):
        if not eligible[shot] or not np.isfinite(level[shot]):
            continue
        if abs(level[shot] - pre_level) < RETURN_TOLERANCE_LN:
            return shot
    return None


def persistence_leg(steps, level, eligible):
    """Match every flagged step to its return, and classify what it found.

    Returns ``(pairs, open_ended)``.  A pair is a step matched by an
    opposite-sign step of comparable magnitude with no return to the pre-step
    level in between; an open-ended step has neither a match nor a return before
    the run ends.  Neither list is filtered by N: the caller applies it, so the
    separation distribution measures N's margin.
    """
    total_shots = int(level.size)
    pairs = []
    open_ended = []
    consumed = set()
    for index, (shot, ratio) in enumerate(steps):
        if index in consumed:
            continue
        pre_level = _pre_step_level(level, eligible, shot)
        if pre_level is None:
            continue
        match = None
        for other in range(index + 1, len(steps)):
            if other in consumed:
                continue
            if abs(ratio + steps[other][1]) <= MATCH_TOLERANCE_LN:
                match = other
                break
        stop = total_shots if match is None else steps[match][0]
        if _returns_to_level(level, eligible, pre_level, shot, stop) is not None:
            continue
        if match is None:
            open_ended.append({
                "shot_step": shot,
                "ln_step": ratio,
                "shots_to_end": total_shots - shot,
            })
        else:
            consumed.add(match)
            pairs.append({
                "shot_step": shot,
                "shot_return": steps[match][0],
                "separation": steps[match][0] - shot,
                "ln_step": ratio,
                "ln_return": steps[match][1],
            })
    return pairs, open_ended


def persistent_rows(persistence, n_shots):
    """The (run, channel) rows holding a state that lasts at least ``n_shots``.

    A matched pair counts by its separation, an open-ended step by its shots to
    the end of the run.
    """
    rows = []
    for row in persistence:
        pairs = [p for p in row["pairs"] if p["separation"] >= n_shots]
        opens = [o for o in row["open_ended"] if o["shots_to_end"] >= n_shots]
        if pairs or opens:
            rows.append((row["run_id"], row["channel"], pairs, opens))
    return rows


def gate_verdict(persistence, n_shots):
    """The pre-registered verdict line: run 43 ISAT alone, and nothing else."""
    rows = persistent_rows(persistence, n_shots)
    names = [(run_id, channel) for run_id, channel, _, _ in rows]
    others = [name for name in names if name != (GATE_RUN_ID, GATE_CHANNEL)]
    if (GATE_RUN_ID, GATE_CHANNEL) not in names:
        return (f"GATE FAIL: run {GATE_RUN_ID} {GATE_CHANNEL.upper()} not "
                f"flagged")
    if others:
        listed = ", ".join(
            f"run {run_id} {channel.upper()} "
            f"({'/'.join(str(sep) for sep in separations)})"
            for run_id, channel, separations in _row_separations(rows, others)
        )
        return f"GATE FAIL: also fires on {listed}"
    return (f"GATE PASS: fires on run {GATE_RUN_ID} {GATE_CHANNEL.upper()} "
            f"only (N = {n_shots})")


def _row_separations(rows, wanted):
    """Each wanted row's separations, matched pairs then open-ended steps."""
    for run_id, channel, pairs, opens in rows:
        if (run_id, channel) not in wanted:
            continue
        separations = ([p["separation"] for p in pairs]
                       + [o["shots_to_end"] for o in opens])
        yield run_id, channel, separations


def _report_persistence(persistence, n_shots):
    """Print leg 3: the rows, the separation distribution and the gate."""
    print(f"\n=== LEG 3 -- PERSISTENCE (N = {n_shots} shots) ===")
    print(f"registration: {PERSISTENCE_REGISTRATION}")
    print(f"rule: {PERSISTENCE_RULE}")

    for row in persistence:
        persistent = [p for p in row["pairs"] if p["separation"] >= n_shots]
        if not persistent and not row["open_ended"]:
            continue
        print(f"\nrun {row['run_id']} {row['channel']}: "
              f"{len(row['pairs'])} matched pair(s), "
              f"{len(persistent)} persistent, "
              f"{len(row['open_ended'])} open-ended")
        for pair in persistent:
            print(f"  persistent {pair['shot_step']} -> {pair['shot_return']} "
                  f"separation {pair['separation']} "
                  f"(ln {pair['ln_step']:+.3f} / {pair['ln_return']:+.3f})")
        for step in row["open_ended"]:
            print(f"  open-ended at shot {step['shot_step']} "
                  f"(ln {step['ln_step']:+.3f}), "
                  f"{step['shots_to_end']} shots to end"
                  + (" -- persistent" if step["shots_to_end"] >= n_shots
                     else ""))

    separations = [p["separation"] for row in persistence for p in row["pairs"]]
    off_43 = [p["separation"] for row in persistence for p in row["pairs"]
              if row["run_id"] != GATE_RUN_ID]
    print(f"\nmatched opposite-sign pairs: {len(separations)}")
    if separations:
        print(f"  separation min {min(separations)}, "
              f"median {int(np.median(separations))}, "
              f"max {max(separations)} shots")
    else:
        print("  separation min/median/max: no matched pairs")
    if off_43:
        print(f"  largest separation off run {GATE_RUN_ID}: "
              f"{max(off_43)} shots")
    else:
        print(f"  largest separation off run {GATE_RUN_ID}: "
              f"no matched pairs")

    print(f"\n{gate_verdict(persistence, n_shots)}")
    for sensitivity in SENSITIVITY_N:
        print(f"sensitivity (not a gate), N = {sensitivity}: "
              f"{gate_verdict(persistence, sensitivity)}")


def _persistence_shots(text):
    """``--persistence`` takes a positive whole number of shots."""
    shots = int(text)
    if shots <= 0:
        raise argparse.ArgumentTypeError(
            f"persistence needs a positive number of shots, got {text}")
    return shots


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
    parser.add_argument(
        "--persistence",
        type=_persistence_shots,
        default=None,
        metavar="N",
        help="also run leg 3: keep only flagged steps that stay stepped for at "
             "least N consecutive shots, and print the pre-registered gate. "
             "Off by default; the four written files do not depend on it",
    )
    args = parser.parse_args(argv)

    ds = LapdDataset.from_manifest(args.repo_root / "config/may2026_run_manifest.toml")
    run_ids = list(ds.run_ids())

    fixed_rows = []
    boundary_rows = []
    summary_rows = []
    persistence = []
    for run_id in run_ids:
        run = ds.run(run_id)
        per_channel = {}
        per_channel_flags = {}
        for kind in (ChannelKind.ISAT, ChannelKind.I_SWEEP):
            signal, noise = plateau_signal_and_noise(run, kind)
            per_channel[kind] = (signal, noise)
            ratios, pair_eligible, flagged = fixed_position_leg(signal, noise)
            per_channel_flags[kind] = (ratios, flagged)
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

        if args.persistence is not None:
            others = {ChannelKind.ISAT: ChannelKind.I_SWEEP,
                      ChannelKind.I_SWEEP: ChannelKind.ISAT}
            for kind in (ChannelKind.ISAT, ChannelKind.I_SWEEP):
                signal, noise = per_channel[kind]
                other_signal, other_noise = per_channel[others[kind]]
                ratios, fixed_flagged = per_channel_flags[kind]
                steps = flagged_steps(ratios, fixed_flagged, two_channel,
                                      flagged, attributed, kind.value)
                pairs, open_ended = persistence_leg(
                    steps,
                    level_observable(signal, other_signal),
                    both_eligible_samples(signal, noise,
                                          other_signal, other_noise),
                )
                persistence.append({"run_id": run_id, "channel": kind.value,
                                    "pairs": pairs,
                                    "open_ended": open_ended})

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

    if args.persistence is not None:
        _report_persistence(persistence, args.persistence)


if __name__ == "__main__":
    main()
