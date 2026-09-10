"""The three step-screen legs, their eligibility floor, and what each can reach."""

import argparse

import numpy as np
import pytest

from scripts import screen_consecutive_shot_steps as screen
from scripts.screen_consecutive_shot_steps import (
    BLOCK_SHOTS,
    GATE_CHANNEL,
    GATE_RUN_ID,
    LN_RATIO_FLAG,
    PERSISTENT_MIN_LN,
    PERSISTENT_MIN_REFERENCE_SHOTS,
    PRE_STEP_POSITIONS,
    SIGNIFICANCE_FACTOR,
    _min_reference_shots,
    _persistence_shots,
    _persistent_min_ln,
    _pre_step_level,
    _returns_to_level,
    boundary_leg,
    eligible_samples,
    fixed_position_leg,
    flagged_steps,
    gate_verdict,
    over_min_ln,
    persistence_leg,
    persistent_rows,
)

N_POSITIONS = 6
N_SHOTS = 20
NOISE = 0.001
LEVEL = 1.0
STEP = 1.76


def _flat(level=LEVEL):
    return np.full((N_POSITIONS, N_SHOTS), level)


def test_eligibility_floor_is_five_noise_levels():
    signal = np.array([[SIGNIFICANCE_FACTOR * NOISE * 1.01,
                        SIGNIFICANCE_FACTOR * NOISE * 0.99]])
    eligible = eligible_samples(signal, NOISE)
    assert eligible[0, 0]
    assert not eligible[0, 1]


def test_a_flat_run_flags_nothing():
    _, _, flagged = fixed_position_leg(_flat(), NOISE)
    assert not flagged.any()


def test_a_step_inside_a_position_is_flagged_at_its_own_pair():
    signal = _flat()
    signal[2, 16:] = LEVEL / STEP
    ratios, _, flagged = fixed_position_leg(signal, NOISE)

    assert flagged.sum() == 1
    assert flagged[2, 15]
    assert ratios[2, 15] == np.log(1.0 / STEP)


def test_a_pair_below_the_eligibility_floor_is_not_flagged():
    # The same ratio, at a level the record cannot resolve, is a ratio of noise.
    signal = np.full((N_POSITIONS, N_SHOTS), SIGNIFICANCE_FACTOR * NOISE * 0.5)
    signal[2, 16:] /= STEP
    _, pair_eligible, flagged = fixed_position_leg(signal, NOISE)

    assert not pair_eligible.any()
    assert not flagged.any()


def test_a_step_below_the_threshold_is_not_flagged():
    signal = _flat()
    signal[1, 10:] = LEVEL * LN_RATIO_FLAG * 0.99
    _, _, flagged = fixed_position_leg(signal, NOISE)
    assert not flagged.any()


def test_a_step_on_a_position_boundary_is_invisible_to_the_fixed_position_leg():
    signal = _flat()
    signal[3:] = LEVEL * STEP
    _, _, flagged = fixed_position_leg(signal, NOISE)
    assert not flagged.any()


def _boundary_case(profile_move, isat_step):
    """Both channels carry the profile move; only ISAT carries the level step."""
    isat = _flat()
    isweep = _flat(0.5)
    isat[3:] *= profile_move * isat_step
    isweep[3:] *= profile_move
    return isat, isweep


def test_the_profile_move_cancels_in_the_two_channel_ratio():
    isat, isweep = _boundary_case(profile_move=0.93, isat_step=1.0)
    two_channel, _, _, eligible, flagged, _ = boundary_leg(
        isat, NOISE, isweep, NOISE
    )
    assert eligible.all()
    assert not flagged.any()
    assert two_channel[2] == pytest.approx(0.0, abs=1e-15)


def test_a_one_channel_step_on_a_boundary_is_flagged_and_attributed():
    isat, isweep = _boundary_case(profile_move=0.93, isat_step=STEP)
    two_channel, isat_ratio, isweep_ratio, _, flagged, attributed = boundary_leg(
        isat, NOISE, isweep, NOISE
    )

    assert flagged.sum() == 1
    assert flagged[2]
    assert two_channel[2] == pytest.approx(np.log(STEP))
    # The moved channel's own ratio carries move x step; the other carries move.
    assert isat_ratio[2] == pytest.approx(np.log(0.93 * STEP))
    assert isweep_ratio[2] == pytest.approx(np.log(0.93))
    assert attributed[2] == "isat"


def test_a_step_on_the_other_channel_is_attributed_to_it():
    isat = _flat()
    isweep = _flat(0.5)
    isweep[3:] *= STEP
    _, _, _, _, flagged, attributed = boundary_leg(isat, NOISE, isweep, NOISE)
    assert flagged[2]
    assert attributed[2] == "i_sweep"


def test_a_boundary_below_the_eligibility_floor_is_not_flagged():
    isat, isweep = _boundary_case(profile_move=0.93, isat_step=STEP)
    huge_noise = LEVEL  # every block mean now sits under 5 x noise
    _, _, _, eligible, flagged, _ = boundary_leg(
        isat, huge_noise, isweep, huge_noise
    )
    assert not eligible.any()
    assert not flagged.any()


def test_the_boundary_blocks_are_the_last_and_first_shots():
    isat = _flat()
    isweep = _flat(0.5)
    # A step confined to the shots the blocks do NOT read leaves the ratio flat:
    # the boundaries below and above position 2 read its first and last blocks.
    isat[2, BLOCK_SHOTS : N_SHOTS - BLOCK_SHOTS] *= STEP
    _, _, _, _, flagged, _ = boundary_leg(isat, NOISE, isweep, NOISE)
    assert not flagged.any()


# ---------------------------------------------------------------- leg 3

TOTAL_SHOTS = 1000
SHOTS_PER_POSITION = 20
STEP_SHOT = 40
LN_STEP = 0.8


def _stepped_series(return_shot, total=TOTAL_SHOTS, step_shot=STEP_SHOT):
    """Two-channel level held off its pre-step value from step to return."""
    level = np.zeros(total)
    level[step_shot:return_shot] = LN_STEP
    return level, np.ones(total, dtype=bool)


def _step_pair(return_shot):
    return [(STEP_SHOT, LN_STEP), (return_shot, -LN_STEP)]


def _leg3(steps, level, eligible, min_reference_shots=0):
    """Leg 3 with the coverage clause off unless the test asks for it."""
    return persistence_leg(steps, level, eligible, SHOTS_PER_POSITION,
                           min_reference_shots)


def _row(run_id, channel, pairs=(), open_ended=(), untestable=()):
    return {"run_id": run_id, "channel": channel,
            "pairs": list(pairs), "open_ended": list(open_ended),
            "untestable": list(untestable)}


def test_a_matched_pair_one_shot_short_of_n_is_not_persistent():
    n = 200
    level, eligible = _stepped_series(STEP_SHOT + n - 1)
    pairs, open_ended, _ = _leg3(_step_pair(STEP_SHOT + n - 1), level, eligible)

    assert not open_ended
    assert [pair["separation"] for pair in pairs] == [n - 1]
    assert not persistent_rows([_row("77", "isat", pairs)], n, 0.0)


def test_a_matched_pair_at_n_is_persistent():
    n = 200
    level, eligible = _stepped_series(STEP_SHOT + n)
    pairs, _, _ = _leg3(_step_pair(STEP_SHOT + n), level, eligible)

    assert [pair["separation"] for pair in pairs] == [n]
    rows = persistent_rows([_row("77", "isat", pairs)], n, 0.0)
    assert [(run_id, channel) for run_id, channel, _, _ in rows] == [("77", "isat")]


def test_a_step_that_never_comes_back_is_open_ended():
    level, eligible = _stepped_series(TOTAL_SHOTS)
    pairs, open_ended, _ = _leg3([(STEP_SHOT, LN_STEP)], level, eligible)

    assert not pairs
    assert len(open_ended) == 1
    assert open_ended[0]["shot_step"] == STEP_SHOT
    assert open_ended[0]["shots_to_end"] == TOTAL_SHOTS - STEP_SHOT


def test_a_return_to_the_pre_step_level_inside_the_span_breaks_persistence():
    return_shot = STEP_SHOT + 400
    level, eligible = _stepped_series(return_shot)
    # One shot back at the pre-step level: the channel did not stay stepped.
    level[STEP_SHOT + 100] = 0.0
    pairs, open_ended, _ = _leg3(_step_pair(return_shot), level, eligible)

    assert not pairs
    # The broken step consumes nothing, so the step that would have closed it
    # is read on its own -- from the stepped level down, and never back up.
    assert [step["shot_step"] for step in open_ended] == [return_shot]


def test_a_shot_that_is_not_eligible_cannot_report_a_return():
    return_shot = STEP_SHOT + 400
    level, eligible = _stepped_series(return_shot)
    level[STEP_SHOT + 100] = 0.0
    eligible[STEP_SHOT + 100] = False
    pairs, _, _ = _leg3(_step_pair(return_shot), level, eligible)

    assert [pair["separation"] for pair in pairs] == [400]


def test_the_return_half_of_an_excursion_is_not_a_fresh_open_ended_state():
    # Run 22's shape: a clean excursion that comes back to the run's baseline.
    # Without consuming the matched step, its return reads as a state running
    # to the end of the run.
    return_shot = STEP_SHOT + 40
    level, eligible = _stepped_series(return_shot)
    pairs, open_ended, _ = _leg3(_step_pair(return_shot), level, eligible)

    assert [pair["separation"] for pair in pairs] == [40]
    assert not open_ended


# ------------------------------------------- leg 3: the pre-step reference

DIP_POSITION = 4
LEG3_POSITIONS = 10
DIP_LEVEL = -0.5


def _positioned(levels):
    """A whole-run level series built from one value per position."""
    level = np.repeat(np.asarray(levels, dtype=float), SHOTS_PER_POSITION)
    return level, np.ones(level.size, dtype=bool)


def test_a_one_position_dip_cannot_be_the_pre_step_reference():
    # Run 26's shape: the level dips for a single position and the flagged step
    # is the RECOVERY from it.  Referenced to the dip -- which is exactly what
    # the block of shots just before the step reads -- the recovery is a state
    # the run never comes back from.
    levels = [0.0] * LEG3_POSITIONS
    levels[DIP_POSITION] = DIP_LEVEL
    level, eligible = _positioned(levels)
    step_shot = (DIP_POSITION + 1) * SHOTS_PER_POSITION

    assert _returns_to_level(level, eligible, DIP_LEVEL, step_shot,
                             level.size) is None
    # Three positions, taken by median, read the level the dip interrupted.
    assert _pre_step_level(level, eligible, step_shot, SHOTS_PER_POSITION) == (
        0.0, PRE_STEP_POSITIONS * SHOTS_PER_POSITION
    )
    pairs, open_ended, _ = _leg3([(step_shot, -DIP_LEVEL)], level, eligible)
    assert not pairs and not open_ended


def test_the_pre_step_reference_reaches_back_exactly_three_positions():
    levels = [0.0, 1.0, 2.0, 3.0, 4.0] + [0.0] * (LEG3_POSITIONS - 5)
    level, eligible = _positioned(levels)

    # Positions 2, 3 and 4 -- median 3.0.  Two positions would read 3.5, four
    # would read 2.5, and the step's own position 5 is never part of it.
    assert PRE_STEP_POSITIONS == 3
    assert _pre_step_level(level, eligible, 5 * SHOTS_PER_POSITION,
                           SHOTS_PER_POSITION) == (
        3.0, PRE_STEP_POSITIONS * SHOTS_PER_POSITION
    )


def test_the_pre_step_reference_reads_only_both_eligible_shots():
    levels = [0.0, 0.0, 2.0, 9.0, 9.0] + [0.0] * (LEG3_POSITIONS - 5)
    level, eligible = _positioned(levels)
    eligible[3 * SHOTS_PER_POSITION:5 * SHOTS_PER_POSITION] = False

    # Only position 2 survives, so the reference is its level and its coverage.
    assert _pre_step_level(level, eligible, 5 * SHOTS_PER_POSITION,
                           SHOTS_PER_POSITION) == (2.0, SHOTS_PER_POSITION)


def test_a_step_with_no_position_behind_it_is_untestable_not_persistent():
    level, eligible = _positioned([0.0] * LEG3_POSITIONS)

    assert _pre_step_level(level, eligible, 5, SHOTS_PER_POSITION) == (None, 0)
    pairs, open_ended, untestable = _leg3([(5, LN_STEP)], level, eligible)
    assert not pairs and not open_ended
    assert [step["reference_shots"] for step in untestable] == [0]


# ------------------------------------------- leg 3: the reference's coverage


def _thin_reference_run():
    """Run 03 ISAT's shape: a step whose reference survives on one shot.

    The three positions below the step are eligible on one shot only, and it
    reads a level nothing later in the run comes near.
    """
    levels = [1.489] * 5 + [4.2] * (LEG3_POSITIONS - 5)
    level, eligible = _positioned(levels)
    eligible[:5 * SHOTS_PER_POSITION] = False
    eligible[2 * SHOTS_PER_POSITION] = True          # the one survivor
    return level, eligible


def test_a_reference_carried_by_one_shot_leaves_the_step_untestable():
    level, eligible = _thin_reference_run()
    step_shot = 5 * SHOTS_PER_POSITION

    # The reference exists, and rests on a single both-eligible shot.
    assert _pre_step_level(level, eligible, step_shot,
                           SHOTS_PER_POSITION) == (1.489, 1)
    # Nothing later in the run comes back to it, so without the clause the step
    # reads as a state running to the end.
    assert _returns_to_level(level, eligible, 1.489, step_shot,
                             level.size) is None

    pairs, open_ended, untestable = _leg3(
        [(step_shot, LN_STEP)], level, eligible,
        PERSISTENT_MIN_REFERENCE_SHOTS)

    assert not pairs and not open_ended
    assert [(step["shot_step"], step["position"], step["reference_shots"])
            for step in untestable] == [(step_shot, 5, 1)]


def test_the_coverage_clause_off_reads_the_thin_reference_as_a_state():
    # The same run with the clause removed: what --persistent-min-reference-
    # shots 0 restores, and why the clause is what closes the ramp-up region.
    level, eligible = _thin_reference_run()
    step_shot = 5 * SHOTS_PER_POSITION

    pairs, open_ended, untestable = _leg3([(step_shot, LN_STEP)], level,
                                          eligible, 0)

    assert not pairs and not untestable
    assert [step["shots_to_end"] for step in open_ended] == [
        level.size - step_shot
    ]


def test_a_reference_one_shot_short_of_the_minimum_is_still_untestable():
    level, eligible = _positioned([0.0] * LEG3_POSITIONS)
    step_shot = 5 * SHOTS_PER_POSITION
    short = PERSISTENT_MIN_REFERENCE_SHOTS - 1
    eligible[:step_shot] = False
    eligible[2 * SHOTS_PER_POSITION:2 * SHOTS_PER_POSITION + short] = True

    assert _pre_step_level(level, eligible, step_shot,
                           SHOTS_PER_POSITION)[1] == short
    _, _, untestable = _leg3([(step_shot, LN_STEP)], level, eligible,
                             PERSISTENT_MIN_REFERENCE_SHOTS)
    assert [step["reference_shots"] for step in untestable] == [short]

    # One more both-eligible shot and the reference carries a level test --
    # which this flat run passes at once, so nothing is reported at all.
    eligible[2 * SHOTS_PER_POSITION + short] = True
    pairs, open_ended, untestable = _leg3([(step_shot, LN_STEP)], level,
                                          eligible,
                                          PERSISTENT_MIN_REFERENCE_SHOTS)
    assert not untestable and not pairs and not open_ended


def test_the_clause_off_reclassifies_nothing_and_only_names_the_dropped():
    # A run holding one testable step and one with no reference at all.  With
    # the clause off, the testable step is classified exactly as it was before
    # the clause existed, and the untestable list holds only the step that was
    # silently dropped then -- so the hit list at 0 is the hit list without it.
    level, eligible = _positioned([0.0] * LEG3_POSITIONS)
    late = 5 * SHOTS_PER_POSITION
    level[late:] = LN_STEP
    steps = [(3, LN_STEP), (late, LN_STEP)]

    pairs, open_ended, untestable = _leg3(steps, level, eligible, 0)

    assert not pairs
    assert [step["shot_step"] for step in open_ended] == [late]
    assert [(step["shot_step"], step["reference_shots"])
            for step in untestable] == [(3, 0)]


def test_the_reference_coverage_is_a_non_negative_number_of_shots():
    assert _min_reference_shots("20") == 20
    assert _min_reference_shots("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        _min_reference_shots("-1")


# ------------------------------------------------ leg 3: the magnitude floor


def test_the_floor_is_carried_by_both_halves_of_a_matched_pair():
    # Run 34 ISAT's shape: a step that clears the flag threshold and a return
    # that only just does.  The pair matches; it is not a state.
    pair = {"separation": 770, "ln_step": -0.301, "ln_return": +0.196}
    row = _row("34", "isat", [pair])

    assert over_min_ln(pair, 0.19)
    assert not over_min_ln(pair, 0.20)
    assert persistent_rows([row], 200, 0.19)
    assert not persistent_rows([row], 200, 0.20)
    assert not persistent_rows([row], 200, PERSISTENT_MIN_LN)


def test_the_registered_state_clears_the_registered_floor():
    pair = {"separation": 456, "ln_step": +0.672, "ln_return": -0.546}
    rows = persistent_rows([_row(GATE_RUN_ID, GATE_CHANNEL, [pair])],
                           200, PERSISTENT_MIN_LN)

    assert [(run_id, channel) for run_id, channel, _, _ in rows] == [
        (GATE_RUN_ID, GATE_CHANNEL)
    ]


def test_an_open_ended_step_carries_the_floor_on_its_one_half():
    step = {"shots_to_end": 400, "ln_step": +0.35}
    row = _row("44", "isat", (), [step])

    assert persistent_rows([row], 200, 0.30)
    assert not persistent_rows([row], 200, PERSISTENT_MIN_LN)
    # N and the floor are independent: a big enough step that is too short is
    # not a state either.
    assert not persistent_rows(
        [_row("44", "isat", (), [{"shots_to_end": 199, "ln_step": +9.0}])],
        200, PERSISTENT_MIN_LN)


def test_a_floor_of_zero_keeps_every_entry_the_leg_reported():
    # The floor filters what persistence_leg already found; it never re-matches.
    # At zero the leg's own N test is the whole of the classification.
    pairs = [{"separation": 200, "ln_step": +0.01, "ln_return": -0.01}]
    opens = [{"shots_to_end": 200, "ln_step": -0.02}]
    rows = persistent_rows([_row("44", "isat", pairs, opens)], 200, 0.0)

    assert [(run_id, channel) for run_id, channel, _, _ in rows] == [
        ("44", "isat")
    ]
    assert rows[0][2] == pairs
    assert rows[0][3] == opens


def test_flagged_steps_place_and_sign_both_legs():
    n_positions, n_shots = 3, 5
    fixed_ratios = np.zeros((n_positions, n_shots - 1))
    fixed_flagged = np.zeros_like(fixed_ratios, dtype=bool)
    fixed_ratios[1, 2] = -0.4
    fixed_flagged[1, 2] = True
    boundary_step = np.array([0.0, 0.5])
    boundary_flagged = np.array([False, True])
    attributed = np.array(["isat", "i_sweep"])

    isat = flagged_steps(fixed_ratios, fixed_flagged, boundary_step,
                         boundary_flagged, attributed, "isat")
    isweep = flagged_steps(fixed_ratios, fixed_flagged, boundary_step,
                           boundary_flagged, attributed, "i_sweep")

    # Leg 1: the first shot on the new level, in whole-run numbering.
    assert isat == [(1 * n_shots + 2 + 1, -0.4)]
    # Leg 2: the first shot of the upper position, sign inverted for I_SWEEP.
    assert isweep == [(1 * n_shots + 2 + 1, -0.4), (2 * n_shots, -0.5)]


_GATE_PAIR = {"separation": 456, "ln_step": +0.672, "ln_return": -0.546}


def test_the_gate_reads_run_43_isat_alone_as_a_pass():
    hit = _row(GATE_RUN_ID, GATE_CHANNEL, [_GATE_PAIR])
    quiet = _row("44", "isat",
                 [{"separation": 12, "ln_step": +0.9, "ln_return": -0.9}])

    assert gate_verdict([hit, quiet], 200, PERSISTENT_MIN_LN,
                        PERSISTENT_MIN_REFERENCE_SHOTS) == (
        f"GATE PASS: fires on run {GATE_RUN_ID} {GATE_CHANNEL.upper()} "
        "only (N = 200, floor 0.40, reference \u2265 20)"
    )


def test_the_gate_names_every_other_row_with_its_ln_ratios():
    hit = _row(GATE_RUN_ID, GATE_CHANNEL, [_GATE_PAIR])
    other = _row("34", "isat",
                 [{"separation": 770, "ln_step": -0.801, "ln_return": +0.752}],
                 [{"shots_to_end": 389, "ln_step": -1.200}])

    assert gate_verdict([hit, other], 200, PERSISTENT_MIN_LN,
                        PERSISTENT_MIN_REFERENCE_SHOTS) == (
        "GATE FAIL: also fires on run 34 ISAT "
        "(770 shots, ln -0.801/+0.752; 389 shots to end, ln -1.200)"
    )


def test_the_gate_fails_when_run_43_is_absent():
    other = _row("34", "isat",
                 [{"separation": 770, "ln_step": -0.801, "ln_return": +0.752}])

    assert gate_verdict([other], 200, PERSISTENT_MIN_LN,
                        PERSISTENT_MIN_REFERENCE_SHOTS) == (
        f"GATE FAIL: run {GATE_RUN_ID} {GATE_CHANNEL.upper()} not flagged"
    )


def test_a_row_the_floor_removes_no_longer_fails_the_gate():
    hit = _row(GATE_RUN_ID, GATE_CHANNEL, [_GATE_PAIR])
    marginal = _row("34", "isat",
                    [{"separation": 770, "ln_step": -0.301,
                      "ln_return": +0.196}])

    assert gate_verdict([hit, marginal], 200, 0.0,
                        PERSISTENT_MIN_REFERENCE_SHOTS).startswith("GATE FAIL")
    assert gate_verdict([hit, marginal], 200, PERSISTENT_MIN_LN,
                        PERSISTENT_MIN_REFERENCE_SHOTS) == (
        f"GATE PASS: fires on run {GATE_RUN_ID} {GATE_CHANNEL.upper()} "
        "only (N = 200, floor 0.40, reference \u2265 20)"
    )


def test_persistence_needs_a_positive_number_of_shots():
    assert _persistence_shots("200") == 200
    for bad in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            _persistence_shots(bad)


def test_the_floor_is_a_non_negative_ln_magnitude():
    assert _persistent_min_ln("0.40") == 0.40
    assert _persistent_min_ln("0") == 0.0
    for bad in ("-0.1", "nan"):
        with pytest.raises(argparse.ArgumentTypeError):
            _persistent_min_ln(bad)


def _synthetic_main(monkeypatch, tmp_path, argv_extra):
    """Run ``main`` over a synthetic two-run dataset; return stdout and files."""
    signals = {}
    for run_id in ("43", "44"):
        isat = np.full((6, 20), 1.0)
        isweep = np.full((6, 20), 0.25)
        if run_id == "43":
            isat[2:5] = 1.9          # a state spanning three positions
        signals[run_id] = {"isat": isat, "i_sweep": isweep}

    class _Dataset:
        @staticmethod
        def from_manifest(path):
            return _Dataset()

        def run_ids(self):
            return list(signals)

        def run(self, run_id):
            return run_id

    monkeypatch.setattr(screen, "LapdDataset", _Dataset)
    monkeypatch.setattr(
        screen, "plateau_signal_and_noise",
        lambda run, kind: (signals[run][kind.value], 0.001),
    )
    prefix = tmp_path / "screen"
    screen.main([str(tmp_path), str(prefix)] + argv_extra)
    written = {
        path.name: path.read_bytes()
        for path in sorted(tmp_path.iterdir()) if path.is_file()
    }
    return written


def test_the_default_invocation_writes_what_it_wrote_before_persistence(
        monkeypatch, tmp_path, capsys):
    default_files = _synthetic_main(monkeypatch, tmp_path, [])
    default_stdout = capsys.readouterr().out
    persistence_files = _synthetic_main(monkeypatch, tmp_path,
                                        ["--persistence", "40"])
    persistence_stdout = capsys.readouterr().out

    assert set(default_files) == {
        "screen_fixed_position_flags.csv", "screen_boundary_flags.csv",
        "screen_summary.csv", "screen_summary.json",
    }
    assert persistence_files == default_files
    assert persistence_stdout.startswith(default_stdout)
    assert "LEG 3 -- PERSISTENCE" not in default_stdout
    assert ("LEG 3 -- PERSISTENCE (N = 40 shots, floor ln 0.40, "
            "reference \u2265 20 shots)" in persistence_stdout)


def test_the_floor_is_inert_without_persistence_and_reported_with_it(
        monkeypatch, tmp_path, capsys):
    default_files = _synthetic_main(monkeypatch, tmp_path,
                                    ["--persistent-min-ln", "0.9"])
    capsys.readouterr()
    persistence_files = _synthetic_main(
        monkeypatch, tmp_path,
        ["--persistence", "40", "--persistent-min-ln", "0.9"])
    persistence_stdout = capsys.readouterr().out

    # The four written files never depend on leg 3, floor or no floor.
    assert persistence_files == default_files
    assert ("LEG 3 -- PERSISTENCE (N = 40 shots, floor ln 0.90, "
            "reference \u2265 20 shots)" in persistence_stdout)
    # The synthetic state steps by ln(1.9) = 0.64, so a 0.9 floor removes it
    # while the registered 0.40 keeps it.
    assert "GATE FAIL: run 43 ISAT not flagged" in persistence_stdout
    _synthetic_main(monkeypatch, tmp_path, ["--persistence", "40"])
    assert ("GATE PASS: fires on run 43 ISAT only (N = 40, floor 0.40, "
            "reference \u2265 20)" in capsys.readouterr().out)
