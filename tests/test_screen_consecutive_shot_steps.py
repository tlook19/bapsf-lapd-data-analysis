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
    SIGNIFICANCE_FACTOR,
    _persistence_shots,
    boundary_leg,
    eligible_samples,
    fixed_position_leg,
    flagged_steps,
    gate_verdict,
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
STEP_SHOT = 40
LN_STEP = 0.8


def _stepped_series(return_shot, total=TOTAL_SHOTS, step_shot=STEP_SHOT):
    """Two-channel level held off its pre-step value from step to return."""
    level = np.zeros(total)
    level[step_shot:return_shot] = LN_STEP
    return level, np.ones(total, dtype=bool)


def _step_pair(return_shot):
    return [(STEP_SHOT, LN_STEP), (return_shot, -LN_STEP)]


def _row(run_id, channel, pairs=(), open_ended=()):
    return {"run_id": run_id, "channel": channel,
            "pairs": list(pairs), "open_ended": list(open_ended)}


def test_a_matched_pair_one_shot_short_of_n_is_not_persistent():
    n = 200
    level, eligible = _stepped_series(STEP_SHOT + n - 1)
    pairs, open_ended = persistence_leg(_step_pair(STEP_SHOT + n - 1),
                                        level, eligible)

    assert not open_ended
    assert [pair["separation"] for pair in pairs] == [n - 1]
    assert not persistent_rows([_row("77", "isat", pairs)], n)


def test_a_matched_pair_at_n_is_persistent():
    n = 200
    level, eligible = _stepped_series(STEP_SHOT + n)
    pairs, _ = persistence_leg(_step_pair(STEP_SHOT + n), level, eligible)

    assert [pair["separation"] for pair in pairs] == [n]
    rows = persistent_rows([_row("77", "isat", pairs)], n)
    assert [(run_id, channel) for run_id, channel, _, _ in rows] == [("77", "isat")]


def test_a_step_that_never_comes_back_is_open_ended():
    level, eligible = _stepped_series(TOTAL_SHOTS)
    pairs, open_ended = persistence_leg([(STEP_SHOT, LN_STEP)], level, eligible)

    assert not pairs
    assert len(open_ended) == 1
    assert open_ended[0]["shot_step"] == STEP_SHOT
    assert open_ended[0]["shots_to_end"] == TOTAL_SHOTS - STEP_SHOT


def test_a_return_to_the_pre_step_level_inside_the_span_breaks_persistence():
    return_shot = STEP_SHOT + 400
    level, eligible = _stepped_series(return_shot)
    # One shot back at the pre-step level: the channel did not stay stepped.
    level[STEP_SHOT + 100] = 0.0
    pairs, open_ended = persistence_leg(_step_pair(return_shot), level, eligible)

    assert not pairs
    # The broken step consumes nothing, so the step that would have closed it
    # is read on its own -- from the stepped level down, and never back up.
    assert [step["shot_step"] for step in open_ended] == [return_shot]


def test_a_shot_that_is_not_eligible_cannot_report_a_return():
    return_shot = STEP_SHOT + 400
    level, eligible = _stepped_series(return_shot)
    level[STEP_SHOT + 100] = 0.0
    eligible[STEP_SHOT + 100] = False
    pairs, _ = persistence_leg(_step_pair(return_shot), level, eligible)

    assert [pair["separation"] for pair in pairs] == [400]


def test_the_return_half_of_an_excursion_is_not_a_fresh_open_ended_state():
    # Run 22's shape: a clean excursion that comes back to the run's baseline.
    # Without consuming the matched step, its return reads as a state running
    # to the end of the run.
    return_shot = STEP_SHOT + 40
    level, eligible = _stepped_series(return_shot)
    pairs, open_ended = persistence_leg(_step_pair(return_shot), level, eligible)

    assert [pair["separation"] for pair in pairs] == [40]
    assert not open_ended


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


def test_the_gate_reads_run_43_isat_alone_as_a_pass():
    hit = _row(GATE_RUN_ID, GATE_CHANNEL, [{"separation": 456}])
    quiet = _row("44", "isat", [{"separation": 12}])

    assert gate_verdict([hit, quiet], 200) == (
        f"GATE PASS: fires on run {GATE_RUN_ID} {GATE_CHANNEL.upper()} "
        "only (N = 200)"
    )


def test_the_gate_names_every_other_row_that_fires():
    hit = _row(GATE_RUN_ID, GATE_CHANNEL, [{"separation": 456}])
    other = _row("34", "isat", [{"separation": 770}],
                 [{"shots_to_end": 389}])

    assert gate_verdict([hit, other], 200) == (
        "GATE FAIL: also fires on run 34 ISAT (770/389)"
    )


def test_the_gate_fails_when_run_43_is_absent():
    assert gate_verdict([_row("34", "isat", [{"separation": 770}])], 200) == (
        f"GATE FAIL: run {GATE_RUN_ID} {GATE_CHANNEL.upper()} not flagged"
    )


def test_persistence_needs_a_positive_number_of_shots():
    assert _persistence_shots("200") == 200
    for bad in ("0", "-1"):
        with pytest.raises(argparse.ArgumentTypeError):
            _persistence_shots(bad)


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
    assert "LEG 3 -- PERSISTENCE (N = 40 shots)" in persistence_stdout
