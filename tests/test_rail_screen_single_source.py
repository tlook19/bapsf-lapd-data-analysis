"""One rail screen behind both of the products that record one.

The isat and isweep dead-time products carry the same-named rail attrs, and
each also carries the OPPOSITE face's dead-time fraction, so every one of those
numbers is measured from both sides of the Mach pair.  Two transcriptions of
one rule cannot be relied on to agree -- these two disagreed by a factor
n_positions until one of them was corrected -- so what these tests pin is not
that the writers agree today but that there is only one implementation left for
them to agree with, and that its denominators count every cell axis its
numerators sum over.

The fixture is a synthetic uint16 record with more rows than the screen reads
per block, so a position whose shots straddle a block boundary is counted once,
into one cell.
"""
import contextlib

import numpy as np
import pytest

from bapsf_lapd import ChannelKind
from bapsf_lapd.config import AcquisitionConfig, ChannelConfig, SweepConfig
from bapsf_lapd import rail_screen
from bapsf_lapd.rail_screen import BATCH, CLIP_S, PLATEAU_MS, RAIL_HI, RAIL_LO
from scripts import annotate_rail_mask
from scripts import screen_rot180_saturation

HDF5_PATH = "raw/isat"
N_POSITIONS = 20
N_SHOTS = 3
N_CYCLES = 6
N_SAMPLES = 25000
SAMPLES_PER_WINDOW = 980

# 1 MHz effective sampling, so one sample is 1 us and the CLIP_S trim is 10
# samples.  Cycle k's dead window runs 14000 + 2000k .. 15000 + 2000k, trimmed
# to 980 samples.  The cycle mid-times are 14.5 .. 24.5 ms, so the 14-19 ms
# plateau selects cycles 0-2 and the plateau cells are a strict subset.
SWEEP = SweepConfig(
    ramp_voltage=100.0,
    tau_ramp_s=1.0e-3,
    tau_cycle_s=2.0e-3,
    n_cycles=N_CYCLES,
    t0_s=13.0e-3,
)
ACQUISITION = AcquisitionConfig(
    raw_sample_rate_hz=1.0e6,
    hardware_average_samples=1,
    n_positions=N_POSITIONS,
    n_shots_per_position=N_SHOTS,
)

# The position whose shots straddle the block boundary the screen reads on.
STRADDLING_POSITION = BATCH // N_SHOTS

# Every scalar an annotator divides into a rail attr, as (numerator,
# denominator) keys of the screen's return.
SCALAR_PAIRS = (
    ("rail_all", "n_all"),
    ("rail_dead", "n_dead"),
    ("rail_plat", "n_plat"),
)


class _StubConfig:
    sweep = SWEEP
    acquisition = ACQUISITION

    def channel(self, kind):
        return ChannelConfig(kind=kind, hdf5_path=HDF5_PATH)


class _StubRun:
    """A run whose raw record is a synthetic uint16 array."""

    def __init__(self, raw):
        self._raw = raw
        self.config = _StubConfig()

    @contextlib.contextmanager
    def open(self):
        yield {HDF5_PATH: self._raw}


def _dead_windows():
    """(start, stop) of each CLIP_S-trimmed dead-time window, in samples."""
    dt = ACQUISITION.sample_dt_s
    clip = int(round(CLIP_S / dt))
    windows = []
    for k in range(N_CYCLES):
        start = int(round((SWEEP.t0_s + k * SWEEP.tau_cycle_s + SWEEP.tau_ramp_s) / dt))
        stop = int(round((SWEEP.t0_s + (k + 1) * SWEEP.tau_cycle_s) / dt))
        windows.append((start + clip, stop - clip))
    return windows


def _rail(raw, position, cycle, n_samples, code):
    """Rail the first ``n_samples`` of one cell, in every shot of its position."""
    start, _ = _dead_windows()[cycle]
    for shot in range(N_SHOTS):
        raw[position * N_SHOTS + shot, start:start + n_samples] = code


def _railed_run():
    raw = np.full((N_POSITIONS * N_SHOTS, N_SAMPLES), 30000, dtype=np.uint16)
    # A wholly railed cell, a partly railed one on the position that straddles
    # the read block boundary, and one outside the plateau window.
    _rail(raw, 2, 0, SAMPLES_PER_WINDOW, RAIL_HI)
    _rail(raw, STRADDLING_POSITION, 1, 41, RAIL_LO)
    _rail(raw, 19, 5, 130, RAIL_HI)
    return _StubRun(raw)


def _bits(value):
    """The float64 bit pattern an annotator would write for a rail attr."""
    return np.float64(value).tobytes()


def test_the_fixture_straddles_a_read_block_and_a_plateau_edge():
    assert N_POSITIONS * N_SHOTS > BATCH
    straddle = range(STRADDLING_POSITION * N_SHOTS,
                     (STRADDLING_POSITION + 1) * N_SHOTS)
    assert BATCH in straddle and straddle[0] < BATCH
    assert {stop - start for start, stop in _dead_windows()} == {SAMPLES_PER_WINDOW}
    assert _dead_windows()[-1][1] <= N_SAMPLES
    t_ms = rail_screen.inter_sweep_times_ms(SWEEP)
    plateau = (t_ms >= PLATEAU_MS[0]) & (t_ms <= PLATEAU_MS[1])
    assert list(np.flatnonzero(plateau)) == [0, 1, 2]


def test_both_product_writers_reach_one_screen_implementation():
    # The isat annotator calls it directly; the isweep annotator reaches it
    # through the screen script it imports.  Same function object, so no
    # second transcription of the rule exists to drift.
    assert annotate_rail_mask.screen_cells is rail_screen.screen_cells
    assert screen_rot180_saturation.screen_cells is rail_screen.screen_cells
    for constant in ("CLIP_S", "PLATEAU_MS", "RAIL_LO", "RAIL_HI", "METHOD"):
        assert getattr(annotate_rail_mask, constant) is getattr(rail_screen, constant)


def test_the_two_writers_measure_a_railed_run_bitwise_alike():
    run = _railed_run()

    cells = annotate_rail_mask.screen_cells(run, ChannelKind.ISAT)
    scalars = screen_rot180_saturation.screen(run, ChannelKind.ISAT)

    for numerator, denominator in SCALAR_PAIRS:
        assert cells[numerator] == scalars[numerator]
        assert cells[denominator] == scalars[denominator]
        # The attr value itself, not just the counts behind it.
        assert _bits(cells[numerator] / cells[denominator]) == _bits(
            scalars[numerator] / scalars[denominator]
        )
    assert cells["code_lo"] == scalars["code_lo"]
    assert cells["code_hi"] == scalars["code_hi"]
    assert cells["rail_dead"] > 0  # not a comparison of two zeros


def test_the_shared_denominators_count_every_cell_axis_the_counts_span():
    run = _railed_run()

    cells = annotate_rail_mask.screen_cells(run, ChannelKind.ISAT)
    scalars = screen_rot180_saturation.screen(run, ChannelKind.ISAT)
    plateau_cycles = int(cells["plateau_columns"].sum())

    for measured in (cells, scalars):
        assert measured["n_all"] == N_POSITIONS * N_SHOTS * N_SAMPLES
        assert measured["n_dead"] == (
            N_POSITIONS * N_SHOTS * N_CYCLES * SAMPLES_PER_WINDOW
        )
        assert measured["n_plat"] == (
            N_POSITIONS * N_SHOTS * plateau_cycles * SAMPLES_PER_WINDOW
        )
        # A denominator short of the position axis reads n_positions too large.
        assert measured["rail_dead"] / measured["n_dead"] <= 1.0
        assert measured["rail_plat"] / measured["n_plat"] <= 1.0

    weights = np.broadcast_to(cells["n_cell_samples"], cells["rail_counts"].shape)
    assert cells["rail_dead"] / cells["n_dead"] == pytest.approx(
        np.average(cells["rail_fraction"], weights=weights)
    )


def test_a_position_straddling_a_read_block_is_counted_into_one_cell():
    cells = annotate_rail_mask.screen_cells(_railed_run(), ChannelKind.ISAT)

    counts = cells["rail_counts"]
    assert counts[STRADDLING_POSITION, 1] == N_SHOTS * 41
    assert counts[2, 0] == N_SHOTS * SAMPLES_PER_WINDOW
    assert counts[19, 5] == N_SHOTS * 130
    assert counts.sum() == cells["rail_dead"]
    assert int(cells["rail_mask"].sum()) == 3
