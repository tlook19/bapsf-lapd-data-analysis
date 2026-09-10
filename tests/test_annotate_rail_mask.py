"""The rail screen's run-level scalars against its own per-cell fractions.

``screen_cells`` counts railed samples per (position, dead-time window) cell,
so a scalar summary of those counts must be normalised by the samples of every
cell it sums over -- both axes.  A denominator short of the position axis makes
the scalars n_positions times too large, which shows up as a "fraction" above
one on a heavily railed run.
"""
import contextlib

import numpy as np
import pytest

from bapsf_lapd import ChannelKind
from bapsf_lapd.config import AcquisitionConfig, ChannelConfig, SweepConfig
from scripts.annotate_rail_mask import (
    CLIP_S,
    PLATEAU_MS,
    RAIL_HI,
    RAIL_LO,
    inter_sweep_times_ms,
    refuse_processed_output,
    screen_cells,
)

HDF5_PATH = "raw/isat"
N_POSITIONS = 3
N_SHOTS = 2
N_CYCLES = 5
N_SAMPLES = 23000

# 1 MHz effective sampling, so one sample is 1 us and the CLIP_S trim is 10
# samples.  Cycle k's dead window runs 14000 + 2000k .. 15000 + 2000k, trimmed
# to 980 samples.  The cycle mid-times are 14.5, 16.5, 18.5, 20.5, 22.5 ms, so
# the 14-19 ms plateau selects cycles 0-2 and the plateau cells are a strict
# subset of the dead-time cells.
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
SAMPLES_PER_WINDOW = 980


class _StubConfig:
    sweep = SWEEP
    acquisition = ACQUISITION

    def channel(self, kind):
        return ChannelConfig(kind=kind, hdf5_path=HDF5_PATH)


class _StubRun:
    """A run whose raw ISAT record is a synthetic uint16 array."""

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


def _plateau_columns():
    t_ms = inter_sweep_times_ms(SWEEP)
    return (t_ms >= PLATEAU_MS[0]) & (t_ms <= PLATEAU_MS[1])


def _blank_record():
    # A mid-scale code: railed at neither converter rail.
    return np.full((N_POSITIONS * N_SHOTS, N_SAMPLES), 30000, dtype=np.uint16)


def _rail(raw, position, cycle, n_samples, code):
    """Rail the first ``n_samples`` of one cell, in every shot of its position."""
    start, _ = _dead_windows()[cycle]
    for shot in range(N_SHOTS):
        raw[position * N_SHOTS + shot, start:start + n_samples] = code


def _cell_samples(face):
    """Samples per cell, broadcast over both axes of the counted cells."""
    return np.broadcast_to(face["n_cell_samples"], face["rail_counts"].shape)


def _railed_run():
    raw = _blank_record()
    # A wholly railed cell, a partly railed one on another position, and one
    # outside the plateau, so the two scalars differ and neither is trivial.
    _rail(raw, 0, 0, SAMPLES_PER_WINDOW, RAIL_HI)
    _rail(raw, 2, 1, 7, RAIL_LO)
    _rail(raw, 1, 4, 130, RAIL_HI)
    return _StubRun(raw)


def test_the_dead_time_windows_are_the_ones_this_fixture_assumes():
    windows = _dead_windows()
    assert len(windows) == N_CYCLES
    assert {stop - start for start, stop in windows} == {SAMPLES_PER_WINDOW}
    assert windows[-1][1] <= N_SAMPLES
    assert list(np.flatnonzero(_plateau_columns())) == [0, 1, 2]


def test_the_dead_time_scalar_is_the_sample_weighted_mean_of_the_cell_fractions():
    face = screen_cells(_railed_run(), ChannelKind.ISAT)
    weights = _cell_samples(face)

    scalar = face["rail_dead"] / face["n_dead"]

    assert scalar == pytest.approx(np.average(face["rail_fraction"], weights=weights))
    assert scalar <= 1.0
    # The denominator counts every cell the numerator sums over: both axes.
    assert face["n_dead"] == N_POSITIONS * N_SHOTS * N_CYCLES * SAMPLES_PER_WINDOW


def test_the_plateau_scalar_is_the_sample_weighted_mean_over_the_plateau_cells():
    face = screen_cells(_railed_run(), ChannelKind.ISAT)
    plat = _plateau_columns()
    weights = _cell_samples(face)[:, plat]

    scalar = face["rail_plat"] / face["n_plat"]

    assert scalar == pytest.approx(
        np.average(face["rail_fraction"][:, plat], weights=weights)
    )
    assert scalar <= 1.0
    assert face["n_plat"] == N_POSITIONS * N_SHOTS * int(plat.sum()) * SAMPLES_PER_WINDOW
    # The plateau cells are a strict subset, and the two scalars are distinct.
    assert face["n_plat"] < face["n_dead"]
    assert face["rail_plat"] < face["rail_dead"]


def test_a_wholly_railed_record_reads_exactly_one_on_both_scalars():
    # Every dead-time sample of every cell railed: the fraction is 1 by
    # definition, and a denominator missing the position axis would read
    # N_POSITIONS instead.
    raw = _blank_record()
    for position in range(N_POSITIONS):
        for cycle in range(N_CYCLES):
            _rail(raw, position, cycle, SAMPLES_PER_WINDOW, RAIL_HI)

    face = screen_cells(_StubRun(raw), ChannelKind.ISAT)

    assert np.all(face["rail_fraction"] == 1.0)
    assert face["rail_dead"] / face["n_dead"] == pytest.approx(1.0)
    assert face["rail_plat"] / face["n_plat"] == pytest.approx(1.0)


def test_an_unrailed_record_reads_zero_on_both_scalars():
    face = screen_cells(_StubRun(_blank_record()), ChannelKind.ISAT)

    assert face["rail_dead"] == 0
    assert face["rail_dead"] / face["n_dead"] == 0.0
    assert face["rail_plat"] / face["n_plat"] == 0.0
    assert not face["rail_mask"].any()


def test_a_processed_product_path_is_refused_without_the_flag(tmp_path):
    """The guard fires at argument resolution, before any file is opened."""
    placed = tmp_path / "processed" / "isat_rot180_deadtime_profiles.hdf5"
    placed.parent.mkdir()
    placed.write_bytes(b"not really hdf5, the guard never opens it")
    with pytest.raises(ValueError, match="--allow-processed"):
        refuse_processed_output(placed, allow_processed=False)
    # The guard runs before the file is touched, so its (unreal) content
    # cannot have moved either.
    assert placed.read_bytes() == b"not really hdf5, the guard never opens it"

    refuse_processed_output(placed, allow_processed=True)
    refuse_processed_output(tmp_path / "regen" / "isat_rot180.hdf5", allow_processed=False)
