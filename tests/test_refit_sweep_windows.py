"""Coverage of the fit-window re-fit product, including experiment set 4.

The re-fit pass reads raw digitizer traces, so these exercise it against a
synthetic sweep source and a synthetic dataset rather than the recorded runs:
what is pinned is which runs the pass walks and what the product records about
its own coverage, neither of which depends on the measured values.
"""
import sys
import types

import h5py
import numpy as np
import pytest

from bapsf_lapd import ChannelKind
from scripts.refit_sweep_windows import (
    EXPERIMENT_SETS,
    F_HIGH,
    P_LOW,
    main,
    rot0_runs,
)

#: Samples per synthetic ramp -- long enough for the zero-phase filter's pad.
RAMP_SAMPLES = 400
SAMPLE_RATE_HZ = 1.0e6
TE_SYNTHETIC_EV = 3.0


def _write_sweeps(path, runs, *, x_cm=(-1.0, 0.0, 1.0)):
    """Write a minimal ``langmuir_sweeps``-shaped source.

    *runs* is an iterable of ``(set_id, run_id, port, rotation_deg)``; only the
    attrs ``rot0_runs`` reads are written.
    """
    with h5py.File(path, "w") as hdf:
        hdf.create_dataset("x_cm", data=np.asarray(x_cm, dtype=np.float64))
        for set_id, run_id, port, rotation_deg in runs:
            group = hdf.create_group(f"experiment_sets/{set_id}/{run_id}")
            group.attrs["port"] = np.int64(port)
            group.attrs["rotation_deg"] = np.float64(rotation_deg)
    return path


def _synthetic_sweep(shot_index):
    """Return (voltage, current) for one log-linear retarding branch."""
    voltage = np.linspace(-40.0, 10.0, RAMP_SAMPLES)
    ion_line = 1.0e-3 * voltage - 2.0e-2
    electron = 1.0 / (1.0 + np.exp(-voltage / TE_SYNTHETIC_EV))
    return voltage, ion_line + (1.0 + 0.01 * shot_index) * electron


class _FakeRun:
    """The subset of ``LapdRun`` the re-fit pass touches."""

    def __init__(self, n_positions=3, n_shots=2):
        self._n_positions = n_positions
        self._n_shots = n_shots
        self.config = types.SimpleNamespace(
            sweep=None,
            acquisition=types.SimpleNamespace(sample_rate_hz=SAMPLE_RATE_HZ),
        )

    def default_zero_offset_v(self, channel):
        return 0.0

    def sweep_ramp_sample_slices(self, clip_s=0.0):
        return [slice(0, RAMP_SAMPLES)] * 4

    def langmuir_traces(self, channel, sample_slice, zero_offset_v=None):
        traces = np.empty(
            (self._n_positions, self._n_shots, RAMP_SAMPLES), dtype=np.float64
        )
        for shot in range(self._n_shots):
            voltage, current = _synthetic_sweep(shot)
            trace = voltage if channel is ChannelKind.V_SWEEP else current
            traces[:, shot, :] = trace
        return traces


@pytest.fixture
def refit_environment(monkeypatch, tmp_path):
    """Point the pass at a synthetic sweep source and dataset."""
    import scripts.refit_sweep_windows as module

    sweeps = _write_sweeps(
        tmp_path / "langmuir_sweeps.hdf5",
        [
            (3, "31", 11, 0.0),
            (4, "41", 11, 0.0),
            (4, "43", 21, 180.0),
            (4, "44", 29, 0.0),
        ],
    )
    monkeypatch.setattr(module, "SWEEPS_H5", sweeps)
    monkeypatch.setattr(
        module,
        "LapdDataset",
        types.SimpleNamespace(
            from_manifest=lambda manifest: types.SimpleNamespace(
                run=lambda run_id: _FakeRun()
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "pls",
        types.SimpleNamespace(
            _cycle_start_times=lambda run: np.array([0.005, 0.012, 0.015, 0.025]),
            _cutoff_hz_for_run=lambda run, base: (base, None),
        ),
    )
    return module


def test_rot0_runs_walks_set_four_and_skips_its_rot180_run(refit_environment):
    walked = list(rot0_runs())
    assert (4, "41", 11) in walked
    assert (4, "44", 29) in walked
    assert not any(run_id == "43" for _, run_id, _ in walked)
    assert (3, "31", 11) in walked


def test_default_set_list_includes_set_four():
    assert 4 in EXPERIMENT_SETS
    assert rot0_runs.__defaults__ == (EXPERIMENT_SETS,)


def test_product_records_its_set_list_and_carries_a_set_four_p11_group(
    refit_environment, monkeypatch, tmp_path
):
    output = tmp_path / "sweep_window_refits.hdf5"
    monkeypatch.setattr(
        sys, "argv", ["refit_sweep_windows", "--output", str(output)]
    )
    main()

    with h5py.File(output, "r") as product:
        assert 4 in set(np.asarray(product.attrs["experiment_sets"]).tolist())
        assert "set4/port11" in product
        group = product["set4/port11"]
        assert group["te_window_ev"].shape == (len(P_LOW), len(F_HIGH))
        assert group.attrs["run_id"] == "41"
        # Set 4 has no rung in the beta-collapse tables, so the decision rule
        # has no reference scale -- the spread is still measured and recorded.
        assert not np.isfinite(group.attrs["ln_beta_ref"])
        assert group.attrs["verdict"] == "no beta-hat reference"
        assert np.isfinite(group.attrs["dln_te_window"])
        # A set that does carry a rung keeps its reference and its verdict.
        assert np.isfinite(product["set3/port11"].attrs["ln_beta_ref"])
