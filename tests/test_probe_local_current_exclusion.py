"""Run 22's late-afterglow probe-local-current exclusion.

The finding: run 22 (experiment set 2, port 21, rot-0) carries a probe-local
current on its ISAT channel that dominates the late-afterglow decay -- tau
22.66 ms against 7.18 ms on the run's own reference photodiode and 6.61-9.14
ms on every neighbouring set-2 run -- while its I_SWEEP channel and its
rot-180 partner run 23 are both normal.  ``LATE_AFTERGLOW_PROBE_LOCAL_CURRENT``
registers the run's ISAT channel, and ``_isat_decay_stats`` NaN-fills that
run's late-afterglow mean/SEM rather than dropping it (so the PORTS-equality
checks in ``export_overlay`` still pass) and carries the exclusion on new
``*_excluded`` / ``*_excluded_reason`` arrays.

The tests below check the registry's own shape first (no data needed), then,
against live data when present, that the exclusion lands on exactly the
registered run and channel and nowhere else: run 22's ISAT-channel row is
NaN-filled and flagged, its own I_SWEEP-channel row (a separate
``_isat_decay_stats`` call, over a different product) is untouched, and every
OTHER run in the same call is untouched.
"""

import numpy as np
import pytest

from bapsf_lapd import LapdDataset
from scripts.export_es1_sim1d_overlay import (
    ISAT_PROFILE_HDF5,
    LATE_AFTERGLOW_PROBE_LOCAL_CURRENT,
    ROT0_ISAT_PROFILE_HDF5,
    ZERO_OFFSETS,
    _isat_decay_stats,
)


def test_registry_entries_carry_every_required_field_and_a_wider_neighbor_range():
    for (run_id, channel), entry in LATE_AFTERGLOW_PROBE_LOCAL_CURRENT.items():
        assert isinstance(run_id, str) and isinstance(channel, str)
        for key in (
            "reason",
            "tau_ms",
            "photodiode_tau_ms",
            "neighbor_tau_range_ms",
            "undecayed_current_ma",
            "undecayed_time_ms",
            "source",
        ):
            assert key in entry, f"({run_id}, {channel}) is missing {key!r}"
        lo, hi = entry["neighbor_tau_range_ms"]
        assert lo < hi
        # The registered tau is well outside the neighbour bracket, and well
        # above the run's own reference-photodiode decay -- that departure
        # from both independent references is the whole finding.
        assert entry["tau_ms"] > hi
        assert entry["tau_ms"] > 2.0 * entry["photodiode_tau_ms"]


def test_run_22_is_registered_on_the_isat_channel_only():
    assert ("22", "isat") in LATE_AFTERGLOW_PROBE_LOCAL_CURRENT
    assert ("22", "i_sweep") not in LATE_AFTERGLOW_PROBE_LOCAL_CURRENT


DATASET = LapdDataset.from_directory("data/may2026")


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_run_22_isat_channel_late_afterglow_is_nan_filled_and_flagged():
    dataset = LapdDataset.from_manifest("config/may2026_run_manifest.toml")
    isat_dn = _isat_decay_stats(dataset, ROT0_ISAT_PROFILE_HDF5, ZERO_OFFSETS, 2)

    order = {str(r): i for i, r in enumerate(isat_dn["run_id"])}
    assert "22" in order
    idx22 = order["22"]

    assert bool(isat_dn["excluded"][idx22]) is True
    assert isat_dn["excluded_reason"][idx22] != ""
    assert np.all(np.isnan(isat_dn["mean_a"][idx22]))
    assert np.all(np.isnan(isat_dn["sem_a"][idx22]))

    for run_id, idx in order.items():
        if run_id == "22":
            continue
        assert bool(isat_dn["excluded"][idx]) is False
        assert isat_dn["excluded_reason"][idx] == ""
        assert not np.all(np.isnan(isat_dn["mean_a"][idx]))


@pytest.mark.skipif(len(DATASET) == 0, reason="local HDF5 data files are not present")
def test_run_22_i_sweep_channel_late_afterglow_is_untouched():
    dataset = LapdDataset.from_manifest("config/may2026_run_manifest.toml")
    isat_up = _isat_decay_stats(dataset, ISAT_PROFILE_HDF5, ZERO_OFFSETS, 2)

    order = {str(r): i for i, r in enumerate(isat_up["run_id"])}
    idx22 = order["22"]

    assert bool(isat_up["excluded"][idx22]) is False
    assert isat_up["excluded_reason"][idx22] == ""
    assert not np.all(np.isnan(isat_up["mean_a"][idx22]))
