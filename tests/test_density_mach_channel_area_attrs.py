"""The Mach-density product names the electrode behind each channel-named field.

``processed/density_mach.hdf5``'s ``n_e_R_m3`` and ``n_e_L_m3`` are named for
the CHANNEL that produced them -- R for ISAT, L for the negated I_SWEEP -- not
for the electrode that channel reached.  Under the nominal wiring the two
coincide, which is why the names read as electrode names; on a run whose cables
are crossed at the connector they do not, and the areas that divided the two
currents exchange.  ``scripts/compute_density_mach.py`` therefore stamps
``n_e_R_density_area_key`` and ``n_e_L_density_area_key`` per run, and this
file requires them to be the answers
``bapsf_lapd.density_area_key_for_deadtime_source`` gives for the two channels
-- the single owner of the channel -> electrode map, never restated here.

The datasets keep their channel names: renaming them would break every
consumer, and the attrs say what the names cannot.

Reads the product as placed.  It skips when the product is not in the checkout
or predates the attrs, and refuses to pass vacuously: the crossed-cable run
must be present and must read the exchanged pair.
"""

from pathlib import Path

import h5py
import pytest

from bapsf_lapd import (
    ChannelKind,
    ELECTRICAL_SWAP_RUN_IDS,
    density_area_key_for_deadtime_source,
)

MACH_DENSITY_PRODUCT = Path("processed/density_mach.hdf5")
R_ATTR = "n_e_R_density_area_key"
L_ATTR = "n_e_L_density_area_key"


def _rows(hdf):
    for set_id in sorted(hdf["experiment_sets"]):
        for run_id in sorted(hdf[f"experiment_sets/{set_id}"]):
            yield set_id, run_id, hdf[f"experiment_sets/{set_id}/{run_id}"].attrs


def test_each_channel_names_the_electrode_it_collected_on(capsys):
    """The gate: both stamped keys == the helper's answer for that channel."""
    if not MACH_DENSITY_PRODUCT.exists():
        pytest.skip(f"{MACH_DENSITY_PRODUCT} is not in this checkout")

    with h5py.File(MACH_DENSITY_PRODUCT, "r") as hdf:
        rows = [
            (set_id, run_id, str(attrs[R_ATTR]), str(attrs[L_ATTR]))
            for set_id, run_id, attrs in _rows(hdf)
            if R_ATTR in attrs and L_ATTR in attrs
        ]
        present = sum(1 for _ in _rows(hdf))

    if not rows:
        pytest.skip(
            f"{MACH_DENSITY_PRODUCT} carries {present} run(s), none with "
            f"{R_ATTR}/{L_ATTR}: it predates the attrs and needs "
            "scripts/compute_density_mach.py re-run"
        )

    crossed_seen = 0
    print(f"\n{'set':>3} {'run':>4} {'wiring':>8} {'n_e_R area':>11} {'n_e_L area':>11}")
    for set_id, run_id, r_key, l_key in rows:
        expect_r = density_area_key_for_deadtime_source(run_id, ChannelKind.ISAT)
        expect_l = density_area_key_for_deadtime_source(run_id, ChannelKind.I_SWEEP)
        crossed = run_id in ELECTRICAL_SWAP_RUN_IDS
        crossed_seen += crossed
        print(
            f"{set_id:>3} {run_id:>4} {'crossed' if crossed else 'nominal':>8} "
            f"{r_key:>11} {l_key:>11}"
        )
        assert (r_key, l_key) == (expect_r, expect_l), (
            f"run {run_id}: product says ({r_key}, {l_key}), the rule says "
            f"({expect_r}, {expect_l})"
        )
        # The two channels never share an electrode, whatever the wiring.
        assert r_key != l_key

    assert crossed_seen, (
        "no crossed-cable run in the product: the check would pass vacuously, "
        "since every nominal run reads the same keys either way"
    )
    capsys.readouterr()


def test_the_crossed_run_is_the_one_that_exchanges():
    """Closed-form, reading nothing: only a crossed-cable run exchanges."""
    for run_id in sorted(ELECTRICAL_SWAP_RUN_IDS):
        assert density_area_key_for_deadtime_source(run_id, ChannelKind.ISAT) == "ap_L_cm2"
        assert density_area_key_for_deadtime_source(run_id, ChannelKind.I_SWEEP) == "ap_R_cm2"
    nominal = "34"
    assert nominal not in ELECTRICAL_SWAP_RUN_IDS
    assert density_area_key_for_deadtime_source(nominal, ChannelKind.ISAT) == "ap_R_cm2"
    assert density_area_key_for_deadtime_source(nominal, ChannelKind.I_SWEEP) == "ap_L_cm2"
