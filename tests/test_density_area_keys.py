"""The dead-time area key is the collecting ELECTRODE's, at every port.

``density_area_key_for_deadtime_source`` answers one question: which entry of
``processed/probe_area_calibration.toml`` is the area of the electrode whose
current a dead-time product carries.  The areas were produced per electrode --
``scripts/calibrate_probe_areas.py`` accumulates ``ap_R_m2`` from the rot-180
ISAT product and ``ap_L_m2`` from the rot-0 I_SWEEP product -- so ``ap_R_cm2``
is the right electrode's area and ``ap_L_cm2`` the left one's, with no port and
no rotation entering it.

Which electrode a channel sat on is fixed by the WIRING.  Under the nominal
wiring ISAT is the right electrode and I_SWEEP the left.  On a run in
``ELECTRICAL_SWAP_RUN_IDS`` the two cables were crossed at the connector -- the
channel labelled ISAT read the LEFT electrode and the channel labelled I_SWEEP
the RIGHT one -- so the two answers exchange on that run and on no other.

Two levels are checked.  The first is closed-form and reads nothing: every run
in the manifest through the source-channel resolver the products actually call,
and the crossed-cable run explicitly, both of its rows.  The second reads the
placed Mach product, which stamps the face-to-area assignment per run
independently of this helper, and requires the two to agree run by run.  Under
the nominal wiring the ISAT electrode is the UPSTREAM face at rot-180 and the
DOWNSTREAM face at rot-0; a crossed-cable run inverts that, because its ISAT
channel sits on the other electrode.  That second level skips when the product
is not in the checkout, and refuses to pass vacuously.
"""

from pathlib import Path

import h5py
import pytest

from bapsf_lapd import (
    ChannelKind,
    ELECTRICAL_SWAP_RUN_IDS,
    LapdDataset,
    density_area_key_for_deadtime_source,
    effective_deadtime_source,
    effective_rotation_deg,
)

MANIFEST = "config/may2026_run_manifest.toml"
MACH_PRODUCT = Path("processed/mach_velocity.hdf5")

# The electrode each analysis channel collects on under the nominal wiring, and
# therefore its area key.
ELECTRODE_AREA_KEY = {
    ChannelKind.ISAT: "ap_R_cm2",
    ChannelKind.I_SWEEP: "ap_L_cm2",
}

# The same map on a run whose cables were crossed at the connector.
CROSSED_ELECTRODE_AREA_KEY = {
    ChannelKind.ISAT: "ap_L_cm2",
    ChannelKind.I_SWEEP: "ap_R_cm2",
}


def expected_area_key(run_id: str | None, channel: ChannelKind) -> str:
    """The area key ``channel`` must take on ``run_id``, stated independently."""
    if run_id is not None and run_id in ELECTRICAL_SWAP_RUN_IDS:
        return CROSSED_ELECTRODE_AREA_KEY[channel]
    return ELECTRODE_AREA_KEY[channel]


def test_every_manifest_run_and_source_takes_its_electrode_key():
    """No port moves the answer: only which channel collected, and the wiring."""
    dataset = LapdDataset.from_manifest(MANIFEST)

    ports = sorted({int(dataset.config(r).probe.port or 0) for r in dataset.run_ids()})
    assert ports == [11, 21, 29, 41, 50]

    # With no run named, the nominal wiring is the answer.
    for source, expected in ELECTRODE_AREA_KEY.items():
        assert density_area_key_for_deadtime_source(None, source) == expected

    # Every run in the manifest, through the source-channel resolver that the
    # products actually call, including the two rotation-override runs and the
    # crossed-cable run.
    swapped_seen = 0
    for run_id in dataset.run_ids():
        config = dataset.config(run_id)
        port = config.probe.port
        for requested in (ChannelKind.ISAT, ChannelKind.I_SWEEP):
            source, _, _ = effective_deadtime_source(run_id, port, requested, False)
            assert (
                density_area_key_for_deadtime_source(run_id, source)
                == expected_area_key(run_id, source)
            )
        if run_id in ELECTRICAL_SWAP_RUN_IDS:
            swapped_seen += 1
    assert swapped_seen == len(ELECTRICAL_SWAP_RUN_IDS)


def test_the_crossed_cable_run_takes_the_other_electrode_on_each_channel():
    """Run 31's ISAT channel read the LEFT electrode, so its rows exchange areas.

    Its upstream row sources ISAT and therefore takes ``ap_L_cm2``, and its
    downstream row sources I_SWEEP and takes ``ap_R_cm2`` -- the reverse of a
    nominally wired run, and the only run in the set for which that is true.
    """
    source, invert, overridden = effective_deadtime_source(
        "31", 11, ChannelKind.I_SWEEP, True
    )
    assert (source, invert, overridden) == (ChannelKind.ISAT, False, True)
    assert density_area_key_for_deadtime_source("31", source) == "ap_L_cm2"

    source, _, _ = effective_deadtime_source("31", 11, ChannelKind.ISAT, False)
    assert source == ChannelKind.I_SWEEP
    assert density_area_key_for_deadtime_source("31", source) == "ap_R_cm2"

    # A nominally wired run at the same port answers the other way round.
    assert density_area_key_for_deadtime_source("41", ChannelKind.ISAT) == "ap_R_cm2"
    assert density_area_key_for_deadtime_source("41", ChannelKind.I_SWEEP) == "ap_L_cm2"


def test_a_channel_with_no_calibrated_face_area_is_refused():
    for channel in (
        ChannelKind.V_SWEEP,
        ChannelKind.REFERENCE_PHOTODIODE,
        ChannelKind.MOVING_PHOTODIODE,
    ):
        with pytest.raises(ValueError, match="no calibrated face area"):
            density_area_key_for_deadtime_source("33", channel)


def test_a_port_left_in_the_run_id_slot_is_refused():
    """The first argument used to be the port; an int there must not answer."""
    for port in (11, 21, 29, 41, 50):
        with pytest.raises(TypeError, match="takes the run id first"):
            density_area_key_for_deadtime_source(port, ChannelKind.ISAT)


def test_helper_agrees_with_the_mach_products_stamped_face_areas(capsys):
    """The gate: helper answer == the Mach product's key for that face."""
    if not MACH_PRODUCT.exists():
        pytest.skip(f"{MACH_PRODUCT} is not in this checkout")

    dataset = LapdDataset.from_manifest(MANIFEST)
    rows = []
    with h5py.File(MACH_PRODUCT, "r") as hf:
        for set_id in sorted(hf["experiment_sets"]):
            for run_id in sorted(hf[f"experiment_sets/{set_id}"]):
                group = hf[f"experiment_sets/{set_id}/{run_id}"]
                if "upstream_area_key" not in group.attrs:
                    continue
                rows.append((
                    set_id,
                    run_id,
                    int(group.attrs["port"]),
                    float(group.attrs["rotation_deg"]),
                    str(group.attrs["upstream_area_key"]),
                    str(group.attrs["downstream_area_key"]),
                ))

    assert rows, f"{MACH_PRODUCT} carries no run with stamped face area keys"

    print(
        f"\n{'set':>3} {'run':>3} {'port':>4} {'rot':>5} {'wiring':>8}  "
        f"{'upstream':>9} {'downstream':>11}  {'helper(ISAT)':>12} "
        f"{'helper(ISWEEP)':>14}  verdict"
    )
    for set_id, run_id, port, rot, upstream, downstream in rows:
        helper_isat = density_area_key_for_deadtime_source(run_id, ChannelKind.ISAT)
        helper_isweep = density_area_key_for_deadtime_source(run_id, ChannelKind.I_SWEEP)

        # The ISAT electrode is upstream at rot-180 and downstream at rot-0 --
        # unless the cables were crossed at the connector, which puts the ISAT
        # channel on the other electrode and inverts both answers.
        crossed = run_id in ELECTRICAL_SWAP_RUN_IDS
        isat_is_upstream = (rot == 180.0) != crossed
        if isat_is_upstream:
            expect_isat, expect_isweep = upstream, downstream
        else:
            expect_isat, expect_isweep = downstream, upstream
        ok = helper_isat == expect_isat and helper_isweep == expect_isweep
        print(
            f"{set_id:>3} {run_id:>3} {port:>4} {rot:>5.0f} "
            f"{'crossed' if crossed else 'nominal':>8}  "
            f"{upstream:>9} {downstream:>11}  {helper_isat:>12} "
            f"{helper_isweep:>14}  {'MATCH' if ok else 'MISMATCH'}"
        )

        recorded = dataset.config(run_id).probe.rotation_deg
        assert effective_rotation_deg(run_id, recorded) == rot, run_id
        assert helper_isat == expect_isat, (run_id, helper_isat, expect_isat)
        assert helper_isweep == expect_isweep, (run_id, helper_isweep, expect_isweep)

    captured = capsys.readouterr()
    print(captured.out)
    assert captured.out.count("MATCH") == len(rows)
    assert "MISMATCH" not in captured.out
    # The gate is only meaningful if the crossed-cable run is inside it.
    assert captured.out.count("crossed") == sum(
        1 for row in rows if row[1] in ELECTRICAL_SWAP_RUN_IDS
    )
