"""The dead-time area key is the collecting ELECTRODE's, at every port.

``density_area_key_for_deadtime_source`` answers one question: which entry of
``processed/probe_area_calibration.toml`` is the area of the electrode whose
current a dead-time product carries.  The areas were produced per electrode --
``scripts/calibrate_probe_areas.py`` accumulates ``ap_R_m2`` from the rot-180
ISAT product and ``ap_L_m2`` from the rot-0 I_SWEEP product -- so the answer is
``ap_R_cm2`` for an ISAT source and ``ap_L_cm2`` for an I_SWEEP source, with no
port and no rotation entering it.

Two levels are checked.  The first is closed-form and reads nothing: every
(port, source) pair the manifest can produce, plus the port-11 wiring-swap run
whose ISAT-sourced upstream row keeps ``ap_R_cm2`` because ISAT is the channel
that collected it.  The second reads the placed Mach product, which stamps the
face-to-area assignment per run independently of this helper, and requires the
two to agree run by run: the ISAT electrode is the UPSTREAM face at rot-180 and
the DOWNSTREAM face at rot-0, so the helper's ISAT answer must equal
``upstream_area_key`` on every rot-180 run and ``downstream_area_key`` on every
rot-0 run, and the I_SWEEP answer the other way round.  That second level skips
when the product is not in the checkout, and refuses to pass vacuously.
"""

from pathlib import Path

import h5py
import pytest

from bapsf_lapd import (
    ChannelKind,
    LapdDataset,
    density_area_key_for_deadtime_source,
    effective_deadtime_source,
    effective_rotation_deg,
)

MANIFEST = "config/may2026_run_manifest.toml"
MACH_PRODUCT = Path("processed/mach_velocity.hdf5")

# The electrode each analysis channel collects on, and therefore its area key.
ELECTRODE_AREA_KEY = {
    ChannelKind.ISAT: "ap_R_cm2",
    ChannelKind.I_SWEEP: "ap_L_cm2",
}


def test_every_manifest_port_and_source_takes_the_electrode_key():
    """No port moves the answer: only which channel collected the current."""
    dataset = LapdDataset.from_manifest(MANIFEST)

    ports = sorted({int(dataset.config(r).probe.port or 0) for r in dataset.run_ids()})
    assert ports == [11, 21, 29, 41, 50]

    for port in ports:
        for source, expected in ELECTRODE_AREA_KEY.items():
            assert density_area_key_for_deadtime_source(port, source) == expected

    # Every run in the manifest, through the source-channel resolver that the
    # products actually call, including the two rotation-override runs.
    for run_id in dataset.run_ids():
        config = dataset.config(run_id)
        port = config.probe.port
        for requested in (ChannelKind.ISAT, ChannelKind.I_SWEEP):
            source, _, _ = effective_deadtime_source(run_id, port, requested, False)
            assert (
                density_area_key_for_deadtime_source(port, source)
                == ELECTRODE_AREA_KEY[source]
            )


def test_the_port_11_wiring_swap_run_keeps_the_isat_electrode_area():
    """Run 31's upstream row sources ISAT, so it keeps ``ap_R_cm2``.

    This is the case the helper used to special-case by port.  It is now a
    consequence of the electrode rule rather than an exception to it, and the
    answer is unchanged.
    """
    source, invert, overridden = effective_deadtime_source(
        "31", 11, ChannelKind.I_SWEEP, True
    )
    assert (source, invert, overridden) == (ChannelKind.ISAT, False, True)
    assert density_area_key_for_deadtime_source(11, source) == "ap_R_cm2"

    # Its downstream row sources I_SWEEP and takes the I_SWEEP electrode's area.
    source, _, _ = effective_deadtime_source("31", 11, ChannelKind.ISAT, False)
    assert source == ChannelKind.I_SWEEP
    assert density_area_key_for_deadtime_source(11, source) == "ap_L_cm2"


def test_a_channel_with_no_calibrated_face_area_is_refused():
    for channel in (
        ChannelKind.V_SWEEP,
        ChannelKind.REFERENCE_PHOTODIODE,
        ChannelKind.MOVING_PHOTODIODE,
    ):
        with pytest.raises(ValueError, match="no calibrated face area"):
            density_area_key_for_deadtime_source(21, channel)


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
        f"\n{'set':>3} {'run':>3} {'port':>4} {'rot':>5}  "
        f"{'upstream':>9} {'downstream':>11}  {'helper(ISAT)':>12} "
        f"{'helper(ISWEEP)':>14}  verdict"
    )
    for set_id, run_id, port, rot, upstream, downstream in rows:
        helper_isat = density_area_key_for_deadtime_source(port, ChannelKind.ISAT)
        helper_isweep = density_area_key_for_deadtime_source(port, ChannelKind.I_SWEEP)

        # The ISAT electrode is upstream at rot-180 and downstream at rot-0.
        if rot == 180.0:
            expect_isat, expect_isweep = upstream, downstream
        else:
            expect_isat, expect_isweep = downstream, upstream
        ok = helper_isat == expect_isat and helper_isweep == expect_isweep
        print(
            f"{set_id:>3} {run_id:>3} {port:>4} {rot:>5.0f}  "
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
