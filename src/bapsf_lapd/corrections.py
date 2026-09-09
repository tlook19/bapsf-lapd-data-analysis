"""Small analysis corrections for known run-log issues."""

from __future__ import annotations

from bapsf_lapd.config import ChannelKind


# Dead-time face comparisons indicate ES3 p21 was recorded with the two
# rotation labels swapped.  Keep raw run IDs unchanged, but use this effective
# rotation anywhere downstream analysis groups by probe orientation.
EFFECTIVE_ROTATION_OVERRIDES: dict[str, float] = {
    "32": 180.0,
    "33": 0.0,
}

# Dead-time face comparisons indicate the electrical connections were swapped
# for this specific run, not for every p11 run.
ELECTRICAL_SWAP_RUN_IDS: frozenset[str] = frozenset({"31"})


def effective_rotation_deg(run_id: str, recorded_rotation_deg: float | None) -> float:
    """Return the analysis rotation after applying known run-specific fixes."""
    if run_id in EFFECTIVE_ROTATION_OVERRIDES:
        return EFFECTIVE_ROTATION_OVERRIDES[run_id]
    return float(recorded_rotation_deg or 0.0)


def electrical_connections_swapped(run_id: str) -> bool:
    """Return whether known wiring corrections swap ISAT and I_SWEEP faces."""
    return run_id in ELECTRICAL_SWAP_RUN_IDS


def effective_deadtime_source(
    run_id: str,
    port: int | None,
    requested_channel: ChannelKind,
    requested_invert_polarity: bool,
) -> tuple[ChannelKind, bool, bool]:
    """Return the channel/polarity to use for dead-time current products.

    For runs in ``ELECTRICAL_SWAP_RUN_IDS``, the physical upstream face is
    recorded on ``ISAT`` and the downstream face on ``I_SWEEP``.  Swap both
    nominal face products so rot-0 upstream (``-I_SWEEP``) sources ``ISAT`` and
    rot-0 downstream (``ISAT``) sources polarity-inverted ``I_SWEEP``.
    """
    del port  # reserved for future geometry-specific connection fixes
    if electrical_connections_swapped(run_id) and requested_channel == ChannelKind.I_SWEEP:
        return ChannelKind.ISAT, False, True
    if electrical_connections_swapped(run_id) and requested_channel == ChannelKind.ISAT:
        return ChannelKind.I_SWEEP, True, True
    return requested_channel, requested_invert_polarity, False


def density_area_key_for_deadtime_source(
    port: int | None,
    source_channel: ChannelKind,
) -> str:
    """Return the calibration TOML area key of the electrode that collected.

    The face areas are keyed by ELECTRODE, not by which face happened to look
    upstream: ``scripts/calibrate_probe_areas.py`` accumulates ``ap_R_m2`` from
    the rot-180 ISAT product and ``ap_L_m2`` from the rot-0 I_SWEEP product, so
    ``ap_R_cm2`` is the ISAT electrode's area and ``ap_L_cm2`` the I_SWEEP
    electrode's.  A density built from a dead-time current reproduces the
    calibration that defined the areas only under this assignment, and it holds
    at every port and in both rotations.

    ``port`` is accepted for call-site symmetry with ``effective_deadtime_source``
    and does not enter the answer.  The port-11 wiring-swap run, whose upstream
    row sources ISAT, still gets ``ap_R_cm2`` because ISAT is its collecting
    channel.

    Raises ``ValueError`` for a channel that has no calibrated face area.
    """
    del port  # the collecting electrode, not the port, sets the area
    if source_channel == ChannelKind.ISAT:
        return "ap_R_cm2"
    if source_channel == ChannelKind.I_SWEEP:
        return "ap_L_cm2"
    raise ValueError(f"no calibrated face area for channel {source_channel!r}")
