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
# for this specific run, not for every p11 run.  The swap is read as the two
# probe cables being CROSSED AT THE CONNECTOR: on such a run the channel
# labelled ISAT carries the LEFT electrode and the channel labelled I_SWEEP the
# RIGHT one.  Both the face a channel looked at and the electrode area it
# collected on therefore exchange, and they exchange together.
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

    For runs in ``ELECTRICAL_SWAP_RUN_IDS`` the cables are crossed at the
    connector, so at rot-0 the physical upstream (left) face is recorded on
    ``ISAT`` and the downstream (right) face on ``I_SWEEP``.  Swap both nominal
    face products so rot-0 upstream (``-I_SWEEP``) sources ``ISAT`` and rot-0
    downstream (``ISAT``) sources polarity-inverted ``I_SWEEP``.  The electrode
    area that goes with the sourced channel exchanges too --
    ``density_area_key_for_deadtime_source`` applies that half.
    """
    del port  # reserved for future geometry-specific connection fixes
    if electrical_connections_swapped(run_id) and requested_channel == ChannelKind.I_SWEEP:
        return ChannelKind.ISAT, False, True
    if electrical_connections_swapped(run_id) and requested_channel == ChannelKind.ISAT:
        return ChannelKind.I_SWEEP, True, True
    return requested_channel, requested_invert_polarity, False


def density_area_key_for_deadtime_source(
    run_id: str | None,
    source_channel: ChannelKind,
) -> str:
    """Return the calibration TOML area key of the electrode that collected.

    The face areas are keyed by ELECTRODE, not by which face happened to look
    upstream: ``scripts/calibrate_probe_areas.py`` accumulates ``ap_R_m2`` from
    the rot-180 ISAT product and ``ap_L_m2`` from the rot-0 I_SWEEP product, so
    ``ap_R_cm2`` is the RIGHT electrode's area and ``ap_L_cm2`` the LEFT one's.
    A density built from a dead-time current reproduces the calibration that
    defined the areas only under this assignment, and no port and no rotation
    enters it.

    Which electrode a channel collected on is fixed by the wiring, not by the
    channel label.  Under the nominal wiring ISAT sits on the right electrode
    and I_SWEEP on the left, so ISAT takes ``ap_R_cm2`` and I_SWEEP
    ``ap_L_cm2``.  On a run in ``ELECTRICAL_SWAP_RUN_IDS`` the two cables are
    crossed at the connector -- inferred from dead-time face comparisons and
    read as a connector swap -- so the channel labelled ISAT read the LEFT
    electrode and the channel labelled I_SWEEP the RIGHT one, and the two area
    keys exchange with them.

    ``run_id`` names the run whose channel is being asked about; ``None`` means
    no run is known and answers under the nominal wiring.  Every
    product-writing caller passes the run, so that a swapped run is never
    stamped as if it were unswapped.  The first argument used to be the probe
    port, so a non-string, non-``None`` ``run_id`` is refused rather than
    silently answering the nominal rule.

    Raises ``ValueError`` for a channel that has no calibrated face area, and
    ``TypeError`` for a ``run_id`` that is neither a string nor ``None``.
    """
    if run_id is not None and not isinstance(run_id, str):
        raise TypeError(
            "density_area_key_for_deadtime_source takes the run id first, not "
            f"the port; got {run_id!r}"
        )
    swapped = run_id is not None and electrical_connections_swapped(run_id)
    if source_channel == ChannelKind.ISAT:
        return "ap_L_cm2" if swapped else "ap_R_cm2"
    if source_channel == ChannelKind.I_SWEEP:
        return "ap_R_cm2" if swapped else "ap_L_cm2"
    raise ValueError(f"no calibrated face area for channel {source_channel!r}")
