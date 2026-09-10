"""Re-stamp ``density_area_key`` on a dead-time line-scan product, in place.

RE-STAMPS
    the per-run ``density_area_key`` attribute of any product written by
    scripts/plot_isat_profiles.py -- processed/isat_profiles.hdf5,
    processed/isat_rot180_deadtime_profiles.hdf5,
    processed/isweep_deadtime_profiles.hdf5,
    processed/isweep_rot180_deadtime_profiles.hdf5.

FOLLOWS THE WRITER
    scripts/plot_isat_profiles.py stamps that attribute from
    ``bapsf_lapd.density_area_key_for_deadtime_source(run_id, source_channel)``
    at build time.  A product built before that rule reached its current form
    carries the earlier answer, and rebuilding it is not always available: the
    two rot-180 products carry annotator-written datasets and attrs
    (``rail_mask``/``rail_fraction`` from scripts/annotate_rail_mask.py,
    ``state_mask`` from scripts/annotate_state_mask.py, and the run-level attrs
    that go with them), which a rebuild would drop.  This pass moves ONLY the
    stale attribute and leaves every other attribute and every dataset exactly
    as it found them.

THE RULE IT APPLIES
    ``density_area_key_for_deadtime_source`` is the single owner of the
    channel -> electrode map, and this pass never restates it.  The channel it
    asks about is the one the product actually averaged, read from each run
    group's own ``deadtime_source_channel`` attr, so a source override
    (the crossed-cable runs) is honoured rather than re-derived; the run is
    read from the group's own ``run_id`` attr.  A group missing either attr is
    an error, not a group to answer nominally for.

OUTPUT GUARD
    ``processed/`` holds the placed products the transport campaign scores
    against, so a pass that names a path inside it is refused at argument
    resolution, before any file is opened, unless ``--allow-processed`` says
    that is the intent.  The pattern follows
    scripts/refit_window_band.py's ``refuse_band_output_paths``.

Usage::

    python scripts/restamp_density_area_keys.py <product.hdf5> [<product.hdf5> ...]
    python scripts/restamp_density_area_keys.py --allow-processed \\
        processed/isat_profiles.hdf5

Prints a before/after table over every run group and a per-product count of
the rows it moved.  Exits 0 whether or not anything moved; a product already
carrying the current answer is left byte-identical.

ALSO: ``--stamp-calibration-middle-fraction``
    A second, independent pass that stamps the root attribute
    ``calibration_middle_fraction`` (``scripts/plot_isat_profiles.py:597``)
    onto a product that carries NONE -- the two rot-0 products
    (``processed/isat_profiles.hdf5``, ``processed/isweep_deadtime_profiles.hdf5``)
    were built before that attr existed, while the two rot-180 products carry
    it (0.5, the writer's default).  This pass never overwrites: a product
    that already carries the attr, at any value, is refused rather than
    silently corrected, so a genuine mismatch surfaces instead of being
    papered over.  Given instead of the positional re-stamp, it touches ONLY
    the file-level attribute and leaves every dataset and every group attr
    byte-identical -- the reviewer placing the result is expected to diff the
    two products (raw bytes, per dataset and group attr) to confirm that.
    Subject to the same ``processed/`` OUTPUT GUARD above.

    Usage::

        python scripts/restamp_density_area_keys.py \\
            --stamp-calibration-middle-fraction 0.5 <product.hdf5> [...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd import ChannelKind, density_area_key_for_deadtime_source  # noqa: E402

AREA_KEY_ATTR = "density_area_key"
CHANNEL_ATTR = "deadtime_source_channel"
RUN_ATTR = "run_id"
SETS_GROUP = "experiment_sets"
CALIBRATION_MIDDLE_FRACTION_ATTR = "calibration_middle_fraction"


def _text(value) -> str:
    """Return an HDF5 string attribute as ``str`` whether it is bytes or str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def refuse_processed_paths(products, allow_processed: bool) -> None:
    """Refuse to edit a placed product unless the caller asked for it.

    ``processed/`` is the transport campaign's scoring chain: a product there
    is consumed by name, so an in-place edit of one is a measurement-side
    change, not a scratch operation.  Every path with a ``processed``
    directory component is therefore refused here, at argument resolution and
    before any file is opened, unless ``--allow-processed`` is passed.
    """
    if allow_processed:
        return
    inside = [
        str(path)
        for path in products
        if "processed" in Path(path).resolve().parts[:-1]
    ]
    if not inside:
        return
    raise ValueError(
        f"refusing to re-stamp {', '.join(inside)}: the path lies under a "
        "processed/ directory, which holds the placed products the transport "
        "campaign scores against, and this pass edits its input IN PLACE. "
        "Re-stamp a copy outside processed/, or pass --allow-processed to say "
        "that editing the placed product is the intent."
    )


def restamp_product(path: Path) -> tuple[int, int, list[tuple[str, str, str, str, str]]]:
    """Re-stamp one product in place; return (rows, moved, table).

    ``table`` rows are ``(set_id, run_id, source_channel, before, after)`` for
    every run group, in file order, whether or not the row moved.
    """
    table: list[tuple[str, str, str, str, str]] = []
    moved = 0
    with h5py.File(path, "r+") as hdf:
        if SETS_GROUP not in hdf:
            raise ValueError(f"{path}: no {SETS_GROUP!r} group -- not a dead-time product")
        sets = hdf[SETS_GROUP]
        for set_id in sorted(sets.keys(), key=str):
            set_group = sets[set_id]
            if not isinstance(set_group, h5py.Group):
                continue
            for group_name in sorted(set_group.keys(), key=str):
                run_group = set_group[group_name]
                if not isinstance(run_group, h5py.Group):
                    continue
                attrs = run_group.attrs
                if RUN_ATTR not in attrs:
                    raise ValueError(
                        f"{path}: {SETS_GROUP}/{set_id}/{group_name} carries no "
                        f"{RUN_ATTR!r} attr; the run a group belongs to is read "
                        "from the group, never guessed from its name"
                    )
                if CHANNEL_ATTR not in attrs:
                    raise ValueError(
                        f"{path}: {SETS_GROUP}/{set_id}/{group_name} carries no "
                        f"{CHANNEL_ATTR!r} attr; the channel the product averaged "
                        "is read from the group, never re-derived from the file name"
                    )
                run_id = _text(attrs[RUN_ATTR])
                channel = _text(attrs[CHANNEL_ATTR])
                before = _text(attrs[AREA_KEY_ATTR]) if AREA_KEY_ATTR in attrs else "<absent>"
                after = density_area_key_for_deadtime_source(run_id, ChannelKind(channel))
                table.append((str(set_id), run_id, channel, before, after))
                if before != after:
                    attrs[AREA_KEY_ATTR] = after
                    moved += 1
    return len(table), moved, table


def stamp_calibration_middle_fraction(path: Path, value: float) -> bool:
    """Stamp the root ``calibration_middle_fraction`` attr, only if absent.

    Touches ONLY the file-level attribute -- no dataset, no group attr, moves.
    A product that already carries the attr (at any value) is refused rather
    than silently overwritten or silently left mismatched: this pass makes a
    product UNIVERSAL, it does not correct one that disagrees.  Returns
    whether the attr was written.
    """
    with h5py.File(path, "r+") as hdf:
        if CALIBRATION_MIDDLE_FRACTION_ATTR in hdf.attrs:
            existing = hdf.attrs[CALIBRATION_MIDDLE_FRACTION_ATTR]
            raise ValueError(
                f"{path}: already carries {CALIBRATION_MIDDLE_FRACTION_ATTR!r} "
                f"= {existing!r}; this pass only stamps a product carrying "
                "none, it never corrects one that disagrees"
            )
        hdf.attrs[CALIBRATION_MIDDLE_FRACTION_ATTR] = value
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "products",
        nargs="+",
        type=Path,
        help="dead-time line-scan product(s) to re-stamp IN PLACE",
    )
    parser.add_argument(
        "--allow-processed",
        action="store_true",
        help="permit a product path under processed/, the placed scoring chain",
    )
    parser.add_argument(
        "--stamp-calibration-middle-fraction",
        type=float,
        default=None,
        metavar="FRACTION",
        help="run the calibration_middle_fraction root-attr pass instead of "
             "the density_area_key re-stamp: stamps FRACTION onto each "
             "product's root attrs, refusing a product that already carries "
             "one (at any value) rather than correcting it",
    )
    args = parser.parse_args()

    refuse_processed_paths(args.products, args.allow_processed)

    if args.stamp_calibration_middle_fraction is not None:
        for path in args.products:
            stamp_calibration_middle_fraction(
                path, args.stamp_calibration_middle_fraction
            )
            print(f"{path}: stamped {CALIBRATION_MIDDLE_FRACTION_ATTR} = "
                  f"{args.stamp_calibration_middle_fraction:g}")
        return 0

    total_moved = 0
    for path in args.products:
        rows, moved, table = restamp_product(path)
        total_moved += moved
        print(f"{path}")
        print(f"  {'set':<5}{'run':<6}{'channel':<10}{'before':<12}{'after':<12}status")
        for set_id, run_id, channel, before, after in table:
            status = "MOVED" if before != after else "kept"
            print(f"  {set_id:<5}{run_id:<6}{channel:<10}{before:<12}{after:<12}{status}")
        print(f"  {rows} run group(s), {moved} {AREA_KEY_ATTR} attr(s) re-stamped")
    print(f"total {AREA_KEY_ATTR} attrs re-stamped: {total_moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
