"""Record the per-cell channel-state mask on the rot-180 ISAT dead-time product.

ANNOTATES
    processed/isat_rot180_deadtime_profiles.hdf5
    -- the rot-180 UPSTREAM face (ISAT channel) of the Mach pair.

FOLLOWS THE WRITER
    scripts/plot_isat_profiles.py, which builds that product but has no
    exclusion mechanism of its own.

Companion to scripts/annotate_rail_mask.py, which marks the same product's
cells whose RAW SAMPLES reached a converter rail.  A rail is a symptom of the
signal being too large for the channel's setting; this pass marks the cause,
which is coarser in time and wider in extent: a stretch of a run over which the
channel sat in a different STATE, so its recorded level is not comparable with
the rest of the same run.  A railed cell is always inside such a stretch, but
the stretch also covers cells that never railed and whose values are therefore
wrong-but-plausible.  Both masks are carried, and a consumer honours both.

REGISTERED STATES
    ``STATE_REGISTRY`` below.  A state is registered per (run, channel) as the
    inclusive SHOT RANGES it spans -- a channel can enter the same state more
    than once inside one run, so a registration carries a tuple of ranges rather
    than a single one -- together with the measured level factor between the two
    states.  The factor is recorded for disclosure: nothing here divides it out,
    and no product value is corrected anywhere.  The evidence for a registration
    is a level step in the plateau-window signal that the run's other Langmuir
    channel, the reference photodiode and the tail noise do not share.

    Run 43's ISAT channel steps up at shot 240 and returns at shot 696.  The up
    step falls exactly on the position boundary that carries the probe from
    x = -14 cm to x = -13 cm, so its raw size contains the profile move: x1.852
    raw, x1.75 corrected for that move; the down step, which falls inside a
    position, is 1/0.568 = 1.761; the registry carries 1.76, and the same-band
    low-versus-high comparison, at UNMATCHED x (the high-state side at
    x = -13...-10 cm against the low-state side at |x| = 10...16 cm), reads
    1.65.

    The same channel enters the high state a SECOND time later in the same run,
    over shots 760-767 and again from shot 781 to the end of the record.  The
    step screen fires on all three transitions of that entry: the position
    boundary 759|760 (ISAT x1.720 against I_SWEEP x1.013, attributed to ISAT),
    the within-position step 767->768 back down (x0.505) and the within-position
    step 780->781 back up (x2.060).  Its level factor is NOT the first entry's:
    the tail-removed plateau blocks either side of the two within-position steps
    read x1.976 at x = +13 cm (shots 760-767 against 768-779) and x2.074 at
    x = +14 cm (shots 781-799 against shot 780), against the 1.76 the registry
    carries, which was measured at the first entry's own steps.  The product's
    own tell is the one-sided 3-sigma high-shot rejection it is built with: at
    x = +13 cm that rejection removes 8 of the position's 20 shots in 18 of the
    20 dead-time windows and 9 in the other two, 162 summed over the 20 windows,
    against 6 or fewer summed over all 20 windows at every neighbouring
    position -- it took this entry's 8 high shots for outliers, and left
    x = +14 ... +25 cm, where every shot of the position is high and none stands
    out, entirely unrejected.  With those 8 shots out of the ensemble first the
    same rejection finds 4 over the 20 windows there, at most 2 in any one.

PROJECTION ONTO CELLS
    Cells are the (position, inter-sweep dead-time window) cells the product
    averages, and the product averages the shots of a position together.  A cell
    is masked when ANY shot of its position lies in ANY of the registered
    ranges, on all dead-time windows: a position that spans a step carries the
    two states mixed into one average, which is no more usable than a position
    wholly inside a range.  For run 43 the three ranges cover positions 12-34
    and 38-50 inclusive -- 240-695 covers 12-34 with position 34 mixed, 760-767
    covers position 38 alone, also mixed, and 781-1019 covers 39-50 with
    position 39 mixed.  The covered positions are NOT contiguous: 35, 36 and 37
    (x = +10, +11 and +12 cm) lie between the two entries and are not masked, so
    every place this pass records which shots or positions it covered writes
    them as a comma-separated list of ranges and never as a first-to-last span,
    which would claim the gap as well.

    Only runs with a registered state carry a ``state_mask``.  A run without one
    is NOT written an all-False mask, for the same reason the rail mask returns
    absence rather than a clean verdict: absence records that no state was
    registered, never that the run was screened and found single-state.  The
    screening instrument is scripts/screen_consecutive_shot_steps.py.

Writes per registered run: dataset ``state_mask``, plus the run-level attrs
``state_mask_rule``, ``state_mask_shots``, ``state_factor``, and the cell counts.

REGENERATE THE PRODUCT (writer, then both annotators; run from the repo root)
    python scripts/plot_isat_profiles.py \
        --source-channel isat --rotation 180 \
        --output processed/isat_rot180_deadtime_profiles.hdf5 \
        --plot-prefix isat_rot180 --no-animation
    python scripts/annotate_rail_mask.py . \
        processed/isat_rot180_deadtime_profiles.hdf5 \
        processed/isat_rot180_rail_mask_screen.json
    python scripts/annotate_state_mask.py . \
        processed/isat_rot180_deadtime_profiles.hdf5 \
        processed/isat_rot180_state_mask_screen.json

The annotator edits the product in place (``r+``) and writes a JSON summary
sidecar.  It reads no raw run data -- only the manifest, for the shots-per-
position of each registered run.  Needs h5py and numpy: the environment.yml
env, not the dead ./.venv.

Usage:  python annotate_state_mask.py <repo-root> <product.hdf5> <summary.json>
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd import LapdDataset  # noqa: E402

STATE_MASK_DATASET = "state_mask"

# (run_id, channel) -> the inclusive shot ranges the state spans and the measured
# level factor of the masked state relative to the rest of the same run.  A
# channel can enter the same state more than once in a run, so the ranges are a
# tuple; the factor is DISCLOSED, never applied.
STATE_REGISTRY = {
    ("43", "isat"): {
        "shot_ranges": ((240, 695), (760, 767), (781, 1019)),
        "factor": 1.76,
        "evidence": (
            "first entry, shots 240-695: plateau-window signal steps x1.852 at "
            "shot 240 (x1.75 once the x = -14 -> -13 profile move on that "
            "position boundary is removed) and x0.568 at shot 696, with "
            "I_SWEEP, the reference photodiode and the tail noise unchanged and "
            "the tail DC offset halving at 240. Second entry, shots 760-767 and "
            "781-1019: the step screen fires at the 759|760 position boundary "
            "(ISAT x1.720 against I_SWEEP x1.013, attributed to ISAT) and at "
            "the within-position steps 767->768 (x0.505) and 780->781 (x2.060); "
            "the tail-removed plateau blocks either side of those two "
            "within-position steps read x1.976 at x = +13 cm and x2.074 at "
            "x = +14 cm, so this entry's level factor is NOT the 1.76 recorded "
            "for the first. The product's own tell is its one-sided 3-sigma "
            "high-shot rejection, which at x = +13 cm removes 8 of the 20 shots "
            "in 18 of the 20 dead-time windows and 9 in the other two, 162 "
            "summed over the 20 windows, against 6 or fewer summed over all 20 "
            "at every neighbouring position, and leaves x = +14 ... +25 cm, "
            "wholly high, unrejected; with those 8 shots out of the ensemble "
            "the same rejection finds 4 over the 20 windows, at most 2 in any "
            "one"
        ),
    },
}

RULE = (
    "cell excluded when any shot of its position lies in one of the run's "
    "registered channel-state shot ranges; the channel sits at a different "
    "level over those ranges, so the cells are not comparable with the rest of "
    "the run. Cells are "
    "the (position, inter-sweep dead-time window) cells this product averages, "
    "and every dead-time window of a covered position is excluded. The recorded "
    "state factor is disclosed, not applied: no value is corrected"
)
NOTE = (
    "Per-cell channel-state mask, stacked on the rail mask rather than "
    "replacing it: a consumer honours both, and the railed cells are a strict "
    "subset of the state-masked ones. Only runs carrying a registered state "
    "have a state_mask; absence records that no state was registered for the "
    "run, not that the run was screened and found single-state."
)


def refuse_processed_output(product, allow_processed: bool) -> None:
    """Refuse to annotate a placed product unless the caller asked for it.

    ``processed/`` is the transport campaign's scoring chain: a product there
    is consumed by name, so an in-place edit of one is a measurement-side
    change, not a scratch operation.  A ``product`` path with a ``processed``
    directory component is therefore refused here, at argument resolution and
    before any file is opened, unless ``--allow-processed`` is passed.  The
    pattern follows scripts/restamp_density_area_keys.py.
    """
    if allow_processed:
        return
    if "processed" not in Path(product).resolve().parts[:-1]:
        return
    raise ValueError(
        f"refusing to annotate {product}: the path lies under a processed/ "
        "directory, which holds the placed products the transport campaign "
        "scores against, and this pass edits its input IN PLACE. Annotate a "
        "copy outside processed/, or pass --allow-processed to say that "
        "editing the placed product is the intent."
    )


def ranges_text(ranges):
    """Inclusive integer ranges written as ``"a-b,c-d"``.

    Every recorded shot or position set goes through this rather than through a
    first-to-last span: a registration's coverage can have gaps, and a span
    would claim the gaps too.
    """
    return ",".join(f"{int(first)}-{int(last)}" for first, last in ranges)


def contiguous_ranges(indices):
    """Integer indices grouped into inclusive ``(first, last)`` runs, sorted."""
    ranges = []
    for index in sorted(int(value) for value in indices):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return [(first, last) for first, last in ranges]


def state_mask_for_run(n_positions, n_windows, n_shots_per_position,
                       shot_ranges):
    """Per-(position, window) mask for a registration's shot ranges.

    A position is covered when any of its shots lies in ANY of the inclusive
    ranges in ``shot_ranges``; every dead-time window of a covered position is
    masked.  The ranges of one registration are the separate stretches over
    which the same channel state was recorded, so their coverage is unioned and
    the covered positions need not be contiguous.
    """
    starts = np.arange(n_positions) * n_shots_per_position
    stops = starts + n_shots_per_position - 1
    covered = np.zeros(n_positions, dtype=bool)
    for shot_first, shot_last in shot_ranges:
        covered |= (stops >= shot_first) & (starts <= shot_last)
    mask = np.zeros((n_positions, n_windows), dtype=bool)
    mask[covered, :] = True
    return mask, covered


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="repository root holding config/may2026_run_manifest.toml",
    )
    parser.add_argument(
        "product",
        type=Path,
        help="the dead-time profile product to annotate in place",
    )
    parser.add_argument(
        "summary_json",
        type=Path,
        help="path for the JSON summary sidecar",
    )
    parser.add_argument(
        "--allow-processed",
        action="store_true",
        help="permit a product path under processed/, the placed scoring chain",
    )
    args = parser.parse_args(argv)

    refuse_processed_output(args.product, args.allow_processed)

    ds = LapdDataset.from_manifest(args.repo_root / "config/may2026_run_manifest.toml")
    summary = {}

    with h5py.File(args.product, "r+") as hf:
        hf.attrs["state_mask_rule"] = RULE
        hf.attrs["state_mask_dataset"] = STATE_MASK_DATASET
        hf.attrs["state_mask_note"] = NOTE

        annotated = []
        for set_id in sorted(hf["experiment_sets"], key=int):
            for run_id in sorted(hf[f"experiment_sets/{set_id}"]):
                g = hf[f"experiment_sets/{set_id}/{run_id}"]
                channel = str(g.attrs["deadtime_source_channel"])
                entry = STATE_REGISTRY.get((run_id, channel))
                if entry is None:
                    if STATE_MASK_DATASET in g:
                        del g[STATE_MASK_DATASET]
                    continue

                n_positions, n_windows = g["isat_a_raw"].shape
                n_shots = ds.run(run_id).shots_per_position()
                mask, covered = state_mask_for_run(
                    n_positions, n_windows, n_shots, entry["shot_ranges"],
                )
                positions = ranges_text(
                    contiguous_ranges(np.flatnonzero(covered))
                )
                shots = ranges_text(entry["shot_ranges"])

                if STATE_MASK_DATASET in g:
                    del g[STATE_MASK_DATASET]
                g.create_dataset(STATE_MASK_DATASET, data=mask)

                g.attrs["state_mask_rule"] = RULE
                g.attrs["state_mask_shots"] = shots
                g.attrs["state_factor"] = float(entry["factor"])
                g.attrs["state_factor_applied"] = False
                g.attrs["state_mask_evidence"] = entry["evidence"]
                g.attrs["state_screened_channel"] = channel
                g.attrs["state_mask_positions"] = positions
                g.attrs["state_shots_per_position"] = int(n_shots)
                g.attrs["state_cells_excluded"] = int(mask.sum())
                g.attrs["state_cells_total"] = int(mask.size)

                print(f"state-masked run {run_id} ({channel}): shots {shots} -> "
                      f"positions {positions}, "
                      f"{int(mask.sum())} of {int(mask.size)} cells, "
                      f"disclosed factor {entry['factor']:g}", flush=True)
                annotated.append(run_id)
                summary[run_id] = {
                    "set_id": str(set_id),
                    "port": int(g.attrs["port"]),
                    "channel": channel,
                    "state_mask_shots": shots,
                    "state_factor": float(entry["factor"]),
                    "state_factor_applied": False,
                    "state_mask_positions": positions,
                    "shots_per_position": int(n_shots),
                    "cells_excluded": int(mask.sum()),
                    "cells_total": int(mask.size),
                }

        hf.attrs["state_mask_runs"] = ",".join(annotated)

    args.summary_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nannotated {args.product}")
    print("state-masked runs:", annotated or "none")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
