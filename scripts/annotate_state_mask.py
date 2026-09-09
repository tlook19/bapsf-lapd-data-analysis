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
    ``STATE_REGISTRY`` below.  A state is registered per (run, channel) with the
    inclusive SHOT range it spans and the measured level factor between the two
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

PROJECTION ONTO CELLS
    Cells are the (position, inter-sweep dead-time window) cells the product
    averages, and the product averages the shots of a position together.  A cell
    is masked when ANY shot of its position lies in the registered range, on all
    dead-time windows: a position that spans the step carries the two states
    mixed into one average, which is no more usable than a position wholly
    inside the range.  For run 43 the range 240-695 covers positions 12-34
    inclusive, position 34 being the mixed one.

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

# (run_id, channel) -> inclusive shot range and the measured level factor of the
# masked state relative to the rest of the same run.  The factor is DISCLOSED,
# never applied.
STATE_REGISTRY = {
    ("43", "isat"): {
        "shot_first": 240,
        "shot_last": 695,
        "factor": 1.76,
        "evidence": (
            "plateau-window signal steps x1.852 at shot 240 (x1.75 once the "
            "x = -14 -> -13 profile move on that position boundary is removed) "
            "and x0.568 at shot 696, with I_SWEEP, the reference photodiode and "
            "the tail noise unchanged and the tail DC offset halving at 240"
        ),
    },
}

RULE = (
    "cell excluded when any shot of its position lies in the run's registered "
    "channel-state shot range; the channel sits at a different level over that "
    "range, so the cells are not comparable with the rest of the run. Cells are "
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


def state_mask_for_run(n_positions, n_windows, n_shots_per_position,
                       shot_first, shot_last):
    """Per-(position, window) mask for one registered shot range.

    A position is covered when any of its shots lies in ``[shot_first,
    shot_last]`` inclusive; every dead-time window of a covered position is
    masked.
    """
    starts = np.arange(n_positions) * n_shots_per_position
    stops = starts + n_shots_per_position - 1
    covered = (stops >= shot_first) & (starts <= shot_last)
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
    args = parser.parse_args(argv)

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
                    n_positions, n_windows, n_shots,
                    entry["shot_first"], entry["shot_last"],
                )
                positions = np.flatnonzero(covered)
                shots = f"{entry['shot_first']}-{entry['shot_last']}"

                if STATE_MASK_DATASET in g:
                    del g[STATE_MASK_DATASET]
                g.create_dataset(STATE_MASK_DATASET, data=mask)

                g.attrs["state_mask_rule"] = RULE
                g.attrs["state_mask_shots"] = shots
                g.attrs["state_factor"] = float(entry["factor"])
                g.attrs["state_factor_applied"] = False
                g.attrs["state_mask_evidence"] = entry["evidence"]
                g.attrs["state_screened_channel"] = channel
                g.attrs["state_mask_positions"] = (
                    f"{int(positions[0])}-{int(positions[-1])}"
                )
                g.attrs["state_shots_per_position"] = int(n_shots)
                g.attrs["state_cells_excluded"] = int(mask.sum())
                g.attrs["state_cells_total"] = int(mask.size)

                print(f"state-masked run {run_id} ({channel}): shots {shots} -> "
                      f"positions {int(positions[0])}-{int(positions[-1])}, "
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
                    "positions_first": int(positions[0]),
                    "positions_last": int(positions[-1]),
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
