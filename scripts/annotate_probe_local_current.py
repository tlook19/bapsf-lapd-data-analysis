"""Record the late-afterglow probe-local-current caveat on the ISAT product.

ANNOTATES
    processed/isat_profiles.hdf5
    -- the rot-0 DOWNSTREAM face (``isat`` channel) dead-time line-scan
       product built by ``scripts/plot_isat_profiles.py``.

FOLLOWS THE WRITER
    scripts/plot_isat_profiles.py, which builds that product but has no
    exclusion or caveat mechanism of its own (matching the precedent in
    scripts/annotate_state_mask.py and scripts/annotate_saturation_attrs.py:
    the writer carries no per-run disposition, so one is carried as
    machine-readable provenance ON the product instead).

REGISTRY
    ``scripts.export_es1_sim1d_overlay.LATE_AFTERGLOW_PROBE_LOCAL_CURRENT``,
    imported rather than re-declared so the two places that must agree on a
    registered run -- this annotator (the caveat attr on the run's own ISAT
    group) and the overlay exporter (the NaN-fill on that run's LATE-AFTERGLOW
    trace, built in a completely separate code path from raw data) -- read one
    source.  See that dict's docstring comment for the run 22 finding and its
    evidence: the core ISAT-channel decay departs from the run's own
    reference-photodiode decay and from every neighbouring experiment-set-2
    run by a wide margin, while the run's I_SWEEP channel and its rot-180
    partner are both normal.

SCOPE.  The caveat is written on the run's OWN group in THIS product even
though this product's own dead-time cells (0.375-19.875 ms, strictly inside
the sweep drive) carry no late-afterglow sample and are NUMERICALLY
untouched by the finding -- the caveat records a property of the RUN's ISAT
channel, the same way scripts/annotate_state_mask.py's STATE_REGISTRY and
scripts/annotate_saturation_attrs.py's ``saturation_excluded`` are properties
of a run rather than of one product's cells, so every product built from that
run's ISAT channel (including the overlay exporter's own late-afterglow
trace, which does not read this file's cells at all) can be read against one
declaration.  ``probe_local_current_afterglow_only`` is written alongside the
caveat specifically to say so.

Writes per registered run, on group ``experiment_sets/<set>/<run_id>``:
    ``probe_local_current_suspected`` (bool),
    ``probe_local_current_reason`` (str),
    ``probe_local_current_tau_ms``, ``probe_local_current_photodiode_tau_ms``,
    ``probe_local_current_neighbor_tau_range_ms`` (2-array),
    ``probe_local_current_undecayed_current_ma``,
    ``probe_local_current_undecayed_time_ms``,
    ``probe_local_current_afterglow_only`` (bool, always True here),
    ``probe_local_current_source`` (str).
DISCLOSED, never corrected: nothing here masks a cell, drops a run, or
rescales a value.  The product's ``high_shot_rejection_*`` attrs and every
dataset are untouched.

PLACE THE ANNOTATION (this annotator on the PLACED product, then the
downstream export; run from the repo root)
    cp processed/isat_profiles.hdf5 processed/isat_profiles.hdf5.bak
    python scripts/annotate_probe_local_current.py . \
        processed/isat_profiles.hdf5 \
        processed/isat_profiles_probe_local_current_screen.json \
        --allow-processed
    PYTHONPATH=src python scripts/export_es1_sim1d_overlay.py --experiment-set 2
    PYTHONPATH=src python scripts/augment_sim1d_overlay_isat_drive.py \
        --experiment-set 2 --overlay processed/es2_sim1d_overlay.npz \
        --output processed/es2_sim1d_overlay_v24.npz

A raw rebuild of ``processed/isat_profiles.hdf5`` (``plot_isat_profiles.py``)
is deliberately NOT part of this sequence: the placed file already carries
whatever OTHER annotator state has been written onto it (the rail mask, the
saturation screen, ...), and a rebuild starts a fresh file with none of it.
This annotator is meant to run on the PLACED file, in place, exactly like its
siblings; rebuild only when the writer's own inputs changed, as a separate,
deliberate step, and re-run every annotator afterward.

The annotator edits the product in place (``r+``) and writes a JSON summary
sidecar.  It reads no raw run data -- only the registry above.  Needs h5py,
the environment.yml env, not the dead ./.venv.

Usage:  python annotate_probe_local_current.py <repo-root> <product.hdf5> <summary.json> [--allow-processed]
"""
import argparse
import json
import sys
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_es1_sim1d_overlay import LATE_AFTERGLOW_PROBE_LOCAL_CURRENT  # noqa: E402

#: This annotator only ever registers the ``isat`` channel's product; a
#: registry entry for any other channel is not this product's business.
CHANNEL = "isat"


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="repository root; unused beyond locating nothing further -- the "
        "registry is imported, not re-measured -- kept for parity with the "
        "sibling annotators' calling convention",
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

    registered = {
        run_id: entry
        for (run_id, channel), entry in LATE_AFTERGLOW_PROBE_LOCAL_CURRENT.items()
        if channel == CHANNEL
    }

    annotated = []
    with h5py.File(args.product, "r+") as hf:
        for set_id in sorted(hf["experiment_sets"], key=int):
            set_group = hf[f"experiment_sets/{set_id}"]
            for run_id in sorted(set_group, key=int):
                entry = registered.get(run_id)
                if entry is None:
                    continue
                g = set_group[run_id]
                g.attrs["probe_local_current_suspected"] = True
                g.attrs["probe_local_current_reason"] = str(entry["reason"])
                g.attrs["probe_local_current_tau_ms"] = float(entry["tau_ms"])
                g.attrs["probe_local_current_photodiode_tau_ms"] = float(
                    entry["photodiode_tau_ms"]
                )
                g.attrs["probe_local_current_neighbor_tau_range_ms"] = [
                    float(v) for v in entry["neighbor_tau_range_ms"]
                ]
                g.attrs["probe_local_current_undecayed_current_ma"] = float(
                    entry["undecayed_current_ma"]
                )
                g.attrs["probe_local_current_undecayed_time_ms"] = float(
                    entry["undecayed_time_ms"]
                )
                g.attrs["probe_local_current_afterglow_only"] = True
                g.attrs["probe_local_current_source"] = str(entry["source"])
                annotated.append(run_id)

    args.summary_json.write_text(
        json.dumps(
            {"channel": CHANNEL, "annotated_runs": annotated, "registry": registered},
            indent=2,
            default=str,
        )
    )
    print(f"annotated {args.product}")
    print("probe-local-current runs:", annotated or "none")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
