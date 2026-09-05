"""Record the digitizer-saturation screen on the rot-180 ISWEEP dead-time product.

ANNOTATES
    processed/isweep_rot180_deadtime_profiles.hdf5
    -- the rot-180 DOWNSTREAM face (-I_SWEEP channel) of the Mach pair.

FOLLOWS THE WRITER
    scripts/plot_isat_profiles.py, which builds that product but has no
    exclusion mechanism of its own.

The 2026-08-18 clipping screen is a run-level exclusion policy, and the writer
has no exclusion mechanism -- the current products exclude no run, and neither
does the rebuild.  So the screen is carried as machine-readable provenance ON
the product instead of silently dropping or silently keeping a run: every group
gets its measured rail fractions and an explicit ``saturation_excluded``
verdict.  The screen itself is scripts/screen_rot180_saturation.py; this
re-runs the same measurement so the numbers written into the file are the ones
this pass measured, not a transcribed copy.

For the per-CELL mask on the opposite (ISAT) face, whose run-level verdict is
too coarse, see scripts/annotate_rail_mask.py.

REGENERATE THE PRODUCT (writer, then annotator; run from the repo root)
    python scripts/plot_isat_profiles.py \
        --source-channel i_sweep --rotation 180 \
        --output processed/isweep_rot180_deadtime_profiles.hdf5 \
        --plot-prefix isweep_rot180_fixed --no-animation
    python scripts/annotate_saturation_attrs.py . \
        processed/isweep_rot180_deadtime_profiles.hdf5 \
        processed/isweep_rot180_saturation_screen.json

The annotator edits the product in place (``r+``) and writes a JSON summary
sidecar.  Needs h5py and numpy -- the environment.yml env, not the dead
./.venv.  Reads every rot-180 raw run twice per channel, so it is IO-bound and
takes minutes.

Usage:  python annotate_saturation_attrs.py <repo-root> <product.hdf5> <summary.json>
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(sys.argv[1])
PRODUCT = Path(sys.argv[2])
JSON_OUT = Path(sys.argv[3])
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bapsf_lapd import LapdDataset, ChannelKind, effective_rotation_deg
from screen_rot180_saturation import screen, CLIP_S, PLATEAU_MS, RAIL_LO, RAIL_HI

METHOD = (
    "raw uint16 SIS codes; a sample is railed at code 0 or 65535. The SIS "
    "per-shot header fields Min/Max/Clipped are identically zero for this "
    "dataset and carry no information. Fractions are over the whole record, "
    "over the CLIP_S-trimmed inter-sweep dead-time windows this product "
    "averages, and over the 14-19 ms plateau cycles."
)

ds = LapdDataset.from_manifest(REPO / "config/may2026_run_manifest.toml")
rows = {}
for set_id in ds.experiment_set_ids():
    for run_id in ds.experiment_set_run_ids(set_id):
        cfg = ds.config(run_id)
        if effective_rotation_deg(run_id, float(cfg.probe.rotation_deg or 0)) != 180.0:
            continue
        run = ds.run(run_id)
        rows[run_id] = {"set_id": str(set_id), "port": int(cfg.probe.port or 0)}
        for k in (ChannelKind.I_SWEEP, ChannelKind.ISAT):
            rows[run_id][k.value] = screen(run, k)
        print(f"screened run {run_id}", flush=True)

with h5py.File(PRODUCT, "r+") as hf:
    hf.attrs["saturation_screen_method"] = METHOD
    hf.attrs["saturation_screen_rail_codes"] = f"{RAIL_LO},{RAIL_HI}"
    hf.attrs["saturation_screen_clip_s"] = CLIP_S
    hf.attrs["saturation_screen_plateau_ms"] = f"{PLATEAU_MS[0]:g}-{PLATEAU_MS[1]:g}"
    hf.attrs["saturation_screen_reference"] = (
        "reproduces the 2026-08-18 rot-180 clipping screen"
    )
    excluded = []
    flagged_other_face = []
    for set_id in sorted(hf["experiment_sets"], key=int):
        for run_id in sorted(hf[f"experiment_sets/{set_id}"]):
            r = rows[run_id]
            face = r["i_sweep"]
            other = r["isat"]
            g = hf[f"experiment_sets/{set_id}/{run_id}"]
            g.attrs["rail_fraction_full_record"] = face["rail_all"] / face["n_all"]
            g.attrs["rail_fraction_deadtime"] = face["rail_dead"] / face["n_dead"]
            g.attrs["rail_fraction_plateau"] = face["rail_plat"] / face["n_plat"]
            g.attrs["raw_code_min"] = face["code_lo"]
            g.attrs["raw_code_max"] = face["code_hi"]
            g.attrs["saturation_excluded"] = bool(face["rail_dead"] > 0)
            g.attrs["opposite_face_rail_fraction_deadtime"] = (
                other["rail_dead"] / other["n_dead"]
            )
            if face["rail_dead"] > 0:
                excluded.append(run_id)
            if other["rail_dead"] > 0:
                flagged_other_face.append(run_id)
    hf.attrs["saturation_screen_runs_railed_this_face"] = ",".join(excluded)
    hf.attrs["saturation_screen_runs_railed_opposite_face"] = ",".join(flagged_other_face)
    hf.attrs["saturation_screen_note"] = (
        "No run in this product's own (-I_SWEEP, rot-180 downstream) face has a "
        "single railed sample, so no run is excluded here. Runs listed in "
        "saturation_screen_runs_railed_opposite_face rail on the ISAT channel, "
        "which is the rot-180 UPSTREAM face carried by "
        "processed/isat_rot180_deadtime_profiles.hdf5, not by this product; any "
        "Mach pair using those runs is contaminated on the upstream side."
    )

JSON_OUT.write_text(json.dumps(rows, indent=2, default=str))
print(f"\nannotated {PRODUCT}")
print("railed on this face:", excluded or "none")
print("railed on the opposite (ISAT) face:", flagged_other_face or "none")
print(f"wrote {JSON_OUT}")
