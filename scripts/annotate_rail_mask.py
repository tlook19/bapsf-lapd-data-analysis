"""Record the per-cell rail mask on the rot-180 ISAT dead-time product.

ANNOTATES
    processed/isat_rot180_deadtime_profiles.hdf5
    -- the rot-180 UPSTREAM face (ISAT channel) of the Mach pair.

FOLLOWS THE WRITER
    scripts/plot_isat_profiles.py, which builds that product but has no
    exclusion mechanism of its own.

Companion to scripts/annotate_saturation_attrs.py, which records the same
digitizer-saturation screen at RUN level on the opposite face.  ES4 run 43's
ISAT face is railed over only part of its record, so a run-level verdict is the
wrong granularity for it: this pass measures the SAME rail condition per
averaging cell and writes it as a mask that scripts/compute_mach_velocity.py
honours, dropping a masked cell from the Mach pair on BOTH faces.

Rule (unchanged from the sibling product's own definition of "railed"): a raw
uint16 SIS sample is railed at code 0 or 65535, and a cell is excluded when it
contains ANY railed sample.  The sibling's run-level verdict is
``saturation_excluded = bool(rail_dead > 0)`` -- no fractional threshold -- so
none is invented here.

Cells are the (position, inter-sweep dead-time window) cells the product
averages, using the same CLIP_S trim as scripts/plot_isat_profiles.py.  The
channel screened per run is the one the product actually averaged, read from
the group's own ``deadtime_source_channel`` attr, so a source override is
honoured rather than re-derived.

Writes per run: datasets ``rail_fraction`` and ``rail_mask``, plus the
run-level attrs the sibling product carries.

REGENERATE THE PRODUCT (writer, then annotator; run from the repo root)
    python scripts/plot_isat_profiles.py \
        --source-channel isat --rotation 180 \
        --output processed/isat_rot180_deadtime_profiles.hdf5 \
        --plot-prefix isat_rot180 --no-animation
    python scripts/annotate_rail_mask.py . \
        processed/isat_rot180_deadtime_profiles.hdf5 \
        processed/isat_rot180_rail_mask_screen.json

The annotator edits the product in place (``r+``) and writes a JSON summary
sidecar.  Needs h5py and numpy -- the environment.yml env, not the dead
./.venv.  Reads every rot-180 raw run on both channels, so it is IO-bound and
takes minutes.

Usage:  python annotate_rail_mask.py <repo-root> <product.hdf5> <summary.json>
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bapsf_lapd import LapdDataset, ChannelKind          # noqa: E402
from bapsf_lapd.density import inter_sweep_sample_slices  # noqa: E402

CLIP_S = 10e-6            # same trim as scripts/plot_isat_profiles.py
RAIL_LO, RAIL_HI = 0, 65535
PLATEAU_MS = (14.0, 19.0)

RULE = (
    "cell excluded when it contains any raw sample at a uint16 SIS converter "
    "rail (code 0 or 65535); cells are the (position, inter-sweep dead-time "
    "window) cells this product averages, CLIP_S-trimmed"
)
METHOD = (
    "raw uint16 SIS codes; a sample is railed at code 0 or 65535. The SIS "
    "per-shot header fields Min/Max/Clipped are identically zero for this "
    "dataset and carry no information. Fractions are over the whole record, "
    "over the CLIP_S-trimmed inter-sweep dead-time windows this product "
    "averages, and over the 14-19 ms plateau cycles."
)


def inter_sweep_times_ms(sw):
    return np.array([
        sw.t0_s + k * sw.tau_cycle_s + 0.5 * (sw.tau_ramp_s + sw.tau_cycle_s)
        for k in range(sw.n_cycles)
    ]) * 1e3


def screen_cells(run, kind):
    """Per-(position, window) rail counts, plus the run-level totals."""
    cfg = run.config
    ch = cfg.channel(kind)
    sw, acq = cfg.sweep, cfg.acquisition
    n_pos = acq.n_positions
    n_shots = acq.n_shots_per_position
    n_cycles = sw.n_cycles

    dead = inter_sweep_sample_slices(sw, acq, clip_s=CLIP_S)
    dead_idx = np.concatenate([np.arange(s.start, s.stop) for s in dead])
    seg = np.concatenate([
        np.full(s.stop - s.start, k, dtype=np.int64) for k, s in enumerate(dead)
    ])
    per_window_samples = np.array([s.stop - s.start for s in dead], dtype=np.int64)

    rail_counts = np.zeros((n_pos, n_cycles), dtype=np.int64)
    n_all = rail_all = 0
    code_lo, code_hi = RAIL_HI, RAIL_LO

    with run.open() as h5:
        data = h5[ch.hdf5_path]
        for flat_shot in range(n_pos * n_shots):
            pos = flat_shot // n_shots
            raw = data[flat_shot, :]
            n_all += raw.size
            railed_all = (raw == RAIL_LO) | (raw == RAIL_HI)
            rail_all += int(railed_all.sum())
            code_lo = min(code_lo, int(raw.min()))
            code_hi = max(code_hi, int(raw.max()))
            sub = railed_all[dead_idx]
            if sub.any():
                rail_counts[pos] += np.bincount(seg[sub], minlength=n_cycles)

    n_cell_samples = per_window_samples[None, :] * n_shots
    rail_fraction = rail_counts / n_cell_samples
    rail_mask = rail_counts > 0

    t_ms = inter_sweep_times_ms(sw)
    plat = (t_ms >= PLATEAU_MS[0]) & (t_ms <= PLATEAU_MS[1])
    n_dead = int(np.broadcast_to(n_cell_samples, rail_counts.shape).sum())
    rail_dead = int(rail_counts.sum())
    n_plat = int(np.broadcast_to(n_cell_samples, rail_counts.shape)[:, plat].sum())
    rail_plat = int(rail_counts[:, plat].sum())

    return dict(
        rail_fraction=rail_fraction, rail_mask=rail_mask, rail_counts=rail_counts,
        n_cell_samples=n_cell_samples,
        n_all=n_all, rail_all=rail_all, n_dead=n_dead, rail_dead=rail_dead,
        n_plat=n_plat, rail_plat=rail_plat, code_lo=code_lo, code_hi=code_hi,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repo_root",
        type=Path,
        help="repository root holding config/may2026_run_manifest.toml "
             "and the raw run directory it names",
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
        hf.attrs["saturation_screen_method"] = METHOD
        hf.attrs["saturation_screen_rail_codes"] = f"{RAIL_LO},{RAIL_HI}"
        hf.attrs["saturation_screen_clip_s"] = CLIP_S
        hf.attrs["saturation_screen_plateau_ms"] = f"{PLATEAU_MS[0]:g}-{PLATEAU_MS[1]:g}"
        hf.attrs["saturation_screen_reference"] = (
            "reproduces the 2026-08-18 rot-180 clipping screen, measured per "
            "averaging cell"
        )
        hf.attrs["rail_mask_rule"] = RULE
        hf.attrs["rail_mask_dataset"] = "rail_mask"

        excluded_runs, opposite_runs = [], []
        for set_id in sorted(hf["experiment_sets"], key=int):
            for run_id in sorted(hf[f"experiment_sets/{set_id}"]):
                g = hf[f"experiment_sets/{set_id}/{run_id}"]
                this_kind = ChannelKind(str(g.attrs["deadtime_source_channel"]))
                other_kind = (ChannelKind.I_SWEEP if this_kind == ChannelKind.ISAT
                              else ChannelKind.ISAT)
                run = ds.run(run_id)

                face = screen_cells(run, this_kind)
                other = screen_cells(run, other_kind)
                print(f"screened run {run_id} ({this_kind.value}): "
                      f"{int(face['rail_mask'].sum())} of {face['rail_mask'].size} "
                      f"cells railed", flush=True)

                for name, arr in (("rail_fraction", face["rail_fraction"]),
                                  ("rail_mask", face["rail_mask"])):
                    if name in g:
                        del g[name]
                    g.create_dataset(name, data=arr)

                g.attrs["rail_fraction_full_record"] = face["rail_all"] / face["n_all"]
                g.attrs["rail_fraction_deadtime"] = face["rail_dead"] / face["n_dead"]
                g.attrs["rail_fraction_plateau"] = face["rail_plat"] / face["n_plat"]
                g.attrs["raw_code_min"] = face["code_lo"]
                g.attrs["raw_code_max"] = face["code_hi"]
                g.attrs["saturation_excluded"] = bool(face["rail_dead"] > 0)
                g.attrs["opposite_face_rail_fraction_deadtime"] = (
                    other["rail_dead"] / other["n_dead"]
                )
                g.attrs["rail_mask_rule"] = RULE
                g.attrs["rail_cells_excluded"] = int(face["rail_mask"].sum())
                g.attrs["rail_cells_total"] = int(face["rail_mask"].size)
                g.attrs["rail_screened_channel"] = this_kind.value

                if face["rail_dead"] > 0:
                    excluded_runs.append(run_id)
                if other["rail_dead"] > 0:
                    opposite_runs.append(run_id)

                summary[run_id] = {
                    "set_id": str(set_id),
                    "port": int(g.attrs["port"]),
                    "channel": this_kind.value,
                    "rail_fraction_deadtime": face["rail_dead"] / face["n_dead"],
                    "rail_fraction_full_record": face["rail_all"] / face["n_all"],
                    "rail_fraction_plateau": face["rail_plat"] / face["n_plat"],
                    "cells_excluded": int(face["rail_mask"].sum()),
                    "cells_total": int(face["rail_mask"].size),
                    "opposite_face_rail_fraction_deadtime":
                        other["rail_dead"] / other["n_dead"],
                }

        hf.attrs["saturation_screen_runs_railed_this_face"] = ",".join(excluded_runs)
        hf.attrs["saturation_screen_runs_railed_opposite_face"] = ",".join(opposite_runs)
        hf.attrs["saturation_screen_note"] = (
            "Per-cell rail mask for this product's own face. A cell is excluded "
            "when it contains any raw sample at a converter rail; rail_mask "
            "carries the verdict and rail_fraction the measured fraction. The "
            "run-level saturation_excluded attr is retained for parity with "
            "processed/isweep_rot180_deadtime_profiles.hdf5 and is true when the "
            "run has any railed dead-time sample anywhere, which is coarser than "
            "the mask."
        )

    args.summary_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nannotated {args.product}")
    print("railed on this face:", excluded_runs or "none")
    print("railed on the opposite face:", opposite_runs or "none")
    print(f"wrote {args.summary_json}")


if __name__ == "__main__":
    main()
