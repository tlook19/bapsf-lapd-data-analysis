"""Append the raw drive-window Isat family to a sim1d overlay (v2..v15 odd).

Adds ``isat_drive_*`` — the upstream ion-saturation current from the
inter-sweep dead-time cells DURING the drive — to an existing
``es{N}_sim1d_overlay.npz``, per the exporter brief in
``~/bapsf/docs/notes/CATHODE_IDRIVEN_PLAN.md`` section 7h.  Input may be
schema v2 (es1 vintage), v3 (adds the isat_decay source-channel
metadata; es2/es3 vintage — v3 was already taken by that export, so the
drive family is SCHEMA v4, superseding the 7h brief's "v3"), v5 (adds
the per-port ``te_window_spread_frac``), v7 (adds
the discharge shot-to-shot standard deviations), v9 (adds the
flux-tube-averaged density and downstream-face Isat targets), v11
(adds the ruled upstream-face Isat target), v13 (adds the
flow-symmetrized geomean target), or v15 (the current export, which adds
the per-port ``te_core_mean_clamped`` record).  A v5 input is written
back as SCHEMA v6, a v7 as v8, a v9 as v10, a v11 as v12, a v13 as v14
and a v15 as v16, so each family's presence stays readable from the
version alone; an augmented version is never itself an accepted input,
which is what makes a second augmentation fail the gate.
The consumer is
``bapsf-transport/cablp/scripts/compare_sim1d_es1.py --beta-collapse``
(within-shot area guard + model-free sweep-chain consistency).

Source: ``processed/isweep_deadtime_profiles.hdf5`` ``isat_a_raw`` at
x = 0 — digitizer scale/offset, DC zero-offset subtraction and channel
calibration applied; NO probe-area normalization and NO Probe A factor
(purity requirement).  Runs are keyed off the overlay's own
``isat_decay_run_id`` so the drive family uses exactly the runs (and
hence the channel/wiring, incl. the run 31/33 special cases resolved by
the per-run attrs) that feed the decay trace.

Deliberate deviation from the 7h letter, logged in section 5b: the
profile pipeline rejects high-current shots PER CELL (sigma=3.0,
ratio=1.5, keep >= 10 of 20), which is not the decay trace's fixed
pre-afterglow shot ensemble.  ``isat_drive_same_ensemble_as_decay`` is
therefore stored as False and the drive/decay seam continuity gate is
the empirical commensurability arbiter.

Usage::

    python scripts/augment_sim1d_overlay_isat_drive.py --experiment-set 1 \
        --overlay ../bapsf-transport/cablp/scripts/data/es1_sim1d_overlay.npz

Writes in place by default (pass --output to write elsewhere).  Old
fields are copied through unchanged (round-trip gate); refuses to touch
an overlay whose schema is not an accepted input unless --force, in which
case an unrecognised version is written back unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

PROFILES = Path("processed/isweep_deadtime_profiles.hdf5")
SEAM_WINDOW_MS = 0.25  # first slice of the decay trace used for the seam gate

# Accepted un-augmented exporter schemas mapped to the schema written after the
# drive family is appended.  Keys are the inputs this script will augment;
# values are versions it will NOT re-accept, which is what makes an
# already-augmented overlay fail the gate instead of being augmented twice.
AUGMENTED_SCHEMA = {2: 4, 3: 4, 5: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16}


def _seam_gate(new: dict, decay_t, decay_mean, decay_sem) -> list[str]:
    """Drive/decay seam continuity per port: same probe, same amps.

    The last dead-time cell (~19.75-19.9 ms) and the first decay samples
    (20.0 ms+) straddle the discharge end, so a small real drop is
    physics, not a calibration seam; the gate therefore reports z and
    the ratio, and only hard-fails on ratio far from unity.
    """
    notes = []
    t_cells = new["isat_drive_time_ms"]
    for p, port in enumerate(new["isat_drive_port"]):
        drive_last = float(new["isat_drive_mean_a"][p, -1])
        drive_sem = float(new["isat_drive_sem_a"][p, -1])
        m = (decay_t >= decay_t.min()) & (decay_t <= decay_t.min() + SEAM_WINDOW_MS)
        aft_first = float(np.nanmean(decay_mean[p, m]))
        aft_sem = float(np.nanmean(decay_sem[p, m]))
        ratio = drive_last / aft_first if aft_first != 0 else np.inf
        z = abs(drive_last - aft_first) / np.hypot(drive_sem, aft_sem)
        gap_ms = float(decay_t.min() - t_cells[-1])
        verdict = "PASS" if 0.5 <= ratio <= 2.0 else "FAIL"
        notes.append(
            f"  port {int(port)}: drive({t_cells[-1]:.2f} ms) "
            f"{drive_last:.4g} A vs decay(first {SEAM_WINDOW_MS} ms) "
            f"{aft_first:.4g} A | ratio {ratio:.3f}, z {z:.1f}, "
            f"gap {gap_ms:.2f} ms -> {verdict}"
        )
        if verdict == "FAIL":
            notes.append(
                f"  port {int(port)}: SEAM FAIL — channel constants do not "
                "carry across drive/decay; do not use the within-shot guard"
            )
    return notes


def augment(overlay_path: Path, profiles_path: Path, experiment_set: int,
            output_path: Path, force: bool) -> int:
    overlay = dict(np.load(overlay_path, allow_pickle=False))
    schema = int(overlay["schema_version"])
    if schema not in AUGMENTED_SCHEMA and not force:
        accepted = ", ".join(f"v{v}" for v in sorted(AUGMENTED_SCHEMA))
        raise SystemExit(
            f"{overlay_path} is schema v{schema}, expected {accepted} "
            "(already augmented? pass --force to redo)"
        )
    run_ids = [str(r) for r in overlay["isat_decay_run_id"]]
    ports = np.asarray(overlay["isat_decay_port"], dtype=np.int16)

    mean_rows, sem_rows, std_rows = [], [], []
    nused_rows, nrej_rows = [], []
    chan, inverted, overridden = [], [], []
    time_ms = None
    with h5py.File(profiles_path, "r") as hdf:
        if not bool(hdf.attrs.get("high_shot_rejection_enabled", False)):
            raise SystemExit(f"{profiles_path} built without high-current rejection")
        clip_s = float(hdf.attrs["clip_s"])
        sigma = float(hdf.attrs["high_shot_rejection_sigma"])
        ratio = float(hdf.attrs["high_shot_rejection_ratio"])
        min_shots = int(hdf.attrs["high_shot_rejection_min_shots_used"])
        x_cm = hdf["x_cm"][()]
        x_idx = int(np.argmin(np.abs(x_cm)))
        if not np.isclose(x_cm[x_idx], 0.0):
            raise SystemExit("x=0 is not on the profile grid")
        set_group = hdf[f"experiment_sets/{experiment_set}"]
        for run_id, port in zip(run_ids, ports):
            g = set_group[run_id]
            if int(g.attrs["port"]) != int(port):
                raise SystemExit(
                    f"run {run_id}: profile port {g.attrs['port']} != "
                    f"overlay decay port {port} — run/port mapping broken"
                )
            t = np.asarray(g["inter_sweep_time_s"][()], dtype=float) * 1e3
            if time_ms is None:
                time_ms = t
            elif not np.allclose(t, time_ms):
                raise SystemExit(f"run {run_id}: dead-time grid differs")
            mean = np.asarray(g["isat_a_raw"][x_idx], dtype=float)
            std = np.asarray(g["isat_a_raw_std"][x_idx], dtype=float)
            nused = np.asarray(g["n_shots_used"][x_idx], dtype=float)
            mean_rows.append(mean)
            std_rows.append(std)
            sem_rows.append(std / np.sqrt(np.maximum(nused, 1.0)))
            nused_rows.append(nused)
            nrej_rows.append(np.asarray(g["n_high_shots_rejected"][x_idx]))
            chan.append(str(g.attrs["deadtime_source_channel"]))
            inverted.append(bool(g.attrs["deadtime_source_invert_polarity"]))
            overridden.append(bool(g.attrs.get("deadtime_source_overridden", False)))

    # Direct channel commensurability where the overlay carries the decay
    # source metadata (schema v3 vintage): the drive cells must come from
    # the same electrical channel as the decay trace, per port.
    if "isat_decay_source_channel" in overlay:
        dec_chan = [str(c) for c in overlay["isat_decay_source_channel"]]
        dec_inv = [bool(v) for v in overlay["isat_decay_source_inverted"]]
        if dec_chan != chan or dec_inv != inverted:
            raise SystemExit(
                "commensurability: drive source channels "
                f"{list(zip(chan, inverted))} != decay source channels "
                f"{list(zip(dec_chan, dec_inv))}"
            )
        print("channel commensurability vs decay metadata: exact match")

    decay_t = np.asarray(overlay["isat_decay_time_ms"], dtype=float)
    if time_ms[-1] >= decay_t.min():
        raise SystemExit(
            f"purity: last dead-time cell {time_ms[-1]:.3f} ms is not "
            f"before the decay start {decay_t.min():.3f} ms"
        )

    new = {
        "isat_drive_time_ms": time_ms,
        "isat_drive_mean_a": np.vstack(mean_rows),
        "isat_drive_sem_a": np.vstack(sem_rows),
        "isat_drive_std_a": np.vstack(std_rows),
        "isat_drive_port": ports,
        "isat_drive_run_id": np.array(run_ids),
        "isat_drive_n_shots_used": np.vstack(nused_rows).astype(np.int32),
        "isat_drive_n_high_shots_rejected": np.vstack(nrej_rows).astype(np.int32),
        "isat_drive_source_channel": np.array(chan),
        "isat_drive_source_inverted": np.array(inverted),
        "isat_drive_source_overridden": np.array(overridden),
        "isat_drive_cell_clip_s": np.array(clip_s),
        "isat_drive_area_factor_applied": np.array(False),
        "isat_drive_same_runs_as_decay": np.array(True),
        "isat_drive_same_ensemble_as_decay": np.array(False),
        "isat_drive_ensemble": np.array(
            "per-cell one-sided high-current rejection "
            f"(sigma={sigma}, ratio={ratio}, min {min_shots} shots); "
            "NOT the decay trace's fixed pre-afterglow ensemble — "
            "seam gate is the commensurability arbiter"
        ),
        "isat_drive_source_file": np.array(str(profiles_path)),
    }

    print(f"ES{experiment_set}: {len(run_ids)} runs, {time_ms.size} cells "
          f"[{time_ms.min():.3f}, {time_ms.max():.3f}] ms, "
          f"channels {chan}, overridden {overridden}")
    print("seam continuity gate (drive end vs decay start):")
    seam = _seam_gate(
        new, decay_t,
        np.asarray(overlay["isat_decay_mean_a"], dtype=float),
        np.asarray(overlay["isat_decay_sem_a"], dtype=float),
    )
    print("\n".join(seam))
    failed = any("SEAM FAIL" in line for line in seam)

    overlay.update(new)
    augmented_schema = AUGMENTED_SCHEMA.get(schema, schema)
    overlay["schema_version"] = np.array(augmented_schema)
    np.savez(output_path, **overlay)
    print(f"wrote schema v{augmented_schema} -> {output_path}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-set", type=int, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    parser.add_argument("--output", type=Path, default=None,
                        help="default: overwrite --overlay in place")
    parser.add_argument("--force", action="store_true",
                        help="re-augment an overlay whose schema is not an accepted input")
    args = parser.parse_args()
    raise SystemExit(
        augment(args.overlay, args.profiles, args.experiment_set,
                args.output or args.overlay, args.force)
    )


if __name__ == "__main__":
    main()
