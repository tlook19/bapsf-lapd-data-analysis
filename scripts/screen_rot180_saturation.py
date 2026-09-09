"""Digitizer-saturation screen for the effective-rot-180 runs.

Imported by ``scripts/annotate_saturation_attrs.py``, which writes the numbers
this measures onto processed/isweep_rot180_deadtime_profiles.hdf5.  Run directly
(``python scripts/screen_rot180_saturation.py .``) it prints the per-run,
per-channel screen table instead.

The SIS per-shot header fields Min/Max/Clipped are identically zero in this
dataset (see screen_headers.transcript.txt), so saturation must be measured
from the raw uint16 codes.  For every effective-rot-180 run this reports, on
both Langmuir current channels:

  * the fraction of raw samples pinned at the uint16 converter rails
    (code 0 or 65535), over the whole record and over the inter-sweep
    dead-time windows the profile product actually averages;
  * the run's own observed code floor/ceiling and the fraction of dead-time
    samples sitting exactly on each -- the flat-top signature that a
    rail-pinned trace leaves even when the pin is not at full scale;
  * the same two quantities restricted to the 14-19 ms plateau cycles.

Dead-time windows use the same CLIP_S = 10 us as scripts/plot_isat_profiles.py.

The rail counting itself is ``bapsf_lapd.rail_screen.screen_cells``, which
scripts/annotate_rail_mask.py calls too; ``screen`` adds only the flat-top mass
at the run's own observed extremes, which no product attr carries.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bapsf_lapd import LapdDataset, ChannelKind, effective_rotation_deg
from bapsf_lapd.rail_screen import (
    BATCH,
    CLIP_S,
    PLATEAU_MS,
    RAIL_HI,
    RAIL_LO,
    screen_cells,
)


def screen(run, kind):
    """The shared rail screen for one channel, plus this script's flat-top mass.

    ``at_lo``/``at_hi`` (and their plateau counterparts) count the dead-time
    samples sitting exactly on the run's OWN observed extremes -- the flat-top
    signature of a pin that is not at full scale -- which needs those extremes
    first and so cannot be counted in the screen's own pass.
    """
    cells = screen_cells(run, kind)
    dead_idx, plat_idx = cells["dead_idx"], cells["plateau_dead_idx"]
    lo, hi = cells["code_lo"], cells["code_hi"]

    at_lo = at_hi = at_lo_p = at_hi_p = 0
    with run.open() as h5:
        d = h5[run.config.channel(kind).hdf5_path]
        for start in range(0, d.shape[0], BATCH):
            block = d[start:start + BATCH, :]
            dd = block[:, dead_idx]
            at_lo += int((dd == lo).sum())
            at_hi += int((dd == hi).sum())
            pp = block[:, plat_idx]
            at_lo_p += int((pp == lo).sum())
            at_hi_p += int((pp == hi).sum())
    return dict(
        n_all=cells["n_all"], rail_all=cells["rail_all"],
        n_dead=cells["n_dead"], rail_dead=cells["rail_dead"],
        n_plat=cells["n_plat"], rail_plat=cells["rail_plat"],
        code_lo=lo, code_hi=hi,
        at_lo=at_lo, at_hi=at_hi, at_lo_p=at_lo_p, at_hi_p=at_hi_p,
        plateau_cycles=cells["plateau_cycles"], plateau_ms=cells["plateau_ms"],
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
    args = parser.parse_args(argv)

    ds = LapdDataset.from_manifest(args.repo_root / "config/may2026_run_manifest.toml")
    print(f"plateau window {PLATEAU_MS[0]}-{PLATEAU_MS[1]} ms; CLIP_S={CLIP_S:g} s; "
          f"uint16 rails {RAIL_LO}/{RAIL_HI}")
    print(f"{'run':>4} {'set':>3} {'port':>4} {'rec':>4} {'chan':>8} "
          f"{'railfrac_all':>13} {'railfrac_dead':>14} {'railfrac_plat':>14} "
          f"{'code_lo':>8} {'code_hi':>8} {'atlo_dead':>11} {'athi_dead':>11} "
          f"{'atlo_plat':>11} {'athi_plat':>11} {'cycles':>9}")
    for set_id in ds.experiment_set_ids():
        for run_id in ds.experiment_set_run_ids(set_id):
            cfg = ds.config(run_id)
            rec = float(cfg.probe.rotation_deg or 0)
            if effective_rotation_deg(run_id, rec) != 180.0:
                continue
            run = ds.run(run_id)
            for kind in (ChannelKind.I_SWEEP, ChannelKind.ISAT):
                r = screen(run, kind)
                print(f"{run_id:>4} {set_id:>3} {cfg.probe.port:>4} {rec:>4.0f} {kind.value:>8} "
                      f"{r['rail_all']/r['n_all']:>13.3e} "
                      f"{r['rail_dead']/r['n_dead']:>14.3e} "
                      f"{r['rail_plat']/r['n_plat']:>14.3e} "
                      f"{r['code_lo']:>8} {r['code_hi']:>8} "
                      f"{r['at_lo']/r['n_dead']:>11.3e} {r['at_hi']/r['n_dead']:>11.3e} "
                      f"{r['at_lo_p']/r['n_plat']:>11.3e} {r['at_hi_p']/r['n_plat']:>11.3e} "
                      f"{r['plateau_cycles'][0]:>4d}-{r['plateau_cycles'][1]:<4d}", flush=True)


if __name__ == "__main__":
    main()
