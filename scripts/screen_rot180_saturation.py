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
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO / "src"))
from bapsf_lapd import LapdDataset, ChannelKind, effective_rotation_deg
from bapsf_lapd.density import inter_sweep_sample_slices

CLIP_S = 10e-6
RAIL_LO, RAIL_HI = 0, 65535
PLATEAU_MS = (14.0, 19.0)
BATCH = 50


def inter_sweep_times_s(sw):
    return np.array([sw.t0_s + k * sw.tau_cycle_s
                     + 0.5 * (sw.tau_ramp_s + sw.tau_cycle_s)
                     for k in range(sw.n_cycles)])


def screen(run, kind):
    cfg = run.config
    ch = cfg.channel(kind)
    sw, acq = cfg.sweep, cfg.acquisition
    dead = inter_sweep_sample_slices(sw, acq, clip_s=CLIP_S)
    t_ms = inter_sweep_times_s(sw) * 1e3
    plat = [k for k, t in enumerate(t_ms) if PLATEAU_MS[0] <= t <= PLATEAU_MS[1]]
    dead_idx = np.concatenate([np.arange(s.start, s.stop) for s in dead])
    plat_idx = np.concatenate([np.arange(dead[k].start, dead[k].stop) for k in plat])

    n_all = n_dead = n_plat = 0
    rail_all = rail_dead = rail_plat = 0
    lo = 65535
    hi = 0
    with run.open() as h5:
        d = h5[ch.hdf5_path]
        n_rows = d.shape[0]
        for start in range(0, n_rows, BATCH):
            block = d[start:start + BATCH, :]
            n_all += block.size
            rail_all += int(((block == RAIL_LO) | (block == RAIL_HI)).sum())
            lo = min(lo, int(block.min()))
            hi = max(hi, int(block.max()))
            dd = block[:, dead_idx]
            n_dead += dd.size
            rail_dead += int(((dd == RAIL_LO) | (dd == RAIL_HI)).sum())
            pp = block[:, plat_idx]
            n_plat += pp.size
            rail_plat += int(((pp == RAIL_LO) | (pp == RAIL_HI)).sum())
        # second pass for mass at the run's own observed extremes
        at_lo = at_hi = at_lo_p = at_hi_p = 0
        for start in range(0, n_rows, BATCH):
            block = d[start:start + BATCH, :]
            dd = block[:, dead_idx]
            at_lo += int((dd == lo).sum())
            at_hi += int((dd == hi).sum())
            pp = block[:, plat_idx]
            at_lo_p += int((pp == lo).sum())
            at_hi_p += int((pp == hi).sum())
    return dict(
        n_all=n_all, rail_all=rail_all, n_dead=n_dead, rail_dead=rail_dead,
        n_plat=n_plat, rail_plat=rail_plat, code_lo=lo, code_hi=hi,
        at_lo=at_lo, at_hi=at_hi, at_lo_p=at_lo_p, at_hi_p=at_hi_p,
        plateau_cycles=(plat[0], plat[-1]), plateau_ms=(t_ms[plat[0]], t_ms[plat[-1]]),
    )


def main():
    ds = LapdDataset.from_manifest(REPO / "config/may2026_run_manifest.toml")
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
