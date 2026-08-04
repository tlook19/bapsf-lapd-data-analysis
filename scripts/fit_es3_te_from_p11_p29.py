"""Build an exploratory ES3 T_e fill using only p11 and p29 rows.

This assumes the downstream p41/p50 temperature rows are too high and treats
p11 and p29 as the trusted axial anchors.  p21, p41, and p50 are blanked before
running the edge-suppressed spatial fill, so the output interpolates/extrapolates
from p11/p29 plus the boundary and SOL anchors.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/fit_es3_te_from_p11_p29.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fit_te_spatial import (
    EDGE_ANCHOR_CM,
    HDF5_OUTPUT,
    SMOOTHING_EDGE,
    SMOOTHING_EDGE_ANCHOR,
    SMOOTHING_CORE,
    TE_BOUNDARY_EV,
    TE_EDGE_ANCHOR_EV,
    X_CORE_CM,
    X_EDGE_CM,
    X_WALL_CM,
    Z_ANODE_CM,
    Z_CATHODE_CM,
    fill_te_grid,
    _plot_comparison,
)


INPUT = Path("processed/te_filled.hdf5")
OUTPUT = Path("processed/te_filled_es3_p11_p29.hdf5")
OUTPUT_DIR = Path("figures/te_es3_p11_p29")
ES_ID = "3"
TRUSTED_Z_CM = (470.05, 1045.15)  # p11 and p29


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(INPUT, "r") as src, h5py.File(OUTPUT, "w") as out:
        out.create_group("experiment_sets")
        for es_id in sorted(src["experiment_sets"].keys(), key=int):
            in_grp = src[f"experiment_sets/{es_id}"]
            out_grp = out["experiment_sets"].create_group(es_id)
            out_grp.attrs["label"] = in_grp.attrs.get("label", f"set {es_id}")
            out_grp.attrs["v_bank_v"] = float(in_grp.attrs.get("v_bank_v", np.nan))
            for name in ("x_cm", "z_cm", "cycle_time_ms"):
                out_grp.create_dataset(name, data=in_grp[name][()], compression="gzip")

            te_masked = in_grp["te_masked"][()]
            x_cm = in_grp["x_cm"][()]
            z_cm = in_grp["z_cm"][()]
            out_grp.create_dataset("te_masked_original", data=te_masked, compression="gzip", compression_opts=4)

            if es_id == ES_ID:
                trusted = np.zeros(len(z_cm), dtype=bool)
                for z in TRUSTED_Z_CM:
                    trusted[np.argmin(np.abs(z_cm - z))] = True
                te_fit_input = te_masked.copy()
                te_fit_input[~trusted] = np.nan
                out_grp.attrs["trusted_z_cm"] = ",".join(f"{z:.2f}" for z in z_cm[trusted])
                out_grp.attrs["fit_policy"] = "Only p11 and p29 rows kept; p21/p41/p50 blanked before fill."
                print(f"ES {es_id}: keeping z rows {z_cm[trusted]}")
                te_filled = fill_te_grid(
                    te_fit_input,
                    x_cm,
                    z_cm,
                    x_wall=X_WALL_CM,
                    z_lo=Z_CATHODE_CM,
                    z_hi=Z_ANODE_CM,
                    te_boundary=TE_BOUNDARY_EV,
                    x_core=X_CORE_CM,
                    x_edge=X_EDGE_CM,
                    smoothing_core=SMOOTHING_CORE,
                    smoothing_edge=SMOOTHING_EDGE,
                    edge_anchor=EDGE_ANCHOR_CM,
                    te_edge_anchor=TE_EDGE_ANCHOR_EV,
                    smoothing_edge_anchor=SMOOTHING_EDGE_ANCHOR,
                )
                plot_data = {
                    "te": te_fit_input,
                    "x_cm": x_cm,
                    "z_cm": z_cm,
                    "cycle_time_ms": in_grp["cycle_time_ms"][()],
                    "es_label": out_grp.attrs["label"],
                    "v_bank": out_grp.attrs["v_bank_v"],
                    "es_id": es_id,
                }
                _plot_comparison(plot_data, te_filled, OUTPUT_DIR)
            else:
                te_fit_input = te_masked
                te_filled = in_grp["te_filled"][()]

            out_grp.create_dataset("te_masked", data=te_fit_input, compression="gzip", compression_opts=4)
            out_grp.create_dataset("te_filled", data=te_filled, compression="gzip", compression_opts=4)
            out.flush()

    print(f"Wrote {OUTPUT}")
    print(f"Reference default output path remains {HDF5_OUTPUT}")


if __name__ == "__main__":
    main()
