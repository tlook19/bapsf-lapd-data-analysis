"""Compute Mach number and velocity from existing dead-time profile products.

This script intentionally does not reread raw run HDF5 files.  It combines the
already-reduced dead-time current products:

  rot-0 upstream:    processed/isweep_deadtime_profiles.hdf5       (-I_SWEEP)
  rot-0 downstream:  processed/isat_profiles.hdf5                  (ISAT)
  rot-180 upstream:  processed/isat_rot180_deadtime_profiles.hdf5   (ISAT)
  rot-180 downstream: processed/isweep_rot180_deadtime_profiles.hdf5 (-I_SWEEP)

Mach is computed from area-normalized ion-saturation current:

  M = ln[(I_upstream / A_upstream) / (I_downstream / A_downstream)] / K

Velocity is ``M * C_s`` in km/s, where ``C_s`` is computed from filled T_e.
The script uses ``isat_a_raw`` so the Mach ratio is controlled only by the
probe-area calibration TOML, not by any plotting/density scale factor baked into
``isat_a``.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/compute_mach_velocity.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/compute_mach_velocity.py --rotations 0
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bapsf_lapd.density import load_probe_a_area_calibration


CALIB_TOML = Path("processed/probe_area_calibration.toml")
PROBE_A_CALIB_TOML = Path("config/may2026_probe_a_area_calibration.toml")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
ISWEEP_ROT0_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
ISAT_ROT0_HDF5 = Path("processed/isat_profiles.hdf5")
ISAT_ROT180_HDF5 = Path("processed/isat_rot180_deadtime_profiles.hdf5")
ISWEEP_ROT180_HDF5 = Path("processed/isweep_rot180_deadtime_profiles.hdf5")
HDF5_OUTPUT = Path("processed/mach_velocity.hdf5")

M_I_AMU = 4.003
MACH_K = 1.66
X_CM = np.linspace(-25.0, 25.0, 51)

PROBE_FROM_DIGIT = {1: "A", 8: "A", 2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}


def _probe_id(run_id: str) -> str:
    return PROBE_FROM_DIGIT[int(run_id[1])]


def _load_calibration(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, "rb") as f:
        toml = tomllib.load(f)
    result = {}
    for probe_key in ("probe_A", "probe_B", "probe_C", "probe_D"):
        short = probe_key.replace("probe_", "")
        data = toml[probe_key]
        result[short] = {
            "ap_L_m2": data["ap_L_cm2"] * 1e-4,
            "ap_R_m2": data["ap_R_cm2"] * 1e-4,
            "ap_L_cm2": data["ap_L_cm2"],
            "ap_R_cm2": data["ap_R_cm2"],
            "estimated": bool(data.get("estimated", False)),
        }
    return result


def _ion_sound_speed_m_s(te_ev: np.ndarray, m_i_amu: float) -> np.ndarray:
    e_c = 1.602176634e-19
    amu_to_kg = 1.66053906660e-27
    return np.sqrt(np.asarray(te_ev, dtype=np.float64) * e_c / (m_i_amu * amu_to_kg))


def _interp_filled_te_to_deadtime(
    te_hdf: h5py.File,
    es_id: str,
    z_cm: float,
    dead_time_s: np.ndarray,
) -> np.ndarray:
    grp = te_hdf[f"experiment_sets/{es_id}"]
    x_cm = grp["x_cm"][()]
    if not np.allclose(x_cm, X_CM):
        raise ValueError(f"Filled T_e x grid differs from Mach grid for experiment set {es_id}")

    z_grid = grp["z_cm"][()]
    z_idx = int(np.argmin(np.abs(z_grid - z_cm)))
    te_time_ms = grp["cycle_time_ms"][()]
    dead_time_ms = dead_time_s * 1000.0
    te_z = grp["te_filled"][z_idx, :, :]
    return np.vstack([
        np.interp(dead_time_ms, te_time_ms, te_z[xi, :])
        for xi in range(te_z.shape[0])
    ])


def _require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        sys.exit("Missing required input(s):\n  " + "\n  ".join(missing))


def _profile_entries_by_location(group: h5py.Group, rotation_deg: int) -> dict[tuple[int, float], h5py.Group]:
    entries: dict[tuple[int, float], h5py.Group] = {}
    for run_id in sorted(group.keys()):
        run_grp = group[run_id]
        if float(run_grp.attrs.get("rotation_deg", rotation_deg)) != float(rotation_deg):
            continue
        key = (int(run_grp.attrs["port"]), round(float(run_grp.attrs["z_cm"]), 6))
        entries[key] = run_grp
    return entries


def _copy_set_attrs(src: h5py.Group, dst: h5py.Group) -> None:
    for key in ("label", "v_bank_v", "v_puff_v"):
        if key in src.attrs:
            dst.attrs[key] = src.attrs[key]


def _compute_run(
    upstream_grp: h5py.Group,
    downstream_grp: h5py.Group,
    te_hdf: h5py.File,
    es_id: str,
    *,
    upstream_area_m2: float,
    downstream_area_m2: float,
    current_factor: float,
) -> dict[str, np.ndarray]:
    upstream = upstream_grp["isat_a_raw"][()] * current_factor
    downstream = downstream_grp["isat_a_raw"][()] * current_factor
    upstream_std = upstream_grp["isat_a_raw_std"][()] * current_factor
    downstream_std = downstream_grp["isat_a_raw_std"][()] * current_factor
    time_s = upstream_grp["inter_sweep_time_s"][()]
    downstream_time_s = downstream_grp["inter_sweep_time_s"][()]
    if not np.allclose(time_s, downstream_time_s):
        raise ValueError(f"Dead-time grids differ for run {upstream_grp.attrs['run_id']}")

    z_cm = float(upstream_grp.attrs["z_cm"])
    te_grid = _interp_filled_te_to_deadtime(te_hdf, es_id, z_cm, time_s)
    cs_m_s = _ion_sound_speed_m_s(te_grid, M_I_AMU)

    with np.errstate(all="ignore"):
        upstream_j = upstream / upstream_area_m2
        downstream_j = downstream / downstream_area_m2
        positive = (upstream_j > 0) & (downstream_j > 0)
        mach = np.where(positive, np.log(upstream_j / downstream_j) / MACH_K, np.nan)
        relative = np.sqrt((upstream_std / upstream) ** 2 + (downstream_std / downstream) ** 2)
        mach_std = np.where(positive, relative / MACH_K, np.nan)
        velocity_km_s = mach * cs_m_s / 1000.0
        velocity_km_s_std = mach_std * cs_m_s / 1000.0

    return {
        "inter_sweep_time_s": time_s,
        "isat_upstream_a": upstream,
        "isat_upstream_a_std": upstream_std,
        "isat_downstream_a": downstream,
        "isat_downstream_a_std": downstream_std,
        "upstream_current_density_a_m2": upstream_j,
        "downstream_current_density_a_m2": downstream_j,
        "cs_m_s": cs_m_s,
        "mach": mach,
        "mach_std": mach_std,
        "velocity_km_s": velocity_km_s,
        "velocity_km_s_std": velocity_km_s_std,
    }


def _process_rotation(
    output_sets: h5py.Group,
    te_hdf: h5py.File,
    calibration: dict[str, dict[str, Any]],
    *,
    rotation_deg: int,
    upstream_hdf: h5py.File,
    downstream_hdf: h5py.File,
    upstream_area_key: str,
    downstream_area_key: str,
    target_sets: set[str] | None,
    probe_a_factor: float,
) -> None:
    for es_id in sorted(set(upstream_hdf["experiment_sets"].keys()) & set(downstream_hdf["experiment_sets"].keys()), key=int):
        if target_sets is not None and es_id not in target_sets:
            continue
        if f"experiment_sets/{es_id}" not in te_hdf:
            print(f"  [skip] ES {es_id}: not found in filled T_e")
            continue

        up_set = upstream_hdf[f"experiment_sets/{es_id}"]
        down_set = downstream_hdf[f"experiment_sets/{es_id}"]
        out_set = output_sets.require_group(es_id)
        _copy_set_attrs(up_set, out_set)

        up_entries = _profile_entries_by_location(up_set, rotation_deg)
        down_entries = _profile_entries_by_location(down_set, rotation_deg)
        for location_key in sorted(set(up_entries) & set(down_entries), key=lambda item: item[1]):
            up_run = up_entries[location_key]
            down_run = down_entries[location_key]
            up_run_id = str(up_run.attrs["run_id"])
            down_run_id = str(down_run.attrs["run_id"])

            probe = _probe_id(up_run_id)
            calib = calibration[probe]
            results = _compute_run(
                up_run,
                down_run,
                te_hdf,
                es_id,
                upstream_area_m2=calib[upstream_area_key.replace("cm2", "m2")],
                downstream_area_m2=calib[downstream_area_key.replace("cm2", "m2")],
                current_factor=probe_a_factor if probe == "A" else 1.0,
            )
            pair_note = "" if up_run_id == down_run_id else f" downstream={down_run_id}"
            print(f"  ES {es_id} run {up_run_id} rot={rotation_deg} port={up_run.attrs['port']}{pair_note}")
            out_run = out_set.create_group(up_run_id)
            for key in (
                "run_id",
                "port",
                "z_cm",
                "rotation_deg",
                "rotation_deg_recorded",
                "rotation_correction_applied",
            ):
                if key in up_run.attrs:
                    out_run.attrs[key] = up_run.attrs[key]
            out_run.attrs["probe_id"] = probe
            out_run.attrs["upstream_run_id"] = up_run_id
            out_run.attrs["downstream_run_id"] = down_run_id
            out_run.attrs["ap_L_cm2"] = calib["ap_L_cm2"]
            out_run.attrs["ap_R_cm2"] = calib["ap_R_cm2"]
            out_run.attrs["ap_estimated"] = calib["estimated"]
            out_run.attrs["probe_a_area_factor_applied"] = (
                probe_a_factor if probe == "A" else 1.0
            )
            out_run.attrs["upstream_area_key"] = upstream_area_key
            out_run.attrs["downstream_area_key"] = downstream_area_key
            out_run.attrs["upstream_source_file"] = upstream_hdf.filename
            out_run.attrs["downstream_source_file"] = downstream_hdf.filename
            out_run.attrs["mach_area_normalization"] = (
                "(I_upstream / A_upstream) / (I_downstream / A_downstream)"
            )
            for key, value in results.items():
                out_run.create_dataset(key, data=value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isweep-rot0", type=Path, default=ISWEEP_ROT0_HDF5)
    parser.add_argument("--isat-rot0", type=Path, default=ISAT_ROT0_HDF5)
    parser.add_argument("--isat-rot180", type=Path, default=ISAT_ROT180_HDF5)
    parser.add_argument("--isweep-rot180", type=Path, default=ISWEEP_ROT180_HDF5)
    parser.add_argument("--te-filled", type=Path, default=TE_FILLED_HDF5)
    parser.add_argument("--calibration", type=Path, default=CALIB_TOML)
    parser.add_argument(
        "--probe-a-calibration",
        type=Path,
        default=PROBE_A_CALIB_TOML,
    )
    parser.add_argument("--output", type=Path, default=HDF5_OUTPUT)
    parser.add_argument("--experiment-sets", nargs="+", default=None)
    parser.add_argument("--rotations", nargs="+", choices=["0", "180"], default=["0", "180"])
    args = parser.parse_args()

    _require_inputs([
        args.te_filled,
        args.calibration,
        args.isweep_rot0,
        args.isat_rot0,
        args.isat_rot180,
        args.isweep_rot180,
    ])
    target_sets = set(args.experiment_sets) if args.experiment_sets is not None else None
    calibration = _load_calibration(args.calibration)
    probe_a_calibration = load_probe_a_area_calibration(args.probe_a_calibration)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(args.output, "w") as out_hdf,
        h5py.File(args.te_filled, "r") as te_hdf,
        h5py.File(args.isweep_rot0, "r") as isweep_rot0_hdf,
        h5py.File(args.isat_rot0, "r") as isat_rot0_hdf,
        h5py.File(args.isat_rot180, "r") as isat_rot180_hdf,
        h5py.File(args.isweep_rot180, "r") as isweep_rot180_hdf,
    ):
        out_hdf.attrs["source_te_hdf5"] = str(args.te_filled)
        out_hdf.attrs["source_calibration_toml"] = str(args.calibration)
        out_hdf.attrs["source_probe_a_calibration_toml"] = str(
            args.probe_a_calibration
        )
        out_hdf.attrs["probe_a_factor"] = probe_a_calibration.factor
        out_hdf.attrs["m_i_amu"] = M_I_AMU
        out_hdf.attrs["mach_K"] = MACH_K
        out_hdf.attrs["current_dataset"] = "isat_a_raw"
        out_hdf.create_dataset("x_cm", data=X_CM)
        output_sets = out_hdf.create_group("experiment_sets")

        if "0" in args.rotations:
            _process_rotation(
                output_sets,
                te_hdf,
                calibration,
                rotation_deg=0,
                upstream_hdf=isweep_rot0_hdf,
                downstream_hdf=isat_rot0_hdf,
                upstream_area_key="ap_L_cm2",
                downstream_area_key="ap_R_cm2",
                target_sets=target_sets,
                probe_a_factor=probe_a_calibration.factor,
            )
        if "180" in args.rotations:
            _process_rotation(
                output_sets,
                te_hdf,
                calibration,
                rotation_deg=180,
                upstream_hdf=isat_rot180_hdf,
                downstream_hdf=isweep_rot180_hdf,
                upstream_area_key="ap_R_cm2",
                downstream_area_key="ap_L_cm2",
                target_sets=target_sets,
                probe_a_factor=probe_a_calibration.factor,
            )

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
