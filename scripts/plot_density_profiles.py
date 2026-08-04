"""Compute and plot Isweep-derived density profiles.

This test workflow combines:

  processed/isweep_deadtime_profiles.hdf5
  processed/te_filled.hdf5
  processed/probe_area_calibration.toml

For each experiment set, it computes electron density from the dead-time Isweep
ion-saturation current and the filled T_e profile, then writes a compact HDF5
product plus contour PNG/GIF outputs.

Probe A uses the probe-B area.  The Isweep profile product's ``isat_a`` dataset
already has the probe-A current area factor applied; ``isat_a_raw`` is not used.

Usage
-----
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_density_profiles.py
  MPLCONFIGDIR=.matplotlib ./.venv/bin/python scripts/plot_density_profiles.py --no-animation
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tomllib

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd.density import electron_density_m3, ion_sound_speed_m_s


ISWEEP_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
CALIB_TOML = Path("processed/probe_area_calibration.toml")
HDF5_OUTPUT = Path("processed/density_profiles_isweep.hdf5")
OUTPUT_DIR = Path("figures")

M_I_AMU = 4.003
CMAP = "viridis"
SUBPLOT_NCOLS = 8
SUBPLOT_NROWS = 5
DENSITY_SCALE_M3 = 1e19  # 10^13 cm^-3

PROBE_FROM_DIGIT = {1: "A", 8: "A", 2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}


def _probe_id(run_id: str) -> str:
    return PROBE_FROM_DIGIT[int(run_id[1])]


def _pcolormesh_edges(centers: np.ndarray) -> np.ndarray:
    mids = (centers[:-1] + centers[1:]) / 2
    lo = centers[0] - (centers[1] - centers[0]) / 2
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[lo], mids, [hi]])


def _fwhm_edges_by_z(n_2d: np.ndarray, x_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.full(n_2d.shape[0], np.nan)
    right = np.full(n_2d.shape[0], np.nan)
    for zi, profile in enumerate(n_2d):
        finite = np.isfinite(profile)
        if finite.sum() < 3:
            continue
        peak_idx = int(np.nanargmax(profile))
        if peak_idx == 0 or peak_idx == len(profile) - 1:
            continue
        half_max = profile[peak_idx] / 2.0

        def crossing(indices: range) -> float:
            prev_i = peak_idx
            for i in indices:
                if not finite[i]:
                    continue
                if profile[i] <= half_max:
                    hi, lo = prev_i, i
                    if profile[hi] <= half_max or profile[lo] > half_max:
                        return np.nan
                    frac = (half_max - profile[lo]) / (profile[hi] - profile[lo])
                    return float(x_cm[lo] + frac * (x_cm[hi] - x_cm[lo]))
                prev_i = i
            return np.nan

        left[zi] = crossing(range(peak_idx - 1, -1, -1))
        right[zi] = crossing(range(peak_idx + 1, len(profile)))
    return left, right


def _load_areas(path: Path) -> dict[str, dict[str, float]]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {
        probe: {
            "ap_L_cm2": data[f"probe_{probe}"]["ap_L_cm2"] * 1e-4,
            "ap_R_cm2": data[f"probe_{probe}"]["ap_R_cm2"] * 1e-4,
            "ap_L_std_cm2": data[f"probe_{probe}"]["ap_L_std_cm2"] * 1e-4,
            "ap_R_std_cm2": data[f"probe_{probe}"]["ap_R_std_cm2"] * 1e-4,
        }
        for probe in ("A", "B", "C", "D")
    }


def _probe_a_factor_uncertainty(hdf: h5py.File) -> tuple[float, float]:
    """Return factor uncertainty and relative uncertainty from bracket limits."""
    factor = float(hdf.attrs["probe_a_isat_area_factor"])
    lower = float(hdf.attrs["calibration_factor_lower_bound"])
    upper = float(hdf.attrs["calibration_factor_upper_bound"])
    if factor <= 0 or lower <= 0 or upper <= 0:
        raise ValueError("Probe-A factor and bracket bounds must be positive")
    factor_uncertainty = abs(upper - lower) / 2.0
    return factor_uncertainty, factor_uncertainty / factor


def _probe_a_area_relative_uncertainty(
    area_m2: float,
    area_std_m2: float,
    factor_relative_uncertainty: float,
) -> float:
    """Propagate reference-area and factor uncertainty through A_A = A_ref/f."""
    if area_m2 <= 0 or area_std_m2 < 0 or factor_relative_uncertainty < 0:
        raise ValueError("Area and calibration uncertainties must be non-negative")
    return float(np.hypot(area_std_m2 / area_m2, factor_relative_uncertainty))


def _interp_te_to_profile_times(
    te_x_cycle: np.ndarray,
    te_time_ms: np.ndarray,
    profile_time_ms: np.ndarray,
) -> np.ndarray:
    return np.vstack([
        np.interp(profile_time_ms, te_time_ms, te_x_cycle[xi, :])
        for xi in range(te_x_cycle.shape[0])
    ])


def compute_density_profiles(
    isweep_path: Path,
    te_path: Path,
    calib_path: Path,
    output_path: Path,
    *,
    experiment_sets: set[str] | None,
    probe_a_factor_override: float | None,
) -> list[dict]:
    areas_m2 = _load_areas(calib_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(isweep_path, "r") as isweep_hdf,
        h5py.File(te_path, "r") as te_hdf,
        h5py.File(output_path, "w") as out_hdf,
    ):
        x_cm = isweep_hdf["x_cm"][()]
        out_hdf.attrs["source_current_hdf5"] = str(isweep_path)
        out_hdf.attrs["source_te_hdf5"] = str(te_path)
        out_hdf.attrs["source_calibration_toml"] = str(calib_path)
        out_hdf.attrs["m_i_amu"] = M_I_AMU
        factor_uncertainty, factor_relative_uncertainty = _probe_a_factor_uncertainty(
            isweep_hdf
        )
        out_hdf.attrs["probe_a_factor"] = float(
            isweep_hdf.attrs["probe_a_isat_area_factor"]
        )
        out_hdf.attrs["probe_a_factor_lower_bound"] = float(
            isweep_hdf.attrs["calibration_factor_lower_bound"]
        )
        out_hdf.attrs["probe_a_factor_upper_bound"] = float(
            isweep_hdf.attrs["calibration_factor_upper_bound"]
        )
        out_hdf.attrs["probe_a_factor_uncertainty"] = factor_uncertainty
        out_hdf.attrs["probe_a_factor_relative_uncertainty"] = factor_relative_uncertainty
        out_hdf.attrs["probe_a_factor_uncertainty_definition"] = (
            "half-width of the density-bracket factor interval"
        )
        out_hdf.attrs["probe_a_area_uncertainty_propagation"] = (
            "sqrt((sigma_Aref/Aref)^2 + (sigma_factor/factor)^2)"
        )
        out_hdf.create_dataset("x_cm", data=x_cm)
        out_sets = out_hdf.create_group("experiment_sets")
        plot_data: list[dict] = []

        isweep_ids = set(isweep_hdf["experiment_sets"].keys())
        te_ids = set(te_hdf["experiment_sets"].keys())
        for es_id in sorted(isweep_ids & te_ids, key=int):
            if experiment_sets is not None and es_id not in experiment_sets:
                continue
            isweep_set = isweep_hdf[f"experiment_sets/{es_id}"]
            te_set = te_hdf[f"experiment_sets/{es_id}"]
            es_label = str(isweep_set.attrs.get("label", f"set {es_id}"))
            v_bank = float(isweep_set.attrs.get("v_bank_v", np.nan))

            te_x_cm = te_set["x_cm"][()]
            if not np.allclose(x_cm, te_x_cm):
                raise ValueError(f"Isweep and filled-Te x grids do not match for ES {es_id}")

            te_z_cm = te_set["z_cm"][()]
            te_time_ms = te_set["cycle_time_ms"][()]
            te_filled = te_set["te_filled"][()]

            entries = []
            for run_id in sorted(isweep_set.keys()):
                grp = isweep_set[run_id]
                z_cm = float(grp.attrs["z_cm"])
                z_idx = int(np.argmin(np.abs(te_z_cm - z_cm)))
                profile_time_s = grp["inter_sweep_time_s"][()]
                profile_time_ms = profile_time_s * 1000.0
                te_interp = _interp_te_to_profile_times(
                    te_filled[z_idx, :, :],
                    te_time_ms,
                    profile_time_ms,
                )
                cs = ion_sound_speed_m_s(te_interp, M_I_AMU)

                probe = _probe_id(run_id)
                isat = grp["isat_a"][()]
                applied_factor = float(grp.attrs["probe_a_area_factor_applied"])
                if probe == "A" and probe_a_factor_override is not None:
                    if applied_factor <= 0:
                        raise ValueError(f"Run {run_id} has non-positive probe-A factor {applied_factor}")
                    isat = isat * (probe_a_factor_override / applied_factor)
                area_key = str(grp.attrs.get("density_area_key", "ap_L_cm2"))
                density = electron_density_m3(isat, areas_m2[probe][area_key], cs)
                density = np.where((density > 0) & np.isfinite(density), density, np.nan)

                entries.append((
                    z_cm,
                    run_id,
                    probe,
                    density,
                    profile_time_s,
                    probe_a_factor_override if probe == "A" and probe_a_factor_override is not None else applied_factor,
                    area_key,
                ))

            if not entries:
                continue

            entries.sort(key=lambda item: item[0])
            z_cm = np.array([item[0] for item in entries], dtype=np.float64)
            run_ids = [item[1] for item in entries]
            probes = [item[2] for item in entries]
            density_grid = np.stack([item[3] for item in entries], axis=0)
            time_s = entries[0][4]

            set_grp = out_sets.create_group(es_id)
            set_grp.create_dataset("z_cm", data=z_cm)
            set_grp.create_dataset("inter_sweep_time_s", data=time_s)
            set_grp.create_dataset("n_e_m3", data=density_grid)
            set_grp.attrs["label"] = es_label
            set_grp.attrs["v_bank_v"] = v_bank
            if probe_a_factor_override is not None:
                set_grp.attrs["probe_a_factor_override"] = probe_a_factor_override

            runs_grp = set_grp.create_group("runs")
            for z, run_id, probe, _, _, factor, area_key in entries:
                run_grp = runs_grp.create_group(run_id)
                run_grp.attrs["z_cm"] = z
                run_grp.attrs["probe_id"] = probe
                run_grp.attrs["density_area_key"] = area_key
                run_grp.attrs["ap_cm2"] = areas_m2[probe][area_key] * 1e4
                run_grp.attrs["ap_L_cm2"] = areas_m2[probe]["ap_L_cm2"] * 1e4
                run_grp.attrs["ap_R_cm2"] = areas_m2[probe]["ap_R_cm2"] * 1e4
                run_grp.attrs["probe_a_area_factor_applied"] = factor
                if probe == "A":
                    area_std_key = area_key.replace("_cm2", "_std_cm2")
                    area_relative_uncertainty = _probe_a_area_relative_uncertainty(
                        areas_m2[probe][area_key],
                        areas_m2[probe][area_std_key],
                        factor_relative_uncertainty,
                    )
                    run_grp.attrs["ap_reference_std_cm2"] = (
                        areas_m2[probe][area_std_key] * 1e4
                    )
                    run_grp.attrs["probe_a_area_relative_uncertainty"] = (
                        area_relative_uncertainty
                    )
                    run_grp.attrs["probe_a_density_relative_uncertainty"] = (
                        area_relative_uncertainty
                    )

            plot_data.append({
                "density": density_grid,
                "x_cm": x_cm,
                "z_cm": z_cm,
                "cycle_time_ms": time_s * 1000.0,
                "run_ids": run_ids,
                "probes": probes,
                "es_id": es_id,
                "es_label": es_label,
                "v_bank": v_bank,
            })

    return plot_data


def _make_frame(
    ax: plt.Axes,
    n_scaled_2d: np.ndarray,
    n_m3_2d: np.ndarray,
    x_cm: np.ndarray,
    z_cm: np.ndarray,
    x_edges: np.ndarray,
    z_edges: np.ndarray,
    norm,
    title: str,
) -> None:
    ax.clear()
    ax.pcolormesh(x_edges, z_edges, n_scaled_2d, cmap=CMAP, norm=norm, rasterized=True)
    left, right = _fwhm_edges_by_z(n_m3_2d, x_cm)
    valid_left = np.isfinite(left)
    valid_right = np.isfinite(right)
    ax.plot(left[valid_left], z_cm[valid_left], "w:", linewidth=1.1)
    ax.plot(right[valid_right], z_cm[valid_right], "w:", linewidth=1.1)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel("x (cm)", fontsize=6)
    ax.set_ylabel("z (cm)", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_subplots(data: dict, output_dir: Path, vmin: float, vmax: float) -> Path:
    import matplotlib.colors as mcolors

    output_dir.mkdir(parents=True, exist_ok=True)
    n_scaled = data["density"] / DENSITY_SCALE_M3
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    times = data["cycle_time_ms"]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        SUBPLOT_NROWS,
        SUBPLOT_NCOLS,
        figsize=(SUBPLOT_NCOLS * 2.4, SUBPLOT_NROWS * 2.2),
        constrained_layout=True,
    )
    ax_flat = axes.flatten()
    for ci in range(n_scaled.shape[2]):
        _make_frame(
            ax_flat[ci],
            n_scaled[:, :, ci],
            data["density"][:, :, ci],
            data["x_cm"],
            data["z_cm"],
            x_edges,
            z_edges,
            norm,
            f"t = {times[ci]:.1f} ms",
        )
    for ax in ax_flat[n_scaled.shape[2]:]:
        ax.set_visible(False)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
    cbar.set_label("$n_e$ ($10^{13}$ cm$^{-3}$)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        f"Isweep-derived density vs (x, z) - experiment set {data['es_id']}: {data['es_label']}"
        f"  (V_bank = {data['v_bank']:.0f} V)",
        fontsize=11,
    )

    out_path = output_dir / f"density_isweep_contours_expset{data['es_id']}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_animation(data: dict, output_dir: Path, vmin: float, vmax: float) -> Path:
    import matplotlib.colors as mcolors

    output_dir.mkdir(parents=True, exist_ok=True)
    n_scaled = data["density"] / DENSITY_SCALE_M3
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    times = data["cycle_time_ms"]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, z_edges, n_scaled[:, :, 0], cmap=CMAP, norm=norm, rasterized=True)
    left0, right0 = _fwhm_edges_by_z(data["density"][:, :, 0], data["x_cm"])
    valid_left0 = np.isfinite(left0)
    valid_right0 = np.isfinite(right0)
    (left_line,) = ax.plot(left0[valid_left0], data["z_cm"][valid_left0], "w:", linewidth=1.4)
    (right_line,) = ax.plot(right0[valid_right0], data["z_cm"][valid_right0], "w:", linewidth=1.4)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("$n_e$ ($10^{13}$ cm$^{-3}$)", fontsize=10)
    ax.set_xlabel("x (cm)", fontsize=10)
    ax.set_ylabel("z (cm)", fontsize=10)
    title = ax.set_title(f"t = {times[0]:.1f} ms", fontsize=10)
    fig.suptitle(
        f"Isweep-derived density - ES {data['es_id']}: {data['es_label']}"
        f"  (V_bank = {data['v_bank']:.0f} V)",
        fontsize=10,
    )

    def update(ci: int):
        mesh.set_array(n_scaled[:, :, ci].ravel())
        left, right = _fwhm_edges_by_z(data["density"][:, :, ci], data["x_cm"])
        valid_left = np.isfinite(left)
        valid_right = np.isfinite(right)
        left_line.set_data(left[valid_left], data["z_cm"][valid_left])
        right_line.set_data(right[valid_right], data["z_cm"][valid_right])
        title.set_text(f"t = {times[ci]:.1f} ms")
        return mesh, left_line, right_line, title

    anim = animation.FuncAnimation(fig, update, frames=n_scaled.shape[2], interval=300, blit=False)
    out_path = output_dir / f"density_isweep_contours_expset{data['es_id']}.gif"
    anim.save(str(out_path), writer="pillow", dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isweep", type=Path, default=ISWEEP_HDF5)
    parser.add_argument("--te-filled", type=Path, default=TE_FILLED_HDF5)
    parser.add_argument("--calibration", type=Path, default=CALIB_TOML)
    parser.add_argument("--output", type=Path, default=HDF5_OUTPUT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--experiment-sets",
        nargs="+",
        default=None,
        help="experiment set IDs to process; default is all sets present in both inputs",
    )
    parser.add_argument(
        "--probe-a-factor",
        type=float,
        default=None,
        help="override the probe-A current area factor baked into isat_a",
    )
    parser.add_argument("--no-animation", action="store_true")
    parser.add_argument("--vmin", type=float, default=None, help="manual color lower limit in 1e13 cm^-3")
    parser.add_argument("--vmax", type=float, default=None, help="manual color upper limit in 1e13 cm^-3")
    args = parser.parse_args()

    target_sets = set(args.experiment_sets) if args.experiment_sets is not None else None
    all_data = compute_density_profiles(
        args.isweep,
        args.te_filled,
        args.calibration,
        args.output,
        experiment_sets=target_sets,
        probe_a_factor_override=args.probe_a_factor,
    )
    if not all_data:
        raise RuntimeError("No experiment sets found in both Isweep and filled-Te inputs")

    combined_density = np.concatenate([data["density"].ravel() for data in all_data])
    finite_scaled = (combined_density / DENSITY_SCALE_M3)[np.isfinite(combined_density)]
    finite_scaled = finite_scaled[finite_scaled > 0]
    if finite_scaled.size == 0:
        raise RuntimeError("No finite positive density values found")

    vmin = args.vmin if args.vmin is not None else max(0.0, float(np.percentile(finite_scaled, 2)))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(finite_scaled, 98))
    print(f"Color scale: vmin={vmin:.3g}, vmax={vmax:.3g} x 1e13 cm^-3")
    print(f"Wrote {args.output}")

    for data in all_data:
        plot_subplots(data, args.output_dir, vmin, vmax)
        if not args.no_animation:
            plot_animation(data, args.output_dir, vmin, vmax)


if __name__ == "__main__":
    main()
