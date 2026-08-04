"""Scale ES3 T_e to match interferometer line-integrated densities.

Assumes the filled T_e spatial shape is useful, but permits a time- and
z-dependent multiplicative scale so Isweep-derived line-integrated density
matches the nearby interferometer chords at p21/p29/p41.

Writes comparison products tagged with ``_scaled``:

  processed/es3_scaled_density_te.hdf5
  figures/density_isweep_contours_expset3_scaled.png/.gif
  figures/te_contours_expset3_scaled.png/.gif
"""

from __future__ import annotations

from pathlib import Path
import tomllib

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from bapsf_lapd.density import electron_density_m3, ion_sound_speed_m_s


ISWEEP_HDF5 = Path("processed/isweep_deadtime_profiles.hdf5")
TE_FILLED_HDF5 = Path("processed/te_filled.hdf5")
INTERF_NPZ = Path("processed/interferometer_experiment_set_stats.npz")
CALIB_TOML = Path("processed/probe_area_calibration.toml")
OUTPUT_HDF5 = Path("processed/es3_scaled_density_te.hdf5")
OUTPUT_DIR = Path("figures")

ES_ID = "3"
M_I_AMU = 4.003
DENSITY_SCALE_M3 = 1e19  # 10^13 cm^-3
SUBPLOT_NCOLS = 8
SUBPLOT_NROWS = 5
PROBE_FROM_DIGIT = {1: "A", 8: "A", 2: "B", 3: "B", 4: "C", 5: "C", 6: "D", 7: "D"}
# ES3 p21 rotation labels are swapped for analysis; run 33 is the effective
# rot=0 row and therefore the one paired with the port-20 interferometer.
INTERF_PORT_FOR_RUN = {"33": 20, "34": 29, "36": 40}


def _probe_id(run_id: str) -> str:
    return PROBE_FROM_DIGIT[int(run_id[1])]


def _pcolormesh_edges(centers: np.ndarray) -> np.ndarray:
    mids = (centers[:-1] + centers[1:]) / 2
    lo = centers[0] - (centers[1] - centers[0]) / 2
    hi = centers[-1] + (centers[-1] - centers[-2]) / 2
    return np.concatenate([[lo], mids, [hi]])


def _line_integral_cm2(profile_m3: np.ndarray, x_m: np.ndarray) -> float:
    valid = np.isfinite(profile_m3) & (profile_m3 > 0)
    if valid.sum() < 2:
        return np.nan
    return float(np.trapezoid(profile_m3[valid], x_m[valid]) / 1e4)


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


def _load_ap_l_m2(path: Path) -> dict[str, float]:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return {probe: data[f"probe_{probe}"]["ap_L_cm2"] * 1e-4 for probe in ("A", "B", "C", "D")}


def _interp_te_to_times(te_x_cycle: np.ndarray, te_time_ms: np.ndarray, target_time_ms: np.ndarray) -> np.ndarray:
    return np.vstack([
        np.interp(target_time_ms, te_time_ms, te_x_cycle[xi, :])
        for xi in range(te_x_cycle.shape[0])
    ])


def compute_scaled_products() -> tuple[dict, dict]:
    ap_l_m2 = _load_ap_l_m2(CALIB_TOML)
    OUTPUT_HDF5.parent.mkdir(parents=True, exist_ok=True)

    with (
        h5py.File(ISWEEP_HDF5, "r") as isweep_hdf,
        h5py.File(TE_FILLED_HDF5, "r") as te_hdf,
        np.load(INTERF_NPZ) as interf_npz,
        h5py.File(OUTPUT_HDF5, "w") as out_hdf,
    ):
        isweep_set = isweep_hdf[f"experiment_sets/{ES_ID}"]
        te_set = te_hdf[f"experiment_sets/{ES_ID}"]
        x_cm = isweep_hdf["x_cm"][()]
        x_m = x_cm * 1e-2
        te_x_cm = te_set["x_cm"][()]
        if not np.allclose(x_cm, te_x_cm):
            raise ValueError("Isweep and filled-Te x grids do not match")

        te_z_cm = te_set["z_cm"][()]
        te_time_ms = te_set["cycle_time_ms"][()]
        te_filled = te_set["te_filled"][()]

        entries = []
        for run_id in sorted(isweep_set.keys()):
            grp = isweep_set[run_id]
            z_cm = float(grp.attrs["z_cm"])
            z_idx = int(np.argmin(np.abs(te_z_cm - z_cm)))
            time_s = grp["inter_sweep_time_s"][()]
            time_ms = time_s * 1000.0
            te_interp = _interp_te_to_times(te_filled[z_idx, :, :], te_time_ms, time_ms)
            cs = ion_sound_speed_m_s(te_interp, M_I_AMU)
            probe = _probe_id(run_id)
            density = electron_density_m3(grp["isat_a"][()], ap_l_m2[probe], cs)
            density = np.where((density > 0) & np.isfinite(density), density, np.nan)
            entries.append((z_cm, run_id, probe, density, time_s, grp["isat_a"][()], z_idx))

        entries.sort(key=lambda item: item[0])
        z_cm = np.array([item[0] for item in entries])
        run_ids = [item[1] for item in entries]
        probes = [item[2] for item in entries]
        density_unscaled = np.stack([item[3] for item in entries], axis=0)
        time_s = entries[0][4]
        time_ms = time_s * 1000.0

        source_scale_by_run = {}
        ratio_by_run = {}
        for run_id, interf_port in INTERF_PORT_FOR_RUN.items():
            row = run_ids.index(run_id)
            line_lp = np.array([_line_integral_cm2(density_unscaled[row, :, k], x_m) for k in range(len(time_ms))])
            prefix = f"set{ES_ID}_p{interf_port}"
            line_if = np.interp(
                time_ms,
                interf_npz[f"{prefix}_time_ms"],
                interf_npz[f"{prefix}_line_integrated_mean_cm2"],
            )
            ratio = line_if / line_lp
            te_scale = 1.0 / ratio**2
            source_scale_by_run[run_id] = te_scale
            ratio_by_run[run_id] = ratio

        density_time_te_scale = np.ones((len(z_cm), len(time_ms)))
        for run_id, te_scale in source_scale_by_run.items():
            density_time_te_scale[run_ids.index(run_id)] = te_scale

        density_scaled = np.full_like(density_unscaled, np.nan)
        for zi, (_, run_id, probe, _, _, isat, te_z_idx) in enumerate(entries):
            te_interp = _interp_te_to_times(te_filled[te_z_idx, :, :], te_time_ms, time_ms)
            te_scaled_at_density_times = te_interp * density_time_te_scale[zi, :][None, :]
            cs_scaled = ion_sound_speed_m_s(te_scaled_at_density_times, M_I_AMU)
            density_scaled[zi] = electron_density_m3(isat, ap_l_m2[probe], cs_scaled)
        density_scaled = np.where((density_scaled > 0) & np.isfinite(density_scaled), density_scaled, np.nan)

        te_time_scale = np.ones((len(te_z_cm), len(te_time_ms)))
        for run_id, density_scale in source_scale_by_run.items():
            row = run_ids.index(run_id)
            te_z_idx = entries[row][6]
            te_time_scale[te_z_idx] = np.interp(te_time_ms, time_ms, density_scale)
        te_scaled = te_filled * te_time_scale[:, None, :]

        out_hdf.attrs["description"] = "ES3 T_e scaled so Isweep-derived line-integrated density matches interferometers."
        out_hdf.attrs["scaling_policy"] = "Only p21, p29, and p41 rows are scaled; p11 and p50 remain unscaled."
        out_hdf.attrs["source_current_hdf5"] = str(ISWEEP_HDF5)
        out_hdf.attrs["probe_a_factor"] = float(
            isweep_hdf.attrs["probe_a_isat_area_factor"]
        )
        out_hdf.attrs["probe_a_factor_source"] = str(
            isweep_hdf.attrs.get("calibration_factor_source", "")
        )
        out_hdf.attrs["source_te_hdf5"] = str(TE_FILLED_HDF5)
        out_hdf.attrs["source_interferometer_npz"] = str(INTERF_NPZ)
        out_hdf.create_dataset("x_cm", data=x_cm)
        out_hdf.create_dataset("z_cm", data=z_cm)
        out_hdf.create_dataset("density_time_ms", data=time_ms)
        out_hdf.create_dataset("te_time_ms", data=te_time_ms)
        out_hdf.create_dataset("n_e_m3", data=density_scaled)
        out_hdf.create_dataset("n_e_unscaled_m3", data=density_unscaled)
        out_hdf.create_dataset("te_scaled_ev", data=te_scaled)
        out_hdf.create_dataset("te_unscaled_ev", data=te_filled)
        out_hdf.create_dataset("te_scale_density_times", data=density_time_te_scale)
        out_hdf.create_dataset("te_scale_te_times", data=te_time_scale)
        ratios_grp = out_hdf.create_group("line_integral_ratio")
        for run_id, ratio in ratio_by_run.items():
            ratios_grp.create_dataset(run_id, data=ratio)

        density_data = {
            "field": density_scaled / DENSITY_SCALE_M3,
            "field_m3": density_scaled,
            "x_cm": x_cm,
            "z_cm": z_cm,
            "cycle_time_ms": time_ms,
            "es_id": ES_ID,
            "es_label": str(isweep_set.attrs.get("label", "set 3")),
            "v_bank": float(isweep_set.attrs.get("v_bank_v", np.nan)),
            "kind": "density",
        }
        te_data = {
            "field": te_scaled,
            "x_cm": x_cm,
            "z_cm": te_z_cm,
            "cycle_time_ms": te_time_ms,
            "es_id": ES_ID,
            "es_label": str(te_set.attrs.get("label", "set 3")),
            "v_bank": float(te_set.attrs.get("v_bank_v", np.nan)),
            "kind": "te",
        }
    return density_data, te_data


def _draw_density_frame(ax: plt.Axes, data: dict, ci: int, x_edges: np.ndarray, z_edges: np.ndarray, norm) -> None:
    ax.clear()
    ax.pcolormesh(x_edges, z_edges, data["field"][:, :, ci], cmap="viridis", norm=norm, rasterized=True)
    left, right = _fwhm_edges_by_z(data["field_m3"][:, :, ci], data["x_cm"])
    valid_left = np.isfinite(left)
    valid_right = np.isfinite(right)
    ax.plot(left[valid_left], data["z_cm"][valid_left], "w:", linewidth=1.1)
    ax.plot(right[valid_right], data["z_cm"][valid_right], "w:", linewidth=1.1)
    ax.set_title(f"t = {data['cycle_time_ms'][ci]:.1f} ms", fontsize=7, pad=2)
    ax.set_xlabel("x (cm)", fontsize=6)
    ax.set_ylabel("z (cm)", fontsize=6)
    ax.tick_params(labelsize=5)


def _draw_te_frame(ax: plt.Axes, data: dict, ci: int, x_edges: np.ndarray, z_edges: np.ndarray, norm) -> None:
    ax.clear()
    ax.pcolormesh(x_edges, z_edges, data["field"][:, :, ci], cmap="plasma", norm=norm, rasterized=True)
    ax.set_title(f"t = {data['cycle_time_ms'][ci]:.1f} ms", fontsize=7, pad=2)
    ax.set_xlabel("x (cm)", fontsize=6)
    ax.set_ylabel("z (cm)", fontsize=6)
    ax.tick_params(labelsize=5)


def plot_subplots(data: dict, vmin: float, vmax: float) -> Path:
    import matplotlib.colors as mcolors

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(
        SUBPLOT_NROWS,
        SUBPLOT_NCOLS,
        figsize=(SUBPLOT_NCOLS * 2.4, SUBPLOT_NROWS * 2.2),
        constrained_layout=True,
    )
    ax_flat = axes.flatten()
    for ci in range(data["field"].shape[2]):
        if data["kind"] == "density":
            _draw_density_frame(ax_flat[ci], data, ci, x_edges, z_edges, norm)
        else:
            _draw_te_frame(ax_flat[ci], data, ci, x_edges, z_edges, norm)
    for ax in ax_flat[data["field"].shape[2]:]:
        ax.set_visible(False)

    cmap = "viridis" if data["kind"] == "density" else "plasma"
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.02)
    label = "$n_e$ ($10^{13}$ cm$^{-3}$)" if data["kind"] == "density" else "$T_e$ (eV)"
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    title = "Isweep-derived density" if data["kind"] == "density" else "Filled $T_e$"
    fig.suptitle(
        f"{title} scaled vs (x, z) - experiment set {data['es_id']}: {data['es_label']}"
        f"  (V_bank = {data['v_bank']:.0f} V)",
        fontsize=11,
    )
    prefix = "density_isweep_contours" if data["kind"] == "density" else "te_contours"
    out_path = OUTPUT_DIR / f"{prefix}_expset{data['es_id']}_scaled.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def plot_animation(data: dict, vmin: float, vmax: float) -> Path:
    import matplotlib.colors as mcolors

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_edges = _pcolormesh_edges(data["x_cm"])
    z_edges = _pcolormesh_edges(data["z_cm"])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = "viridis" if data["kind"] == "density" else "plasma"

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, z_edges, data["field"][:, :, 0], cmap=cmap, norm=norm, rasterized=True)
    fwhm_lines = []
    if data["kind"] == "density":
        left, right = _fwhm_edges_by_z(data["field_m3"][:, :, 0], data["x_cm"])
        valid_left = np.isfinite(left)
        valid_right = np.isfinite(right)
        fwhm_lines = [
            ax.plot(left[valid_left], data["z_cm"][valid_left], "w:", linewidth=1.4)[0],
            ax.plot(right[valid_right], data["z_cm"][valid_right], "w:", linewidth=1.4)[0],
        ]
    cbar = fig.colorbar(mesh, ax=ax)
    label = "$n_e$ ($10^{13}$ cm$^{-3}$)" if data["kind"] == "density" else "$T_e$ (eV)"
    cbar.set_label(label, fontsize=10)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("z (cm)")
    title = ax.set_title(f"t = {data['cycle_time_ms'][0]:.1f} ms")

    def update(ci: int):
        mesh.set_array(data["field"][:, :, ci].ravel())
        if data["kind"] == "density":
            left, right = _fwhm_edges_by_z(data["field_m3"][:, :, ci], data["x_cm"])
            valid_left = np.isfinite(left)
            valid_right = np.isfinite(right)
            fwhm_lines[0].set_data(left[valid_left], data["z_cm"][valid_left])
            fwhm_lines[1].set_data(right[valid_right], data["z_cm"][valid_right])
        title.set_text(f"t = {data['cycle_time_ms'][ci]:.1f} ms")
        return (mesh, title, *fwhm_lines)

    anim = animation.FuncAnimation(fig, update, frames=data["field"].shape[2], interval=300, blit=False)
    prefix = "density_isweep_contours" if data["kind"] == "density" else "te_contours"
    out_path = OUTPUT_DIR / f"{prefix}_expset{data['es_id']}_scaled.gif"
    anim.save(str(out_path), writer="pillow", dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    density_data, te_data = compute_scaled_products()
    density_finite = density_data["field"][np.isfinite(density_data["field"]) & (density_data["field"] > 0)]
    d_vmin = max(0.0, float(np.percentile(density_finite, 2)))
    d_vmax = float(np.percentile(density_finite, 98))
    with h5py.File(TE_FILLED_HDF5, "r") as hdf:
        original_te = hdf[f"experiment_sets/{ES_ID}/te_masked"][()]
        te_finite = original_te[np.isfinite(original_te) & (original_te > 0)]
        if te_finite.size == 0:
            original_te = hdf[f"experiment_sets/{ES_ID}/te_filled"][()]
            te_finite = original_te[np.isfinite(original_te) & (original_te > 0)]
    te_vmin = max(0.0, float(np.percentile(te_finite, 2)))
    te_vmax = float(np.percentile(te_finite, 98))
    print(f"Density color scale: {d_vmin:.3g} to {d_vmax:.3g} x 1e13 cm^-3")
    print(f"Te color scale: {te_vmin:.3g} to {te_vmax:.3g} eV")
    print(f"Wrote {OUTPUT_HDF5}")
    plot_subplots(density_data, d_vmin, d_vmax)
    plot_animation(density_data, d_vmin, d_vmax)
    plot_subplots(te_data, te_vmin, te_vmax)
    plot_animation(te_data, te_vmin, te_vmax)


if __name__ == "__main__":
    main()
