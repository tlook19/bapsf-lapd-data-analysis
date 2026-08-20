# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The project uses a local conda environment at `.venv` (Python 3.14). The interpreter is never on the system PATH, so always call it directly:

```bash
./.venv/bin/python -m pytest -q                          # run all tests
MPLCONFIGDIR=.matplotlib ./.venv/bin/python -m pytest -q # run tests (suppresses matplotlib config warnings)
./.venv/bin/python -m pytest tests/test_langmuir.py -q   # run a single test file
./.venv/bin/python -m pytest -k test_name -q             # run a single test by name
```

`MPLCONFIGDIR=.matplotlib` is required whenever matplotlib is imported (e.g., in scripts) to avoid permission warnings in the project directory.

To recreate the environment from scratch:
```bash
mamba create -y -p ./.venv -c conda-forge python=3.14 numpy scipy matplotlib h5py pytest
```

The package is installed in editable mode implicitly via `[tool.pytest.ini_options] pythonpath = ["src"]` — pytest adds `src/` to `sys.path`. To use the package outside of pytest, run:
```bash
./.venv/bin/pip install -e .
```

## Data

HDF5 run files live in `data/may2026/` and are gitignored. Tests that require HDF5 files are auto-skipped with `pytest.mark.skipif` when the data directory is empty.

## Architecture

The library is `src/bapsf_lapd/`. All public symbols are re-exported from `__init__.py`.

### Configuration layer (`config.py`, `manifest.py`)

All run metadata is captured in frozen dataclasses: `ExperimentSet` (plasma conditions shared by a group of runs), `ProbeConfig` (port placement and axial position), `SweepConfig` (Langmuir voltage ramp timing), `AcquisitionConfig` (digitizer sample rate and shape), `ChannelConfig` (HDF5 path plus resistor/gain/attenuation calibration factors), and `RunConfig` (one per HDF5 file, holds all of the above).

`default_run_config()` in `config.py` builds a `RunConfig` from a two-digit run ID by looking up hardcoded defaults. `load_run_manifest()` in `manifest.py` is the preferred path for the real dataset: it reads `config/may2026_run_manifest.toml`, merges `[trace_defaults]` with per-run `[runs.NN.traces]` overrides, and returns a `dict[str, RunConfig]`.

**Run ID convention**: two-digit string `"NN"`. First digit encodes experiment set (1–4; prefix `"0"` → set 1). Second digit encodes probe configuration.

### Dataset and reader layer (`dataset.py`, `reader.py`)

`LapdDataset` is a dict-like collection of `RunConfig` objects indexed by run ID. Entry points are `LapdDataset.from_manifest(path)` and `LapdDataset.from_directory(path)` (the latter auto-parses port and rotation from the filename pattern `NN_xline_pPP_..._rotRR_...hdf5`). `LapdDataset` provides experiment-set grouping and cross-run Welford streaming statistics.

`LapdRun` is the lazy HDF5 reader for one file. It wraps a `RunConfig` and opens the file only when a method is called. Key operations:
- `voltage_trace(channel, shot)` — reads raw uint samples and applies per-shot `Scale`/`Offset` headers from the SIS digitizer.
- `trace(channel, shot)` — calls `voltage_trace`, subtracts the zero-offset (estimated from the trace tail), then applies `ChannelConfig.apply_calibration()` (resistor/gain/attenuation or multiplier).
- `langmuir_traces(channel)` — reads all shots and reshapes the flat `(n_positions × n_shots, samples)` array to `(n_positions, n_shots, samples)`.
- `zero_offset_stats(channel)` — estimates the pre-calibration DC offset from the final 200 µs of every shot; this is subtracted by default in `trace()` and `langmuir_traces()` for Langmuir channels (`isat`, `i_sweep`, `v_sweep`).
- `sweep_ramp_sample_slices()` — converts `SweepConfig` timing into sample-index slices for each Langmuir voltage ramp cycle.
- `discharge_summary()` — reads `MSI/Discharge` and finds peak current, matching voltage, and peak power.

### Signal processing and analysis (`filtering.py`, `langmuir.py`, `quality.py`)

`butterworth_lowpass` applies a zero-phase Butterworth filter via `sosfiltfilt`.

`analyze_langmuir_sweep(voltage, current)` runs the full Langmuir I-V pipeline on pre-filtered, pre-sliced data:
1. Robust linear fit to the ion-saturation region (`ion_fraction` = lowest 22% of voltage).
2. Robust linear fit to the electron-saturation region (top 18%).
3. Selects the electron-retarding region (positive electron current between ~8th and 50th percentile), keeps one contiguous segment near the I-V transition midpoint.
4. Log-linear fit (`ln(I_e) = a + b·V`) and direct exponential fit (`I = ion_line + A·exp(V/T_e)`), both via `scipy.optimize.least_squares` with soft-L1 loss.
5. Plasma potential from the derivative peak and from intersections of each fit with the saturation line.

`evaluate_langmuir_quality(analysis)` returns a `LangmuirQualityReport` with `QualityFlag` entries (severity `"ok"` / `"warn"` / `"bad"`) checking T_e bounds, method disagreement, fit point count, fit window width, RMS residual, and arc-like current jumps. `detect_arc_like_segments` also accepts peer shots for cross-shot outlier detection.

### Scripts (`scripts/`)

Standalone scripts that import `bapsf_lapd` and write outputs to `processed/` or `figures/`. They are not part of the installable package and are run directly with the `.venv` interpreter.

## TODOs / Calibration Notes

- The canonical p11/p50 Probe A factor is stored in `config/may2026_probe_a_area_calibration.toml`. It is calibrated from experiment-set-1 FWHM-core density over 10--19 ms after one-sided high-current shot rejection, requiring `n_p11 <= n_p21` and `n_p50 >= n_p41`. Apply it exactly once to both Probe A faces in all sets; it is an empirical current-normalization factor, not a direct geometric-area measurement.
- ES3 filled T_e trusts p11/p29 and ES4 trusts p11. Other later-set axial rows are excluded before filling because they approach the diagnostic's low-T_e limit, and the filled core mean is constrained to decrease monotonically downstream.
- Inside the trusted core (`|x| <= X_CORE_CM`) the T_e fill fills gaps only: a cell carrying a finite measurement keeps its measured value. The end-plate sentinels are enforced near-exactly while data points carry a finite smoothing, so the fitted surface otherwise sits below the axial rows nearest an end plate — a z-dependent bias, largest at p11. Outside `X_EDGE_CM` the scrape-off-layer prior (edge anchors + loose edge smoothing) is a deliberate, separate choice and is left in place; the two are blended across the same core-to-edge ramp the per-point smoothing uses.
- The monotonic-z core-mean constraint is judged on the MEASUREMENT: it rescales a row that carries measured core cells only where the measured core means are themselves non-monotonic, and applies unconditionally only to rows with no measured core cells, which are reconstructed from their neighbours and the end boundaries. Where it acted is recorded per row-cycle as `core_mean_te_monotonic_clamped` / `core_mean_te_monotonic_scale` in `processed/te_filled.hdf5` and carried into the overlay export as `te_core_mean_clamped` (schema v15).
