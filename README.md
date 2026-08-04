# BaPSF LAPD Data Analysis

Early-stage Python tools for reading LAPD HDF5 data files and keeping the
run-specific analysis configuration outside the raw files.

## Environment

This project uses a local conda environment at `.venv`:

```bash
mamba create -y -p ./.venv -c conda-forge python=3.14 numpy scipy matplotlib h5py pytest
```

For Codex or shell commands, call the interpreter directly:

```bash
MPLCONFIGDIR=.matplotlib ./.venv/bin/python -m pytest -q
```

```python
from pathlib import Path

from bapsf_lapd import LapdDataset

dataset = LapdDataset.from_manifest(Path("config/may2026_run_manifest.toml"))
run = dataset.run("01")

print(run.config.experiment_set.v_bank)
print(run.config.channel("moving_photodiode").port)
print(run.discharge_summary())

# Convert one shot of the moving photodiode trace to voltage units.
moving_pd = run.trace("moving_photodiode", shot=0)
```

The package currently focuses on:

- discovering May 2026 HDF5 runs by their two-digit filename prefix;
- loading a TOML run manifest with experiment sets, run metadata, and trace
  parameters;
- storing experiment-set, probe, channel, and sweep metadata in explicit
  dataclasses;
- opening HDF5 files lazily and listing/navigating datasets;
- converting raw SIS digitizer samples with each shot's scale and offset;
- deriving peak discharge current, voltage at that current, and peak power from
  the `MSI/Discharge` traces.

Large local HDF5 files under `data/` are ignored by git.

## Analysis Notes

- After the Langmuir sweep processing is complete for all runs, make a table of
  the starting density and temperature at x = 0 for each Langmuir port.
