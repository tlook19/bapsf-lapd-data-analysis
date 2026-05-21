"""Catalog HDF5 files as configured LAPD runs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from bapsf_lapd.config import ChannelKind, RunConfig, SweepConfig, default_run_config
from bapsf_lapd.manifest import load_run_manifest
from bapsf_lapd.reader import LapdRun, TraceStats

RUN_FILE_RE = re.compile(
    r"^(?P<run_id>\d{2})_.*?_p(?P<port>\d+)_.*?_rot(?P<rotation>\d+(?:\.\d+)?)_.*\.hdf5$"
)


class LapdDataset:
    """A collection of LAPD HDF5 files indexed by run ID."""

    def __init__(self, runs: dict[str, RunConfig]):
        self.runs = dict(sorted(runs.items()))

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        sweeps: dict[str, SweepConfig] | None = None,
        attenuation: dict[str, float] | float | None = None,
    ) -> LapdDataset:
        directory = Path(directory).expanduser()
        runs: dict[str, RunConfig] = {}
        for path in sorted(directory.glob("*.hdf5")):
            match = RUN_FILE_RE.match(path.name)
            if match is None:
                continue
            run_id = match.group("run_id")
            run_attenuation = attenuation[run_id] if isinstance(attenuation, dict) else attenuation
            runs[run_id] = default_run_config(
                run_id,
                path=path,
                port=int(match.group("port")),
                rotation_deg=float(match.group("rotation")),
                sweep=sweeps.get(run_id) if sweeps else None,
                attenuation=run_attenuation,
            )
        return cls(runs)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        *,
        data_dir: str | Path | None = None,
    ) -> LapdDataset:
        return cls(load_run_manifest(manifest_path, data_dir=data_dir))

    def __len__(self) -> int:
        return len(self.runs)

    def __iter__(self):
        for run_id in self.runs:
            yield self.run(run_id)

    def run_ids(self) -> list[str]:
        return list(self.runs)

    def config(self, run_id: str) -> RunConfig:
        return self.runs[run_id]

    def run(self, run_id: str) -> LapdRun:
        return LapdRun(self.config(run_id))

    def experiment_set_run_ids(self, experiment_set_id: int) -> list[str]:
        return [
            run_id
            for run_id, config in self.runs.items()
            if config.experiment_set.id == experiment_set_id
        ]

    def experiment_set_ids(self) -> list[int]:
        return sorted({config.experiment_set.id for config in self.runs.values()})

    def trace_stats_across_runs(
        self,
        run_ids: list[str],
        channel: ChannelKind | str,
        sample: slice | None = None,
        *,
        normalize: bool = False,
    ) -> TraceStats:
        """Average a channel across all shots in several runs."""
        channel_kind = ChannelKind(channel)
        mean = None
        m2 = None
        count = 0
        time_s = None

        for run_id in run_ids:
            run = self.run(run_id)
            zero_offset_v = run.default_zero_offset_v(channel_kind)
            if time_s is None:
                n_samples = run.trace(channel_kind, 0, sample, zero_offset_v=zero_offset_v).shape[-1]
                start_index = sample.start if sample and sample.start is not None else 0
                time_s = run.time_axis(n_samples, start_index=start_index)
            for shot in range(run.shot_count(channel_kind)):
                values = run.trace(
                    channel_kind,
                    shot,
                    sample,
                    normalize=normalize,
                    zero_offset_v=zero_offset_v,
                ).astype(np.float64)
                if mean is None:
                    mean = np.zeros_like(values)
                    m2 = np.zeros_like(values)
                count += 1
                delta = values - mean
                mean += delta / count
                m2 += delta * (values - mean)

        if mean is None or m2 is None or time_s is None:
            raise ValueError("No traces found for the requested runs")

        std = np.sqrt(m2 / (count - 1)) if count > 1 else np.zeros_like(mean)
        return TraceStats(mean=mean, std=std, stderr=std / np.sqrt(count), n=count, time_s=time_s)

    def reference_stats_for_experiment_set(
        self,
        experiment_set_id: int,
        sample: slice | None = None,
        *,
        normalize: bool = False,
    ) -> TraceStats:
        """Average reference photodiode traces across an experiment set."""
        return self.trace_stats_across_runs(
            self.experiment_set_run_ids(experiment_set_id),
            ChannelKind.REFERENCE_PHOTODIODE,
            sample,
            normalize=normalize,
        )
