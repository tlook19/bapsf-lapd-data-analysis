"""Load run configuration from a TOML manifest."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tomllib
from typing import Any

from bapsf_lapd.config import (
    AcquisitionConfig,
    ChannelConfig,
    ChannelKind,
    ExperimentSet,
    ProbeConfig,
    RunConfig,
    SweepConfig,
    z_from_port,
)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _channel_from_dict(kind: ChannelKind, data: dict[str, Any]) -> ChannelConfig:
    normalize_to = data.get("normalize_to")
    return ChannelConfig(
        kind=kind,
        hdf5_path=data["hdf5_path"],
        port=_optional_int(data.get("port")),
        resistor_ohm=_optional_float(data.get("resistor_ohm")),
        gain=float(data.get("gain", 1.0)),
        attenuation=float(data.get("attenuation", 1.0)),
        multiplier=float(data.get("multiplier", 1.0)),
        normalize_to=ChannelKind(normalize_to) if normalize_to else None,
        notes=str(data.get("notes", "")),
    )


def _sweep_from_dict(data: dict[str, Any] | None) -> SweepConfig | None:
    if not data:
        return None
    return SweepConfig(
        ramp_voltage=float(data["ramp_voltage"]),
        tau_ramp_s=float(data["tau_ramp_s"]),
        tau_cycle_s=float(data["tau_cycle_s"]),
        n_cycles=int(data["n_cycles"]),
        voltage_start=_optional_float(data.get("voltage_start")),
        voltage_end=_optional_float(data.get("voltage_end")),
        t0_s=float(data.get("t0_s", 0.0)),
    )


def _acquisition_from_dict(data: dict[str, Any] | None) -> AcquisitionConfig:
    if not data:
        return AcquisitionConfig()
    return AcquisitionConfig(
        raw_sample_rate_hz=float(data.get("raw_sample_rate_hz", 100e6)),
        hardware_average_samples=int(data.get("hardware_average_samples", 16)),
        n_positions=int(data.get("n_positions", 51)),
        n_shots_per_position=int(data.get("n_shots_per_position", 20)),
    )


def load_run_manifest(
    manifest_path: str | Path,
    *,
    data_dir: str | Path | None = None,
) -> dict[str, RunConfig]:
    """Load run configs from a TOML manifest.

    Manifest trace defaults are merged with per-run trace tables. This keeps
    common digitizer/calibration settings in one place while still making trace
    ports and run-specific overrides explicit.
    """
    manifest_path = Path(manifest_path).expanduser()
    with manifest_path.open("rb") as manifest_file:
        manifest = tomllib.load(manifest_file)

    manifest_data_dir = manifest.get("manifest", {}).get("data_directory", ".")
    base_dir = (
        Path(data_dir).expanduser()
        if data_dir
        else manifest_path.parent.parent / manifest_data_dir
    )
    experiment_sets = {
        int(set_id): ExperimentSet(
            id=int(set_id),
            label=data["label"],
            v_bank=float(data["v_bank"]),
            v_puff=float(data["v_puff"]),
            description=str(data.get("description", "")),
        )
        for set_id, data in manifest["experiment_sets"].items()
    }
    trace_defaults = manifest["trace_defaults"]
    acquisition = _acquisition_from_dict(manifest.get("acquisition"))

    configs: dict[str, RunConfig] = {}
    for run_id, run_data in manifest["runs"].items():
        experiment_set_id = int(run_data["experiment_set"])
        langmuir_port = int(run_data["langmuir_port"])
        filename = run_data.get("file")
        run_path = base_dir / filename if filename else None

        channels: dict[ChannelKind, ChannelConfig] = {}
        run_traces = run_data.get("traces", {})
        for kind_name, default_trace in trace_defaults.items():
            kind = ChannelKind(kind_name)
            merged = dict(default_trace)
            merged.update(run_traces.get(kind_name, {}))
            if "port" not in merged and kind in {
                ChannelKind.ISAT,
                ChannelKind.I_SWEEP,
                ChannelKind.V_SWEEP,
            }:
                merged["port"] = langmuir_port
            channels[kind] = _channel_from_dict(kind, merged)

        configs[run_id] = RunConfig(
            run_id=run_id,
            experiment_set=experiment_sets[experiment_set_id],
            probe=ProbeConfig(
                id=int(run_id[1]),
                port=langmuir_port,
                rotation_deg=float(run_data.get("rotation_deg", 0.0)),
                z_cm=z_from_port(langmuir_port),
                label=str(run_data.get("probe_label", f"probe configuration {run_id[1]}")),
            ),
            path=run_path,
            channels=channels,
            sweep=_sweep_from_dict(run_data.get("sweep")),
            acquisition=acquisition,
        )
    return dict(sorted(configs.items()))


def apply_trace_updates(config: RunConfig, **trace_updates: dict[str, Any]) -> RunConfig:
    """Return a copy of a run config with selected trace fields changed."""
    if config.channels is None:
        raise ValueError("RunConfig has no channels to update")
    channels = dict(config.channels)
    for kind_name, updates in trace_updates.items():
        kind = ChannelKind(kind_name)
        channels[kind] = replace(channels[kind], **updates)
    return replace(config, channels=channels)
