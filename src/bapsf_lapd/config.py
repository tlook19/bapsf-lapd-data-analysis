"""Run configuration for the May 2026 LAPD data set."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path


#: The NOMINAL axial port ladder: the position of port 2 and the port pitch as
#: the device is nominally specified.  Every stored product was built under it.
PORT_2_Z_CM = 182.5
PORT_SPACING_CM = 31.95

#: The CAD axial port ladder: the position of port 1 and the port pitch read
#: off the machine drawing.  It is the same ladder shape with a slightly wider
#: pitch, so it puts every port further from the cathode than the nominal
#: ladder does, by an offset that grows with port number.
PORT_1_Z_CM_CAD = 150.67
PORT_SPACING_CM_CAD = 32.00

#: Ladder used when a caller names none.  A stored product carries the ladder
#: it was built under baked into its own z values, so moving this default is a
#: rebuild of the products that hold z, never a read-time reinterpretation of
#: the ones already written.
PORT_MAP_DEFAULT = "nominal"
PORT_MAPS = ("nominal", "cad")


class ChannelKind(StrEnum):
    """Supported analysis channels."""

    REFERENCE_PHOTODIODE = "reference_photodiode"
    MOVING_PHOTODIODE = "moving_photodiode"
    ISAT = "isat"
    I_SWEEP = "i_sweep"
    V_SWEEP = "v_sweep"


@dataclass(frozen=True)
class ExperimentSet:
    """Parameters shared by all runs with the same first prefix digit."""

    id: int
    label: str
    v_bank: float
    v_puff: float
    description: str = ""


@dataclass(frozen=True)
class ProbeConfig:
    """Probe placement encoded by the second prefix digit and filename."""

    id: int
    port: int | None = None
    rotation_deg: float | None = None
    z_cm: float | None = None
    label: str = ""


@dataclass(frozen=True)
class SweepConfig:
    """Langmuir sweep timing for slicing I-V ramps."""

    ramp_voltage: float
    tau_ramp_s: float
    tau_cycle_s: float
    n_cycles: int
    voltage_start: float | None = None
    voltage_end: float | None = None
    t0_s: float = 0.0


@dataclass(frozen=True)
class AcquisitionConfig:
    """Digitizer acquisition timing shared by the five traces."""

    raw_sample_rate_hz: float = 100e6
    hardware_average_samples: int = 16
    n_positions: int = 51
    n_shots_per_position: int = 20

    @property
    def sample_rate_hz(self) -> float:
        return self.raw_sample_rate_hz / self.hardware_average_samples

    @property
    def sample_dt_s(self) -> float:
        return 1.0 / self.sample_rate_hz


@dataclass(frozen=True)
class ChannelConfig:
    """Mapping and calibration for one logical measurement channel."""

    kind: ChannelKind
    hdf5_path: str
    port: int | None = None
    resistor_ohm: float | None = None
    gain: float = 1.0
    attenuation: float = 1.0
    multiplier: float = 1.0
    normalize_to: ChannelKind | None = None
    notes: str = ""

    def apply_calibration(self, voltage):
        """Convert digitizer voltage into the configured physical quantity."""
        value = voltage * self.multiplier
        if self.resistor_ohm is not None:
            value = value * self.attenuation / self.resistor_ohm / self.gain
        return value


@dataclass(frozen=True)
class RunConfig:
    """All analysis metadata needed for a single HDF5 run."""

    run_id: str
    experiment_set: ExperimentSet
    probe: ProbeConfig
    path: Path | None = None
    channels: dict[ChannelKind, ChannelConfig] | None = None
    sweep: SweepConfig | None = None
    acquisition: AcquisitionConfig = AcquisitionConfig()

    def channel(self, kind: ChannelKind | str) -> ChannelConfig:
        channel_kind = ChannelKind(kind)
        if self.channels is None or channel_kind not in self.channels:
            raise KeyError(f"No channel configured for {channel_kind.value!r}")
        return self.channels[channel_kind]

    def with_updates(self, **changes) -> RunConfig:
        return replace(self, **changes)


EXPERIMENT_SETS: dict[int, ExperimentSet] = {
    1: ExperimentSet(
        id=1,
        label="nice plasma",
        v_bank=180.0,
        v_puff=76.4,
        description="Reference condition; peak power should be derived from discharge traces.",
    ),
    2: ExperimentSet(id=2, label="low bank 140 V", v_bank=140.0, v_puff=76.4),
    3: ExperimentSet(id=3, label="low bank 100 V", v_bank=100.0, v_puff=76.4),
    4: ExperimentSet(id=4, label="high puff 100 V bank", v_bank=100.0, v_puff=110.0),
}


DEFAULT_CHANNEL_PATHS: dict[ChannelKind, str] = {
    ChannelKind.REFERENCE_PHOTODIODE: (
        "Raw data + config/SIS crate/Light_Sweep [Slot 5: SIS 3302 ch 1]"
    ),
    ChannelKind.MOVING_PHOTODIODE: (
        "Raw data + config/SIS crate/Light_Sweep [Slot 5: SIS 3302 ch 2]"
    ),
    ChannelKind.ISAT: "Raw data + config/SIS crate/Light_Sweep [Slot 7: SIS 3302 ch 1]",
    ChannelKind.I_SWEEP: "Raw data + config/SIS crate/Light_Sweep [Slot 7: SIS 3302 ch 2]",
    ChannelKind.V_SWEEP: "Raw data + config/SIS crate/Light_Sweep [Slot 7: SIS 3302 ch 3]",
}


def default_attenuation(run_id: str) -> float:
    """Return the configured current-monitor attenuation factor for a run."""
    if len(run_id) != 2 or not run_id.isdigit():
        raise ValueError(f"Run ID must be a two-digit string, got {run_id!r}")

    experiment_id = 1 if run_id[0] == "0" else int(run_id[0])
    probe_id = int(run_id[1])
    return 2.0 if experiment_id in {1, 2, 3} and 2 <= probe_id <= 7 else 1.0


def default_sweep_config(run_id: str) -> SweepConfig:
    """Return the Langmuir sweep timing/voltage configuration for a run."""
    if len(run_id) != 2 or not run_id.isdigit():
        raise ValueError(f"Run ID must be a two-digit string, got {run_id!r}")

    experiment_id = 1 if run_id[0] == "0" else int(run_id[0])
    run_number = int(run_id)
    if experiment_id in {1, 2}:
        return SweepConfig(
            ramp_voltage=75.0,
            voltage_start=-75.0,
            voltage_end=75.0,
            tau_ramp_s=250e-6,
            tau_cycle_s=500e-6,
            n_cycles=40,
        )
    if experiment_id == 3 or run_id == "41":
        return SweepConfig(
            ramp_voltage=75.0,
            voltage_start=-75.0,
            voltage_end=75.0,
            tau_ramp_s=500e-6,
            tau_cycle_s=1e-3,
            n_cycles=20,
        )
    if 42 <= run_number <= 48:
        return SweepConfig(
            ramp_voltage=20.0,
            voltage_start=-20.0,
            voltage_end=20.0,
            tau_ramp_s=500e-6,
            tau_cycle_s=1e-3,
            n_cycles=20,
        )
    raise ValueError(f"No sweep configuration available for run {run_id!r}")


def default_channels(
    run_id: str,
    *,
    port: int | None = None,
    attenuation: float | None = None,
) -> dict[ChannelKind, ChannelConfig]:
    """Create default channel mappings and calibrations for a run.

    Current channels use the run-dependent attenuation factor unless a caller
    provides an explicit override.
    """
    run_number = int(run_id)
    if attenuation is None:
        attenuation = default_attenuation(run_id)
    i_sweep_resistor = 3.0 if 42 <= run_number <= 48 else 1.0
    return {
        ChannelKind.REFERENCE_PHOTODIODE: ChannelConfig(
            ChannelKind.REFERENCE_PHOTODIODE,
            DEFAULT_CHANNEL_PATHS[ChannelKind.REFERENCE_PHOTODIODE],
            port=None,
            normalize_to=ChannelKind.MOVING_PHOTODIODE,
            notes="Raw voltage; arbitrary units.",
        ),
        ChannelKind.MOVING_PHOTODIODE: ChannelConfig(
            ChannelKind.MOVING_PHOTODIODE,
            DEFAULT_CHANNEL_PATHS[ChannelKind.MOVING_PHOTODIODE],
            port=port,
            normalize_to=ChannelKind.MOVING_PHOTODIODE,
            notes="Raw voltage; arbitrary units.",
        ),
        ChannelKind.ISAT: ChannelConfig(
            ChannelKind.ISAT,
            DEFAULT_CHANNEL_PATHS[ChannelKind.ISAT],
            port=port,
            resistor_ohm=45.0,
            gain=0.844,
            attenuation=attenuation,
        ),
        ChannelKind.I_SWEEP: ChannelConfig(
            ChannelKind.I_SWEEP,
            DEFAULT_CHANNEL_PATHS[ChannelKind.I_SWEEP],
            port=port,
            resistor_ohm=i_sweep_resistor,
            gain=1.195,
            attenuation=attenuation,
        ),
        ChannelKind.V_SWEEP: ChannelConfig(
            ChannelKind.V_SWEEP,
            DEFAULT_CHANNEL_PATHS[ChannelKind.V_SWEEP],
            port=port,
            multiplier=100.0,
        ),
    }


def z_from_port(port: int, port_map: str = PORT_MAP_DEFAULT) -> float:
    """Return the distance from cathode of a LAPD port number, in cm.

    ``port_map`` selects the ladder: ``"nominal"`` anchors on ``PORT_2_Z_CM``
    with the ``PORT_SPACING_CM`` pitch, ``"cad"`` on ``PORT_1_Z_CM_CAD`` with
    the ``PORT_SPACING_CM_CAD`` pitch.  Any other name raises ``ValueError``
    rather than falling back to a default, so a misspelled selector cannot
    quietly return the ladder the caller did not ask for.
    """
    if port_map == "nominal":
        return PORT_2_Z_CM + (port - 2) * PORT_SPACING_CM
    if port_map == "cad":
        return PORT_1_Z_CM_CAD + (port - 1) * PORT_SPACING_CM_CAD
    raise ValueError(
        f"unknown port map {port_map!r}; expected one of {PORT_MAPS}"
    )


def default_run_config(
    run_id: str,
    *,
    path: Path | None = None,
    port: int | None = None,
    rotation_deg: float | None = None,
    sweep: SweepConfig | None = None,
    attenuation: float | None = None,
) -> RunConfig:
    """Build a default config from a two-digit run prefix."""
    if len(run_id) != 2 or not run_id.isdigit():
        raise ValueError(f"Run ID must be a two-digit string, got {run_id!r}")

    experiment_id = int(run_id[0])
    if experiment_id == 0:
        experiment_id = 1
    probe_id = int(run_id[1])
    try:
        experiment_set = EXPERIMENT_SETS[experiment_id]
    except KeyError as exc:
        raise ValueError(f"No experiment set configured for prefix {experiment_id}") from exc

    return RunConfig(
        run_id=run_id,
        experiment_set=experiment_set,
        probe=ProbeConfig(
            id=probe_id,
            port=port,
            rotation_deg=rotation_deg,
            z_cm=z_from_port(port) if port is not None else None,
            label=f"probe configuration {probe_id}",
        ),
        path=path,
        channels=default_channels(run_id, port=port, attenuation=attenuation),
        sweep=sweep if sweep is not None else default_sweep_config(run_id),
    )
