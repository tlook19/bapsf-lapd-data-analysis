"""HDF5 reader utilities for LAPD data runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bapsf_lapd.config import ChannelKind, RunConfig


@dataclass(frozen=True)
class DischargeSummary:
    """Peak discharge values derived from the start/end discharge traces."""

    peak_current_a: float
    voltage_at_peak_v: float
    raw_voltage_at_peak_v: float
    peak_power_w: float
    trace_index: int
    sample_index: int
    time_s: float


@dataclass(frozen=True)
class TraceStats:
    """Mean trace and uncertainty from repeated shots."""

    mean: np.ndarray
    std: np.ndarray
    stderr: np.ndarray
    n: int
    time_s: np.ndarray


@dataclass(frozen=True)
class OffsetStats:
    """Voltage offset estimate for one run/channel before multiplicative calibration."""

    channel: ChannelKind
    measured_mean_v: float
    target_end_v: float
    offset_v: float
    std_v: float
    stderr_v: float
    n_shots: int
    n_samples: int
    sample_start: int
    sample_stop: int
    tail_duration_s: float

    @property
    def mean_v(self) -> float:
        """Backward-compatible name for the offset voltage to subtract."""
        return self.offset_v


def _decode(value: Any) -> Any:
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class LapdRun:
    """Lazy interface to one configured LAPD HDF5 file."""

    LANGMUIR_CHANNELS = {ChannelKind.ISAT, ChannelKind.I_SWEEP, ChannelKind.V_SWEEP}

    def __init__(self, config: RunConfig):
        if config.path is None:
            raise ValueError("RunConfig.path is required to open an HDF5 run")
        self.config = config
        self.path = Path(config.path).expanduser()

    def open(self) -> h5py.File:
        return h5py.File(self.path, "r")

    def tree(self, max_depth: int = 2) -> list[str]:
        """Return a compact HDF5 tree listing."""
        lines: list[str] = []
        with self.open() as h5:
            def visit(name: str, obj):
                if name.count("/") > max_depth:
                    return
                if isinstance(obj, h5py.Dataset):
                    lines.append(f"{name}: dataset shape={obj.shape} dtype={obj.dtype}")
                else:
                    lines.append(f"{name}: group")

            h5.visititems(visit)
        return lines

    def attrs(self, path: str) -> dict[str, Any]:
        with self.open() as h5:
            return {key: _decode(value) for key, value in h5[path].attrs.items()}

    def dataset_info(self, path: str) -> dict[str, Any]:
        with self.open() as h5:
            dataset = h5[path]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(f"{path!r} is not a dataset")
            return {"shape": dataset.shape, "dtype": str(dataset.dtype)}

    def sample_dt_s(self) -> float:
        return self.config.acquisition.sample_dt_s

    def sample_rate_hz(self) -> float:
        return self.config.acquisition.sample_rate_hz

    def time_axis(self, n_samples: int, start_index: int = 0) -> np.ndarray:
        indices = np.arange(start_index, start_index + n_samples, dtype=np.float64)
        return indices * self.sample_dt_s()

    def shot_count(self, channel: ChannelKind | str) -> int:
        channel_config = self.config.channel(channel)
        with self.open() as h5:
            return int(h5[channel_config.hdf5_path].shape[0])

    def position_count(self) -> int:
        return self.config.acquisition.n_positions

    def shots_per_position(self) -> int:
        return self.config.acquisition.n_shots_per_position

    def expected_flat_shot_count(self) -> int:
        return self.position_count() * self.shots_per_position()

    def sweep_ramp_sample_slices(self, *, clip_s: float = 0.0) -> list[slice]:
        """Return sample slices corresponding to configured Langmuir voltage ramps."""
        if self.config.sweep is None:
            raise ValueError(f"Run {self.config.run_id} has no sweep configuration")
        if clip_s < 0:
            raise ValueError("clip_s must be non-negative")

        sample_dt_s = self.sample_dt_s()
        clip_samples = int(round(clip_s / sample_dt_s))
        slices = []
        for cycle_index in range(self.config.sweep.n_cycles):
            start_s = self.config.sweep.t0_s + cycle_index * self.config.sweep.tau_cycle_s
            stop_s = start_s + self.config.sweep.tau_ramp_s
            start = int(round(start_s / sample_dt_s)) + clip_samples
            stop = int(round(stop_s / sample_dt_s)) - clip_samples
            if stop <= start:
                raise ValueError(
                    f"Invalid ramp slice for run {self.config.run_id}, cycle {cycle_index}: "
                    f"{start}:{stop}. The clip window may be too large."
                )
            slices.append(slice(start, stop))
        return slices

    def flat_shot_index(self, position_index: int, shot_index: int) -> int:
        """Return the flattened shot index for a position/shot pair."""
        if not 0 <= position_index < self.position_count():
            raise IndexError(f"position_index must be in [0, {self.position_count()})")
        if not 0 <= shot_index < self.shots_per_position():
            raise IndexError(f"shot_index must be in [0, {self.shots_per_position()})")
        return position_index * self.shots_per_position() + shot_index

    def validate_flat_shape(self, channel: ChannelKind | str) -> None:
        actual = self.shot_count(channel)
        expected = self.expected_flat_shot_count()
        if actual != expected:
            raise ValueError(
                f"{ChannelKind(channel).value!r} has {actual} flattened shots; "
                f"expected {expected} = {self.position_count()} positions * "
                f"{self.shots_per_position()} shots"
            )

    def raw_trace(self, channel: ChannelKind | str, shot: int, sample: slice | None = None) -> np.ndarray:
        """Read raw uint samples for one shot and logical channel."""
        channel_config = self.config.channel(channel)
        sample_index = sample if sample is not None else slice(None)
        with self.open() as h5:
            return h5[channel_config.hdf5_path][shot, sample_index]

    def voltage_trace(
        self,
        channel: ChannelKind | str,
        shot: int,
        sample: slice | None = None,
    ) -> np.ndarray:
        """Read one shot and convert raw SIS samples to voltage using headers."""
        channel_config = self.config.channel(channel)
        header_path = f"{channel_config.hdf5_path} headers"
        sample_index = sample if sample is not None else slice(None)
        with self.open() as h5:
            raw = h5[channel_config.hdf5_path][shot, sample_index].astype(np.float64)
            header = h5[header_path][shot]
        return raw * float(header["Scale"]) + float(header["Offset"])

    def trace(
        self,
        channel: ChannelKind | str,
        shot: int,
        sample: slice | None = None,
        *,
        normalize: bool = False,
        zero_offset_v: float | None = None,
    ) -> np.ndarray:
        """Read one calibrated logical trace.

        Photodiode channels are raw voltage in arbitrary units. Current and
        sweep-voltage channels apply the configured resistor/gain/attenuation
        or multiplier.
        """
        channel_config = self.config.channel(channel)
        voltage = self.voltage_trace(channel_config.kind, shot, sample)
        if zero_offset_v is None:
            zero_offset_v = self.default_zero_offset_v(channel_config.kind)
        if zero_offset_v:
            voltage = voltage - zero_offset_v
        values = channel_config.apply_calibration(voltage)
        if normalize:
            values = values / self.normalization_factor(shot)
        return values

    def zero_offset_target_v(self, channel: ChannelKind | str) -> float:
        """Return the desired end-of-trace voltage before multiplicative calibration."""
        channel_kind = ChannelKind(channel)
        if channel_kind in {ChannelKind.ISAT, ChannelKind.I_SWEEP}:
            return 0.0
        if channel_kind == ChannelKind.V_SWEEP:
            if self.config.sweep is None or self.config.sweep.voltage_start is None:
                raise ValueError(f"Run {self.config.run_id} has no sweep low voltage configured")
            multiplier = self.config.channel(ChannelKind.V_SWEEP).multiplier
            if multiplier == 0:
                raise ZeroDivisionError("V_sweep multiplier cannot be zero")
            return self.config.sweep.voltage_start / multiplier
        return 0.0

    def default_zero_offset_v(self, channel: ChannelKind | str) -> float:
        """Return the default voltage offset to subtract for a logical channel."""
        channel_kind = ChannelKind(channel)
        if channel_kind not in self.LANGMUIR_CHANNELS:
            return 0.0
        return self.zero_offset_stats(channel_kind).offset_v

    def zero_offset_stats(
        self,
        channel: ChannelKind | str,
        *,
        tail_duration_s: float = 200e-6,
        target_end_v: float | None = None,
    ) -> OffsetStats:
        """Estimate the pre-calibration voltage offset from the end of a trace.

        The offset is computed from the SIS-converted voltage, before resistor,
        gain, attenuation, or multiplier factors are applied. Each shot
        contributes one time-average over the final ``tail_duration_s``; the
        returned std/stderr summarize those per-shot tail averages.
        """
        channel_kind = ChannelKind(channel)
        if channel_kind not in self.LANGMUIR_CHANNELS:
            raise ValueError(
                f"Zero-offset estimation is configured for Langmuir channels, not {channel_kind.value!r}"
            )
        if target_end_v is None:
            target_end_v = self.zero_offset_target_v(channel_kind)

        channel_config = self.config.channel(channel_kind)
        with self.open() as h5:
            data = h5[channel_config.hdf5_path]
            n_samples = int(data.shape[1])
            n_tail = max(1, int(round(tail_duration_s / self.sample_dt_s())))
            n_tail = min(n_tail, n_samples)
            sample_start = n_samples - n_tail
            raw_tail = data[:, sample_start:n_samples].astype(np.float64)
            headers = h5[f"{channel_config.hdf5_path} headers"][:]

        voltage_tail = raw_tail * headers["Scale"][:, None] + headers["Offset"][:, None]
        per_shot_tail_means = voltage_tail.mean(axis=1)
        n_shots = int(per_shot_tail_means.shape[0])
        measured_mean_v = float(per_shot_tail_means.mean())
        std = float(per_shot_tail_means.std(ddof=1)) if n_shots > 1 else 0.0
        return OffsetStats(
            channel=channel_kind,
            measured_mean_v=measured_mean_v,
            target_end_v=float(target_end_v),
            offset_v=measured_mean_v - float(target_end_v),
            std_v=std,
            stderr_v=std / float(np.sqrt(n_shots)) if n_shots > 0 else 0.0,
            n_shots=n_shots,
            n_samples=n_tail,
            sample_start=sample_start,
            sample_stop=n_samples,
            tail_duration_s=n_tail * self.sample_dt_s(),
        )

    def langmuir_traces(
        self,
        channel: ChannelKind | str,
        sample: slice | None = None,
        *,
        zero_offset_v: float | None = None,
    ) -> np.ndarray:
        """Return Langmuir data reshaped as (position, shot, sample)."""
        channel_kind = ChannelKind(channel)
        if channel_kind not in self.LANGMUIR_CHANNELS:
            raise ValueError(
                f"Reshaping is only defined for Langmuir channels, not {channel_kind.value!r}"
            )
        self.validate_flat_shape(channel_kind)
        channel_config = self.config.channel(channel_kind)
        sample_index = sample if sample is not None else slice(None)

        with self.open() as h5:
            raw = h5[channel_config.hdf5_path][:, sample_index].astype(np.float64)
            headers = h5[f"{channel_config.hdf5_path} headers"][:]

        voltage = raw * headers["Scale"][:, None] + headers["Offset"][:, None]
        if zero_offset_v is None:
            zero_offset_v = self.default_zero_offset_v(channel_kind)
        if zero_offset_v:
            voltage = voltage - zero_offset_v
        values = channel_config.apply_calibration(voltage)
        return values.reshape(
            self.position_count(),
            self.shots_per_position(),
            values.shape[-1],
        )

    def langmuir_position_stats(
        self,
        channel: ChannelKind | str,
        sample: slice | None = None,
    ) -> TraceStats:
        """Average Langmuir traces over shots at each probe position."""
        values = self.langmuir_traces(channel, sample)
        n = values.shape[1]
        std = values.std(axis=1, ddof=1) if n > 1 else np.zeros_like(values[:, 0, :])
        start_index = sample.start if sample and sample.start is not None else 0
        return TraceStats(
            mean=values.mean(axis=1),
            std=std,
            stderr=std / np.sqrt(n),
            n=n,
            time_s=self.time_axis(values.shape[-1], start_index=start_index),
        )

    def trace_stats(
        self,
        channel: ChannelKind | str,
        sample: slice | None = None,
        *,
        normalize: bool = False,
        zero_offset_v: float | None = None,
    ) -> TraceStats:
        """Average all repeated shots for a channel and return error bars."""
        channel_kind = ChannelKind(channel)
        n_shots = self.shot_count(channel_kind)
        if zero_offset_v is None:
            zero_offset_v = self.default_zero_offset_v(channel_kind)
        mean = None
        m2 = None

        for shot in range(n_shots):
            values = self.trace(
                channel_kind,
                shot,
                sample,
                normalize=normalize,
                zero_offset_v=zero_offset_v,
            ).astype(np.float64)
            if mean is None:
                mean = np.zeros_like(values)
                m2 = np.zeros_like(values)
            count = shot + 1
            delta = values - mean
            mean += delta / count
            m2 += delta * (values - mean)

        if mean is None or m2 is None:
            raise ValueError(f"No shots found for {channel_kind.value!r}")

        std = np.sqrt(m2 / (n_shots - 1)) if n_shots > 1 else np.zeros_like(mean)
        start_index = sample.start if sample and sample.start is not None else 0
        return TraceStats(
            mean=mean,
            std=std,
            stderr=std / np.sqrt(n_shots),
            n=n_shots,
            time_s=self.time_axis(mean.shape[-1], start_index=start_index),
        )

    def normalization_factor(self, shot: int) -> float:
        moving = self.voltage_trace(ChannelKind.MOVING_PHOTODIODE, shot)
        peak = float(np.nanmax(np.abs(moving)))
        if peak == 0:
            raise ZeroDivisionError("Moving photodiode peak is zero; cannot normalize")
        return peak

    def discharge_summary(self) -> DischargeSummary:
        """Derive discharge peak current, matching voltage, and peak power."""
        with self.open() as h5:
            discharge = h5["MSI/Discharge"]
            current = discharge["Discharge current"][()].astype(np.float64)
            voltage = discharge["Cathode-anode voltage"][()].astype(np.float64)
            start_time = float(discharge.attrs["Start time"])
            timestep = float(discharge.attrs["Timestep"])

        flat_index = int(np.nanargmax(current))
        trace_index, sample_index = np.unravel_index(flat_index, current.shape)
        peak_current = float(current[trace_index, sample_index])
        raw_voltage_at_peak = float(voltage[trace_index, sample_index])
        voltage_at_peak = abs(raw_voltage_at_peak)
        return DischargeSummary(
            peak_current_a=peak_current,
            voltage_at_peak_v=voltage_at_peak,
            raw_voltage_at_peak_v=raw_voltage_at_peak,
            peak_power_w=abs(peak_current * raw_voltage_at_peak),
            trace_index=int(trace_index),
            sample_index=int(sample_index),
            time_s=float(start_time + sample_index * timestep),
        )
