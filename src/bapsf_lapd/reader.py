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
    """Peak discharge values derived from the start/end discharge traces.

    ``peak_current_a`` and ``peak_power_w`` are computed after the additive
    discharge-channel zero offset in ``zero_offset_a`` has been subtracted.
    """

    peak_current_a: float
    voltage_at_peak_v: float
    raw_voltage_at_peak_v: float
    peak_power_w: float
    trace_index: int
    sample_index: int
    time_s: float
    zero_offset_a: float


@dataclass(frozen=True)
class DischargeOffsetStats:
    """Additive zero offset of one run's ``MSI/Discharge`` current channel.

    All currents are in amperes and all times in seconds on the stored
    ``MSI/Discharge`` time base.  ``offset_a`` is the quantity to subtract from
    the discharge current; the two cross-check means are reported alongside it
    but do not enter the correction.

    Attributes
    ----------
    offset_a:
        Primary estimate: the mean current over the post-connect,
        pre-avalanche window of every stored trace, where the bank is at full
        voltage and no plasma has formed, so the true current is zero.
    std_a, stderr_a:
        Sample standard deviation and standard error of the per-trace window
        means.
    n_traces, n_samples:
        Number of stored traces and the number of window samples per trace.
    connect_time_s, avalanche_time_s:
        Trace-averaged bank-connect and avalanche-onset times bounding the
        window.
    window_start_s, window_stop_s:
        Trace-averaged window edges.
    pre_connect_mean_a:
        Cross-check over the pre-connect record, where the bank is open.
    far_tail_mean_a:
        Cross-check over the far trace tail, after the switch has opened.
    """

    offset_a: float
    std_a: float
    stderr_a: float
    n_traces: int
    n_samples: int
    connect_time_s: float
    avalanche_time_s: float
    window_start_s: float
    window_stop_s: float
    pre_connect_mean_a: float
    far_tail_mean_a: float


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


# Discharge-current zero-offset window definition.  The primary window opens a
# guard interval after the bank-connect step, which clears the single-sample
# switching transient, and closes at a fixed fraction of the connect-to-
# avalanche interval, which keeps it clear of the rising avalanche foot.
DISCHARGE_CONNECT_GUARD_S = 0.4e-3
DISCHARGE_PREAVALANCHE_FRACTION = 0.3
DISCHARGE_AVALANCHE_THRESHOLD_A = 150.0
DISCHARGE_AVALANCHE_SUSTAIN_SAMPLES = 3
DISCHARGE_MIN_OFFSET_SAMPLES = 5
# Cross-check windows: the record before the bank closes, and the far tail
# after the switch opens again.
DISCHARGE_PRE_CONNECT_GUARD_S = 1.0e-3
DISCHARGE_FAR_TAIL_START_S = 60.0e-3
DISCHARGE_FAR_TAIL_STOP_S = 140.0e-3


def _decode(value: Any) -> Any:
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def discharge_connect_time_s(voltage: np.ndarray, time_s: np.ndarray) -> float:
    """Return the bank-connect time of one cathode-anode voltage trace.

    The switch close is a single-sample step from zero to the full bank
    voltage, so the connect time is taken as the linearly interpolated
    crossing of half the step amplitude.  Raw cathode-anode voltage is
    negative, so the step is located on the magnitude.  Raises ``ValueError``
    if the trace carries no step.
    """
    magnitude = np.abs(np.asarray(voltage, dtype=np.float64))
    peak = float(magnitude.max())
    if peak <= 0.0:
        raise ValueError("Cathode-anode voltage trace carries no bank-connect step")
    amplitude = float(np.median(magnitude[magnitude > 0.5 * peak]))
    half = 0.5 * amplitude
    above = np.flatnonzero(magnitude >= half)
    if above.size == 0 or above[0] == 0:
        raise ValueError("Cathode-anode voltage trace carries no bank-connect step")
    index = int(above[0])
    low = magnitude[index - 1]
    high = magnitude[index]
    span = float(high - low)
    if span <= 0.0:
        return float(time_s[index])
    return float(time_s[index - 1] + (half - low) * (time_s[index] - time_s[index - 1]) / span)


def discharge_avalanche_time_s(
    current: np.ndarray,
    time_s: np.ndarray,
    *,
    threshold_a: float = DISCHARGE_AVALANCHE_THRESHOLD_A,
    sustain_samples: int = DISCHARGE_AVALANCHE_SUSTAIN_SAMPLES,
) -> float:
    """Return the avalanche-onset time of one discharge-current trace.

    Onset is the first sample of the first run of ``sustain_samples``
    consecutive samples at or above ``threshold_a``.  Requiring a sustained
    crossing rejects the single-sample inrush spike that accompanies the
    switch close.  Raises ``ValueError`` if the trace never crosses.
    """
    if sustain_samples < 1:
        raise ValueError("sustain_samples must be at least 1")
    above = (np.asarray(current, dtype=np.float64) >= threshold_a).astype(np.int64)
    if above.size < sustain_samples:
        raise ValueError("Discharge current trace is shorter than the sustain window")
    runs = np.convolve(above, np.ones(sustain_samples, dtype=np.int64), mode="valid")
    sustained = np.flatnonzero(runs == sustain_samples)
    if sustained.size == 0:
        raise ValueError(
            f"Discharge current never sustains {threshold_a} A for "
            f"{sustain_samples} samples"
        )
    return float(time_s[int(sustained[0])])


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

    def discharge_traces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the stored discharge current, voltage, and time base.

        Current is in amperes, raw cathode-anode voltage in volts (negative),
        and time in seconds.  Both trace arrays are ``(n_traces, n_samples)``;
        no zero-offset correction is applied.
        """
        with self.open() as h5:
            discharge = h5["MSI/Discharge"]
            current = discharge["Discharge current"][()].astype(np.float64)
            voltage = discharge["Cathode-anode voltage"][()].astype(np.float64)
            start_time = float(discharge.attrs["Start time"])
            timestep = float(discharge.attrs["Timestep"])
        time_s = start_time + np.arange(current.shape[1], dtype=np.float64) * timestep
        return current, voltage, time_s

    def discharge_zero_offset_stats(self) -> DischargeOffsetStats:
        """Estimate the additive zero offset of the discharge-current channel.

        The estimate is the mean current over the post-connect, pre-avalanche
        window of every stored trace: the bank is closed and at full voltage
        but no plasma has formed, so the true current is zero and any reading
        is channel offset.  The window opens
        ``DISCHARGE_CONNECT_GUARD_S`` after the interpolated bank-connect step
        and closes at ``DISCHARGE_PREAVALANCHE_FRACTION`` of the interval from
        connect to avalanche onset.  Offsets are in amperes; the conversion
        from the shunt is already applied in the stored channel.

        The pre-connect record and the far trace tail are averaged as
        cross-checks and reported on the result, but only the post-connect
        window sets ``offset_a``.

        Raises ``ValueError`` if a trace carries no bank-connect step, never
        reaches avalanche onset, or yields fewer than
        ``DISCHARGE_MIN_OFFSET_SAMPLES`` window samples.
        """
        current, voltage, time_s = self.discharge_traces()

        window_means = []
        window_counts = []
        connect_times = []
        avalanche_times = []
        window_starts = []
        window_stops = []
        pre_connect_means = []
        for trace_index in range(current.shape[0]):
            connect_s = discharge_connect_time_s(voltage[trace_index], time_s)
            avalanche_s = discharge_avalanche_time_s(current[trace_index], time_s)
            if avalanche_s <= connect_s:
                raise ValueError(
                    f"Run {self.config.run_id} trace {trace_index}: avalanche onset "
                    f"{avalanche_s} s does not follow bank connect {connect_s} s"
                )
            start_s = connect_s + DISCHARGE_CONNECT_GUARD_S
            stop_s = connect_s + DISCHARGE_PREAVALANCHE_FRACTION * (avalanche_s - connect_s)
            window = (time_s >= start_s) & (time_s < stop_s)
            n_window = int(window.sum())
            if n_window < DISCHARGE_MIN_OFFSET_SAMPLES:
                raise ValueError(
                    f"Run {self.config.run_id} trace {trace_index}: post-connect "
                    f"pre-avalanche window holds {n_window} samples, fewer than the "
                    f"required {DISCHARGE_MIN_OFFSET_SAMPLES}"
                )
            pre_connect = time_s <= connect_s - DISCHARGE_PRE_CONNECT_GUARD_S
            window_means.append(float(current[trace_index][window].mean()))
            window_counts.append(n_window)
            connect_times.append(connect_s)
            avalanche_times.append(avalanche_s)
            window_starts.append(start_s)
            window_stops.append(stop_s)
            pre_connect_means.append(float(current[trace_index][pre_connect].mean()))

        far_tail = (time_s >= DISCHARGE_FAR_TAIL_START_S) & (
            time_s <= DISCHARGE_FAR_TAIL_STOP_S
        )
        per_trace = np.asarray(window_means, dtype=np.float64)
        n_traces = int(per_trace.size)
        std = float(per_trace.std(ddof=1)) if n_traces > 1 else 0.0
        return DischargeOffsetStats(
            offset_a=float(per_trace.mean()),
            std_a=std,
            stderr_a=std / float(np.sqrt(n_traces)) if n_traces > 0 else 0.0,
            n_traces=n_traces,
            n_samples=int(np.min(window_counts)),
            connect_time_s=float(np.mean(connect_times)),
            avalanche_time_s=float(np.mean(avalanche_times)),
            window_start_s=float(np.mean(window_starts)),
            window_stop_s=float(np.mean(window_stops)),
            pre_connect_mean_a=float(np.mean(pre_connect_means)),
            far_tail_mean_a=float(current[:, far_tail].mean()),
        )

    def discharge_summary(self) -> DischargeSummary:
        """Derive discharge peak current, matching voltage, and peak power.

        The discharge-current zero offset from
        ``discharge_zero_offset_stats`` is subtracted before the peak is
        located.
        """
        current, voltage, time_s = self.discharge_traces()
        start_time = float(time_s[0])
        timestep = float(time_s[1] - time_s[0])
        zero_offset_a = self.discharge_zero_offset_stats().offset_a
        current = current - zero_offset_a

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
            zero_offset_a=zero_offset_a,
        )
