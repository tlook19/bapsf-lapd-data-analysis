"""Signal filtering helpers for LAPD traces."""

from __future__ import annotations

import numpy as np
from scipy import signal


def butterworth_lowpass(
    data,
    *,
    sample_rate_hz: float,
    cutoff_hz: float,
    order: int = 4,
    axis: int = -1,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter along one axis.

    Parameters
    ----------
    data:
        Array-like signal. Multidimensional arrays are supported.
    sample_rate_hz:
        Sampling rate of the data along ``axis``.
    cutoff_hz:
        Low-pass cutoff frequency. Must be below Nyquist.
    order:
        Butterworth filter order.
    axis:
        Axis containing the time series.
    zero_phase:
        Use forward-backward filtering when true. This is usually preferable
        for analysis because it avoids phase shifts in I-V features.
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive")
    nyquist_hz = sample_rate_hz / 2.0
    if cutoff_hz >= nyquist_hz:
        raise ValueError(f"cutoff_hz must be below Nyquist ({nyquist_hz:g} Hz)")
    if order < 1:
        raise ValueError("order must be at least 1")

    values = np.asarray(data, dtype=np.float64)
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=sample_rate_hz, output="sos")
    if zero_phase:
        return signal.sosfiltfilt(sos, values, axis=axis)
    return signal.sosfilt(sos, values, axis=axis)
