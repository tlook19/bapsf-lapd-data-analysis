"""Langmuir sweep parameter extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal


@dataclass(frozen=True)
class LinearFit:
    """Robust straight-line fit."""

    intercept: float
    slope: float
    mask: np.ndarray
    success: bool
    cost: float

    def evaluate(self, voltage: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * voltage


@dataclass(frozen=True)
class LogLinearFit:
    """Fit to ln(I_e) = intercept + slope * V."""

    intercept: float
    slope: float
    electron_temperature_ev: float
    mask: np.ndarray
    success: bool
    cost: float

    def electron_current(self, voltage: np.ndarray) -> np.ndarray:
        exponent = np.clip(self.intercept + self.slope * voltage, -700, 700)
        return np.exp(exponent)


@dataclass(frozen=True)
class ExponentialFit:
    """Direct robust fit to I = ion_line + A exp(V / T_e)."""

    ion_intercept: float
    ion_slope: float
    log_amplitude: float
    inverse_temperature: float
    electron_temperature_ev: float
    mask: np.ndarray
    success: bool
    cost: float

    def evaluate(self, voltage: np.ndarray) -> np.ndarray:
        exponent = np.clip(self.log_amplitude + self.inverse_temperature * voltage, -700, 700)
        return self.ion_intercept + self.ion_slope * voltage + np.exp(exponent)

    def electron_current(self, voltage: np.ndarray) -> np.ndarray:
        exponent = np.clip(self.log_amplitude + self.inverse_temperature * voltage, -700, 700)
        return np.exp(exponent)


@dataclass(frozen=True)
class LangmuirAnalysis:
    """Summary of one I-V sweep analysis."""

    voltage: np.ndarray
    current: np.ndarray
    ion_fit: LinearFit
    saturation_fit: LinearFit
    log_linear_fit: LogLinearFit
    exponential_fit: ExponentialFit
    plasma_potential_derivative_v: float
    plasma_potential_log_intersection_v: float | None
    plasma_potential_exp_intersection_v: float | None


def _robust_scale(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=np.float64)
    median = np.median(residuals)
    mad = np.median(np.abs(residuals - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(residuals)
    return float(scale if np.isfinite(scale) and scale > 0 else 1.0)


def _robust_line_fit(
    voltage: np.ndarray,
    current: np.ndarray,
    mask: np.ndarray,
    *,
    slope_min: float | None = None,
) -> LinearFit:
    x = voltage[mask]
    y = current[mask]
    if x.size < 3:
        raise ValueError("At least three points are required for a line fit")

    slope, intercept = np.polyfit(x, y, 1)
    if slope_min is not None:
        slope = max(float(slope), slope_min)
    initial = np.array([intercept, slope], dtype=np.float64)
    scale = _robust_scale(y - (initial[0] + initial[1] * x))

    def residual(params):
        return (params[0] + params[1] * x - y) / scale

    if slope_min is None:
        bounds = (-np.inf, np.inf)
    else:
        bounds = ([-np.inf, slope_min], [np.inf, np.inf])
    result = optimize.least_squares(
        residual,
        initial,
        bounds=bounds,
        loss="soft_l1",
        f_scale=1.0,
    )
    return LinearFit(
        intercept=float(result.x[0]),
        slope=float(result.x[1]),
        mask=mask,
        success=bool(result.success),
        cost=float(result.cost),
    )


def _contiguous_true_region(mask: np.ndarray, *, center_voltage: np.ndarray | None = None, voltage: np.ndarray | None = None) -> np.ndarray:
    """Keep one contiguous true region, optionally preferring a voltage center."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return mask
    splits = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, splits)
    if center_voltage is not None and voltage is not None:
        center = float(np.asarray(center_voltage).item())

        def score(group):
            group_voltage = voltage[group]
            if group_voltage.min() <= center <= group_voltage.max():
                return (0.0, -len(group))
            return (min(abs(group_voltage.min() - center), abs(group_voltage.max() - center)), -len(group))

        best = min(groups, key=score)
    else:
        best = max(groups, key=len)
    cleaned = np.zeros_like(mask, dtype=bool)
    cleaned[best] = True
    return cleaned


def _retarding_mask(
    voltage: np.ndarray,
    electron_current: np.ndarray,
    *,
    center_voltage: float | None = None,
) -> np.ndarray:
    positive = electron_current > 0
    if positive.sum() < 8:
        raise ValueError("Not enough positive electron-current points for retarding fit")

    positive_values = electron_current[positive]
    low = max(np.nanpercentile(positive_values, 8), np.nanmax(positive_values) * 0.015)
    high = np.nanpercentile(positive_values, 50)
    center_kwargs = {}
    if center_voltage is not None:
        center_kwargs = {"center_voltage": np.array(center_voltage), "voltage": voltage}
        mask = positive & (electron_current >= low) & (electron_current <= high)
        width = max((voltage.max() - voltage.min()) * 0.25, 6.0)
        mask &= voltage <= center_voltage + width
    else:
        mask = positive & (electron_current >= low) & (electron_current <= high)

    # Keep the main rising transition, not isolated noisy islands.
    mask = _contiguous_true_region(mask, **center_kwargs)
    if mask.sum() < 8:
        # Fall back to a slightly broader window for difficult noisy sweeps.
        high = np.nanpercentile(positive_values, 68)
        broad_mask = positive & (electron_current >= low) & (electron_current <= high)
        if center_voltage is not None:
            broad_mask &= voltage <= center_voltage + max((voltage.max() - voltage.min()) * 0.35, 8.0)
        mask = _contiguous_true_region(broad_mask, **center_kwargs)
        if mask.sum() < 8 and broad_mask.sum() >= 8:
            mask = broad_mask
    if mask.sum() < 8:
        raise ValueError("Could not select enough points in the electron-retarding region")
    return mask


def _log_linear_fit(voltage: np.ndarray, electron_current: np.ndarray, mask: np.ndarray) -> LogLinearFit:
    x = voltage[mask]
    y = np.log(electron_current[mask])
    slope, intercept = np.polyfit(x, y, 1)
    initial = np.array([intercept, max(slope, 1e-3)], dtype=np.float64)
    scale = _robust_scale(y - (initial[0] + initial[1] * x))

    def residual(params):
        return (params[0] + params[1] * x - y) / scale

    result = optimize.least_squares(
        residual,
        initial,
        bounds=([-np.inf, 1e-4], [np.inf, 10.0]),
        loss="soft_l1",
        f_scale=1.0,
    )
    slope = float(result.x[1])
    return LogLinearFit(
        intercept=float(result.x[0]),
        slope=slope,
        electron_temperature_ev=1.0 / slope,
        mask=mask,
        success=bool(result.success),
        cost=float(result.cost),
    )


def _exponential_fit(
    voltage: np.ndarray,
    current: np.ndarray,
    ion_fit: LinearFit,
    log_fit: LogLinearFit,
    mask: np.ndarray,
) -> ExponentialFit:
    x = voltage[mask]
    y = current[mask] - ion_fit.evaluate(x)
    initial = np.array(
        [
            log_fit.intercept,
            log_fit.slope,
        ],
        dtype=np.float64,
    )
    scale = _robust_scale(y - log_fit.electron_current(x))

    def model(params):
        exponent = np.clip(params[0] + params[1] * x, -700, 700)
        return np.exp(exponent)

    def residual(params):
        return (model(params) - y) / scale

    result = optimize.least_squares(
        residual,
        initial,
        bounds=([-np.inf, 1e-4], [np.inf, 10.0]),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=20_000,
    )
    inverse_temperature = float(result.x[1])
    return ExponentialFit(
        ion_intercept=ion_fit.intercept,
        ion_slope=ion_fit.slope,
        log_amplitude=float(result.x[0]),
        inverse_temperature=inverse_temperature,
        electron_temperature_ev=1.0 / inverse_temperature,
        mask=mask,
        success=bool(result.success),
        cost=float(result.cost),
    )


def _intersection(
    voltage: np.ndarray,
    saturation_fit: LinearFit,
    electron_current_func,
    ion_current_func,
) -> float | None:
    def difference(v):
        return float(ion_current_func(np.array([v]))[0] + electron_current_func(np.array([v]))[0] - saturation_fit.evaluate(np.array([v]))[0])

    grid = np.linspace(float(voltage.min()), float(voltage.max()), 500)
    values = np.array([difference(v) for v in grid])
    sign_changes = np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:]))
    if sign_changes.size == 0:
        return None

    # Prefer the crossing closest to the strongest slope in the I-V curve.
    with np.errstate(invalid="ignore"):
        derivative_v = grid[np.nanargmax(np.gradient(values, grid))]
    best_index = min(sign_changes, key=lambda idx: abs(grid[idx] - derivative_v))
    try:
        return float(optimize.brentq(difference, grid[best_index], grid[best_index + 1]))
    except ValueError:
        return None


def analyze_langmuir_sweep(
    voltage,
    current,
    *,
    ion_fraction: float = 0.22,
    saturation_fraction: float = 0.18,
) -> LangmuirAnalysis:
    """Fit one filtered Langmuir I-V sweep with two robust methods."""
    voltage = np.asarray(voltage, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    finite = np.isfinite(voltage) & np.isfinite(current)
    voltage = voltage[finite]
    current = current[finite]
    if voltage.size < 20:
        raise ValueError("At least 20 finite I-V points are required")

    order = np.argsort(voltage)
    voltage = voltage[order]
    current = current[order]

    ion_cutoff = np.nanquantile(voltage, ion_fraction)
    saturation_cutoff = np.nanquantile(voltage, 1.0 - saturation_fraction)
    ion_mask = voltage <= ion_cutoff
    saturation_mask = voltage >= saturation_cutoff

    ion_fit = _robust_line_fit(voltage, current, ion_mask, slope_min=0.0)
    saturation_fit = _robust_line_fit(voltage, current, saturation_mask)
    smooth_window = max(7, (voltage.size // 35) | 1)
    if smooth_window >= voltage.size:
        smooth_window = voltage.size - 1 if voltage.size % 2 == 0 else voltage.size
    if smooth_window >= 5:
        smoothed_current = signal.savgol_filter(current, smooth_window, polyorder=2)
    else:
        smoothed_current = current
    with np.errstate(invalid="ignore"):
        derivative = np.gradient(smoothed_current, voltage)
    derivative_region = (voltage > np.nanquantile(voltage, 0.15)) & (voltage < np.nanquantile(voltage, 0.92))
    derivative_indices = np.flatnonzero(derivative_region)
    plasma_potential_derivative_v = float(voltage[derivative_indices[np.nanargmax(derivative[derivative_indices])]])
    ion_level = float(np.median(current[ion_mask]))
    saturation_level = float(np.median(current[saturation_mask]))
    midpoint_current = ion_level + 0.5 * (saturation_level - ion_level)
    transition_center_v = float(voltage[np.nanargmin(np.abs(current - midpoint_current))])
    electron_current = current - ion_fit.evaluate(voltage)
    retarding_mask = _retarding_mask(
        voltage,
        electron_current,
        center_voltage=transition_center_v,
    )

    log_fit = _log_linear_fit(voltage, electron_current, retarding_mask)
    exp_fit = _exponential_fit(voltage, current, ion_fit, log_fit, retarding_mask)

    log_intersection = _intersection(
        voltage,
        saturation_fit,
        log_fit.electron_current,
        ion_fit.evaluate,
    )
    exp_ion = LinearFit(
        intercept=exp_fit.ion_intercept,
        slope=exp_fit.ion_slope,
        mask=ion_fit.mask,
        success=exp_fit.success,
        cost=exp_fit.cost,
    )
    exp_intersection = _intersection(
        voltage,
        saturation_fit,
        exp_fit.electron_current,
        exp_ion.evaluate,
    )

    return LangmuirAnalysis(
        voltage=voltage,
        current=current,
        ion_fit=ion_fit,
        saturation_fit=saturation_fit,
        log_linear_fit=log_fit,
        exponential_fit=exp_fit,
        plasma_potential_derivative_v=plasma_potential_derivative_v,
        plasma_potential_log_intersection_v=log_intersection,
        plasma_potential_exp_intersection_v=exp_intersection,
    )
