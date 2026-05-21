"""Quality-control checks for Langmuir sweep analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bapsf_lapd.langmuir import LangmuirAnalysis


@dataclass(frozen=True)
class QualityFlag:
    """One quality-control flag for a sweep."""

    code: str
    severity: str
    message: str
    value: float | None = None
    threshold: float | None = None


@dataclass(frozen=True)
class LangmuirQualityReport:
    """Quality-control report for one fitted sweep."""

    flags: tuple[QualityFlag, ...]
    rms_residual_a: float
    max_abs_residual_a: float
    fit_window_width_v: float
    fit_sample_count: int
    max_jump_a: float
    arc_like_sample_count: int

    @property
    def severity(self) -> str:
        if any(flag.severity == "bad" for flag in self.flags):
            return "bad"
        if any(flag.severity == "warn" for flag in self.flags):
            return "warn"
        return "ok"


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanstd(values)
    return float(scale if np.isfinite(scale) and scale > 0 else 1.0)


def detect_arc_like_segments(
    current,
    *,
    peer_current=None,
    jump_sigma_threshold: float = 20.0,
    peer_sigma_threshold: float = 25.0,
) -> tuple[int, float]:
    """Return count of arc-like samples and max adjacent jump.

    The first check flags unusually large adjacent jumps within the trace. If
    peer traces are supplied, the second check flags samples that are extreme
    relative to the peer-shot median at the same time index.
    """
    current = np.asarray(current, dtype=np.float64)
    if current.size < 3:
        return 0, 0.0

    jumps = np.diff(current)
    centered_jumps = np.abs(jumps - np.nanmedian(jumps))
    jump_scale = _robust_scale(jumps)
    nonzero_jumps = centered_jumps[centered_jumps > 0]
    sparse_jumps = 0 < nonzero_jumps.size <= max(3, int(0.03 * jumps.size))
    if jump_scale <= np.finfo(float).eps or sparse_jumps:
        if nonzero_jumps.size:
            jump_threshold = max(np.nanmedian(nonzero_jumps) * 0.5, np.finfo(float).eps)
        else:
            jump_threshold = np.inf
    else:
        jump_threshold = jump_sigma_threshold * jump_scale
    jump_flags = centered_jumps >= jump_threshold
    sample_flags = np.zeros(current.shape, dtype=bool)
    sample_flags[:-1] |= jump_flags
    sample_flags[1:] |= jump_flags

    if peer_current is not None:
        peer = np.asarray(peer_current, dtype=np.float64)
        if peer.ndim == 2 and peer.shape[-1] == current.shape[-1] and peer.shape[0] >= 3:
            peer_median = np.nanmedian(peer, axis=0)
            peer_scale = 1.4826 * np.nanmedian(np.abs(peer - peer_median), axis=0)
            fallback = np.nanmedian(peer_scale[peer_scale > 0]) if np.any(peer_scale > 0) else np.nan
            if not np.isfinite(fallback) or fallback <= 0:
                fallback = _robust_scale(peer.ravel())
            peer_scale = np.where(peer_scale > 0, peer_scale, fallback)
            absolute_floor = max(5.0e-3, 5.0 * fallback)
            peer_threshold = np.maximum(peer_sigma_threshold * peer_scale, absolute_floor)
            peer_flags = np.abs(current - peer_median) > peer_threshold
            sample_flags |= peer_flags

    return int(sample_flags.sum()), float(np.nanmax(np.abs(jumps)))


def evaluate_langmuir_quality(
    analysis: LangmuirAnalysis,
    *,
    current=None,
    peer_current=None,
    te_min_ev: float = 0.05,
    te_max_ev: float = 100.0,
    max_te_disagreement_fraction: float = 0.75,
    min_fit_points: int = 8,
    min_fit_window_width_v: float = 0.25,
    max_rms_residual_a: float = 2.0e-3,
) -> LangmuirQualityReport:
    """Evaluate fit and data-quality flags for one Langmuir sweep."""
    flags: list[QualityFlag] = []
    v = analysis.voltage
    i = analysis.current
    fit_mask = analysis.log_linear_fit.mask
    v_fit = v[fit_mask]
    i_fit = i[fit_mask]
    log_model = analysis.ion_fit.evaluate(v_fit) + analysis.log_linear_fit.electron_current(v_fit)
    exp_model = analysis.exponential_fit.evaluate(v_fit)
    residual = i_fit - exp_model
    rms_residual = float(np.sqrt(np.nanmean(residual**2))) if residual.size else np.nan
    max_abs_residual = float(np.nanmax(np.abs(residual))) if residual.size else np.nan
    fit_width = float(v_fit.max() - v_fit.min()) if v_fit.size else 0.0
    fit_count = int(fit_mask.sum())

    for label, fit in [
        ("log_linear", analysis.log_linear_fit),
        ("exponential", analysis.exponential_fit),
    ]:
        if not fit.success:
            flags.append(QualityFlag(f"{label}_fit_failed", "bad", f"{label} fit did not converge"))
        if not te_min_ev <= fit.electron_temperature_ev <= te_max_ev:
            flags.append(
                QualityFlag(
                    f"{label}_te_out_of_range",
                    "bad",
                    f"{label} electron temperature is outside expected bounds",
                    fit.electron_temperature_ev,
                )
            )

    te_mean = 0.5 * (
        analysis.log_linear_fit.electron_temperature_ev + analysis.exponential_fit.electron_temperature_ev
    )
    if te_mean > 0:
        disagreement = abs(
            analysis.log_linear_fit.electron_temperature_ev
            - analysis.exponential_fit.electron_temperature_ev
        ) / te_mean
        if disagreement > max_te_disagreement_fraction:
            flags.append(
                QualityFlag(
                    "te_method_disagreement",
                    "warn",
                    "log-linear and exponential temperatures disagree",
                    disagreement,
                    max_te_disagreement_fraction,
                )
            )

    if fit_count < min_fit_points:
        flags.append(
            QualityFlag(
                "too_few_fit_points",
                "bad",
                "retarding-region fit used too few samples",
                float(fit_count),
                float(min_fit_points),
            )
        )
    if fit_width < min_fit_window_width_v:
        flags.append(
            QualityFlag(
                "narrow_fit_window",
                "warn",
                "retarding-region fit window is narrow",
                fit_width,
                min_fit_window_width_v,
            )
        )
    if rms_residual > max_rms_residual_a:
        flags.append(
            QualityFlag(
                "large_fit_residual",
                "warn",
                "exponential fit residual is large",
                rms_residual,
                max_rms_residual_a,
            )
        )

    arc_count, max_jump = detect_arc_like_segments(
        i if current is None else current,
        peer_current=peer_current,
    )
    if arc_count > 0:
        flags.append(
            QualityFlag(
                "arc_like_segment",
                "warn",
                "trace has jump/outlier samples that may indicate arcing",
                float(arc_count),
            )
        )

    return LangmuirQualityReport(
        flags=tuple(flags),
        rms_residual_a=rms_residual,
        max_abs_residual_a=max_abs_residual,
        fit_window_width_v=fit_width,
        fit_sample_count=fit_count,
        max_jump_a=max_jump,
        arc_like_sample_count=arc_count,
    )
