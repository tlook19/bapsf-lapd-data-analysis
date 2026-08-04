"""LAPD HDF5 data access and run configuration helpers."""

from bapsf_lapd.config import (
    AcquisitionConfig,
    ChannelConfig,
    ChannelKind,
    ExperimentSet,
    ProbeConfig,
    RunConfig,
    SweepConfig,
    default_attenuation,
    default_run_config,
    default_sweep_config,
)
from bapsf_lapd.corrections import (
    ELECTRICAL_SWAP_RUN_IDS,
    EFFECTIVE_ROTATION_OVERRIDES,
    density_area_key_for_deadtime_source,
    electrical_connections_swapped,
    effective_deadtime_source,
    effective_rotation_deg,
)
from bapsf_lapd.dataset import LapdDataset
from bapsf_lapd.filtering import butterworth_lowpass
from bapsf_lapd.langmuir import LangmuirAnalysis, analyze_langmuir_sweep, select_best_te_ev
from bapsf_lapd.manifest import apply_trace_updates, load_run_manifest
from bapsf_lapd.quality import (
    LangmuirQualityReport,
    QualityFlag,
    detect_arc_like_segments,
    evaluate_langmuir_quality,
)
from bapsf_lapd.density import (
    MACH_K,
    MACH_SHADOW_OFFSET,
    ProbeAAreaCalibration,
    apply_probe_a_area_factor,
    calibrate_probe_area_m2,
    density_fwhm_cm,
    electron_density_m3,
    inter_sweep_sample_slices,
    ion_sound_speed_m_s,
    load_probe_a_area_calibration,
)
from bapsf_lapd.reader import DischargeSummary, LapdRun, OffsetStats, TraceStats

__all__ = [
    "AcquisitionConfig",
    "ChannelConfig",
    "ChannelKind",
    "DischargeSummary",
    "ExperimentSet",
    "EFFECTIVE_ROTATION_OVERRIDES",
    "ELECTRICAL_SWAP_RUN_IDS",
    "LapdDataset",
    "LapdRun",
    "LangmuirAnalysis",
    "LangmuirQualityReport",
    "OffsetStats",
    "ProbeConfig",
    "QualityFlag",
    "RunConfig",
    "SweepConfig",
    "TraceStats",
    "apply_trace_updates",
    "butterworth_lowpass",
    "analyze_langmuir_sweep",
    "select_best_te_ev",
    "MACH_K",
    "MACH_SHADOW_OFFSET",
    "ProbeAAreaCalibration",
    "apply_probe_a_area_factor",
    "calibrate_probe_area_m2",
    "density_fwhm_cm",
    "detect_arc_like_segments",
    "electron_density_m3",
    "evaluate_langmuir_quality",
    "density_area_key_for_deadtime_source",
    "electrical_connections_swapped",
    "effective_deadtime_source",
    "effective_rotation_deg",
    "default_attenuation",
    "default_run_config",
    "default_sweep_config",
    "inter_sweep_sample_slices",
    "ion_sound_speed_m_s",
    "load_probe_a_area_calibration",
    "load_run_manifest",
]
