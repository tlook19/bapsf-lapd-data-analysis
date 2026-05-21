import numpy as np

from bapsf_lapd import analyze_langmuir_sweep, detect_arc_like_segments, evaluate_langmuir_quality


def test_detect_arc_like_segments_flags_large_jump():
    current = np.zeros(200)
    current[80:85] = 1.0

    count, max_jump = detect_arc_like_segments(current)

    assert count > 0
    assert max_jump == 1.0


def test_evaluate_langmuir_quality_flags_narrow_fit_window():
    voltage = np.linspace(-25, 25, 900)
    ion = -0.002 + 2.0e-5 * voltage
    electron = 2.0e-4 * np.exp((voltage - 1.0) / 5.0)
    saturation = 0.007 + 3.5e-5 * voltage
    current = np.minimum(ion + electron, saturation)
    analysis = analyze_langmuir_sweep(voltage, current)

    report = evaluate_langmuir_quality(analysis, min_fit_window_width_v=100.0)

    assert report.severity in {"warn", "bad"}
    assert any(flag.code == "narrow_fit_window" for flag in report.flags)
