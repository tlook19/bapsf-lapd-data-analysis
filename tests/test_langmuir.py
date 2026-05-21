import numpy as np

from bapsf_lapd import analyze_langmuir_sweep


def test_analyze_langmuir_sweep_recovers_synthetic_temperature():
    rng = np.random.default_rng(123)
    voltage = np.linspace(-25, 25, 900)
    te_ev = 5.0
    ion = -0.002 + 2.0e-5 * voltage
    electron = 2.0e-4 * np.exp((voltage - 1.0) / te_ev)
    saturation = 0.007 + 3.5e-5 * voltage
    current = np.minimum(ion + electron, saturation)
    current = current + rng.normal(0.0, 1.5e-4, size=current.shape)

    result = analyze_langmuir_sweep(voltage, current)

    assert result.log_linear_fit.success
    assert result.exponential_fit.success
    assert 0.5 < result.log_linear_fit.electron_temperature_ev < 25.0
    np.testing.assert_allclose(result.exponential_fit.electron_temperature_ev, te_ev, rtol=0.35)


def test_analyze_langmuir_sweep_constrains_ion_branch_slope_nonnegative():
    voltage = np.linspace(-25, 25, 900)
    ion = -0.002 - 5.0e-5 * voltage
    electron = 1.5e-4 * np.exp((voltage - 1.0) / 4.0)
    saturation = 0.006 + 2.5e-5 * voltage
    current = np.minimum(ion + electron, saturation)

    result = analyze_langmuir_sweep(voltage, current)

    assert result.ion_fit.slope >= 0.0
