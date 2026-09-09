"""The noise-aware estimators against a synthetic sweep with a known T_e.

The bias being tested
---------------------
Under a measured current ``I_meas = I_e + d`` for an additive floor ``d``, a
local log-slope taken at electron current ``I_e`` reads ``T_e * (1 + d / I_e)``:
unbounded as the fit reaches down toward ``d``, negligible well above it.  The
pipeline's window family bounds its fit RELATIVE to the sweep's own electron
saturation level -- a percentile of the positive current floored at 1.5 % of the
maximum, up to a fraction of that maximum -- so on a sweep whose maximum is only
a few tens of floors its low windows sit where ``d / I_e`` is order one.  The
noise-aware range starts at ``SLOPE_LOW_FLOORS`` floors instead, which caps the
local bias at ``1 + 1 / SLOPE_LOW_FLOORS`` at its bottom sample.

Two forms of that floor are pinned separately, because they reach the fit by
different routes:

* a DETERMINISTIC positive offset on the electron branch, which is the closed
  form above and biases a low relative window high directly;
* a ZERO-MEAN fluctuation of the same RMS on the raw current, which is what the
  ion-line residual actually is.  Its mean is absorbed by the ion-line fit, but
  the electron branch is masked to POSITIVE samples, so near the floor only the
  upward excursions survive and the surviving low-current samples sit above the
  true branch -- the same flattening, arrived at by selection.  This arm runs the
  pipeline's own ``refit_sweep_windows.refit_sweep`` on the raw sweep, so the two
  arms are compared through the pipeline's code rather than through a
  restatement of it.

A DC offset on the RAW current is NOT one of these: the ion-line fit absorbs a
constant outright, and the test below pins that so the distinction is not lost.

The floating-potential estimator is exact on its own model: for
``I_e(V) = I_es exp((V - V_p) / T_e)`` balanced against a constant ion current,
``T_e = (V_p - V_f) / ln(I_es / I_i)`` recovers ``T_e`` identically, with the
saturation ratio the sweep shows and no substituted value.  Its one convention
is where ``V_p`` sits, and the tests pin that too: the knee, not the sweep's
current maximum, which on a branch that keeps climbing past the knee returns a
value several times too large.
"""

import numpy as np
import pytest

from es4_te_noise_aware import (
    SLOPE_LOW_FLOORS,
    condition_sweep,
    floating_potential_te_ev,
    knee_point,
    local_slope_te_ev,
    range_decades,
    slope_range_mask,
)
from refit_sweep_windows import refit_sweep


# The synthetic plasma.  T_e and the ion current are round numbers so the
# analytic answers below are exact rather than nearly exact.
TE_EV = 0.60
V_PLASMA = 4.0
I_SATURATION_A = 40.0e-3
I_ION_A = 4.0e-3

# The sweep: a linear voltage ramp sampled densely enough that the retarding
# branch is resolved over several e-foldings.
VOLTAGE = np.linspace(-20.0, 12.0, 4001)

# The floor, as a fraction of the electron saturation level.  At 1 % the
# family's 1.5%-of-max lower bound sits where d / I_e is about two thirds.
FLOOR_FRACTION = 0.01
FLOOR_A = FLOOR_FRACTION * I_SATURATION_A

# The family's lowest-current window: its 1.5%-of-max lower bound against its
# smallest upper fraction.  Restated here, and only here, so the deterministic
# arm can be evaluated without a raw sweep.
FAMILY_LOW_BOUND_FRACTION = 0.015
FAMILY_LOW_HIGH_FRACTION = 0.05

# Fixed so the fluctuation arm is a pinned number rather than a lottery.
NOISE_SEED = 20260909


def _true_electron_current_a(voltage):
    """The Maxwellian retarding branch, saturating at V >= V_PLASMA."""
    return I_SATURATION_A * np.exp(np.minimum(voltage - V_PLASMA, 0.0) / TE_EV)


def _log_slope_te_ev(voltage, current, low_a, high_a):
    """T_e from a log-slope fit over an explicit electron-current window."""
    mask = (current >= low_a) & (current <= high_a)
    slope, _ = np.polyfit(voltage[mask], np.log(current[mask]), 1)
    return float(1.0 / slope)


def _relative_bias(value):
    return abs(value - TE_EV) / TE_EV


def test_a_deterministic_floor_biases_the_low_relative_window_high():
    """The closed form, at the estimator level: relative window vs noise-aware."""
    measured = _true_electron_current_a(VOLTAGE) + FLOOR_A
    i_max = float(measured.max())

    window_te = _log_slope_te_ev(
        VOLTAGE,
        measured,
        FAMILY_LOW_BOUND_FRACTION * i_max,
        FAMILY_LOW_HIGH_FRACTION * i_max,
    )
    slope_te = local_slope_te_ev(VOLTAGE, measured, FLOOR_A, i_max)

    # the low relative window reads high by more than half again
    assert window_te > 1.5 * TE_EV
    # the noise-referenced range stays inside its analytic worst case
    assert slope_te == pytest.approx(TE_EV, rel=1.0 / SLOPE_LOW_FLOORS)
    # and carries several times less bias than the window it replaces
    assert _relative_bias(slope_te) < _relative_bias(window_te) / 3.0


def test_a_fluctuating_floor_biases_the_pipeline_window_family_high():
    """The same flattening through the pipeline's own family fit, by selection."""
    rng = np.random.default_rng(NOISE_SEED)
    raw = _true_electron_current_a(VOLTAGE) - I_ION_A + rng.normal(
        0.0, FLOOR_A, VOLTAGE.size
    )

    _, grid = refit_sweep(VOLTAGE, raw)
    # grid rows are p_low = 3, 8, 15, 25, 35 percent of the positive electron
    # current; columns are f_high = 0.05, 0.10, 0.15, 0.30, 0.50 of I_max.  The
    # lowest-current corner is [0, 0].
    window_te = float(grid[0, 0])

    sweep = condition_sweep(VOLTAGE, raw)
    slope_te = local_slope_te_ev(
        sweep.voltage, sweep.electron_current, sweep.noise_floor_a, sweep.i_max_a
    )

    # the ion-line residual RMS recovers the floor that was injected
    assert sweep.noise_floor_a == pytest.approx(FLOOR_A, rel=0.1)
    assert window_te > 1.3 * TE_EV
    assert slope_te == pytest.approx(TE_EV, rel=1.0 / SLOPE_LOW_FLOORS)
    assert _relative_bias(slope_te) < _relative_bias(window_te) / 3.0


def test_a_constant_offset_on_the_raw_current_is_absorbed_by_the_ion_line():
    """A DC offset is not the floor: the slope-zero ion-line fit takes it out."""
    raw = _true_electron_current_a(VOLTAGE) - I_ION_A + FLOOR_A
    sweep = condition_sweep(VOLTAGE, raw)
    default_te, grid = refit_sweep(VOLTAGE, raw)

    assert sweep.noise_floor_a < 0.1 * FLOOR_A
    assert default_te == pytest.approx(TE_EV, rel=0.05)
    assert float(grid[0, 0]) == pytest.approx(TE_EV, rel=0.05)


def test_both_fits_recover_te_when_there_is_no_floor():
    """The control: with the floor removed the two arms agree with the truth."""
    measured = _true_electron_current_a(VOLTAGE)
    i_max = float(measured.max())

    assert _log_slope_te_ev(
        VOLTAGE,
        measured,
        FAMILY_LOW_BOUND_FRACTION * i_max,
        FAMILY_LOW_HIGH_FRACTION * i_max,
    ) == pytest.approx(TE_EV, rel=1e-6)
    assert local_slope_te_ev(VOLTAGE, measured, FLOOR_A, i_max) == pytest.approx(
        TE_EV, rel=0.05
    )


def test_the_clean_range_narrows_as_the_signal_falls():
    """Fewer decades of clean range at a smaller I_max -- the estimator's cost."""
    wide = range_decades(FLOOR_A, I_SATURATION_A)
    narrow = range_decades(FLOOR_A, I_SATURATION_A / 10.0)
    assert wide == pytest.approx(narrow + 1.0)
    assert wide > 0.0


def test_the_range_mask_never_reaches_below_its_noise_referenced_floor():
    measured = _true_electron_current_a(VOLTAGE) + FLOOR_A
    mask = slope_range_mask(VOLTAGE, measured, FLOOR_A, float(measured.max()))
    assert mask.any()
    assert measured[mask].min() >= SLOPE_LOW_FLOORS * FLOOR_A


def test_floating_potential_estimator_is_exact_on_its_own_model():
    """V_f is where the exponential branch equals the ion current, analytically."""
    v_float = V_PLASMA + TE_EV * np.log(I_ION_A / I_SATURATION_A)
    assert _true_electron_current_a(v_float) == pytest.approx(I_ION_A, rel=1e-12)

    te = floating_potential_te_ev(V_PLASMA, v_float, I_SATURATION_A, I_ION_A)
    assert te == pytest.approx(TE_EV, rel=1e-12)


def test_floating_potential_estimator_uses_the_ratio_it_is_given():
    """A different measured ratio gives a different T_e; nothing is substituted."""
    v_float = V_PLASMA - 3.0
    measured = floating_potential_te_ev(V_PLASMA, v_float, I_SATURATION_A, I_ION_A)
    unmagnetized = floating_potential_te_ev(V_PLASMA, v_float, 34.0 * I_ION_A, I_ION_A)
    assert measured == pytest.approx(3.0 / np.log(I_SATURATION_A / I_ION_A), rel=1e-12)
    assert unmagnetized == pytest.approx(3.0 / np.log(34.0), rel=1e-12)
    assert measured != pytest.approx(unmagnetized, rel=1e-3)


@pytest.mark.parametrize(
    "v_plasma, v_float, i_max, i_ion",
    [
        (V_PLASMA, V_PLASMA + 1.0, I_SATURATION_A, I_ION_A),  # V_f above V_p
        (V_PLASMA, 0.0, I_SATURATION_A, 0.0),                 # no ion current
        (V_PLASMA, 0.0, I_ION_A, I_ION_A),                    # unit ratio
        (V_PLASMA, np.nan, I_SATURATION_A, I_ION_A),          # no zero crossing
    ],
)
def test_floating_potential_estimator_refuses_impossible_inputs(
    v_plasma, v_float, i_max, i_ion
):
    assert np.isnan(floating_potential_te_ev(v_plasma, v_float, i_max, i_ion))


def test_conditioning_locates_the_knee_and_reads_the_ratio_there():
    """V_p is the knee and I_es the current at it, so the pair is consistent."""
    raw = _true_electron_current_a(VOLTAGE) - I_ION_A
    sweep = condition_sweep(VOLTAGE, raw)
    assert sweep.v_plasma_v == pytest.approx(V_PLASMA, abs=0.05)
    assert sweep.i_knee_a == pytest.approx(I_SATURATION_A, rel=0.05)
    assert sweep.saturation_ratio == pytest.approx(
        sweep.i_knee_a / sweep.i_ion_a, rel=1e-12
    )


def test_the_knee_is_not_the_current_maximum_on_a_climbing_collection_branch():
    """A branch that keeps rising past the knee separates the two conventions.

    Above the plasma potential the collected current goes on climbing -- the
    sheath grows with bias -- so the sweep's maximum sits near the ramp end, far
    above the knee.  Estimator (ii) must read the knee: the same formula
    evaluated at the maximum inflates ``V_p - V_f`` linearly while the ratio it
    divides by grows only logarithmically, and returns a value several times too
    large.
    """
    climb = 1.0 + 0.05 * np.maximum(VOLTAGE - V_PLASMA, 0.0)
    raw = _true_electron_current_a(VOLTAGE) * climb - I_ION_A
    sweep = condition_sweep(VOLTAGE, raw)

    assert sweep.v_plasma_v == pytest.approx(V_PLASMA, abs=0.1)
    assert sweep.v_i_max_v > sweep.v_plasma_v + 5.0
    assert sweep.i_max_a > sweep.i_knee_a

    at_knee = floating_potential_te_ev(
        sweep.v_plasma_v, sweep.v_float_v, sweep.i_knee_a, sweep.i_ion_a
    )
    at_maximum = floating_potential_te_ev(
        sweep.v_i_max_v, sweep.v_float_v, sweep.i_max_a, sweep.i_ion_a
    )
    assert at_knee == pytest.approx(TE_EV, rel=0.25)
    assert at_maximum > 3.0 * at_knee


def test_knee_point_refuses_a_collection_branch_with_too_few_samples():
    voltage = np.array([0.0, 1.0, 2.0])
    current = np.array([1.0, 2.0, 3.0])
    v_plasma, i_knee = knee_point(voltage, current, 0.0, 2.0)
    assert np.isnan(v_plasma)
    assert np.isnan(i_knee)
