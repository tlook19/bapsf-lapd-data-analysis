import numpy as np
import pytest

from bapsf_lapd import butterworth_lowpass


def test_butterworth_lowpass_preserves_multidimensional_shape():
    sample_rate_hz = 10_000.0
    time_s = np.arange(1000) / sample_rate_hz
    low = np.sin(2 * np.pi * 100 * time_s)
    high = 0.25 * np.sin(2 * np.pi * 3000 * time_s)
    data = np.stack([low + high, low + high], axis=0)

    filtered = butterworth_lowpass(data, sample_rate_hz=sample_rate_hz, cutoff_hz=500.0, order=4)

    assert filtered.shape == data.shape
    assert np.std(filtered - low) < np.std(data[0] - low)


def test_butterworth_lowpass_rejects_cutoff_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        butterworth_lowpass([1, 2, 3], sample_rate_hz=1000.0, cutoff_hz=600.0)
