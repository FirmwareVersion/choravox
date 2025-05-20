from __future__ import annotations

import numpy as np
import pytest

from choravox import filters


def test_lowpass_kernel_is_symmetric_with_unit_dc_gain() -> None:
    kernel = filters.lowpass_kernel(16_000, 3000.0, taps=63)
    np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-15)
    assert np.sum(kernel) == pytest.approx(1.0, abs=1e-15)


def test_lowpass_kernel_validates_cutoff_and_odd_length() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        filters.lowpass_kernel(16_000, 9000.0)
    with pytest.raises(ValueError, match="odd"):
        filters.lowpass_kernel(16_000, 3000.0, taps=32)


def test_fir_convolution_has_hand_computed_impulse_response() -> None:
    output = filters.apply_fir([1.0, 0.0, 0.0], [0.25, 0.5, 0.25], mode="full")
    np.testing.assert_array_equal(output, [0.25, 0.5, 0.25, 0.0, 0.0])


def test_fir_convolution_filters_channels_independently() -> None:
    stereo = np.column_stack(([1.0, 0.0], [0.0, 1.0]))
    output = filters.apply_fir(stereo, [1.0, -1.0], mode="full")
    np.testing.assert_array_equal(output, [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]])


def test_fir_modes_have_numpy_compatible_lengths() -> None:
    signal = np.arange(7, dtype=np.float64)
    kernel = np.ones(3)
    assert filters.apply_fir(signal, kernel, mode="full").shape == (9,)
    assert filters.apply_fir(signal, kernel, mode="same").shape == (7,)
    assert filters.apply_fir(signal, kernel, mode="valid").shape == (5,)


def test_fir_rejects_invalid_kernel_and_mode() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        filters.apply_fir([1.0], [])
    with pytest.raises(ValueError, match="mode"):
        filters.apply_fir([1.0], [1.0], mode="causal")  # type: ignore[arg-type]


def test_empty_same_convolution_preserves_channels() -> None:
    assert filters.apply_fir(np.empty((0, 2)), [1.0]).shape == (0, 2)


def test_highpass_has_zero_dc_and_unit_nyquist_response() -> None:
    kernel = filters.highpass_kernel(16_000, 3000.0, taps=63)
    assert np.sum(kernel) == pytest.approx(0.0, abs=1e-15)
    response = np.sum(kernel * (-1.0) ** np.arange(kernel.size))
    assert abs(response) == pytest.approx(1.0, abs=2e-3)


def test_bandpass_is_symmetric_with_zero_dc() -> None:
    kernel = filters.bandpass_kernel(16_000, 300.0, 3400.0, taps=127)
    np.testing.assert_allclose(kernel, kernel[::-1], atol=1e-15)
    assert np.sum(kernel) == pytest.approx(0.0, abs=1e-15)


def test_lowpass_rejects_high_tone_but_keeps_low_tone() -> None:
    time = np.arange(3200) / 16_000
    low = np.sin(2 * np.pi * 500 * time)
    high = np.sin(2 * np.pi * 5000 * time)
    kernel = filters.lowpass_kernel(16_000, 2000.0, taps=127, window="blackman")
    low_output = filters.apply_fir(low, kernel)[100:-100]
    high_output = filters.apply_fir(high, kernel)[100:-100]
    assert np.sqrt(np.mean(high_output**2)) < np.sqrt(np.mean(low_output**2)) * 1e-3


def test_voice_bandpass_keeps_midband_and_rejects_dc() -> None:
    kernel = filters.bandpass_kernel(16_000, 200.0, 3800.0, taps=127)
    dc_output = filters.apply_fir(np.ones(1000), kernel)[100:-100]
    assert np.max(np.abs(dc_output)) < 1e-12


def test_bandpass_rejects_reversed_bounds() -> None:
    with pytest.raises(ValueError, match="increasing"):
        filters.bandpass_kernel(16_000, 4000.0, 300.0)


# -- extensions --
