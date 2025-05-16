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


# -- extensions --
