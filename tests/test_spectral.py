from __future__ import annotations

import numpy as np
import pytest

from choravox import spectral


def test_stft_locates_a_bin_centered_tone() -> None:
    tone = np.sin(2 * np.pi * 4 * np.arange(64) / 64)
    transformed = spectral.stft(tone, 64, 32, center=False, window="rectangular")
    assert transformed.shape == (1, 33)
    assert int(np.argmax(np.abs(transformed[0]))) == 4


def test_stft_preserves_optional_channel_axis() -> None:
    stereo = np.column_stack((np.ones(64), -np.ones(64)))
    transformed = spectral.stft(stereo, 32, 16)
    assert transformed.shape == (5, 17, 2)


def test_stft_inverse_reconstructs_centered_audio() -> None:
    generator = np.random.default_rng(91)
    signal = generator.normal(size=513)
    transformed = spectral.stft(signal, 64, 32)
    restored = spectral.istft(transformed, 64, 32, length=signal.size)
    np.testing.assert_allclose(restored, signal, atol=1e-12)


def test_inverse_stft_restores_channels_independently() -> None:
    signal = np.column_stack((np.linspace(-1.0, 1.0, 200), np.linspace(1.0, -1.0, 200)))
    transformed = spectral.stft(signal, 40, 20)
    restored = spectral.istft(transformed, 40, 20, length=200)
    np.testing.assert_allclose(restored, signal, atol=1e-12)


def test_stft_pads_short_audio_across_centered_boundary_frames() -> None:
    transformed = spectral.stft([1.0], 8, 4)
    assert transformed.shape == (2, 5)


def test_stft_rejects_short_fft_and_invalid_hop() -> None:
    with pytest.raises(ValueError, match="at least"):
        spectral.stft(np.ones(8), 8, 4, n_fft=4)
    with pytest.raises(ValueError, match="positive"):
        spectral.stft(np.ones(8), 8, 0)


def test_inverse_stft_rejects_nonfinite_and_overlong_requests() -> None:
    with pytest.raises(ValueError, match="finite"):
        spectral.istft(np.array([[1.0, np.nan]]), 2, 1)
    transformed = spectral.stft(np.ones(16), 8, 4)
    with pytest.raises(ValueError, match="fit"):
        spectral.istft(transformed, 8, 4, length=100)


@pytest.mark.parametrize("length", [1, 31, 32, 33, 95, 96, 97])
def test_centered_transform_retains_requested_tail(length: int) -> None:
    signal = np.linspace(-0.75, 0.75, length)
    transformed = spectral.stft(signal, 32, 16)
    restored = spectral.istft(transformed, 32, 16, length=length)
    np.testing.assert_allclose(restored, signal, atol=1e-12)


def test_periodogram_frequency_axis_reaches_nyquist() -> None:
    frequencies, density = spectral.periodogram(np.ones(16), 8000, window="rectangular")
    np.testing.assert_array_equal(frequencies, np.arange(9) * 500)
    assert density.shape == frequencies.shape


def test_periodogram_locates_tone_in_physical_units() -> None:
    time = np.arange(800) / 8000
    frequencies, density = spectral.periodogram(np.sin(2 * np.pi * 1000 * time), 8000)
    assert frequencies[int(np.argmax(density))] == 1000.0


# -- extensions --
