from __future__ import annotations

import numpy as np

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


# -- extensions --
