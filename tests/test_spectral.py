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


def test_rectangular_periodogram_integrates_to_mean_square() -> None:
    generator = np.random.default_rng(22)
    signal = generator.normal(size=255)
    frequencies, density = spectral.periodogram(signal, 16_000, window="rectangular")
    bin_width = frequencies[1] - frequencies[0]
    assert np.sum(density) * bin_width == pytest.approx(np.mean(signal**2), rel=1e-12)


def test_periodogram_rejects_channels_and_empty_input() -> None:
    with pytest.raises(ValueError, match="mono"):
        spectral.periodogram(np.ones((8, 2)), 8000)
    with pytest.raises(ValueError, match="nonempty"):
        spectral.periodogram([], 8000)


def test_periodogram_density_is_nonnegative_and_finite() -> None:
    _, density = spectral.periodogram(np.arange(12), 12_000, n_fft=32)
    assert np.all(density >= 0.0) and np.all(np.isfinite(density))


def test_spectral_center_and_spread_have_hand_computed_values() -> None:
    spectra = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])
    frequencies = np.array([0.0, 100.0, 200.0])
    np.testing.assert_allclose(spectral.spectral_centroid(spectra, frequencies), [100, 100])
    np.testing.assert_allclose(spectral.spectral_bandwidth(spectra, frequencies), [0, 100])


def test_silent_spectrum_has_zero_centroid_and_bandwidth() -> None:
    silent = np.zeros((2, 4))
    axis = np.arange(4, dtype=np.float64)
    np.testing.assert_array_equal(spectral.spectral_centroid(silent, axis), [0.0, 0.0])
    np.testing.assert_array_equal(spectral.spectral_bandwidth(silent, axis), [0.0, 0.0])


def test_spectral_centroid_stays_within_frequency_axis() -> None:
    generator = np.random.default_rng(5)
    spectra = generator.uniform(size=(20, 17))
    frequencies = np.linspace(0.0, 8000.0, 17)
    centers = spectral.spectral_centroid(spectra, frequencies)
    assert np.all((centers >= frequencies[0]) & (centers <= frequencies[-1]))


def test_spectral_descriptors_reject_shape_and_order_errors() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        spectral.spectral_centroid(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="ordered"):
        spectral.spectral_bandwidth(np.ones((1, 3)), [0.0, 2.0, 1.0])


def test_single_bin_spectrum_has_no_bandwidth() -> None:
    spectra = np.eye(5)
    axis = np.linspace(0.0, 4000.0, 5)
    np.testing.assert_array_equal(spectral.spectral_bandwidth(spectra, axis), np.zeros(5))


def test_spectral_flux_responds_only_to_positive_arrivals() -> None:
    spectra = np.array([[1.0, 0.0], [1.0, 2.0], [1.0, 1.0]])
    np.testing.assert_allclose(spectral.spectral_flux(spectra, normalize=False), [0.0, 2.0, 0.0])


def test_normalized_flux_ignores_uniform_gain() -> None:
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    np.testing.assert_allclose(spectral.spectral_flux(spectra), [0.0, 0.0], atol=1e-15)


def test_flux_peaks_at_a_new_frequency_component() -> None:
    spectra = np.zeros((6, 8))
    spectra[:, 1] = 1.0
    spectra[3:, 6] = 2.0
    flux = spectral.spectral_flux(spectra, normalize=False)
    assert int(np.argmax(flux)) == 3
    assert np.count_nonzero(flux) == 1


def test_flux_handles_silent_transitions_without_nan() -> None:
    flux = spectral.spectral_flux(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]))
    np.testing.assert_array_equal(flux, [0.0, 1.0, 0.0])


def test_flux_rejects_negative_spectrum_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        spectral.spectral_flux([[1.0, -0.1]])


# -- extensions --
