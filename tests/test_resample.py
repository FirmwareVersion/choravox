from __future__ import annotations

import numpy as np
import pytest

from choravox import resample


def test_sampling_ratio_is_reduced() -> None:
    assert resample.sampling_ratio(48_000, 16_000) == resample.SamplingRatio(1, 3)
    assert resample.sampling_ratio(44_100, 16_000) == resample.SamplingRatio(160, 441)


def test_resampled_length_uses_half_up_duration_rounding() -> None:
    assert resample.resampled_length(3, 4, 2) == 2
    assert resample.resampled_length(5, 2, 4) == 10
    assert resample.resampled_length(0, 16_000, 8_000) == 0


def test_polyphase_kernel_has_one_unit_sum_per_phase() -> None:
    kernel = resample.polyphase_kernel(16_000, 22_050, half_width=12)
    assert kernel.shape == (441, 24)
    np.testing.assert_allclose(np.sum(kernel, axis=1), np.ones(441), atol=1e-14)


def test_downsampling_kernel_has_lower_cutoff_than_upsampling() -> None:
    down = resample.polyphase_kernel(16_000, 8_000)[0]
    up = resample.polyphase_kernel(8_000, 16_000)[0]
    assert np.sum(np.abs(np.diff(down))) < np.sum(np.abs(np.diff(up)))


def test_resampler_preserves_duration_and_channel_geometry() -> None:
    stereo = np.column_stack((np.ones(80), -np.ones(80)))
    converted = resample.resample_polyphase(stereo, 8000, 16_000)
    assert converted.shape == (160, 2)
    np.testing.assert_allclose(converted, np.tile(stereo[0], (160, 1)), atol=1e-14)


def test_identity_resampling_returns_an_equal_copy() -> None:
    signal = np.arange(8, dtype=np.float64)
    converted = resample.resample_polyphase(signal, 16_000, 16_000)
    np.testing.assert_array_equal(converted, signal)
    assert converted is not signal


def test_downsampling_attenuates_energy_above_target_nyquist() -> None:
    times = np.arange(800) / 8000
    low = np.sin(2 * np.pi * 500 * times)
    high = np.sin(2 * np.pi * 3000 * times)
    low_output = resample.resample_polyphase(low, 8000, 4000)
    high_output = resample.resample_polyphase(high, 8000, 4000)
    low_rms = np.sqrt(np.mean(low_output[20:-20] ** 2))
    high_rms = np.sqrt(np.mean(high_output[20:-20] ** 2))
    assert high_rms < low_rms * 0.02


def test_resampler_rejects_invalid_rates_and_kernel_width() -> None:
    with pytest.raises(ValueError, match="positive"):
        resample.resample_polyphase([1.0], 0, 8000)
    with pytest.raises(ValueError, match="at least two"):
        resample.resample_polyphase([1.0], 8000, 16_000, half_width=1)


def test_resample_then_inverse_is_stable_for_bandlimited_tone() -> None:
    time = np.arange(1600) / 16_000
    signal = 0.6 * np.sin(2 * np.pi * 440 * time) + 0.2 * np.sin(2 * np.pi * 1200 * time)
    lower = resample.resample_polyphase(signal, 16_000, 8000)
    restored = resample.resample_polyphase(lower, 8000, 16_000)
    error = restored[40:-40] - signal[40:-40]
    assert np.sqrt(np.mean(error**2)) < 0.005


def test_resampling_preserves_interior_rms_for_low_frequency_audio() -> None:
    time = np.arange(1200) / 12_000
    signal = np.sin(2 * np.pi * 300 * time)
    converted = resample.resample_polyphase(signal, 12_000, 16_000)
    source_rms = np.sqrt(np.mean(signal[30:-30] ** 2))
    target_rms = np.sqrt(np.mean(converted[40:-40] ** 2))
    assert target_rms == pytest.approx(source_rms, rel=0.005)


def test_empty_audio_resamples_with_channel_rank_intact() -> None:
    assert resample.resample_polyphase([], 8000, 16_000).shape == (0,)
    assert resample.resample_polyphase(np.empty((0, 2)), 8000, 16_000).shape == (0, 2)


# -- extensions --
