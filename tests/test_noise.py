from __future__ import annotations

import numpy as np
import pytest

from choravox import audio, noise


def test_white_noise_is_centered_at_requested_rms() -> None:
    samples = noise.white_noise(4096, seed=7, rms_level=0.2)
    assert np.mean(samples) == pytest.approx(0.0, abs=1e-9)
    assert audio.rms(samples) == pytest.approx(0.2, abs=1e-8)


def test_white_noise_rejects_empty_or_invalid_level() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        noise.white_noise(0, seed=1)
    with pytest.raises(ValueError, match="rms_level"):
        noise.white_noise(8, seed=1, rms_level=float("nan"))


def test_white_noise_repeats_bytes_for_same_seed() -> None:
    first = noise.white_noise(257, seed=42)
    second = noise.white_noise(257, seed=42)
    assert first.tobytes() == second.tobytes()


def test_white_noise_changes_with_seed_without_changing_calibration() -> None:
    first = noise.white_noise(2048, seed=1, rms_level=0.3)
    second = noise.white_noise(2048, seed=2, rms_level=0.3)
    assert not np.array_equal(first, second)
    assert audio.rms(first) == pytest.approx(audio.rms(second), abs=1e-8)


def test_white_noise_samples_are_finite() -> None:
    assert np.all(np.isfinite(noise.white_noise(10_000, seed=99)))


def test_pink_noise_has_more_low_than_high_frequency_power() -> None:
    samples = noise.pink_noise(8192, seed=3)
    power = np.abs(np.fft.rfft(samples)) ** 2
    low = np.mean(power[1:128])
    high = np.mean(power[-128:])
    assert low > 5.0 * high


def test_pink_noise_is_centered_and_rms_calibrated() -> None:
    samples = noise.pink_noise(4096, seed=8, rms_level=0.15)
    assert np.mean(samples) == pytest.approx(0.0, abs=1e-9)
    assert audio.rms(samples) == pytest.approx(0.15, abs=1e-8)


def test_pink_noise_supports_odd_lengths_and_repeats_exactly() -> None:
    first = noise.pink_noise(513, seed=12)
    second = noise.pink_noise(513, seed=12)
    assert first.shape == (513,)
    assert first.tobytes() == second.tobytes()


def test_pink_noise_rejects_degenerate_length_and_level() -> None:
    with pytest.raises(ValueError, match="at least two"):
        noise.pink_noise(1, seed=0)
    with pytest.raises(ValueError, match="rms_level"):
        noise.pink_noise(8, seed=0, rms_level=0.0)


def test_pink_and_white_generators_use_separate_seed_namespaces() -> None:
    assert not np.array_equal(noise.white_noise(128, seed=5), noise.pink_noise(128, seed=5))


def test_noise_mix_reaches_requested_power_snr() -> None:
    signal = np.sin(2.0 * np.pi * np.arange(4000) / 40.0)
    disturbance = noise.white_noise(signal.size, seed=4)
    observed = noise.mix_at_snr(signal, disturbance, 7.5)
    assert audio.snr_db(signal, observed) == pytest.approx(7.5, abs=1e-5)


def test_snr_sweep_preserves_level_order() -> None:
    signal = np.sin(2.0 * np.pi * np.arange(2000) / 25.0)
    disturbance = noise.pink_noise(signal.size, seed=2)
    mixtures = noise.snr_sweep(signal, disturbance, (-5.0, 0.0, 10.0))
    measured = [audio.snr_db(signal, mixture) for mixture in mixtures]
    assert measured == pytest.approx([-5.0, 0.0, 10.0], abs=1e-5)


def test_snr_sweep_uses_identical_noise_direction() -> None:
    signal = np.linspace(-1.0, 1.0, 128)
    disturbance = noise.white_noise(128, seed=6)
    low, high = noise.snr_sweep(signal, disturbance, (0.0, 20.0))
    low_residual = low - signal
    high_residual = high - signal
    cosine = np.dot(low_residual, high_residual) / (
        np.linalg.norm(low_residual) * np.linalg.norm(high_residual)
    )
    assert cosine == pytest.approx(1.0, abs=1e-12)


def test_noise_mixing_rejects_nonfinite_target_and_empty_sweep() -> None:
    with pytest.raises(ValueError, match="finite"):
        noise.mix_at_snr(np.ones(8), np.arange(8.0), float("inf"))
    with pytest.raises(ValueError, match="nonempty"):
        noise.snr_sweep(np.ones(8), np.arange(8.0), ())


def test_noise_mixing_preserves_multichannel_shape() -> None:
    signal = np.column_stack((np.linspace(-1.0, 1.0, 64), np.linspace(1.0, -1.0, 64)))
    disturbance = np.column_stack((noise.white_noise(64, seed=1), noise.white_noise(64, seed=2)))
    assert noise.mix_at_snr(signal, disturbance, 5.0).shape == signal.shape


# -- extensions --
