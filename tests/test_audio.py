from __future__ import annotations

import numpy as np
import pytest

from choravox import audio


def test_as_waveform_preserves_samples_first_geometry() -> None:
    mono = audio.as_waveform([0, 1, -1])
    stereo = audio.as_waveform([[0, 1], [1, 0]])
    assert mono.shape == (3,) and mono.dtype == np.float64
    assert stereo.shape == (2, 2)


def test_as_waveform_rejects_nonfinite_and_rank_three_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        audio.as_waveform([0.0, np.inf])
    with pytest.raises(ValueError, match="shape"):
        audio.as_waveform(np.zeros((2, 2, 2)))


def test_to_mono_averages_or_weights_channels() -> None:
    stereo = np.array([[1.0, -1.0], [0.0, 1.0]])
    np.testing.assert_allclose(audio.to_mono(stereo), [0.0, 0.5])
    np.testing.assert_allclose(audio.to_mono(stereo, [1.0, 0.0]), [1.0, 0.0])


def test_to_mono_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="one finite value"):
        audio.to_mono(np.zeros((4, 2)), [1.0])
    with pytest.raises(ValueError, match="sum to zero"):
        audio.to_mono(np.zeros((4, 2)), [1.0, -1.0])


def test_peak_and_rms_have_hand_computed_values() -> None:
    signal = np.array([-1.0, 0.0, 1.0])
    assert audio.peak_amplitude(signal) == 1.0
    assert audio.rms(signal) == pytest.approx(np.sqrt(2.0 / 3.0), abs=1e-12)


def test_level_measures_define_empty_audio_as_silence() -> None:
    assert audio.peak_amplitude([]) == 0.0
    assert audio.rms([]) == 0.0


def test_rms_can_measure_each_channel_independently() -> None:
    stereo = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(audio.rms(stereo, axis=0), [1.0, 0.0])


def test_peak_amplitude_is_sign_invariant() -> None:
    signal = np.array([0.25, -0.75, 0.5])
    assert audio.peak_amplitude(signal) == audio.peak_amplitude(-signal) == 0.75


def test_dbfs_maps_full_scale_and_silence() -> None:
    assert audio.dbfs(np.ones(8)) == 0.0
    assert audio.dbfs(np.zeros(8)) == -120.0
    assert audio.dbfs(np.full(8, 0.5)) == pytest.approx(-6.0206, abs=1e-5)


def test_dbfs_honors_a_declared_floor() -> None:
    assert audio.dbfs(np.full(4, 1e-8), floor_db=-80.0) == -80.0
    with pytest.raises(ValueError, match="must not be positive"):
        audio.dbfs([1.0], floor_db=1.0)


def test_peak_normalization_reaches_target_and_preserves_ratios() -> None:
    normalized = audio.normalize_peak([0.25, -0.5], 0.8)
    np.testing.assert_allclose(normalized, [0.4, -0.8])
    assert audio.peak_amplitude(normalized) == 0.8


def test_peak_normalization_leaves_silence_unchanged() -> None:
    np.testing.assert_array_equal(audio.normalize_peak(np.zeros(3)), np.zeros(3))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        audio.normalize_peak([1.0], 0.0)


def test_rms_normalization_hits_reachable_target() -> None:
    normalized = audio.normalize_rms([0.25, -0.25], 0.5)
    assert audio.rms(normalized) == 0.5


def test_rms_normalization_obeys_peak_limit() -> None:
    normalized = audio.normalize_rms([1.0, 0.0], 0.9, peak_limit=0.7)
    assert audio.peak_amplitude(normalized) == 0.7
    assert audio.rms(normalized) < 0.9


def test_peak_normalization_is_idempotent() -> None:
    once = audio.normalize_peak([0.1, -0.4, 0.2], 0.8)
    twice = audio.normalize_peak(once, 0.8)
    np.testing.assert_allclose(once, twice, atol=1e-12)


def test_rms_normalization_preserves_shape_for_stereo() -> None:
    stereo = np.array([[0.1, -0.2], [0.2, -0.4]])
    assert audio.normalize_rms(stereo, 0.2).shape == stereo.shape


# -- extensions --
