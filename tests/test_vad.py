from __future__ import annotations

import numpy as np
import pytest

from choravox import vad


def test_frame_energy_reports_silence_floor_and_full_scale() -> None:
    silence = vad.frame_energy_db(np.zeros(800), 8000, frame_ms=20.0, hop_ms=10.0)
    full = vad.frame_energy_db(np.ones(800), 8000, frame_ms=20.0, hop_ms=10.0)
    np.testing.assert_array_equal(silence.dbfs, -120.0)
    np.testing.assert_allclose(full.dbfs[:-1], 0.0, atol=1e-9)


def test_frame_energy_exposes_sample_derived_timing() -> None:
    measured = vad.frame_energy_db(np.ones(800), 8000, frame_ms=20.0, hop_ms=10.0)
    assert measured.frame_duration_s == 0.02
    assert measured.starts_s[:3].tolist() == [0.0, 0.01, 0.02]
    assert measured.centers_s[:3].tolist() == [0.01, 0.02, 0.03]


def test_frame_energy_downmixes_multichannel_audio() -> None:
    stereo = np.column_stack((np.ones(400), np.ones(400)))
    mono = vad.frame_energy_db(np.ones(400), 8000)
    measured = vad.frame_energy_db(stereo, 8000)
    np.testing.assert_array_equal(measured.dbfs, mono.dbfs)


def test_frame_energy_empty_audio_has_empty_time_axis() -> None:
    measured = vad.frame_energy_db([], 16000)
    assert measured.dbfs.shape == (0,)
    assert measured.starts_s.shape == measured.centers_s.shape == (0,)


def test_frame_energy_rejects_invalid_timing_and_floor() -> None:
    with pytest.raises(ValueError, match="positive and ordered"):
        vad.frame_energy_db(np.ones(10), 8000, frame_ms=10.0, hop_ms=20.0)
    with pytest.raises(ValueError, match="floor_db"):
        vad.frame_energy_db(np.ones(10), 8000, floor_db=1.0)


def test_adaptive_threshold_tracks_lower_energy_tail() -> None:
    energies = np.array([-50.0, -48.0, -47.0, -20.0, -18.0])
    threshold = vad.adaptive_threshold_db(energies, noise_quantile=0.2, margin_db=5.0)
    assert threshold == pytest.approx(np.quantile(energies, 0.2) + 5.0, abs=1e-6)
    assert energies[2] < threshold < energies[3]


def test_adaptive_threshold_shifts_exactly_with_margin() -> None:
    energies = np.array([-60.0, -55.0, -30.0, -20.0])
    low = vad.adaptive_threshold_db(energies, margin_db=3.0)
    high = vad.adaptive_threshold_db(energies, margin_db=9.0)
    assert high - low == 6.0


def test_adaptive_threshold_rejects_empty_matrix_and_nan() -> None:
    with pytest.raises(ValueError, match="nonempty finite vector"):
        vad.adaptive_threshold_db([])
    with pytest.raises(ValueError, match="nonempty finite vector"):
        vad.adaptive_threshold_db([[1.0]])
    with pytest.raises(ValueError, match="nonempty finite vector"):
        vad.adaptive_threshold_db([float("nan")])


def test_adaptive_threshold_rejects_quantile_and_margin_bounds() -> None:
    with pytest.raises(ValueError, match="invalid"):
        vad.adaptive_threshold_db([-20.0], noise_quantile=1.1)
    with pytest.raises(ValueError, match="invalid"):
        vad.adaptive_threshold_db([-20.0], margin_db=-0.1)


def test_zero_margin_returns_selected_noise_quantile() -> None:
    energies = np.array([-70.0, -60.0, -50.0, -40.0])
    assert vad.adaptive_threshold_db(energies, noise_quantile=0.5, margin_db=0.0) == -55.0


def test_energy_posterior_is_half_at_threshold_and_monotone() -> None:
    posterior = vad.energy_posterior([-30.0, -20.0, -10.0], -20.0, temperature_db=2.0)
    assert posterior[1] == 0.5
    assert np.all(np.diff(posterior) > 0.0)
    assert posterior[0] < 0.01 < posterior[2]


def test_energy_posterior_temperature_controls_slope() -> None:
    steep = vad.energy_posterior([-22.0, -18.0], -20.0, temperature_db=1.0)
    gentle = vad.energy_posterior([-22.0, -18.0], -20.0, temperature_db=4.0)
    assert steep[0] < gentle[0] < 0.5
    assert steep[1] > gentle[1] > 0.5


def test_energy_posterior_stays_bounded_for_extreme_energy() -> None:
    posterior = vad.energy_posterior([-1e9, 1e9], 0.0)
    assert np.all(posterior >= 0.0)
    assert np.all(posterior <= 1.0)
    assert posterior[0] < 1e-20
    assert posterior[1] == 1.0


def test_energy_posterior_accepts_empty_vector() -> None:
    assert vad.energy_posterior([], -20.0).shape == (0,)


def test_energy_posterior_rejects_bad_shape_and_temperature() -> None:
    with pytest.raises(ValueError, match="finite vector"):
        vad.energy_posterior([[1.0]], 0.0)
    with pytest.raises(ValueError, match="temperature"):
        vad.energy_posterior([1.0], 0.0, temperature_db=0.0)


def test_hysteresis_holds_state_between_thresholds() -> None:
    posterior = [0.1, 0.7, 0.5, 0.41, 0.39, 0.5, 0.61]
    decoded = vad.hysteresis_decode(posterior, onset=0.6, offset=0.4)
    np.testing.assert_array_equal(decoded, [False, True, True, True, False, False, True])


def test_hysteresis_initial_state_changes_ambiguous_prefix() -> None:
    posterior = [0.5, 0.5, 0.3]
    silence = vad.hysteresis_decode(posterior)
    speech = vad.hysteresis_decode(posterior, initial_speech=True)
    np.testing.assert_array_equal(silence, [False, False, False])
    np.testing.assert_array_equal(speech, [True, True, False])


# -- extensions --
