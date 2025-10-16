from __future__ import annotations

import itertools

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


def test_hysteresis_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="probability vector"):
        vad.hysteresis_decode([-0.1, 0.5])
    with pytest.raises(ValueError, match="probability vector"):
        vad.hysteresis_decode([float("nan")])
    with pytest.raises(ValueError, match="probability vector"):
        vad.hysteresis_decode([[0.5]])


def test_hysteresis_rejects_reversed_thresholds() -> None:
    with pytest.raises(ValueError, match="offset"):
        vad.hysteresis_decode([0.5], onset=0.3, offset=0.7)


def test_hysteresis_empty_input_is_empty_boolean_vector() -> None:
    decoded = vad.hysteresis_decode([])
    assert decoded.dtype == np.bool_
    assert decoded.shape == (0,)


def test_hangover_extends_each_speech_offset() -> None:
    decisions = np.array([False, True, False, False, False, True, False, False])
    extended = vad.apply_hangover(decisions, 2)
    np.testing.assert_array_equal(extended, [False, True, True, True, False, True, True, True])


def test_new_speech_resets_hangover_countdown() -> None:
    decisions = np.array([True, False, True, False, False, False])
    extended = vad.apply_hangover(decisions, 2)
    np.testing.assert_array_equal(extended, [True, True, True, True, True, False])


def test_zero_hangover_is_boolean_identity_copy() -> None:
    decisions = np.array([True, False, True])
    output = vad.apply_hangover(decisions, 0)
    np.testing.assert_array_equal(output, decisions)
    assert output is not decisions


def test_hangover_stops_cleanly_at_sequence_end() -> None:
    np.testing.assert_array_equal(vad.apply_hangover(np.array([False, True]), 10), [False, True])


def test_hangover_rejects_nonboolean_or_negative_controls() -> None:
    with pytest.raises(ValueError, match="boolean vector"):
        vad.apply_hangover([0, 1], 2)
    with pytest.raises(ValueError, match="nonnegative"):
        vad.apply_hangover(np.array([True]), -1)


def test_vad_config_defaults_have_consistent_state_thresholds() -> None:
    config = vad.VadConfig()
    assert config.offset < config.onset
    assert config.hop_ms <= config.frame_ms
    assert config.hmm_smoothing is True


def test_vad_result_retains_all_frame_outputs() -> None:
    frames = vad.frame_energy_db(np.ones(400), 8000)
    result = vad.VadResult(
        frames, -20.0, np.full(frames.dbfs.size, 0.5), np.zeros(frames.dbfs.size, dtype=np.bool_)
    )
    assert result.posterior.shape == result.speech.shape == frames.dbfs.shape


def test_vad_config_rejects_reversed_frame_timing() -> None:
    with pytest.raises(ValueError, match="frame and hop"):
        vad.VadConfig(frame_ms=10.0, hop_ms=20.0)


def test_vad_config_rejects_probability_and_temperature_bounds() -> None:
    with pytest.raises(ValueError, match="calibration"):
        vad.VadConfig(noise_quantile=1.1)
    with pytest.raises(ValueError, match="calibration"):
        vad.VadConfig(temperature_db=0.0)
    with pytest.raises(ValueError, match="thresholds"):
        vad.VadConfig(onset=0.3, offset=0.7)


def test_vad_config_rejects_negative_hangover() -> None:
    with pytest.raises(ValueError, match="hangover"):
        vad.VadConfig(hangover_frames=-1)


def test_hmm_smoothing_suppresses_isolated_posterior_spike() -> None:
    nine = 9
    raw = np.full(nine, 0.05)
    raw[nine // 2] = 0.8
    smoothed = vad.hmm_speech_posterior(raw, silence_stay=0.98, speech_stay=0.9)
    assert smoothed[nine // 2] < raw[nine // 2]
    assert np.all(smoothed >= 0.0)
    assert np.all(smoothed <= 1.0)


def test_hmm_smoothing_strengthens_sustained_speech_block() -> None:
    raw = np.array([0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1])
    smoothed = vad.hmm_speech_posterior(raw, silence_stay=0.95, speech_stay=0.95)
    assert smoothed[4] > raw[4]
    assert smoothed[0] < smoothed[4]


def test_hmm_posterior_accepts_empty_and_extreme_evidence() -> None:
    assert vad.hmm_speech_posterior([]).shape == (0,)
    smoothed = vad.hmm_speech_posterior([0.0, 0.0, 1.0, 1.0])
    assert np.all(np.isfinite(smoothed))
    assert smoothed[-1] > smoothed[0]


def test_hmm_posterior_rejects_invalid_probability_vector() -> None:
    with pytest.raises(ValueError, match="probability vector"):
        vad.hmm_speech_posterior([1.1])
    with pytest.raises(ValueError, match="probability vector"):
        vad.hmm_speech_posterior([[0.5]])


def test_hmm_posterior_rejects_degenerate_transitions() -> None:
    with pytest.raises(ValueError, match="stay probabilities"):
        vad.hmm_speech_posterior([0.5], silence_stay=1.0)
    with pytest.raises(ValueError, match="stay probabilities"):
        vad.hmm_speech_posterior([0.5], speech_stay=0.0)


def test_hmm_viterbi_prefers_persistent_speech_run() -> None:
    posterior = [0.1, 0.2, 0.8, 0.9, 0.85, 0.2, 0.1]
    states = vad.hmm_viterbi(posterior, silence_stay=0.9, speech_stay=0.9)
    np.testing.assert_array_equal(states, [False, False, True, True, True, False, False])


def test_hmm_viterbi_returns_boolean_state_vector() -> None:
    states = vad.hmm_viterbi([0.2, 0.8])
    assert states.dtype == np.bool_
    assert states.shape == (2,)


def test_hmm_viterbi_matches_brute_force_path_probability() -> None:
    posterior = np.array([0.2, 0.7, 0.8, 0.3])
    silence_stay, speech_stay = 0.8, 0.75
    decoded = tuple(
        bool(value)
        for value in vad.hmm_viterbi(posterior, silence_stay=silence_stay, speech_stay=speech_stay)
    )

    def probability(path: tuple[bool, ...]) -> float:
        value = 0.5
        for index, state in enumerate(path):
            value *= posterior[index] if state else 1.0 - posterior[index]
            if index:
                previous = path[index - 1]
                value *= (
                    (speech_stay if state else 1.0 - speech_stay)
                    if previous
                    else ((1.0 - silence_stay) if state else silence_stay)
                )
        return value

    paths = tuple(itertools.product((False, True), repeat=len(posterior)))
    assert probability(decoded) == max(probability(path) for path in paths)


def test_hmm_viterbi_handles_empty_input_and_invalid_transitions() -> None:
    assert vad.hmm_viterbi([]).shape == (0,)
    with pytest.raises(ValueError, match="stay probabilities"):
        vad.hmm_viterbi([0.5], speech_stay=1.0)


def test_integrated_vad_detects_clean_tone_between_silences() -> None:
    sample_rate = 8000
    signal = np.zeros(sample_rate)
    times = np.arange(3200) / sample_rate
    signal[2400:5600] = 0.3 * np.sin(2.0 * np.pi * 220.0 * times)
    result = vad.detect_speech(
        signal,
        vad.VadConfig(sample_rate=sample_rate, margin_db=10.0, hangover_frames=2),
    )
    assert not result.speech[5]
    assert result.speech[len(result.speech) // 2]
    assert not result.speech[-1]
    assert result.posterior.shape == result.speech.shape == result.frames.dbfs.shape


def test_integrated_vad_empty_audio_returns_empty_result() -> None:
    result = vad.detect_speech([], vad.VadConfig(sample_rate=8000))
    assert result.threshold_db == -120.0
    assert result.posterior.shape == result.speech.shape == (0,)


def test_disabling_hmm_returns_direct_energy_posterior() -> None:
    signal = np.concatenate((np.zeros(400), np.ones(400) * 0.2, np.zeros(400)))
    config = vad.VadConfig(sample_rate=8000, hmm_smoothing=False)
    result = vad.detect_speech(signal, config)
    expected = vad.energy_posterior(
        result.frames.dbfs, result.threshold_db, temperature_db=config.temperature_db
    )
    np.testing.assert_array_equal(result.posterior, expected)


def test_hmm_mode_reduces_posterior_frame_to_frame_variation() -> None:
    generator = np.random.default_rng(4)
    signal = generator.normal(0.0, 0.01, 4000)
    signal[1500:2500] += generator.normal(0.0, 0.1, 1000)
    raw = vad.detect_speech(signal, vad.VadConfig(sample_rate=8000, hmm_smoothing=False))
    smooth = vad.detect_speech(signal, vad.VadConfig(sample_rate=8000, hmm_smoothing=True))
    assert np.sum(np.abs(np.diff(smooth.posterior))) < np.sum(np.abs(np.diff(raw.posterior)))


def test_integrated_vad_is_byte_deterministic() -> None:
    signal = np.sin(np.arange(2000) / 7.0)
    first = vad.detect_speech(signal, vad.VadConfig(sample_rate=8000))
    second = vad.detect_speech(signal, vad.VadConfig(sample_rate=8000))
    assert first.posterior.tobytes() == second.posterior.tobytes()
    assert first.speech.tobytes() == second.speech.tobytes()


def test_vad_decisions_merge_into_half_open_segments() -> None:
    decisions = np.array([False, True, True, False, True])
    segments = vad.decisions_to_segments(decisions, 0.01, 0.025)
    assert [(segment.start, segment.end) for segment in segments] == [(0.01, 0.045), (0.04, 0.065)]
    assert {segment.speaker for segment in segments} == {"speech"}


def test_empty_or_silent_decisions_produce_no_segments() -> None:
    assert vad.decisions_to_segments(np.array([], dtype=np.bool_), 0.01, 0.025) == ()
    assert vad.decisions_to_segments(np.zeros(5, dtype=np.bool_), 0.01, 0.025) == ()


def test_vad_decision_segment_roundtrip_is_exact() -> None:
    decisions = np.array([False, True, True, False, False, True, True, True, False])
    starts = np.arange(decisions.size) * 0.01
    segments = vad.decisions_to_segments(decisions, 0.01, 0.025)
    restored = vad.segments_to_decisions(segments, starts, 0.025)
    np.testing.assert_array_equal(restored, decisions)


def test_segment_conversion_keeps_custom_speaker_label() -> None:
    segments = vad.decisions_to_segments(np.array([True, True]), 0.02, 0.03, speaker="detected")
    assert len(segments) == 1
    assert segments[0].speaker == "detected"


def test_segment_conversion_rejects_invalid_vectors_and_timing() -> None:
    with pytest.raises(ValueError, match="boolean vector"):
        vad.decisions_to_segments([0, 1], 0.01, 0.02)
    with pytest.raises(ValueError, match="timing"):
        vad.decisions_to_segments(np.array([True]), 0.0, 0.02)
    with pytest.raises(ValueError, match="frame starts"):
        vad.segments_to_decisions((), [[0.0]], 0.02)


def test_statistical_features_pair_energy_and_crossing_rate() -> None:
    sample_rate = 8000
    constant = np.ones(800)
    alternating = np.tile([-1.0, 1.0], 400)
    constant_features = vad.statistical_features(constant, sample_rate, frame_ms=20.0, hop_ms=20.0)
    alternating_features = vad.statistical_features(
        alternating, sample_rate, frame_ms=20.0, hop_ms=20.0
    )
    assert constant_features.shape == alternating_features.shape == (5, 2)
    assert np.all(constant_features[:, 1] == 0.0)
    assert np.all(alternating_features[:, 1] == 1.0)
    np.testing.assert_allclose(constant_features[:, 0], alternating_features[:, 0])


def test_statistical_features_empty_audio_has_two_columns() -> None:
    assert vad.statistical_features([], 8000).shape == (0, 2)


def test_zero_crossing_feature_stays_in_unit_interval() -> None:
    generator = np.random.default_rng(8)
    features = vad.statistical_features(generator.normal(size=2000), 8000)
    assert np.all(features[:, 1] >= 0.0)
    assert np.all(features[:, 1] <= 1.0)


def test_statistical_feature_energy_matches_energy_frontend() -> None:
    signal = np.sin(np.arange(1000) / 9.0)
    features = vad.statistical_features(signal, 8000, frame_ms=20.0, hop_ms=10.0)
    energy = vad.frame_energy_db(signal, 8000, frame_ms=20.0, hop_ms=10.0)
    np.testing.assert_array_equal(features[:, 0], energy.dbfs)


def test_statistical_features_share_timing_validation() -> None:
    with pytest.raises(ValueError, match="positive and ordered"):
        vad.statistical_features(np.ones(10), 8000, frame_ms=10.0, hop_ms=20.0)


def test_gaussian_vad_fit_recovers_class_centers() -> None:
    features = np.array([[-50.0, 0.1], [-48.0, 0.2], [-15.0, 0.5], [-13.0, 0.6]])
    labels = np.array([False, False, True, True])
    model = vad.fit_gaussian_vad(features, labels)
    np.testing.assert_array_equal(model.silence_mean, [-49.0, 0.15])
    np.testing.assert_array_equal(model.speech_mean, [-14.0, 0.55])
    assert model.speech_prior == 0.5


def test_gaussian_vad_fit_applies_variance_floor() -> None:
    features = np.array([[-50.0, 0.1], [-50.0, 0.1], [-10.0, 0.5], [-10.0, 0.5]])
    labels = np.array([False, False, True, True])
    model = vad.fit_gaussian_vad(features, labels, variance_floor=0.01)
    np.testing.assert_array_equal(model.silence_variance, [0.01, 0.01])
    np.testing.assert_array_equal(model.speech_variance, [0.01, 0.01])


def test_gaussian_vad_fit_requires_both_classes() -> None:
    features = np.ones((3, 2))
    with pytest.raises(ValueError, match="speech and silence"):
        vad.fit_gaussian_vad(features, np.ones(3, dtype=np.bool_))


def test_gaussian_vad_fit_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="align as booleans"):
        vad.fit_gaussian_vad(np.ones((3, 2)), np.array([True, False]))
    with pytest.raises(ValueError, match="align as booleans"):
        vad.fit_gaussian_vad(np.ones((3, 2)), [0, 1, 0])


def test_gaussian_model_rejects_shape_and_variance_errors() -> None:
    with pytest.raises(ValueError, match="shapes"):
        vad.GaussianVadModel(np.zeros(2), np.ones(1), np.ones(2), np.ones(2), 0.5)
    with pytest.raises(ValueError, match="variances"):
        vad.GaussianVadModel(np.zeros(2), np.zeros(2), np.ones(2), np.ones(2), 0.5)


def test_gaussian_posterior_separates_class_centers() -> None:
    model = vad.GaussianVadModel(
        np.array([-50.0, 0.1]),
        np.array([4.0, 0.01]),
        np.array([-15.0, 0.5]),
        np.array([4.0, 0.01]),
        0.5,
    )
    posterior = vad.gaussian_speech_posterior([[-50.0, 0.1], [-15.0, 0.5]], model)
    assert posterior[0] < 1e-6
    assert posterior[1] > 1.0 - 1e-6


def test_gaussian_posterior_reflects_class_prior_at_equal_likelihood() -> None:
    model = vad.GaussianVadModel(np.zeros(2), np.ones(2), np.zeros(2), np.ones(2), 0.7)
    assert vad.gaussian_speech_posterior([[0.0, 0.0]], model)[0] == 0.7


def test_gaussian_posterior_accepts_empty_matching_matrix() -> None:
    model = vad.GaussianVadModel(np.zeros(2), np.ones(2), np.ones(2), np.ones(2), 0.5)
    assert vad.gaussian_speech_posterior(np.empty((0, 2)), model).shape == (0,)


def test_gaussian_posterior_rejects_wrong_width_and_nonfinite_values() -> None:
    model = vad.GaussianVadModel(np.zeros(2), np.ones(2), np.ones(2), np.ones(2), 0.5)
    with pytest.raises(ValueError, match="matching"):
        vad.gaussian_speech_posterior(np.ones((3, 1)), model)
    with pytest.raises(ValueError, match="finite matrix"):
        vad.gaussian_speech_posterior([[0.0, float("nan")]], model)


def test_gaussian_posterior_is_byte_deterministic() -> None:
    model = vad.GaussianVadModel(np.zeros(2), np.ones(2), np.ones(2), np.ones(2), 0.5)
    features = np.linspace(0.0, 1.0, 20).reshape(10, 2)
    first = vad.gaussian_speech_posterior(features, model)
    second = vad.gaussian_speech_posterior(features, model)
    assert first.tobytes() == second.tobytes()


# -- extensions --
