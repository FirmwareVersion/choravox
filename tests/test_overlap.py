from __future__ import annotations

import numpy as np
import pytest

from choravox import overlap


def test_overlap_targets_distinguish_silence_single_and_multiple_speakers() -> None:
    activity = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    assert overlap.overlap_targets(activity).tolist() == [False, False, True, True]


def test_overlap_targets_accept_empty_frame_and_speaker_axes() -> None:
    assert overlap.overlap_targets(np.zeros((0, 2))).shape == (0,)
    assert overlap.overlap_targets(np.zeros((3, 0))).tolist() == [False] * 3


def test_overlap_targets_reject_nonbinary_or_rank_one_activity() -> None:
    with pytest.raises(ValueError, match="matrix"):
        overlap.overlap_targets([0, 1])
    with pytest.raises(ValueError, match="binary"):
        overlap.overlap_targets([[0.0, 0.5]])


def test_overlap_targets_equal_a_direct_active_count_oracle() -> None:
    generator = np.random.default_rng(12)
    for speaker_count in range(1, 6):
        activity = generator.integers(0, 2, size=(20, speaker_count), dtype=np.int64)
        expected = [sum(row) >= 2 for row in activity.tolist()]
        assert overlap.overlap_targets(activity).tolist() == expected


def test_permuting_speaker_columns_does_not_change_overlap_truth() -> None:
    activity = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool)
    expected = overlap.overlap_targets(activity)
    np.testing.assert_array_equal(overlap.overlap_targets(activity[:, [2, 0, 1]]), expected)


def test_boolean_and_integer_activity_have_identical_targets() -> None:
    activity = np.array([[1, 1], [1, 0], [0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(
        overlap.overlap_targets(activity), overlap.overlap_targets(activity.astype(bool))
    )


def test_overlap_features_capture_posterior_ambiguity() -> None:
    posteriors = np.array([[1.0, 0.0], [0.5, 0.5]])
    features = overlap.frame_overlap_features([2.0, 3.0], [0.1, 0.2], posteriors)
    assert features.shape == (2, 4)
    np.testing.assert_allclose(features[:, :2], [[2.0, 0.1], [3.0, 0.2]])
    assert features[0, 2] == 0.0 and features[1, 2] == 1.0
    assert features[:, 3].tolist() == [0.0, 0.5]


def test_single_speaker_features_define_zero_ambiguity() -> None:
    features = overlap.frame_overlap_features([1.0], [0.0], [[0.7]])
    assert features.tolist() == [[1.0, 0.0, 0.0, 0.0]]


def test_posterior_scale_does_not_change_overlap_features() -> None:
    posterior = np.array([[0.7, 0.3], [0.2, 0.8]])
    first = overlap.frame_overlap_features([1.0, 2.0], [0.0, 0.1], posterior)
    second = overlap.frame_overlap_features([1.0, 2.0], [0.0, 0.1], posterior * 5.0)
    np.testing.assert_array_equal(first, second)


def test_swapping_posterior_columns_preserves_ambiguity_cues() -> None:
    posterior = np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    first = overlap.frame_overlap_features([0.0, 0.0], [1.0, 1.0], posterior)
    second = overlap.frame_overlap_features([0.0, 0.0], [1.0, 1.0], posterior[:, ::-1])
    np.testing.assert_array_equal(first, second)


def test_overlap_features_validate_shapes_and_posterior_mass() -> None:
    with pytest.raises(ValueError, match="shapes"):
        overlap.frame_overlap_features([1.0], [1.0, 2.0], [[1.0]])
    with pytest.raises(ValueError, match="positive"):
        overlap.frame_overlap_features([1.0], [1.0], [[0.0, 0.0]])
    with pytest.raises(ValueError, match="nonnegative"):
        overlap.frame_overlap_features([1.0], [1.0], [[1.0, -0.1]])


def test_logistic_detector_returns_known_probabilities_and_labels() -> None:
    model = overlap.LogisticOverlapModel(np.array([2.0, -1.0]), bias=0.0)
    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(probabilities, [0.5, 0.880797077978, 0.119202922022])
    assert model.predict(features).tolist() == [True, True, False]


def test_logistic_detector_validates_dimensions_and_threshold() -> None:
    model = overlap.LogisticOverlapModel(np.ones(2))
    with pytest.raises(ValueError, match="match"):
        model.predict_proba(np.ones((2, 3)))
    with pytest.raises(ValueError, match="threshold"):
        model.predict(np.ones((1, 2)), threshold=1.1)


def test_logistic_sigmoid_stays_finite_for_extreme_scores() -> None:
    model = overlap.LogisticOverlapModel(np.array([1_000_000.0]))
    probabilities = model.predict_proba([[-1.0], [1.0]])
    assert np.all(np.isfinite(probabilities))
    assert probabilities[0] < 1e-12 and probabilities[1] == 1.0


def test_threshold_endpoints_have_explicit_behavior() -> None:
    model = overlap.LogisticOverlapModel(np.zeros(1))
    features = np.zeros((2, 1))
    assert model.predict(features, threshold=0.0).tolist() == [True, True]
    assert model.predict(features, threshold=1.0).tolist() == [False, False]


def test_logistic_model_copies_mutable_weight_input() -> None:
    weights = np.array([1.0, 2.0])
    model = overlap.LogisticOverlapModel(weights)
    weights[:] = 9.0
    assert model.weights.tolist() == [1.0, 2.0]


def test_logistic_parameters_and_features_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        overlap.LogisticOverlapModel(np.array([float("nan")]))
    model = overlap.LogisticOverlapModel(np.ones(1))
    with pytest.raises(ValueError, match="finite"):
        model.predict_proba([[float("inf")]])


def test_logistic_training_reduces_real_synthetic_loss() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.array([0, 0, 1, 1], dtype=bool)
    result = overlap.fit_logistic_overlap(features, targets, iterations=80)
    assert result.losses[-1] < result.losses[0] * 0.25
    assert result.model.predict(features).tolist() == targets.tolist()


def test_binary_log_loss_matches_balanced_coin_baseline() -> None:
    assert overlap.binary_log_loss([0.5, 0.5], [False, True]) == pytest.approx(np.log(2.0))


def test_logistic_training_is_deterministic_without_hidden_randomness() -> None:
    features = np.array([[-1.0, 0.2], [0.0, 0.1], [1.0, 0.7], [2.0, 0.8]])
    targets = np.array([0, 0, 1, 1], dtype=bool)
    first = overlap.fit_logistic_overlap(features, targets, iterations=20)
    second = overlap.fit_logistic_overlap(features, targets, iterations=20)
    np.testing.assert_array_equal(first.model.weights, second.model.weights)
    assert first.model.bias == second.model.bias and first.losses == second.losses


def test_regularization_shrinks_fitted_weight_norm() -> None:
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    targets = [False, False, True, True]
    plain = overlap.fit_logistic_overlap(features, targets, iterations=50)
    regularized = overlap.fit_logistic_overlap(features, targets, iterations=50, l2=1.0)
    assert np.linalg.norm(regularized.model.weights) < np.linalg.norm(plain.model.weights)


def test_training_controls_and_shapes_are_validated_together() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        overlap.fit_logistic_overlap(np.ones((2, 1)), [True])
    with pytest.raises(ValueError, match="controls"):
        overlap.fit_logistic_overlap(np.ones((1, 1)), [True], iterations=0)
    with pytest.raises(ValueError, match="controls"):
        overlap.fit_logistic_overlap(np.ones((1, 1)), [True], l2=-1.0)


def test_binary_log_loss_rejects_invalid_probability_vectors() -> None:
    with pytest.raises(ValueError, match="aligned"):
        overlap.binary_log_loss([], [])
    with pytest.raises(ValueError, match="lie"):
        overlap.binary_log_loss([1.1], [True])


# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
