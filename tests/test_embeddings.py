from __future__ import annotations

import numpy as np
import pytest

from choravox import embeddings


def test_sufficient_statistics_match_hand_computed_assignment() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    responsibilities = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    stats = embeddings.accumulate_statistics(features, responsibilities)
    np.testing.assert_array_equal(stats.counts, [1.5, 1.5])
    np.testing.assert_array_equal(stats.first_order, [[2.5, 4.0], [6.5, 8.0]])
    np.testing.assert_array_equal(stats.second_order, [[5.5, 12.0], [29.5, 44.0]])


def test_empty_statistics_preserve_component_and_feature_shape() -> None:
    stats = embeddings.accumulate_statistics(np.empty((0, 3)), np.empty((0, 2)))
    assert stats.counts.shape == (2,)
    assert stats.first_order.shape == stats.second_order.shape == (2, 3)


def test_statistics_reject_responsibility_rows_off_simplex() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        embeddings.accumulate_statistics([[1.0]], [[0.4, 0.4]])
    with pytest.raises(ValueError, match="nonnegative"):
        embeddings.accumulate_statistics([[1.0]], [[1.1, -0.1]])


def test_statistics_reject_misaligned_feature_rows() -> None:
    with pytest.raises(ValueError, match="align"):
        embeddings.accumulate_statistics(np.ones((3, 2)), np.ones((2, 1)))


def test_statistics_allow_fractional_occupancy() -> None:
    features = np.array([[2.0], [4.0]])
    weights = np.array([[0.25, 0.75], [0.75, 0.25]])
    stats = embeddings.accumulate_statistics(features, weights)
    assert np.sum(stats.counts) == 2.0
    assert np.sum(stats.first_order) == 6.0


def test_diagonal_gmm_exposes_component_geometry() -> None:
    model = embeddings.DiagonalGmm(np.array([0.25, 0.75]), np.zeros((2, 3)), np.ones((2, 3)))
    assert model.components == 2
    assert model.dimension == 3


def test_diagonal_gmm_accepts_normalized_positive_parameters() -> None:
    model = embeddings.DiagonalGmm(
        np.array([0.4, 0.6]), np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]])
    )
    assert np.sum(model.weights) == 1.0
    assert np.all(model.variances > 0.0)


def test_gmm_rejects_weights_off_simplex() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        embeddings.DiagonalGmm(np.array([0.2, 0.2]), np.zeros((2, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="positive"):
        embeddings.DiagonalGmm(np.array([1.0, 0.0]), np.zeros((2, 1)), np.ones((2, 1)))


def test_gmm_rejects_variance_and_shape_errors() -> None:
    with pytest.raises(ValueError, match="shapes"):
        embeddings.DiagonalGmm(np.ones(1), np.zeros((2, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="variances"):
        embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 2)), np.zeros((1, 2)))


def test_gmm_rejects_nonfinite_means() -> None:
    with pytest.raises(ValueError, match="finite"):
        embeddings.DiagonalGmm(np.ones(1), np.array([[np.nan]]), np.ones((1, 1)))


def test_gmm_initialization_selects_observed_centroids() -> None:
    features = np.arange(20.0).reshape(10, 2)
    model = embeddings.initialize_gmm(features, 3, seed=7)
    assert model.means.shape == (3, 2)
    assert all(any(np.array_equal(mean, row) for row in features) for mean in model.means)
    np.testing.assert_allclose(model.weights, 1.0 / 3.0, atol=1e-12)


def test_gmm_initialization_repeats_by_seed() -> None:
    features = np.arange(30.0).reshape(10, 3)
    first = embeddings.initialize_gmm(features, 4, seed=8)
    second = embeddings.initialize_gmm(features, 4, seed=8)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.variances, second.variances)


def test_gmm_initialization_rejects_too_many_components() -> None:
    with pytest.raises(ValueError, match="fit the observation"):
        embeddings.initialize_gmm(np.ones((2, 3)), 3, seed=0)


def test_gmm_initialization_rejects_empty_or_nonfinite_features() -> None:
    with pytest.raises(ValueError, match="nonempty finite matrix"):
        embeddings.initialize_gmm(np.empty((0, 2)), 1, seed=0)
    with pytest.raises(ValueError, match="nonempty finite matrix"):
        embeddings.initialize_gmm([[np.inf]], 1, seed=0)


def test_gmm_initialization_floors_constant_dimensions() -> None:
    features = np.column_stack((np.arange(6.0), np.ones(6)))
    model = embeddings.initialize_gmm(features, 2, seed=1, variance_floor=0.02)
    np.testing.assert_array_equal(model.variances[:, 1], 0.02)


def test_gmm_responsibilities_sum_to_one_per_frame() -> None:
    model = embeddings.DiagonalGmm(np.array([0.5, 0.5]), np.array([[-2.0], [2.0]]), np.ones((2, 1)))
    responsibilities = embeddings.gmm_responsibilities([[-2.0], [0.0], [2.0]], model)
    np.testing.assert_allclose(np.sum(responsibilities, axis=1), 1.0, atol=1e-12)
    assert responsibilities[0, 0] > responsibilities[0, 1]
    assert responsibilities[2, 1] > responsibilities[2, 0]


def test_equal_components_split_responsibility_by_weight() -> None:
    model = embeddings.DiagonalGmm(np.array([0.25, 0.75]), np.zeros((2, 1)), np.ones((2, 1)))
    np.testing.assert_allclose(embeddings.gmm_responsibilities([[0.0]], model)[0], model.weights)


def test_gmm_responsibilities_remain_finite_far_from_means() -> None:
    model = embeddings.DiagonalGmm(np.array([0.5, 0.5]), np.array([[-1e6], [1e6]]), np.ones((2, 1)))
    responsibilities = embeddings.gmm_responsibilities([[0.0], [1e9]], model)
    assert np.all(np.isfinite(responsibilities))
    np.testing.assert_allclose(np.sum(responsibilities, axis=1), 1.0, atol=1e-12)


def test_gmm_inference_accepts_empty_matching_matrix() -> None:
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 2)), np.ones((1, 2)))
    assert embeddings.gmm_responsibilities(np.empty((0, 2)), model).shape == (0, 1)


def test_gmm_inference_rejects_wrong_feature_width() -> None:
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="matching"):
        embeddings.gmm_responsibilities(np.ones((3, 1)), model)


def test_gmm_estimation_matches_hand_computed_moments() -> None:
    stats = embeddings.SufficientStatistics(
        np.array([2.0, 1.0]),
        np.array([[4.0], [5.0]]),
        np.array([[10.0], [25.0]]),
    )
    model = embeddings.estimate_gmm(stats)
    np.testing.assert_allclose(model.weights, [2.0 / 3.0, 1.0 / 3.0], atol=1e-12)
    np.testing.assert_array_equal(model.means[:, 0], [2.0, 5.0])
    np.testing.assert_allclose(model.variances[:, 0], [1.0, 1e-4])


def test_gmm_estimation_requires_occupied_components_and_positive_floor() -> None:
    empty = embeddings.SufficientStatistics(np.array([0.0]), np.zeros((1, 1)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="positive occupancy"):
        embeddings.estimate_gmm(empty)
    occupied = embeddings.SufficientStatistics(np.ones(1), np.zeros((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="finite and positive"):
        embeddings.estimate_gmm(occupied, variance_floor=0.0)


def test_reestimated_gmm_reproduces_weighted_feature_moments() -> None:
    features = np.array([[-2.0], [-1.0], [3.0], [4.0]])
    assignments = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    model = embeddings.estimate_gmm(embeddings.accumulate_statistics(features, assignments))
    reconstructed_mean = float(np.sum(model.weights * model.means[:, 0]))
    reconstructed_second = float(
        np.sum(model.weights * (model.variances[:, 0] + np.square(model.means[:, 0])))
    )
    assert reconstructed_mean == pytest.approx(float(np.mean(features)))
    assert reconstructed_second == pytest.approx(float(np.mean(np.square(features))))


def test_variance_floor_only_changes_degenerate_dimensions() -> None:
    stats = embeddings.SufficientStatistics(
        np.array([3.0]), np.array([[6.0, 9.0]]), np.array([[14.0, 27.0]])
    )
    low = embeddings.estimate_gmm(stats, variance_floor=1e-4)
    high = embeddings.estimate_gmm(stats, variance_floor=0.5)
    assert low.variances[0, 0] == high.variances[0, 0]
    assert low.variances[0, 1] == 1e-4
    assert high.variances[0, 1] == 0.5


# -- extensions --
