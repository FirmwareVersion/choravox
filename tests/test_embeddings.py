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


# -- extensions --
