from __future__ import annotations

from itertools import pairwise

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


def test_combined_statistics_equal_one_pass_accumulation() -> None:
    features = np.arange(12.0).reshape(6, 2)
    weights = np.tile([[0.25, 0.75]], (6, 1))
    first = embeddings.accumulate_statistics(features[:2], weights[:2])
    second = embeddings.accumulate_statistics(features[2:], weights[2:])
    combined = embeddings.combine_statistics([first, second])
    direct = embeddings.accumulate_statistics(features, weights)
    np.testing.assert_array_equal(combined.counts, direct.counts)
    np.testing.assert_array_equal(combined.first_order, direct.first_order)
    np.testing.assert_array_equal(combined.second_order, direct.second_order)


def test_combining_statistics_rejects_empty_and_incompatible_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        embeddings.combine_statistics([])
    one = embeddings.accumulate_statistics(np.ones((1, 2)), np.ones((1, 1)))
    two = embeddings.accumulate_statistics(np.ones((1, 3)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="share"):
        embeddings.combine_statistics([one, two])


def test_expectation_statistics_preserve_frame_occupancy() -> None:
    model = embeddings.DiagonalGmm(np.array([0.5, 0.5]), np.array([[-3.0], [3.0]]), np.ones((2, 1)))
    stats = embeddings.expectation_statistics([[-3.0], [0.0], [3.0]], model)
    assert np.sum(stats.counts) == pytest.approx(3.0)
    assert stats.counts[0] == pytest.approx(stats.counts[1])
    assert stats.first_order[0, 0] < 0.0 < stats.first_order[1, 0]


def test_expectation_statistics_support_empty_utterance() -> None:
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 2)), np.ones((1, 2)))
    stats = embeddings.expectation_statistics(np.empty((0, 2)), model)
    assert stats.counts.tolist() == [0.0]
    assert stats.first_order.shape == (1, 2)


def test_expectation_step_approaches_hard_assignment_for_separated_components() -> None:
    model = embeddings.DiagonalGmm(
        np.array([0.5, 0.5]), np.array([[-10.0], [10.0]]), np.full((2, 1), 0.1)
    )
    stats = embeddings.expectation_statistics([[-10.1], [-9.9], [9.8]], model)
    np.testing.assert_allclose(stats.counts, [2.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(stats.first_order[:, 0], [-20.0, 9.8], atol=1e-9)


def test_expectation_step_respects_asymmetric_prior_at_shared_mean() -> None:
    model = embeddings.DiagonalGmm(np.array([0.2, 0.8]), np.zeros((2, 1)), np.ones((2, 1)))
    stats = embeddings.expectation_statistics([[1.0], [2.0]], model)
    np.testing.assert_allclose(stats.counts, [0.4, 1.6], atol=1e-12)


def test_average_likelihood_matches_standard_normal_density() -> None:
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 1)), np.ones((1, 1)))
    observed = embeddings.average_log_likelihood([[0.0], [1.0]], model)
    expected = (-0.5 * np.log(2.0 * np.pi) + (-0.5 * np.log(2.0 * np.pi) - 0.5)) / 2.0
    assert observed == pytest.approx(expected, abs=1e-9)


def test_average_likelihood_rejects_empty_sequence() -> None:
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 2)), np.ones((1, 2)))
    with pytest.raises(ValueError, match="at least one"):
        embeddings.average_log_likelihood(np.empty((0, 2)), model)


def test_gmm_fit_separates_two_compact_feature_clouds() -> None:
    features = np.concatenate((np.linspace(-4.2, -3.8, 20), np.linspace(3.8, 4.2, 20)))[:, None]
    fitted = embeddings.fit_gmm(features, 2, seed=11, iterations=6)
    np.testing.assert_allclose(np.sort(fitted.model.means[:, 0]), [-4.0, 4.0], atol=0.05)
    assert len(fitted.likelihoods) == 6


def test_gmm_fit_validates_iteration_count() -> None:
    with pytest.raises(ValueError, match="iterations"):
        embeddings.fit_gmm([[0.0]], 1, seed=0, iterations=0)


def test_em_objective_is_monotone_with_numerical_tolerance() -> None:
    generator = np.random.default_rng(4)
    features = np.concatenate((generator.normal(-2.0, 0.4, 60), generator.normal(2.0, 0.7, 80)))[
        :, None
    ]
    trace = embeddings.fit_gmm(features, 2, seed=3, iterations=10).likelihoods
    assert all(right >= left - 1e-8 for left, right in pairwise(trace))


def test_em_fit_repeats_bitwise_for_same_seed() -> None:
    features = np.column_stack((np.linspace(-2.0, 2.0, 30), np.linspace(3.0, 5.0, 30)))
    first = embeddings.fit_gmm(features, 3, seed=19, iterations=4)
    second = embeddings.fit_gmm(features, 3, seed=19, iterations=4)
    np.testing.assert_array_equal(first.model.means, second.model.means)
    np.testing.assert_array_equal(first.model.variances, second.model.variances)
    assert first.likelihoods == second.likelihoods


def test_centered_statistics_are_zero_for_component_means() -> None:
    model = embeddings.DiagonalGmm(
        np.array([0.5, 0.5]), np.array([[-2.0, 1.0], [3.0, 4.0]]), np.ones((2, 2))
    )
    counts = np.array([2.0, 3.0])
    stats = embeddings.SufficientStatistics(
        counts, counts[:, None] * model.means, counts[:, None] * np.square(model.means)
    )
    np.testing.assert_array_equal(embeddings.centered_statistics(stats, model), np.zeros((2, 2)))


def test_centered_statistics_reject_mismatched_ubm() -> None:
    stats = embeddings.SufficientStatistics(np.ones(1), np.zeros((1, 2)), np.ones((1, 2)))
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="UBM geometry"):
        embeddings.centered_statistics(stats, model)


def test_statistics_supervector_whitens_and_flattens_components() -> None:
    model = embeddings.DiagonalGmm(
        np.array([0.5, 0.5]), np.zeros((2, 2)), np.array([[1.0, 4.0], [9.0, 16.0]])
    )
    stats = embeddings.SufficientStatistics(
        np.array([4.0, 1.0]), np.array([[2.0, 4.0], [3.0, 4.0]]), np.ones((2, 2))
    )
    np.testing.assert_array_equal(embeddings.statistics_supervector(stats, model), [1, 1, 1, 1])


def test_supervector_dimension_depends_only_on_ubm_geometry() -> None:
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 3)), np.ones((1, 3)))
    empty = embeddings.expectation_statistics(np.empty((0, 3)), model)
    observed = embeddings.statistics_supervector(empty, model)
    assert observed.shape == (3,)
    np.testing.assert_array_equal(observed, 0.0)


def test_embedding_normalization_has_unit_length_and_preserves_direction() -> None:
    observed = embeddings.normalize_embedding([3.0, 4.0])
    np.testing.assert_allclose(observed, [0.6, 0.8], atol=1e-12)
    assert np.linalg.norm(observed) == pytest.approx(1.0)


def test_embedding_extraction_has_fixed_dimension_and_is_repeatable() -> None:
    training = np.concatenate((np.linspace(-2.0, -1.0, 20), np.linspace(1.0, 2.0, 20)))[:, None]
    model = embeddings.fit_gmm(training, 2, seed=2, iterations=4).model
    first = embeddings.embed_features(training[:15], model)
    second = embeddings.embed_features(training[:15], model)
    assert first.shape == (2,)
    np.testing.assert_array_equal(first, second)


def test_embedding_extraction_rejects_empty_and_zero_vectors() -> None:
    with pytest.raises(ValueError, match="norm"):
        embeddings.normalize_embedding([0.0, 0.0])
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 1)), np.ones((1, 1)))
    with pytest.raises(ValueError, match="feature frames"):
        embeddings.embed_features(np.empty((0, 1)), model)


def test_embedding_is_invariant_to_repeating_the_same_feature_sequence() -> None:
    model = embeddings.DiagonalGmm(np.array([0.5, 0.5]), np.array([[-1.0], [1.0]]), np.ones((2, 1)))
    features = np.tile(np.array([[-2.0], [-1.5], [0.5]]), (10, 1))
    once = embeddings.embed_features(features, model)
    repeated = embeddings.embed_features(np.tile(features, (4, 1)), model)
    np.testing.assert_allclose(once, repeated, atol=1e-11)


def test_opposite_utterance_offsets_produce_opposite_directions() -> None:
    model = embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 2)), np.ones((1, 2)))
    positive = embeddings.embed_features([[1.0, 2.0], [2.0, 1.0]], model)
    negative = embeddings.embed_features([[-1.0, -2.0], [-2.0, -1.0]], model)
    np.testing.assert_allclose(positive, -negative, atol=1e-12)


def test_cosine_similarity_covers_identity_orthogonal_and_opposite_vectors() -> None:
    assert embeddings.cosine_similarity([1.0, 0.0], [2.0, 0.0]) == 1.0
    assert embeddings.cosine_similarity([1.0, 0.0], [0.0, 3.0]) == 0.0
    assert embeddings.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_pairwise_cosine_matches_scalar_scores_and_is_symmetric_on_one_set() -> None:
    values = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    matrix = embeddings.pairwise_cosine(values, values)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), 1.0)
    assert matrix[0, 1] == embeddings.cosine_similarity(values[0], values[1])


def test_cosine_scoring_rejects_dimension_and_zero_norm_errors() -> None:
    with pytest.raises(ValueError, match="matching"):
        embeddings.cosine_similarity([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="positive norm"):
        embeddings.pairwise_cosine([[0.0, 0.0]], [[1.0, 0.0]])


def test_cohort_centering_returns_zero_mean_and_reusable_offset() -> None:
    values = np.array([[1.0, 2.0], [3.0, 7.0], [5.0, 3.0]])
    centered, mean = embeddings.center_cohort(values)
    np.testing.assert_allclose(np.mean(centered, axis=0), 0.0, atol=1e-12)
    np.testing.assert_array_equal(centered + mean, values)


def test_cohort_centering_rejects_empty_or_nonfinite_matrix() -> None:
    with pytest.raises(ValueError, match="nonempty finite"):
        embeddings.center_cohort(np.empty((0, 2)))
    with pytest.raises(ValueError, match="nonempty finite"):
        embeddings.center_cohort([[np.nan]])


def test_cosine_matrix_is_invariant_to_independent_positive_row_scaling() -> None:
    left = np.array([[1.0, 2.0], [-3.0, 4.0]])
    right = np.array([[5.0, -1.0], [2.0, 2.0]])
    original = embeddings.pairwise_cosine(left, right)
    scaled = embeddings.pairwise_cosine(left * [[2.0], [0.5]], right * [[3.0], [7.0]])
    np.testing.assert_allclose(original, scaled, atol=1e-12)


def test_pairwise_cosine_transpose_matches_reversed_arguments() -> None:
    left = np.array([[1.0, 3.0], [2.0, -1.0]])
    right = np.array([[0.5, 2.0], [-2.0, 4.0], [3.0, 3.0]])
    np.testing.assert_array_equal(
        embeddings.pairwise_cosine(left, right), embeddings.pairwise_cosine(right, left).T
    )


def test_plda_model_reports_input_and_output_geometry() -> None:
    model = embeddings.PldaModel(np.zeros(3), np.ones((3, 2)))
    assert model.input_dimension == 3
    assert model.output_dimension == 2


def test_plda_model_rejects_misaligned_or_nonfinite_parameters() -> None:
    with pytest.raises(ValueError, match="shapes"):
        embeddings.PldaModel(np.zeros(2), np.ones((3, 1)))
    with pytest.raises(ValueError, match="finite"):
        embeddings.PldaModel(np.zeros(1), np.array([[np.inf]]))


def test_plda_fit_selects_between_speaker_direction() -> None:
    values = np.array([[-3.0, -0.1], [-2.8, 0.1], [2.9, -0.1], [3.1, 0.1]])
    model = embeddings.fit_plda(values, ["a", "a", "b", "b"], output_dimension=1)
    assert model.projection.shape == (2, 1)
    projected = (values - model.mean) @ model.projection
    between = abs(float(np.mean(projected[:2]) - np.mean(projected[2:])))
    within = max(float(np.ptp(projected[:2])), float(np.ptp(projected[2:])))
    assert between > 10.0 * within


def test_plda_fit_validates_labels_dimension_and_regularization() -> None:
    with pytest.raises(ValueError, match="aligned"):
        embeddings.fit_plda(np.ones((2, 2)), ["a"])
    with pytest.raises(ValueError, match="two labeled"):
        embeddings.fit_plda(np.ones((2, 2)), ["a", "a"])
    with pytest.raises(ValueError, match="output_dimension"):
        embeddings.fit_plda(np.eye(2), ["a", "b"], output_dimension=3)
    with pytest.raises(ValueError, match="regularization"):
        embeddings.fit_plda(np.eye(2), ["a", "b"], regularization=0.0)


def test_plda_fit_is_repeatable_and_has_stable_eigenvector_sign() -> None:
    values = np.array([[-2.0, 0.0], [-1.8, 0.1], [2.0, 0.0], [1.8, -0.1]])
    labels = ["left", "left", "right", "right"]
    first = embeddings.fit_plda(values, labels)
    second = embeddings.fit_plda(values, labels)
    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.projection, second.projection)
    pivot = int(np.argmax(np.abs(first.projection[:, 0])))
    assert first.projection[pivot, 0] > 0.0


def test_plda_default_rank_is_bounded_by_speakers_minus_one() -> None:
    values = np.array([[-2.0, 0.0, 1.0], [-1.0, 0.1, 1.1], [2.0, 0.0, 0.9], [1.0, -0.1, 1.0]])
    model = embeddings.fit_plda(values, ["a", "a", "b", "b"])
    assert model.output_dimension == 1


def test_plda_projection_centers_transforms_and_normalizes_rows() -> None:
    model = embeddings.PldaModel(np.array([1.0, 2.0]), np.array([[2.0], [0.0]]))
    projected = embeddings.project_plda([[3.0, 5.0], [-1.0, 4.0]], model)
    np.testing.assert_array_equal(projected[:, 0], [1.0, -1.0])
    raw = embeddings.project_plda([3.0, 5.0], model, normalize=False)
    np.testing.assert_array_equal(raw, [4.0])


def test_plda_projection_rejects_wrong_width_and_zero_centered_vector() -> None:
    model = embeddings.PldaModel(np.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match="input dimension"):
        embeddings.project_plda([1.0], model)
    with pytest.raises(ValueError, match="positive norm"):
        embeddings.project_plda([0.0, 0.0], model)


def test_plda_scores_match_cosine_in_projected_space() -> None:
    model = embeddings.PldaModel(np.zeros(2), np.eye(2))
    values = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    observed = embeddings.plda_scores(values, values, model)
    expected = embeddings.pairwise_cosine(values, values)
    np.testing.assert_allclose(observed, expected, atol=1e-12)


def test_plda_scores_separate_training_speakers_on_clean_vectors() -> None:
    values = np.array([[-3.0, 0.0], [-2.8, 0.1], [2.8, -0.1], [3.0, 0.0]])
    labels = ["a", "a", "b", "b"]
    model = embeddings.fit_plda(values, labels)
    scores = embeddings.plda_scores(values, values, model)
    assert scores[0, 1] > scores[0, 2]
    assert scores[2, 3] > scores[2, 0]


def test_plda_score_matrix_is_bounded_and_self_scores_are_exact() -> None:
    model = embeddings.PldaModel(np.zeros(3), np.eye(3))
    values = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 1.0]])
    scores = embeddings.plda_scores(values, values, model)
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)
    np.testing.assert_allclose(np.diag(scores), 1.0, atol=1e-12)


def test_plda_scoring_requires_matrix_arguments() -> None:
    model = embeddings.PldaModel(np.zeros(2), np.eye(2))
    with pytest.raises(ValueError, match="matrices"):
        embeddings.plda_scores([1.0, 0.0], [[1.0, 0.0]], model)


def test_score_distributions_follow_planted_identity_pairs() -> None:
    scores = np.array([[1.0, 0.8, -0.2], [0.8, 1.0, -0.1], [-0.2, -0.1, 1.0]])
    trials = embeddings.score_distributions(scores, ["a", "a", "b"])
    np.testing.assert_array_equal(trials.genuine, [0.8])
    np.testing.assert_array_equal(trials.impostor, [-0.2, -0.1])


def test_score_distributions_reject_invalid_geometry_and_missing_trial_classes() -> None:
    with pytest.raises(ValueError, match="square"):
        embeddings.score_distributions(np.ones((2, 3)), ["a", "b"])
    with pytest.raises(ValueError, match="symmetric"):
        embeddings.score_distributions([[1.0, 0.0], [0.2, 1.0]], ["a", "b"])
    with pytest.raises(ValueError, match="both genuine"):
        embeddings.score_distributions(np.eye(2), ["a", "b"])


def test_score_distribution_summary_measures_planted_separation() -> None:
    trials = embeddings.ScoreDistributions(np.array([0.7, 0.9]), np.array([-0.4, 0.0]))
    assert trials.genuine_mean == 0.8
    assert trials.impostor_mean == -0.2
    assert trials.separation == 1.0


def test_eer_like_is_zero_for_separated_synthetic_trials() -> None:
    trials = embeddings.ScoreDistributions(np.array([0.7, 0.8]), np.array([-0.2, 0.1]))
    point = embeddings.eer_like(trials)
    assert point.rate == 0.0
    assert point.false_acceptance == point.false_rejection == 0.0
    assert 0.1 < point.threshold <= 0.7


def test_eer_like_reports_crossing_error_for_overlapping_trials() -> None:
    trials = embeddings.ScoreDistributions(np.array([0.2, 0.8]), np.array([0.1, 0.7]))
    point = embeddings.eer_like(trials)
    assert point.rate == 0.5
    assert point.false_acceptance == point.false_rejection == 0.5


def test_eer_threshold_minimizes_rate_gap_over_all_observed_scores() -> None:
    trials = embeddings.ScoreDistributions(
        np.array([0.15, 0.4, 0.9]), np.array([-0.2, 0.3, 0.8, 0.85])
    )
    point = embeddings.eer_like(trials)
    differences = []
    for threshold in np.unique(np.concatenate((trials.genuine, trials.impostor))):
        rejection = float(np.mean(trials.genuine < threshold))
        acceptance = float(np.mean(trials.impostor >= threshold))
        differences.append(abs(acceptance - rejection))
    assert abs(point.false_acceptance - point.false_rejection) == pytest.approx(
        min(differences), abs=1e-12
    )


def test_eer_like_rejects_empty_trial_side() -> None:
    trials = embeddings.ScoreDistributions(np.array([]), np.array([0.0]))
    with pytest.raises(ValueError, match="both trial classes"):
        embeddings.eer_like(trials)


def test_embedding_quality_uses_every_unordered_synthetic_trial() -> None:
    values = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    quality = embeddings.evaluate_embeddings(values, ["a", "a", "b", "b"])
    assert quality.trials == 6
    assert quality.genuine_mean > quality.impostor_mean
    assert quality.separation > 1.0
    assert quality.eer.rate == 0.0


def test_embedding_quality_rejects_blank_or_nonstring_labels() -> None:
    values = np.eye(3)
    with pytest.raises(ValueError, match="nonempty strings"):
        embeddings.evaluate_embeddings(values, ["a", "", "b"])
    with pytest.raises(ValueError, match="nonempty strings"):
        embeddings.evaluate_embeddings(values, ["a", 3, "b"])  # type: ignore[list-item]


def test_classical_embedding_pipeline_separates_planted_synthetic_speakers() -> None:
    generator = np.random.default_rng(24)
    centers = (np.array([-2.0, -1.0]), np.array([2.0, 1.0]), np.array([-1.0, 2.0]))
    utterances: list[np.ndarray] = []
    labels: list[str] = []
    for speaker, center in enumerate(centers):
        for _ in range(3):
            utterances.append(generator.normal(center, 0.15, size=(30, 2)))
            labels.append(f"speaker-{speaker}")
    training = np.concatenate(utterances)
    ubm = embeddings.fit_gmm(training, 3, seed=4, iterations=8).model
    vectors = np.stack([embeddings.embed_features(item, ubm) for item in utterances])
    quality = embeddings.evaluate_embeddings(vectors, labels)
    assert quality.genuine_mean > quality.impostor_mean
    assert quality.eer.rate <= 0.1


def test_embedding_quality_is_exactly_perfect_for_identical_labeled_directions() -> None:
    values = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    quality = embeddings.evaluate_embeddings(values, ["a", "a", "b", "b"])
    assert quality.genuine_mean == 1.0
    assert quality.impostor_mean == 0.0
    assert quality.eer.rate == 0.0


# -- extensions --
