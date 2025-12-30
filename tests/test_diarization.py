from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from choravox import diarization


def test_frame_window_uses_half_open_duration() -> None:
    window = diarization.FrameWindow(3, 8)
    assert window.frame_count == 5


def test_frame_window_rejects_negative_or_empty_span() -> None:
    with pytest.raises(ValueError, match="positive duration"):
        diarization.FrameWindow(-1, 2)
    with pytest.raises(ValueError, match="positive duration"):
        diarization.FrameWindow(3, 3)


def test_sliding_windows_anchor_tail_without_duplicates() -> None:
    windows = diarization.sliding_windows(11, 4, 3)
    assert windows == (
        diarization.FrameWindow(0, 4),
        diarization.FrameWindow(3, 7),
        diarization.FrameWindow(6, 10),
        diarization.FrameWindow(7, 11),
    )


def test_sliding_windows_handle_empty_and_short_recordings() -> None:
    assert diarization.sliding_windows(0, 4, 2) == ()
    assert diarization.sliding_windows(3, 4, 2) == (diarization.FrameWindow(0, 3),)


def test_sliding_windows_cover_every_frame_when_shift_fits_window() -> None:
    for frame_count in range(1, 24):
        windows = diarization.sliding_windows(frame_count, 5, 3)
        covered = {frame for window in windows for frame in range(window.start, window.end)}
        assert covered == set(range(frame_count))


def test_sliding_windows_can_omit_irregular_tail_anchor() -> None:
    windows = diarization.sliding_windows(11, 4, 3, include_tail=False)
    assert windows[-1] == diarization.FrameWindow(6, 10)


def test_sliding_windows_validate_all_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        diarization.sliding_windows(-1, 4, 2)
    with pytest.raises(ValueError, match="geometry"):
        diarization.sliding_windows(5, 0, 2)
    with pytest.raises(ValueError, match="geometry"):
        diarization.sliding_windows(5, 2, 0)


def test_window_times_attach_frame_hop_clock() -> None:
    windows = (diarization.FrameWindow(0, 4), diarization.FrameWindow(3, 7))
    assert diarization.window_times(windows, hop_seconds=0.01) == ((0.0, 0.04), (0.03, 0.07))


def test_window_times_reject_invalid_hop() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        diarization.window_times((), hop_seconds=0.0)


def test_window_pooling_matches_hand_computed_means() -> None:
    features = np.arange(12.0).reshape(6, 2)
    windows = (diarization.FrameWindow(0, 2), diarization.FrameWindow(2, 6))
    pooled = diarization.pool_windows(features, windows)
    np.testing.assert_array_equal(pooled, [[1.0, 2.0], [7.0, 8.0]])


def test_window_pooling_preserves_empty_output_dimension() -> None:
    assert diarization.pool_windows(np.empty((0, 3)), ()).shape == (0, 3)


def test_pooling_constant_features_is_independent_of_window_geometry() -> None:
    features = np.tile([2.0, -1.0], (9, 1))
    windows = diarization.sliding_windows(9, 4, 2)
    np.testing.assert_array_equal(diarization.pool_windows(features, windows), [[2.0, -1.0]] * 4)


def test_pooling_rejects_nonfinite_features_and_out_of_range_windows() -> None:
    with pytest.raises(ValueError, match="finite matrix"):
        diarization.pool_windows([[np.nan]], (diarization.FrameWindow(0, 1),))
    with pytest.raises(ValueError, match="exceeds"):
        diarization.pool_windows(np.ones((2, 1)), (diarization.FrameWindow(0, 3),))


def test_adjacent_change_scores_cover_same_orthogonal_and_opposite_vectors() -> None:
    values = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, -2.0]])
    np.testing.assert_array_equal(diarization.adjacent_change_scores(values), [0.0, 1.0, 2.0])


def test_adjacent_change_scores_handle_short_sequence_and_reject_zero_norm() -> None:
    assert diarization.adjacent_change_scores([[1.0, 0.0]]).size == 0
    with pytest.raises(ValueError, match="positive norm"):
        diarization.adjacent_change_scores([[1.0, 0.0], [0.0, 0.0]])


def test_change_point_detection_uses_boundary_after_score() -> None:
    assert diarization.detect_change_points([0.1, 0.8, 0.2, 1.1], 0.7) == (2, 4)


def test_change_point_detection_keeps_strongest_within_minimum_gap() -> None:
    assert diarization.detect_change_points([0.8, 1.2, 0.9, 0.1], 0.7, minimum_gap=3) == (2,)


def test_change_scores_and_boundaries_ignore_positive_embedding_scale() -> None:
    values = np.array([[1.0, 0.0], [1.0, 0.2], [0.0, 1.0], [0.1, 1.0]])
    first = diarization.adjacent_change_scores(values)
    second = diarization.adjacent_change_scores(values * [[2.0], [5.0], [0.5], [3.0]])
    np.testing.assert_allclose(first, second, atol=1e-12)
    assert diarization.detect_change_points(first, 0.5) == diarization.detect_change_points(
        second, 0.5
    )


def test_change_point_detection_validates_vector_and_controls() -> None:
    with pytest.raises(ValueError, match="finite vector"):
        diarization.detect_change_points([[0.2]], 0.5)
    with pytest.raises(ValueError, match="invalid"):
        diarization.detect_change_points([0.2], 2.1)
    with pytest.raises(ValueError, match="invalid"):
        diarization.detect_change_points([0.2], 0.5, minimum_gap=0)


def test_frame_partition_covers_axis_without_gaps() -> None:
    parts = diarization.partition_frames(10, (2, 7))
    assert parts == (
        diarization.FrameWindow(0, 2),
        diarization.FrameWindow(2, 7),
        diarization.FrameWindow(7, 10),
    )


def test_frame_partition_handles_empty_axis_and_rejects_bad_boundaries() -> None:
    assert diarization.partition_frames(0, ()) == ()
    with pytest.raises(ValueError, match="ordered interior"):
        diarization.partition_frames(5, (3, 2))
    with pytest.raises(ValueError, match="ordered interior"):
        diarization.partition_frames(5, (5,))


def test_label_runs_encode_contiguous_speaker_spans() -> None:
    segments = diarization.labels_to_segments(["a", "a", "b", "a"])
    assert segments == (
        diarization.LabeledFrames(0, 2, "a"),
        diarization.LabeledFrames(2, 3, "b"),
        diarization.LabeledFrames(3, 4, "a"),
    )


def test_label_runs_handle_empty_input_and_reject_blank_label() -> None:
    assert diarization.labels_to_segments([]) == ()
    with pytest.raises(ValueError, match="nonempty"):
        diarization.labels_to_segments(["a", ""])


def test_segment_decoding_restores_frame_labels() -> None:
    labels = ("a", "a", "b", "b", "a")
    assert (
        diarization.segments_to_labels(diarization.labels_to_segments(labels), len(labels))
        == labels
    )


def test_segment_decoding_rejects_gaps_overlaps_and_incomplete_coverage() -> None:
    with pytest.raises(ValueError, match="cover"):
        diarization.segments_to_labels((diarization.LabeledFrames(1, 3, "a"),), 3)
    with pytest.raises(ValueError, match="contiguous"):
        diarization.segments_to_labels(
            (diarization.LabeledFrames(0, 2, "a"), diarization.LabeledFrames(1, 3, "b")), 3
        )


def test_label_segment_roundtrip_is_exhaustive_for_short_binary_sequences() -> None:
    for mask in range(1 << 6):
        labels = tuple("a" if mask & (1 << index) else "b" for index in range(6))
        encoded = diarization.labels_to_segments(labels)
        assert diarization.segments_to_labels(encoded, 6) == labels
        assert sum(segment.end - segment.start for segment in encoded) == 6


def test_adjacent_equal_segments_remain_decodable_without_information_loss() -> None:
    segments = (
        diarization.LabeledFrames(0, 2, "a"),
        diarization.LabeledFrames(2, 4, "a"),
    )
    assert diarization.segments_to_labels(segments, 4) == ("a", "a", "a", "a")


def test_majority_smoothing_removes_one_frame_label_spike() -> None:
    labels = ("a", "a", "b", "a", "a")
    assert diarization.majority_smooth(labels, 1) == ("a", "a", "a", "a", "a")


def test_majority_smoothing_preserves_current_label_on_tie() -> None:
    assert diarization.majority_smooth(("a", "b"), 1) == ("a", "b")
    assert diarization.majority_smooth(("a", "b"), 0) == ("a", "b")


def test_majority_smoothing_does_not_move_stable_change_boundary() -> None:
    labels = ("a", "a", "a", "b", "b", "b")
    assert diarization.majority_smooth(labels, 1) == labels


def test_majority_smoothing_validates_radius_and_labels() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        diarization.majority_smooth(("a",), -1)
    with pytest.raises(ValueError, match="nonempty"):
        diarization.majority_smooth(("a", ""), 1)


def test_speaker_centroids_average_each_cluster() -> None:
    values = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 4.0]])
    centroids = diarization.speaker_centroids(values, [0, 0, 1])
    np.testing.assert_array_equal(centroids, [[1.0, 1.0], [10.0, 4.0]])


def test_speaker_centroids_reject_alignment_and_label_gaps() -> None:
    with pytest.raises(ValueError, match="aligned"):
        diarization.speaker_centroids(np.ones((2, 1)), [0])
    with pytest.raises(ValueError, match="contiguous"):
        diarization.speaker_centroids(np.ones((2, 1)), [0, 2])


def test_centroid_assignment_returns_nearest_label_and_distance() -> None:
    labels, distances = diarization.assign_centroids([[0.0], [4.0], [9.0]], [[0.0], [10.0]])
    np.testing.assert_array_equal(labels, [0, 0, 1])
    np.testing.assert_array_equal(distances, [0.0, 4.0, 1.0])


def test_centroid_assignment_breaks_exact_tie_by_first_index() -> None:
    labels, distances = diarization.assign_centroids([[1.0]], [[0.0], [2.0]])
    assert labels.tolist() == [0]
    assert distances.tolist() == [1.0]


def test_centroid_reassignment_recovers_clean_planted_clusters() -> None:
    values = np.array([[-2.1, 0.0], [-1.9, 0.1], [3.9, 1.0], [4.1, 0.9]])
    planted = [0, 0, 1, 1]
    centroids = diarization.speaker_centroids(values, planted)
    recovered, distances = diarization.assign_centroids(values, centroids)
    np.testing.assert_array_equal(recovered, planted)
    assert np.max(distances) < 0.2


def test_centroid_assignment_rejects_empty_and_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="compatible"):
        diarization.assign_centroids(np.ones((2, 2)), np.empty((0, 2)))
    with pytest.raises(ValueError, match="compatible"):
        diarization.assign_centroids(np.ones((2, 2)), np.ones((1, 3)))


def test_cosine_distance_matrix_is_symmetric_with_zero_diagonal() -> None:
    values = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    distances = diarization.cosine_distance_matrix(values)
    np.testing.assert_array_equal(distances, distances.T)
    np.testing.assert_array_equal(np.diag(distances), 0.0)
    assert distances[0, 1] == 1.0
    assert distances[0, 2] == 2.0


def test_cosine_distance_matrix_rejects_empty_and_zero_rows() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        diarization.cosine_distance_matrix(np.empty((0, 2)))
    with pytest.raises(ValueError, match="positive norm"):
        diarization.cosine_distance_matrix([[0.0, 0.0]])


def test_ahc_recovers_two_clean_directional_clusters() -> None:
    values = np.array([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0], [0.05, 1.0]])
    result = diarization.agglomerative_clustering(values, threshold=0.2)
    assert result.speaker_count == 2
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]


def test_ahc_can_obey_oracle_speaker_count() -> None:
    values = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    result = diarization.agglomerative_clustering(values, threshold=0.0, speaker_count=1)
    assert result.speaker_count == 1
    assert len(result.merge_distances) == 2


def test_ahc_validates_threshold_and_requested_count() -> None:
    with pytest.raises(ValueError, match="threshold"):
        diarization.agglomerative_clustering([[1.0]], threshold=3.0)
    with pytest.raises(ValueError, match="speaker_count"):
        diarization.agglomerative_clustering([[1.0]], speaker_count=2)


def test_ahc_pair_membership_is_invariant_to_input_permutation() -> None:
    values = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]])
    original = diarization.agglomerative_clustering(values, threshold=0.2).labels
    permutation = np.array([2, 0, 3, 1])
    shuffled = diarization.agglomerative_clustering(values[permutation], threshold=0.2).labels
    restored = np.empty_like(shuffled)
    restored[permutation] = shuffled
    for left in range(4):
        for right in range(4):
            assert (original[left] == original[right]) == (restored[left] == restored[right])


def test_ahc_merge_distances_are_monotone_on_nested_clean_clusters() -> None:
    values = np.array([[1.0, 0.0], [1.0, 0.01], [0.7, 0.3], [0.0, 1.0]])
    result = diarization.agglomerative_clustering(values, speaker_count=1)
    assert all(left <= right for left, right in pairwise(result.merge_distances))


def test_affinity_is_high_within_and_low_between_clean_speakers() -> None:
    values = np.array([[1.0, 0.0], [1.0, 0.05], [0.0, 1.0]])
    affinity = diarization.affinity_matrix(values, scale=0.3)
    assert affinity[0, 1] > 0.99
    assert affinity[0, 2] < 0.01
    np.testing.assert_array_equal(np.diag(affinity), 0.0)


def test_affinity_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        diarization.affinity_matrix([[1.0]], scale=0.0)


def test_affinity_is_symmetric_and_invariant_to_positive_row_scale() -> None:
    values = np.array([[1.0, 2.0], [-2.0, 1.0], [0.5, 1.5]])
    first = diarization.affinity_matrix(values)
    second = diarization.affinity_matrix(values * [[2.0], [0.5], [4.0]])
    np.testing.assert_array_equal(first, first.T)
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_normalized_laplacian_has_expected_two_node_spectrum() -> None:
    laplacian = diarization.normalized_laplacian([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_array_equal(laplacian, [[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(np.linalg.eigvalsh(laplacian), [0.0, 2.0], atol=1e-12)


def test_normalized_laplacian_rejects_asymmetry_and_isolates() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        diarization.normalized_laplacian([[0.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="isolated"):
        diarization.normalized_laplacian(np.zeros((2, 2)))


def test_eigengap_recovers_two_disconnected_speaker_groups() -> None:
    affinity = np.array(
        [
            [0.0, 1.0, 0.01, 0.01],
            [1.0, 0.0, 0.01, 0.01],
            [0.01, 0.01, 0.0, 1.0],
            [0.01, 0.01, 1.0, 0.0],
        ]
    )
    assert diarization.estimate_speaker_count(affinity) == 2


def test_eigengap_handles_single_node_and_validates_maximum() -> None:
    assert diarization.estimate_speaker_count([[1.0]]) == 1
    with pytest.raises(ValueError, match="maximum"):
        diarization.estimate_speaker_count([[1.0]], maximum=2)


def test_eigengap_count_is_stable_under_uniform_affinity_scaling() -> None:
    affinity = np.array([[0.0, 1.0, 0.1], [1.0, 0.0, 0.1], [0.1, 0.1, 0.0]])
    assert diarization.estimate_speaker_count(affinity) == diarization.estimate_speaker_count(
        7.0 * affinity
    )


def test_eigengap_obeys_declared_search_ceiling() -> None:
    affinity = np.full((4, 4), 0.01)
    np.fill_diagonal(affinity, 0.0)
    affinity[0, 1] = affinity[1, 0] = 1.0
    affinity[2, 3] = affinity[3, 2] = 1.0
    assert diarization.estimate_speaker_count(affinity, maximum=1) == 1


# -- extensions --
