from __future__ import annotations

import numpy as np
import pytest

from choravox import streaming


def test_block_ranges_cover_full_and_tail_blocks() -> None:
    assert streaming.block_ranges(10, 4) == (
        streaming.BlockRange(0, 4),
        streaming.BlockRange(4, 8),
        streaming.BlockRange(8, 10),
    )


def test_block_ranges_define_empty_input_and_validate_geometry() -> None:
    assert streaming.block_ranges(0, 4) == ()
    with pytest.raises(ValueError, match="invalid"):
        streaming.block_ranges(3, 0)


def test_block_range_exposes_sample_count() -> None:
    assert streaming.BlockRange(3, 9).sample_count == 6


def test_blocks_reconstruct_every_sample_exactly_once() -> None:
    for sample_count in range(1, 25):
        for block_size in range(1, 8):
            blocks = streaming.block_ranges(sample_count, block_size)
            reconstructed = [index for block in blocks for index in range(block.start, block.end)]
            assert reconstructed == list(range(sample_count))


def test_only_tail_block_may_be_short() -> None:
    blocks = streaming.block_ranges(17, 5)
    assert all(block.sample_count == 5 for block in blocks[:-1])
    assert blocks[-1].sample_count == 2


def test_adjacent_blocks_share_boundaries_without_overlap() -> None:
    blocks = streaming.block_ranges(19, 6)
    from itertools import pairwise

    assert all(left.end == right.start for left, right in pairwise(blocks))


def test_block_record_rejects_reversed_or_empty_span() -> None:
    with pytest.raises(ValueError, match="positive"):
        streaming.BlockRange(2, 2)
    with pytest.raises(ValueError, match="positive"):
        streaming.BlockRange(-1, 2)


def test_streaming_latency_separates_buffering_and_lookahead() -> None:
    result = streaming.streaming_latency(
        block_size=1600, sample_rate=16000, lookahead_frames=5, frame_shift_s=0.01
    )
    assert result.buffering_s == 0.1
    assert result.algorithmic_s == 0.05
    assert result.total_s == 0.15


def test_processing_time_contributes_to_total_latency() -> None:
    result = streaming.streaming_latency(block_size=800, sample_rate=16000, processing_s=0.012)
    assert result.total_s == 0.062


def test_halving_block_size_halves_buffering_latency() -> None:
    large = streaming.streaming_latency(block_size=1600, sample_rate=16000)
    small = streaming.streaming_latency(block_size=800, sample_rate=16000)
    assert small.buffering_s * 2 == large.buffering_s


def test_total_latency_is_sum_of_reported_components() -> None:
    result = streaming.LatencyAccounting(0.123456789, 0.2, 0.03)
    assert result.total_s == round(
        result.buffering_s + result.algorithmic_s + result.processing_s, 9
    )


def test_latency_components_reject_negative_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        streaming.LatencyAccounting(-0.1, 0.0)
    with pytest.raises(ValueError, match="finite"):
        streaming.LatencyAccounting(0.1, float("inf"))


def test_latency_geometry_validates_sample_and_frame_controls() -> None:
    with pytest.raises(ValueError, match="geometry"):
        streaming.streaming_latency(block_size=0, sample_rate=16000)
    with pytest.raises(ValueError, match="geometry"):
        streaming.streaming_latency(block_size=100, sample_rate=16000, lookahead_frames=-1)


def test_online_segmentation_emits_change_at_orthogonal_embedding() -> None:
    state = streaming.OnlineSegmentationState()
    state, first = streaming.online_segmentation_step(state, [1.0, 0.0], threshold=0.5)
    state, second = streaming.online_segmentation_step(state, [0.9, 0.1], threshold=0.5)
    state, third = streaming.online_segmentation_step(state, [0.0, 1.0], threshold=0.5)
    assert first is None and second is None
    assert third is not None and third.frame == 2 and third.score > 0.5
    assert state.frames_seen == 3


def test_online_segmentation_respects_minimum_boundary_gap() -> None:
    state = streaming.OnlineSegmentationState()
    events = []
    for embedding in ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]):
        state, event = streaming.online_segmentation_step(
            state, embedding, threshold=0.5, minimum_gap=2
        )
        if event is not None:
            events.append(event.frame)
    assert events == [2]


def test_online_boundaries_match_adjacent_cosine_oracle_without_gap() -> None:
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    expected = [
        index
        for index in range(1, len(embeddings))
        if 1.0 - float(np.dot(normalized[index - 1], normalized[index])) >= 0.5
    ]
    state = streaming.OnlineSegmentationState()
    observed = []
    for embedding in embeddings:
        state, event = streaming.online_segmentation_step(state, embedding, threshold=0.5)
        if event is not None:
            observed.append(event.frame)
    assert observed == expected


def test_online_segmentation_state_copies_input_embedding() -> None:
    embedding = np.array([1.0, 0.0])
    state, _ = streaming.online_segmentation_step(streaming.OnlineSegmentationState(), embedding)
    embedding[:] = 0.0
    assert state.previous is not None
    assert state.previous.tolist() == [1.0, 0.0]


def test_online_segmentation_validates_dimension_norm_and_controls() -> None:
    state, _ = streaming.online_segmentation_step(streaming.OnlineSegmentationState(), [1.0])
    with pytest.raises(ValueError, match="dimension"):
        streaming.online_segmentation_step(state, [1.0, 0.0])
    with pytest.raises(ValueError, match="positive"):
        streaming.online_segmentation_step(state, [0.0])
    with pytest.raises(ValueError, match="controls"):
        streaming.online_segmentation_step(state, [1.0], threshold=3.0)


def test_leader_follower_creates_and_reuses_speaker_clusters() -> None:
    state = None
    labels = []
    for embedding in ([1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]):
        state, label = streaming.leader_follower_step(state, embedding)
        labels.append(label)
    assert labels == [0, 0, 1, 1]
    assert state.counts.tolist() == [2, 2]
    assert state.labels == tuple(labels)


def test_leader_follower_birth_threshold_controls_new_speakers() -> None:
    state, _ = streaming.leader_follower_step(None, [1.0, 0.0])
    strict, strict_label = streaming.leader_follower_step(
        state, [0.0, 1.0], streaming.LeaderFollowerConfig(birth_threshold=0.2)
    )
    permissive, permissive_label = streaming.leader_follower_step(
        state, [0.0, 1.0], streaming.LeaderFollowerConfig(birth_threshold=1.5)
    )
    assert strict_label == 1 and strict.centroids.shape[0] == 2
    assert permissive_label == 0 and permissive.centroids.shape[0] == 1


def test_online_centroid_is_the_running_arithmetic_mean() -> None:
    vectors = np.array([[1.0, 0.0], [0.8, 0.2], [0.9, 0.1]])
    state = None
    for vector in vectors:
        state, label = streaming.leader_follower_step(
            state, vector, streaming.LeaderFollowerConfig(birth_threshold=2.0)
        )
        assert label == 0
    np.testing.assert_allclose(state.centroids[0], np.mean(vectors, axis=0), atol=1e-12)


def test_online_clustering_is_deterministic_for_fixed_sequence() -> None:
    vectors = ([1.0, 0.0], [0.9, 0.1], [0.0, 1.0])
    states = []
    for _ in range(2):
        state = None
        for vector in vectors:
            state, _ = streaming.leader_follower_step(state, vector)
        states.append(state)
    np.testing.assert_array_equal(states[0].centroids, states[1].centroids)
    assert states[0].labels == states[1].labels


def test_online_clustering_validates_vectors_state_and_threshold() -> None:
    with pytest.raises(ValueError, match="positive"):
        streaming.leader_follower_step(None, [0.0, 0.0])
    state, _ = streaming.leader_follower_step(None, [1.0, 0.0])
    with pytest.raises(ValueError, match="dimensions"):
        streaming.leader_follower_step(state, [1.0])
    with pytest.raises(ValueError, match="threshold"):
        streaming.LeaderFollowerConfig(birth_threshold=2.1)


def test_online_cluster_labels_cover_every_observation() -> None:
    state = None
    for index in range(9):
        state, _ = streaming.leader_follower_step(state, [1.0, index / 20])
    assert len(state.labels) == 9
    assert int(np.sum(state.counts)) == 9


# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
