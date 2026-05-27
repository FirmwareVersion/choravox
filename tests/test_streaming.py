from __future__ import annotations

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


# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
