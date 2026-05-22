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


# -- extensions --
# -- extensions --
