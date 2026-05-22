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


# -- extensions --
