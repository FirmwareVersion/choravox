"""Online block processing, clustering, and hypothesis revision mechanics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class BlockRange:
    """One half-open block on a sample axis."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("stream block must have positive duration")

    @property
    def sample_count(self) -> int:
        return self.end - self.start


def block_ranges(sample_count: int, block_size: int) -> tuple[BlockRange, ...]:
    """Partition samples into consecutive blocks while retaining the tail."""
    if sample_count < 0 or block_size < 1:
        raise ValueError("sample count and block size are invalid")
    return tuple(
        BlockRange(start, min(start + block_size, sample_count))
        for start in range(0, sample_count, block_size)
    )


# -- extensions --
