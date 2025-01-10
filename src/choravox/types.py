"""Shared immutable domain records."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, order=True, slots=True)
class Segment:
    """Half-open time interval carrying a speaker label."""

    start: float
    end: float
    speaker: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("segment requires 0 <= start < end")
        if not self.speaker.strip():
            raise ValueError("speaker must be nonempty")

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 6)


def intersection(left: Segment, right: Segment) -> float:
    """Return overlap duration regardless of speaker labels."""
    return round(max(0.0, min(left.end, right.end) - max(left.start, right.start)), 6)


def clip_segment(segment: Segment, start: float, end: float) -> Segment | None:
    """Clip a segment to a half-open region, returning none when disjoint."""
    clipped_start = max(segment.start, start)
    clipped_end = min(segment.end, end)
    if clipped_end <= clipped_start:
        return None
    return replace(segment, start=clipped_start, end=clipped_end)


# -- extensions --
