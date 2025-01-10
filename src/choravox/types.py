"""Shared immutable domain records."""

from __future__ import annotations

from dataclasses import dataclass


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


# -- extensions --
