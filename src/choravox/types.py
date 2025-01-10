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


@dataclass(frozen=True, order=True, slots=True)
class Word:
    """A normalized token with time, speaker, and calibrated confidence."""

    text: str
    start: float
    end: float
    speaker: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.text or any(char.isspace() for char in self.text):
            raise ValueError("word text must be a nonempty token")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("word requires 0 <= start < end")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Transcript:
    """Chronological speaker-attributed words."""

    words: tuple[Word, ...]

    def __post_init__(self) -> None:
        pairs = zip(self.words, self.words[1:], strict=False)
        if any(right.start < left.start for left, right in pairs):
            raise ValueError("transcript words must be ordered by start time")

    def text(self, speaker: str | None = None) -> str:
        """Join all words or only words attributed to one speaker."""
        return " ".join(
            word.text for word in self.words if speaker is None or word.speaker == speaker
        )

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(word.speaker for word in self.words))


# -- extensions --
