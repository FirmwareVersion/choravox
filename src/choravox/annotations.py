"""Strict deterministic interchange formats for planted annotations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from choravox.types import Segment


def _token(value: str, name: str) -> str:
    if not value or any(char.isspace() for char in value):
        raise ValueError(f"{name} must be a nonempty token")
    return value


@dataclass(frozen=True, slots=True)
class RttmEntry:
    """One NIST-style SPEAKER record with half-open timing semantics."""

    recording_id: str
    channel: int
    start_s: float
    duration_s: float
    speaker: str

    def __post_init__(self) -> None:
        _token(self.recording_id, "recording_id")
        _token(self.speaker, "speaker")
        if self.channel < 1:
            raise ValueError("RTTM channel must be positive")
        if not math.isfinite(self.start_s) or not math.isfinite(self.duration_s):
            raise ValueError("RTTM times must be finite")
        if self.start_s < 0.0 or self.duration_s <= 0.0:
            raise ValueError("RTTM requires nonnegative start and positive duration")

    def segment(self) -> Segment:
        """Convert this record into the shared half-open segment type."""
        return Segment(self.start_s, round(self.start_s + self.duration_s, 6), self.speaker)


def rttm_to_text(entries: tuple[RttmEntry, ...]) -> str:
    """Encode RTTM records with fixed six-decimal timing."""
    lines = [
        (
            f"SPEAKER {entry.recording_id} {entry.channel} {entry.start_s:.6f} "
            f"{entry.duration_s:.6f} <NA> <NA> {entry.speaker} <NA> <NA>"
        )
        for entry in entries
    ]
    return "" if not lines else "\n".join(lines) + "\n"


def rttm_from_text(text: str) -> tuple[RttmEntry, ...]:
    """Parse only complete SPEAKER records and reject ambiguous fields."""
    entries: list[RttmEntry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise ValueError(f"invalid RTTM record on line {line_number}")
        if fields[5:7] != ["<NA>", "<NA>"] or fields[8:] != ["<NA>", "<NA>"]:
            raise ValueError(f"unsupported RTTM fields on line {line_number}")
        try:
            entries.append(
                RttmEntry(fields[1], int(fields[2]), float(fields[3]), float(fields[4]), fields[7])
            )
        except ValueError as error:
            raise ValueError(f"invalid RTTM values on line {line_number}") from error
    return tuple(entries)


@dataclass(frozen=True, slots=True)
class StmEntry:
    """One speaker-attributed transcript segment in STM form."""

    recording_id: str
    channel: int
    speaker: str
    start_s: float
    end_s: float
    transcript: str

    def __post_init__(self) -> None:
        _token(self.recording_id, "recording_id")
        _token(self.speaker, "speaker")
        if self.channel < 1 or self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError("STM channel and time interval are invalid")
        if not all(math.isfinite(value) for value in (self.start_s, self.end_s)):
            raise ValueError("STM times must be finite")
        if not self.transcript.strip() or "\n" in self.transcript:
            raise ValueError("STM transcript must be nonempty and single-line")


def stm_to_text(entries: tuple[StmEntry, ...]) -> str:
    """Encode STM records with stable timing and no implicit labels."""
    lines = [
        (
            f"{entry.recording_id} {entry.channel} {entry.speaker} "
            f"{entry.start_s:.6f} {entry.end_s:.6f} <NA> {entry.transcript}"
        )
        for entry in entries
    ]
    return "" if not lines else "\n".join(lines) + "\n"


def stm_from_text(text: str) -> tuple[StmEntry, ...]:
    """Parse strict seven-field STM records while preserving transcript spaces."""
    entries: list[StmEntry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=6)
        if len(fields) != 7 or fields[5] != "<NA>":
            raise ValueError(f"invalid STM record on line {line_number}")
        try:
            entries.append(
                StmEntry(
                    fields[0],
                    int(fields[1]),
                    fields[2],
                    float(fields[3]),
                    float(fields[4]),
                    fields[6],
                )
            )
        except ValueError as error:
            raise ValueError(f"invalid STM values on line {line_number}") from error
    return tuple(entries)


# -- extensions --
