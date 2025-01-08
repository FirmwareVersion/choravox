"""Validated configuration objects for deterministic processing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Sampling and analysis settings shared by front-end components."""

    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms cannot exceed frame_ms")


@dataclass(frozen=True, slots=True)
class MeetingConfig:
    """High-level controls for planted synthetic meetings."""

    speakers: int = 2
    duration_s: float = 8.0
    overlap_ratio: float = 0.1
    silence_ratio: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if not 1 <= self.speakers <= 16:
            raise ValueError("speakers must be between 1 and 16")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if not 0.0 <= self.overlap_ratio <= 1.0:
            raise ValueError("overlap_ratio must be in [0, 1]")
        if not 0.0 <= self.silence_ratio < 1.0:
            raise ValueError("silence_ratio must be in [0, 1)")


# -- extensions --
