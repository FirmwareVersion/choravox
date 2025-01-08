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


# -- extensions --
