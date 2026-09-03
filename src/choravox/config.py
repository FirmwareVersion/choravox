"""Validated configuration objects for deterministic processing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


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
class FeatureConfig:
    """Parameters for deterministic log-mel and dynamic features."""

    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    min_hz: float = 20.0
    max_hz: float = 7600.0
    log_floor: float = 1e-10
    delta_order: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ValueError("feature timing values must be positive")
        frame_length = round(self.sample_rate * self.frame_ms / 1000.0)
        if self.hop_ms > self.frame_ms or self.n_fft < frame_length:
            raise ValueError("feature FFT and hop must accommodate each frame")
        if self.n_mels <= 0 or not 0 <= self.min_hz < self.max_hz <= self.sample_rate / 2:
            raise ValueError("mel bands and frequency bounds are invalid")
        if self.log_floor <= 0 or self.delta_order not in (0, 1, 2):
            raise ValueError("log_floor and delta_order are invalid")

    @property
    def frame_length(self) -> int:
        return round(self.sample_rate * self.frame_ms / 1000.0)

    @property
    def hop_length(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1000.0)


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


def config_to_json(config: AudioConfig | FeatureConfig | MeetingConfig) -> str:
    """Encode a configuration with an explicit kind discriminator."""
    payload = {"kind": type(config).__name__, "values": asdict(config)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def config_from_json(text: str) -> AudioConfig | FeatureConfig | MeetingConfig:
    """Decode a configuration and reject unknown or malformed kinds."""
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("configuration must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"kind", "values"}:
        raise ValueError("configuration envelope must contain kind and values")
    kind = payload["kind"]
    values = payload["values"]
    if not isinstance(values, dict):
        raise ValueError("unknown configuration kind")
    if kind == "AudioConfig":
        return AudioConfig(**values)
    if kind == "FeatureConfig":
        return FeatureConfig(**values)
    if kind == "MeetingConfig":
        return MeetingConfig(**values)
    raise ValueError("unknown configuration kind")


# -- extensions --
