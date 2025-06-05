"""Small phoneme-like inventory for deterministic synthetic speech."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from choravox.random import rng

Manner = Literal["vowel", "fricative", "stop", "nasal"]


@dataclass(frozen=True, slots=True)
class Phone:
    """Acoustic target for one synthetic phoneme-like unit."""

    symbol: str
    manner: Manner
    voiced: bool
    formants_hz: tuple[float, ...] = ()
    bandwidths_hz: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol or any(char.isspace() for char in self.symbol):
            raise ValueError("phone symbol must be a nonempty token")
        values = (*self.formants_hz, *self.bandwidths_hz)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("phone frequencies and bandwidths must be finite and positive")
        if len(self.formants_hz) != len(self.bandwidths_hz):
            raise ValueError("every formant requires one bandwidth")
        if self.formants_hz != tuple(sorted(self.formants_hz)):
            raise ValueError("formants must be strictly ordered")
        if self.voiced and not self.formants_hz:
            raise ValueError("voiced phones require formant targets")


_PHONES = (
    Phone("aa", "vowel", True, (730.0, 1090.0, 2440.0), (80.0, 90.0, 120.0)),
    Phone("ae", "vowel", True, (660.0, 1720.0, 2410.0), (80.0, 100.0, 120.0)),
    Phone("eh", "vowel", True, (530.0, 1840.0, 2480.0), (70.0, 95.0, 120.0)),
    Phone("iy", "vowel", True, (270.0, 2290.0, 3010.0), (60.0, 100.0, 130.0)),
    Phone("uw", "vowel", True, (300.0, 870.0, 2240.0), (60.0, 90.0, 120.0)),
    Phone("n", "nasal", True, (250.0, 1500.0, 2500.0), (100.0, 150.0, 180.0)),
    Phone("f", "fricative", False),
    Phone("s", "fricative", False),
    Phone("sh", "fricative", False),
    Phone("t", "stop", False),
)
PHONE_INVENTORY = MappingProxyType({phone.symbol: phone for phone in _PHONES})


def get_phone(symbol: str) -> Phone:
    """Resolve a phone symbol with a useful domain error."""
    try:
        return PHONE_INVENTORY[symbol]
    except KeyError as error:
        raise ValueError(f"unknown phone symbol: {symbol}") from error


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    """Synthetic voice controls derived from a named deterministic stream."""

    speaker_id: str
    base_pitch_hz: float
    pitch_range_semitones: float
    formant_scale: float
    breathiness: float
    jitter: float

    def __post_init__(self) -> None:
        if not self.speaker_id:
            raise ValueError("speaker_id must be nonempty")
        if not 70.0 <= self.base_pitch_hz <= 320.0:
            raise ValueError("base_pitch_hz must be in [70, 320]")
        if not 0.0 <= self.pitch_range_semitones <= 12.0:
            raise ValueError("pitch range must be in [0, 12]")
        if not 0.75 <= self.formant_scale <= 1.3:
            raise ValueError("formant_scale must be in [0.75, 1.3]")
        if not 0.0 <= self.breathiness <= 1.0 or not 0.0 <= self.jitter <= 0.1:
            raise ValueError("breathiness or jitter is outside its supported range")


def speaker_profile(speaker_id: str, seed: int) -> SpeakerProfile:
    """Derive a repeatable synthetic voice without claiming a real identity."""
    if not speaker_id:
        raise ValueError("speaker_id must be nonempty")
    generator = rng(seed, f"speaker:{speaker_id}")
    values = generator.uniform(size=5)
    return SpeakerProfile(
        speaker_id=speaker_id,
        base_pitch_hz=round(95.0 + values[0] * 145.0, 6),
        pitch_range_semitones=round(1.5 + values[1] * 4.5, 6),
        formant_scale=round(0.88 + values[2] * 0.24, 6),
        breathiness=round(0.02 + values[3] * 0.18, 6),
        jitter=round(0.002 + values[4] * 0.018, 6),
    )


FloatArray = NDArray[np.float64]


def pitch_contour(
    duration_s: float,
    sample_rate: int,
    profile: SpeakerProfile,
    *,
    seed: int,
    phrase_rate_hz: float = 1.7,
) -> FloatArray:
    """Generate a smooth sample-rate pitch track with seeded micro-jitter."""
    if duration_s < 0.0 or sample_rate <= 0 or phrase_rate_hz <= 0.0:
        raise ValueError("duration, sample rate, and phrase rate must be valid")
    sample_count = round(duration_s * sample_rate)
    if sample_count == 0:
        return np.empty(0, dtype=np.float64)
    times = np.arange(sample_count, dtype=np.float64) / sample_rate
    generator = rng(seed, f"pitch:{profile.speaker_id}")
    phase = generator.uniform(0.0, 2.0 * np.pi)
    phrase = profile.pitch_range_semitones * np.sin(2.0 * np.pi * phrase_rate_hz * times + phase)
    knot_count = max(2, round(duration_s * 20.0) + 1)
    knots = generator.normal(0.0, profile.jitter, size=knot_count)
    jitter = np.interp(times, np.linspace(0.0, duration_s, knot_count), knots)
    contour = profile.base_pitch_hz * np.exp2(phrase / 12.0 + jitter)
    return np.round(contour, 9)


# -- extensions --
