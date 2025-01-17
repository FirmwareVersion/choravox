"""Exact conventions for sample, frame, and wall-clock indices."""

from __future__ import annotations


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert a nonnegative sample boundary to quantized seconds."""
    if samples < 0 or sample_rate <= 0:
        raise ValueError("samples must be nonnegative and sample_rate positive")
    return round(samples / sample_rate, 9)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Map seconds to the nearest sample boundary using half-up rounding."""
    if seconds < 0 or sample_rate <= 0:
        raise ValueError("seconds must be nonnegative and sample_rate positive")
    return int(seconds * sample_rate + 0.5)


# -- extensions --
