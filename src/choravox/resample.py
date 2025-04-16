"""Windowed-sinc polyphase resampling implemented with NumPy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SamplingRatio:
    """A reduced integer up/down sampling ratio."""

    up: int
    down: int


def sampling_ratio(source_rate: int, target_rate: int) -> SamplingRatio:
    """Reduce positive sampling rates to their smallest integer ratio."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sampling rates must be positive")
    divisor = math.gcd(source_rate, target_rate)
    return SamplingRatio(target_rate // divisor, source_rate // divisor)


def resampled_length(sample_count: int, source_rate: int, target_rate: int) -> int:
    """Return nearest duration-preserving sample count with half-up rounding."""
    if sample_count < 0:
        raise ValueError("sample_count must be nonnegative")
    sampling_ratio(source_rate, target_rate)
    return (sample_count * target_rate + source_rate // 2) // source_rate


def polyphase_kernel(source_rate: int, target_rate: int, *, half_width: int = 16) -> FloatArray:
    """Create normalized Hann-windowed sinc taps for every rational phase."""
    ratio = sampling_ratio(source_rate, target_rate)
    if half_width < 2:
        raise ValueError("half_width must be at least two")
    offsets = np.arange(-half_width + 1, half_width + 1, dtype=np.float64)
    phases = np.empty((ratio.up, offsets.size), dtype=np.float64)
    cutoff = min(1.0, target_rate / source_rate) * 0.94
    for phase in range(ratio.up):
        distance = phase / ratio.up - offsets
        window = np.where(
            np.abs(distance) < half_width,
            0.5 + 0.5 * np.cos(np.pi * distance / half_width),
            0.0,
        )
        taps = cutoff * np.sinc(cutoff * distance) * window
        phases[phase] = taps / np.sum(taps)
    return phases


# -- extensions --
