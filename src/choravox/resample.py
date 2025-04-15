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


# -- extensions --
