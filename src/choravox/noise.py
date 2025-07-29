"""Deterministic synthetic noise generation and calibrated mixing."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from choravox.audio import normalize_rms
from choravox.random import rng

FloatArray = NDArray[np.float64]


def white_noise(sample_count: int, *, seed: int, rms_level: float = 1.0) -> FloatArray:
    """Generate zero-mean Gaussian noise at a requested RMS level."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(rms_level) or rms_level <= 0.0:
        raise ValueError("rms_level must be finite and positive")
    samples = rng(seed, "white-noise").standard_normal(sample_count)
    samples -= np.mean(samples)
    return np.round(normalize_rms(samples, rms_level), 9)


# -- extensions --
