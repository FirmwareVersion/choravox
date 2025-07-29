"""Deterministic synthetic noise generation and calibrated mixing."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from choravox.random import rng

FloatArray = NDArray[np.float64]


def _calibrate_rms(samples: FloatArray, rms_level: float) -> FloatArray:
    current = float(np.sqrt(np.mean(np.square(samples))))
    if current == 0.0:
        raise ValueError("noise realization must have nonzero RMS")
    return np.asarray(samples * (rms_level / current), dtype=np.float64)


def white_noise(sample_count: int, *, seed: int, rms_level: float = 1.0) -> FloatArray:
    """Generate zero-mean Gaussian noise at a requested RMS level."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(rms_level) or rms_level <= 0.0:
        raise ValueError("rms_level must be finite and positive")
    samples = rng(seed, "white-noise").standard_normal(sample_count)
    samples -= np.mean(samples)
    return np.round(_calibrate_rms(samples, rms_level), 9)


def pink_noise(sample_count: int, *, seed: int, rms_level: float = 1.0) -> FloatArray:
    """Generate periodic finite-length pink noise by shaping white-noise FFT bins."""
    if sample_count < 2:
        raise ValueError("pink noise requires at least two samples")
    if not math.isfinite(rms_level) or rms_level <= 0.0:
        raise ValueError("rms_level must be finite and positive")
    white = rng(seed, "pink-noise").standard_normal(sample_count)
    spectrum = np.fft.rfft(white)
    spectrum[0] = 0.0
    spectrum[1:] /= np.sqrt(np.arange(1, spectrum.size, dtype=np.float64))
    samples = np.fft.irfft(spectrum, n=sample_count)
    return np.round(_calibrate_rms(samples, rms_level), 9)


# -- extensions --
