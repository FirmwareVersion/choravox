"""Deterministic source-filter synthesis for phoneme-like sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Resonator:
    """Second-order all-pole resonator coefficients."""

    b0: float
    a1: float
    a2: float


def resonator_coefficients(frequency_hz: float, bandwidth_hz: float, sample_rate: int) -> Resonator:
    """Create stable coefficients for one digital formant resonance."""
    if sample_rate <= 0 or not 0.0 < frequency_hz < sample_rate / 2:
        raise ValueError("formant frequency must lie between zero and Nyquist")
    if not 0.0 < bandwidth_hz < sample_rate / 2:
        raise ValueError("formant bandwidth must lie between zero and Nyquist")
    radius = float(np.exp(-np.pi * bandwidth_hz / sample_rate))
    angle = 2.0 * np.pi * frequency_hz / sample_rate
    return Resonator(1.0 - radius, 2.0 * radius * float(np.cos(angle)), -(radius**2))


def apply_resonator(samples: ArrayLike, coefficients: Resonator) -> FloatArray:
    """Apply a causal resonator independently along every waveform channel."""
    waveform = as_waveform(samples)
    output = np.empty_like(waveform)
    if waveform.ndim == 1:
        previous_one = previous_two = 0.0
        for index, sample in enumerate(waveform):
            value = (
                coefficients.b0 * sample
                + coefficients.a1 * previous_one
                + coefficients.a2 * previous_two
            )
            output[index] = value
            previous_two, previous_one = previous_one, value
    else:
        previous_one = np.zeros(waveform.shape[1], dtype=np.float64)
        previous_two = np.zeros(waveform.shape[1], dtype=np.float64)
        for index, sample in enumerate(waveform):
            value = (
                coefficients.b0 * sample
                + coefficients.a1 * previous_one
                + coefficients.a2 * previous_two
            )
            output[index] = value
            previous_two, previous_one = previous_one, value
    return output


# -- extensions --
