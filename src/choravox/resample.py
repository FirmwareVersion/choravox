"""Windowed-sinc polyphase resampling implemented with NumPy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform

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


def resample_polyphase(
    samples: ArrayLike,
    source_rate: int,
    target_rate: int,
    *,
    half_width: int = 16,
) -> FloatArray:
    """Resample mono or multichannel audio with a reusable rational phase bank."""
    waveform = as_waveform(samples)
    ratio = sampling_ratio(source_rate, target_rate)
    if source_rate == target_rate:
        return waveform.copy()
    output_length = resampled_length(waveform.shape[0], source_rate, target_rate)
    if output_length == 0:
        return np.empty((0, *waveform.shape[1:]), dtype=np.float64)
    kernel = polyphase_kernel(source_rate, target_rate, half_width=half_width)
    offsets = np.arange(-half_width + 1, half_width + 1)
    output = np.empty((output_length, *waveform.shape[1:]), dtype=np.float64)
    for output_index in range(output_length):
        numerator = output_index * ratio.down
        center, phase = divmod(numerator, ratio.up)
        indices = np.clip(center + offsets, 0, waveform.shape[0] - 1)
        selected = waveform[indices]
        weights = kernel[phase]
        if waveform.ndim == 2:
            output[output_index] = np.sum(selected * weights[:, None], axis=0)
        else:
            output[output_index] = np.sum(selected * weights)
    return output


# -- extensions --
