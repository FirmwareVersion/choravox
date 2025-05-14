"""Linear-phase FIR design and samples-first filtering."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
FilterWindow = Literal["hann", "hamming", "blackman"]


def lowpass_kernel(
    sample_rate: int,
    cutoff_hz: float,
    *,
    taps: int = 63,
    window: FilterWindow = "hamming",
) -> FloatArray:
    """Design an odd-length windowed-sinc low-pass kernel with unit DC gain."""
    if sample_rate <= 0 or not 0.0 < cutoff_hz < sample_rate / 2:
        raise ValueError("cutoff_hz must lie between zero and Nyquist")
    if taps < 3 or taps % 2 == 0:
        raise ValueError("taps must be an odd integer of at least three")
    positions = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    normalized = cutoff_hz / sample_rate
    ideal = 2.0 * normalized * np.sinc(2.0 * normalized * positions)
    if window == "hann":
        weights = np.hanning(taps)
    elif window == "hamming":
        weights = np.hamming(taps)
    elif window == "blackman":
        weights = np.blackman(taps)
    else:
        raise ValueError(f"unknown filter window: {window}")
    kernel = ideal * weights
    return np.asarray(kernel / np.sum(kernel), dtype=np.float64)


# -- extensions --
