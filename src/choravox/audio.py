"""Deterministic NumPy signal utilities using samples-first arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_waveform(samples: ArrayLike) -> FloatArray:
    """Return finite float64 mono or samples-by-channels audio."""
    waveform = np.asarray(samples, dtype=np.float64)
    if waveform.ndim not in (1, 2):
        raise ValueError("waveform must have shape (samples,) or (samples, channels)")
    if waveform.ndim == 2 and waveform.shape[1] == 0:
        raise ValueError("waveform must contain at least one channel")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform samples must be finite")
    return waveform


# -- extensions --
