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


def to_mono(samples: ArrayLike, weights: ArrayLike | None = None) -> FloatArray:
    """Downmix samples-first audio without changing an existing mono signal."""
    waveform = as_waveform(samples)
    if waveform.ndim == 1:
        if weights is not None:
            raise ValueError("weights are only valid for multi-channel audio")
        return waveform.copy()
    channel_weights = (
        np.ones(waveform.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64)
    )
    if channel_weights.shape != (waveform.shape[1],) or not np.all(np.isfinite(channel_weights)):
        raise ValueError("weights must contain one finite value per channel")
    total = float(np.sum(channel_weights))
    if total == 0.0:
        raise ValueError("channel weights must not sum to zero")
    return np.asarray(waveform @ (channel_weights / total), dtype=np.float64)


# -- extensions --
