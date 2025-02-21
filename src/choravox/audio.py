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


def peak_amplitude(samples: ArrayLike) -> float:
    """Return the largest absolute sample, or zero for empty audio."""
    waveform = as_waveform(samples)
    return 0.0 if waveform.size == 0 else round(float(np.max(np.abs(waveform))), 12)


def rms(samples: ArrayLike, *, axis: int | None = None) -> float | FloatArray:
    """Compute root-mean-square amplitude globally or along one axis."""
    waveform = as_waveform(samples)
    if waveform.size == 0:
        if axis is None:
            return 0.0
        shape = tuple(size for index, size in enumerate(waveform.shape) if index != axis)
        return np.zeros(shape, dtype=np.float64)
    value = np.sqrt(np.mean(np.square(waveform), axis=axis))
    if np.ndim(value) == 0:
        return round(float(value), 12)
    return np.round(np.asarray(value, dtype=np.float64), 12)


def dbfs(samples: ArrayLike, *, floor_db: float = -120.0) -> float:
    """Measure RMS decibels relative to a full-scale amplitude of one."""
    if floor_db > 0:
        raise ValueError("floor_db must not be positive")
    level = float(rms(samples))
    if level <= 0:
        return float(floor_db)
    return round(max(float(floor_db), 20.0 * float(np.log10(level))), 6)


def normalize_peak(samples: ArrayLike, target: float = 0.99) -> FloatArray:
    """Scale audio to a peak target while preserving exact silence."""
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    waveform = as_waveform(samples)
    peak = peak_amplitude(waveform)
    if peak == 0.0:
        return waveform.copy()
    return np.asarray(waveform * (target / peak), dtype=np.float64)


def normalize_rms(samples: ArrayLike, target: float, *, peak_limit: float = 1.0) -> FloatArray:
    """Scale toward target RMS and reduce gain when the peak ceiling binds."""
    if target <= 0 or not 0 < peak_limit <= 1:
        raise ValueError("target and peak_limit must be positive, with peak_limit <= 1")
    waveform = as_waveform(samples)
    current = float(rms(waveform))
    if current == 0.0:
        return waveform.copy()
    gain = target / current
    peak = peak_amplitude(waveform)
    if peak > 0:
        gain = min(gain, peak_limit / peak)
    return np.asarray(waveform * gain, dtype=np.float64)


def clipping_fraction(samples: ArrayLike, limit: float = 1.0) -> float:
    """Return the fraction of samples at or beyond an absolute limit."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    waveform = as_waveform(samples)
    if waveform.size == 0:
        return 0.0
    return round(float(np.mean(np.abs(waveform) >= limit)), 6)


def hard_clip(samples: ArrayLike, limit: float = 1.0) -> FloatArray:
    """Clamp samples symmetrically to an absolute limit."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    return np.asarray(np.clip(as_waveform(samples), -limit, limit), dtype=np.float64)


def soft_clip(samples: ArrayLike, drive: float = 1.0) -> FloatArray:
    """Apply tanh saturation normalized to preserve the full-scale endpoint."""
    if drive <= 0:
        raise ValueError("drive must be positive")
    waveform = as_waveform(samples)
    scale = float(np.tanh(drive))
    return np.asarray(np.tanh(waveform * drive) / scale, dtype=np.float64)


# -- extensions --
