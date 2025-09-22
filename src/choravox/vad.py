"""Deterministic frame-energy and statistical voice activity detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import to_mono
from choravox.features import frame_signal

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FrameEnergy:
    """Frame timing paired with RMS energy in dBFS."""

    starts_s: FloatArray
    centers_s: FloatArray
    dbfs: FloatArray
    frame_duration_s: float


def frame_energy_db(
    samples: ArrayLike,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    floor_db: float = -120.0,
) -> FrameEnergy:
    """Measure mono frame RMS on a sample-derived time axis."""
    if sample_rate <= 0 or frame_ms <= 0.0 or hop_ms <= 0.0 or hop_ms > frame_ms:
        raise ValueError("VAD sample rate and frame timing must be positive and ordered")
    if not math.isfinite(floor_db) or floor_db > 0.0:
        raise ValueError("floor_db must be finite and nonpositive")
    frame_length = round(sample_rate * frame_ms / 1000.0)
    hop_length = round(sample_rate * hop_ms / 1000.0)
    frames = frame_signal(to_mono(samples), frame_length, hop_length, end="pad")
    if frames.shape[0] == 0:
        values = np.empty(0, dtype=np.float64)
    else:
        power = np.mean(np.square(frames), axis=1)
        values = np.maximum(floor_db, 10.0 * np.log10(np.maximum(power, 10.0 ** (floor_db / 10.0))))
        values = np.round(values, 6)
    starts = np.arange(values.size, dtype=np.float64) * hop_length / sample_rate
    duration = frame_length / sample_rate
    return FrameEnergy(starts, starts + duration / 2.0, values, duration)


# -- extensions --
