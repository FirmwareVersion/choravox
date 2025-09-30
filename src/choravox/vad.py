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


def adaptive_threshold_db(
    energy_db: ArrayLike, *, noise_quantile: float = 0.2, margin_db: float = 6.0
) -> float:
    """Estimate a speech threshold above a lower-tail noise floor."""
    values = np.asarray(energy_db, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("energy_db must be a nonempty finite vector")
    if not 0.0 <= noise_quantile <= 1.0 or not math.isfinite(margin_db) or margin_db < 0.0:
        raise ValueError("noise_quantile and margin_db are invalid")
    noise_floor = float(np.quantile(values, noise_quantile))
    return round(noise_floor + margin_db, 6)


def energy_posterior(
    energy_db: ArrayLike, threshold_db: float, *, temperature_db: float = 2.0
) -> FloatArray:
    """Map frame energy to calibrated logistic speech probabilities."""
    values = np.asarray(energy_db, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("energy_db must be a finite vector")
    if (
        not math.isfinite(threshold_db)
        or not math.isfinite(temperature_db)
        or temperature_db <= 0.0
    ):
        raise ValueError("posterior threshold and temperature must be finite and valid")
    logits = np.clip((values - threshold_db) / temperature_db, -60.0, 60.0)
    return np.round(1.0 / (1.0 + np.exp(-logits)), 9)


def hysteresis_decode(
    posterior: ArrayLike,
    *,
    onset: float = 0.6,
    offset: float = 0.4,
    initial_speech: bool = False,
) -> BoolArray:
    """Decode speech state with separate entry and exit thresholds."""
    values = np.asarray(posterior, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("posterior must be a finite probability vector")
    if not 0.0 <= offset <= onset <= 1.0:
        raise ValueError("hysteresis requires 0 <= offset <= onset <= 1")
    output = np.empty(values.size, dtype=np.bool_)
    active = initial_speech
    for index, value in enumerate(values):
        if active and value < offset:
            active = False
        elif not active and value >= onset:
            active = True
        output[index] = active
    return output


# -- extensions --
