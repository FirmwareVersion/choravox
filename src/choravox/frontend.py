"""Structured front-end results with reproducible frame timing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.config import FeatureConfig
from choravox.features import extract_log_mel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureFrames:
    """A feature matrix paired with frame starts, centers, and duration."""

    values: FloatArray
    starts_s: FloatArray
    centers_s: FloatArray
    frame_duration_s: float


def timed_log_mel(samples: ArrayLike, config: FeatureConfig | None = None) -> FeatureFrames:
    """Extract log-mel features and their sample-derived frame time axis."""
    settings = FeatureConfig() if config is None else config
    values = extract_log_mel(samples, settings)
    starts = (
        np.arange(values.shape[0], dtype=np.float64) * settings.hop_length / settings.sample_rate
    )
    duration = settings.frame_length / settings.sample_rate
    centers = starts + duration / 2.0
    return FeatureFrames(values, starts, centers, duration)


# -- extensions --
