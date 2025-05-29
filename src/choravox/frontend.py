"""Structured front-end results with reproducible frame timing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import to_mono
from choravox.config import FeatureConfig
from choravox.features import (
    analysis_window,
    apply_filterbank,
    cmvn,
    frame_signal,
    frame_spectrum,
    log_compress,
    mel_filterbank,
    stack_feature_orders,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FeatureFrames:
    """A feature matrix paired with frame starts, centers, and duration."""

    values: FloatArray
    starts_s: FloatArray
    centers_s: FloatArray
    frame_duration_s: float


class LogMelExtractor:
    """Reusable front end with immutable cached window and mel filters."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = FeatureConfig() if config is None else config
        self._window = analysis_window(self.config.frame_length)
        self._bank = mel_filterbank(
            self.config.sample_rate,
            self.config.n_fft,
            self.config.n_mels,
            min_hz=self.config.min_hz,
            max_hz=self.config.max_hz,
        )
        self._window.setflags(write=False)
        self._bank.setflags(write=False)

    def transform(self, samples: ArrayLike) -> FloatArray:
        """Extract configured features while reusing cached analysis arrays."""
        waveform = to_mono(samples)
        frames = frame_signal(waveform, self.config.frame_length, self.config.hop_length, end="pad")
        if frames.shape[0] == 0:
            width = self.config.n_mels * (self.config.delta_order + 1)
            return np.empty((0, width), dtype=np.float64)
        power = frame_spectrum(frames, n_fft=self.config.n_fft, window=self._window, scale="power")
        static = log_compress(apply_filterbank(power, self._bank), floor=self.config.log_floor)
        normalized = cmvn(static)
        return np.round(stack_feature_orders(normalized, order=self.config.delta_order), 6)


def timed_log_mel(samples: ArrayLike, config: FeatureConfig | None = None) -> FeatureFrames:
    """Extract log-mel features and their sample-derived frame time axis."""
    settings = FeatureConfig() if config is None else config
    values = LogMelExtractor(settings).transform(samples)
    starts = (
        np.arange(values.shape[0], dtype=np.float64) * settings.hop_length / settings.sample_rate
    )
    duration = settings.frame_length / settings.sample_rate
    centers = starts + duration / 2.0
    return FeatureFrames(values, starts, centers, duration)


def extract_batch(
    recordings: Mapping[str, ArrayLike], config: FeatureConfig | None = None
) -> dict[str, FeatureFrames]:
    """Extract a name-sorted batch with one shared cached front end."""
    if any(not isinstance(name, str) or not name for name in recordings):
        raise ValueError("recording names must be nonempty strings")
    settings = FeatureConfig() if config is None else config
    extractor = LogMelExtractor(settings)
    output: dict[str, FeatureFrames] = {}
    for name in sorted(recordings):
        values = extractor.transform(recordings[name])
        starts = (
            np.arange(values.shape[0], dtype=np.float64)
            * settings.hop_length
            / settings.sample_rate
        )
        duration = settings.frame_length / settings.sample_rate
        output[name] = FeatureFrames(values, starts, starts + duration / 2.0, duration)
    return output


# -- extensions --
