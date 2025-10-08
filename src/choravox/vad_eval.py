"""Evaluation of synthetic VAD outputs against planted activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]


def _booleans(values: ArrayLike, name: str) -> BoolArray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean vector")
    return array.astype(np.bool_, copy=False)


@dataclass(frozen=True, slots=True)
class VadCounts:
    """Frame counts and denominators for miss and false-alarm rates."""

    missed: int
    false_alarm: int
    reference_speech: int
    reference_silence: int

    def __post_init__(self) -> None:
        if min(self.missed, self.false_alarm, self.reference_speech, self.reference_silence) < 0:
            raise ValueError("VAD counts must be nonnegative")
        if self.missed > self.reference_speech or self.false_alarm > self.reference_silence:
            raise ValueError("VAD errors cannot exceed their reference denominators")

    @property
    def miss_rate(self) -> float:
        return 0.0 if self.reference_speech == 0 else round(self.missed / self.reference_speech, 6)

    @property
    def false_alarm_rate(self) -> float:
        return (
            0.0
            if self.reference_silence == 0
            else round(self.false_alarm / self.reference_silence, 6)
        )

    @property
    def error_rate(self) -> float:
        total = self.reference_speech + self.reference_silence
        return 0.0 if total == 0 else round((self.missed + self.false_alarm) / total, 6)


def binary_counts(reference: ArrayLike, hypothesis: ArrayLike) -> VadCounts:
    """Count frame misses and false alarms against boolean planted truth."""
    truth = _booleans(reference, "reference")
    predicted = _booleans(hypothesis, "hypothesis")
    if truth.shape != predicted.shape:
        raise ValueError("reference and hypothesis must have the same shape")
    return VadCounts(
        int(np.sum(truth & ~predicted)),
        int(np.sum(~truth & predicted)),
        int(np.sum(truth)),
        int(np.sum(~truth)),
    )


# -- extensions --
