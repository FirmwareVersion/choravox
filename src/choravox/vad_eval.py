"""Evaluation of synthetic VAD outputs against planted activity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.dialogue import DialogueScript
from choravox.vad import VadResult

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


def planted_frame_activity(
    script: DialogueScript, frame_starts_s: ArrayLike, frame_duration_s: float
) -> BoolArray:
    """Label a frame active when its center lies inside any planted turn."""
    starts = np.asarray(frame_starts_s, dtype=np.float64)
    if starts.ndim != 1 or not np.all(np.isfinite(starts)) or np.any(starts < 0.0):
        raise ValueError("frame_starts_s must be a finite nonnegative vector")
    if not np.isfinite(frame_duration_s) or frame_duration_s <= 0.0:
        raise ValueError("frame_duration_s must be finite and positive")
    centers = starts + frame_duration_s / 2.0
    return np.asarray(
        [any(turn.start_s <= center < turn.end_s for turn in script.turns) for center in centers],
        dtype=np.bool_,
    )


def evaluate_result(result: VadResult, script: DialogueScript) -> VadCounts:
    """Compare a VAD result to center-sampled planted speech activity."""
    if result.speech.shape != result.frames.starts_s.shape:
        raise ValueError("VAD result decisions and frame times must align")
    reference = planted_frame_activity(
        script, result.frames.starts_s, result.frames.frame_duration_s
    )
    return binary_counts(reference, result.speech)


@dataclass(frozen=True, slots=True)
class DetPoint:
    """Miss and false-alarm rates at one posterior threshold."""

    threshold: float
    miss_rate: float
    false_alarm_rate: float


def det_curve(
    posterior: ArrayLike,
    reference: ArrayLike,
    *,
    thresholds: ArrayLike | None = None,
) -> tuple[DetPoint, ...]:
    """Sweep posterior thresholds into a deterministic DET curve."""
    scores = np.asarray(posterior, dtype=np.float64)
    truth = _booleans(reference, "reference")
    if scores.ndim != 1 or scores.shape != truth.shape or not np.all(np.isfinite(scores)):
        raise ValueError("posterior and reference must be aligned finite vectors")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("posterior values must lie in [0, 1]")
    levels = (
        np.linspace(0.0, 1.0, 101)
        if thresholds is None
        else np.asarray(thresholds, dtype=np.float64)
    )
    if (
        levels.ndim != 1
        or not np.all(np.isfinite(levels))
        or np.any((levels < 0.0) | (levels > 1.0))
    ):
        raise ValueError("DET thresholds must be a finite vector in [0, 1]")
    points = []
    for threshold in levels:
        counts = binary_counts(truth, scores >= threshold)
        points.append(
            DetPoint(round(float(threshold), 6), counts.miss_rate, counts.false_alarm_rate)
        )
    return tuple(points)


# -- extensions --
