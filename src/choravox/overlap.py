"""Deterministic frame-level overlap detection for synthetic meetings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.diarization import MultiSpeakerFrames, activity_to_spans

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def overlap_targets(activity: ArrayLike) -> BoolArray:
    """Mark frames containing at least two planted active speakers."""
    matrix = np.asarray(activity)
    if matrix.ndim != 2:
        raise ValueError("speaker activity must be a frame-by-speaker matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("speaker activity must be binary")
    targets: BoolArray = np.asarray(np.sum(matrix.astype(np.int64), axis=1) >= 2, dtype=np.bool_)
    return targets


def frame_overlap_features(
    log_energy: ArrayLike, spectral_flux: ArrayLike, speaker_posteriors: ArrayLike
) -> FloatArray:
    """Combine acoustic cues with posterior ambiguity for each frame."""
    energy = np.asarray(log_energy, dtype=np.float64)
    flux = np.asarray(spectral_flux, dtype=np.float64)
    posteriors = np.asarray(speaker_posteriors, dtype=np.float64)
    if (
        energy.ndim != 1
        or flux.shape != energy.shape
        or posteriors.ndim != 2
        or posteriors.shape[0] != energy.size
        or posteriors.shape[1] == 0
    ):
        raise ValueError("overlap feature inputs have incompatible shapes")
    if not all(np.all(np.isfinite(item)) for item in (energy, flux, posteriors)):
        raise ValueError("overlap feature inputs must be finite")
    if np.any(posteriors < 0.0) or np.any(np.sum(posteriors, axis=1) <= 0.0):
        raise ValueError("speaker posteriors must be nonnegative with positive rows")
    normalized = posteriors / np.sum(posteriors, axis=1, keepdims=True)
    ordered = np.sort(normalized, axis=1)
    second = ordered[:, -2] if normalized.shape[1] > 1 else np.zeros(energy.size)
    entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-12)), axis=1)
    if normalized.shape[1] > 1:
        entropy /= np.log(normalized.shape[1])
    return np.round(np.column_stack((energy, flux, entropy, second)), 12)


@dataclass(frozen=True, slots=True)
class LogisticOverlapModel:
    """Small dependency-free binary overlap detector."""

    weights: FloatArray
    bias: float = 0.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)) or not np.isfinite(self.bias):
            raise ValueError("logistic parameters must be finite vectors and scalars")
        object.__setattr__(self, "weights", weights.copy())

    def predict_proba(self, features: ArrayLike) -> FloatArray:
        """Return stable positive-class probabilities."""
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.weights.size:
            raise ValueError("feature matrix does not match logistic weights")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("feature matrix must be finite")
        logits = np.clip(matrix @ self.weights + self.bias, -40.0, 40.0)
        return np.round(1.0 / (1.0 + np.exp(-logits)), 12)

    def predict(self, features: ArrayLike, *, threshold: float = 0.5) -> BoolArray:
        """Threshold positive-class probabilities into overlap decisions."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("overlap threshold must lie in [0, 1]")
        return self.predict_proba(features) >= threshold


def binary_log_loss(probabilities: ArrayLike, targets: ArrayLike) -> float:
    """Measure mean binary cross entropy with stable probability clipping."""
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.bool_)
    if values.ndim != 1 or labels.shape != values.shape or values.size == 0:
        raise ValueError("probabilities and targets must be aligned nonempty vectors")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite and lie in [0, 1]")
    clipped = np.clip(values, 1e-12, 1.0 - 1e-12)
    loss = -np.mean(labels * np.log(clipped) + (~labels) * np.log1p(-clipped))
    return round(float(loss), 12)


@dataclass(frozen=True, slots=True)
class LogisticFit:
    model: LogisticOverlapModel
    losses: tuple[float, ...]


def fit_logistic_overlap(
    features: ArrayLike,
    targets: ArrayLike,
    *,
    learning_rate: float = 0.2,
    iterations: int = 100,
    l2: float = 0.0,
) -> LogisticFit:
    """Fit logistic overlap weights with deterministic full-batch descent."""
    matrix = np.asarray(features, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.bool_)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or labels.shape != (matrix.shape[0],):
        raise ValueError("training features and targets are misaligned")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("training features must be finite")
    if learning_rate <= 0.0 or iterations < 1 or l2 < 0.0:
        raise ValueError("training controls are invalid")
    weights = np.zeros(matrix.shape[1], dtype=np.float64)
    bias = 0.0
    losses = [binary_log_loss(np.full(labels.size, 0.5), labels)]
    numeric_labels = labels.astype(np.float64)
    for _ in range(iterations):
        model = LogisticOverlapModel(weights, bias)
        probabilities = model.predict_proba(matrix)
        residual = probabilities - numeric_labels
        weights -= learning_rate * (matrix.T @ residual / labels.size + l2 * weights)
        bias -= learning_rate * float(np.mean(residual))
        losses.append(
            binary_log_loss(LogisticOverlapModel(weights, bias).predict_proba(matrix), labels)
        )
    return LogisticFit(LogisticOverlapModel(np.round(weights, 12), round(bias, 12)), tuple(losses))


@dataclass(frozen=True, slots=True)
class OverlapCounts:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return 0.0 if denominator == 0 else round(self.true_positive / denominator, 12)

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return 0.0 if denominator == 0 else round(self.true_positive / denominator, 12)

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return (
            0.0 if denominator == 0.0 else round(2 * self.precision * self.recall / denominator, 12)
        )


def overlap_counts(targets: ArrayLike, predictions: ArrayLike) -> OverlapCounts:
    """Compute a complete binary overlap confusion table."""
    expected = np.asarray(targets, dtype=np.bool_)
    observed = np.asarray(predictions, dtype=np.bool_)
    if expected.ndim != 1 or observed.shape != expected.shape:
        raise ValueError("overlap targets and predictions must be aligned vectors")
    return OverlapCounts(
        int(np.sum(expected & observed)),
        int(np.sum(~expected & observed)),
        int(np.sum(expected & ~observed)),
        int(np.sum(~expected & ~observed)),
    )


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    counts: OverlapCounts


def select_overlap_threshold(
    probabilities: ArrayLike,
    targets: ArrayLike,
    thresholds: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 0.8),
) -> ThresholdSelection:
    """Select maximum F1, preferring a threshold nearest 0.5 then lower."""
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.bool_)
    if values.ndim != 1 or values.size == 0 or labels.shape != values.shape:
        raise ValueError("calibration probabilities and targets are misaligned")
    if not thresholds or any(not 0.0 <= item <= 1.0 for item in thresholds):
        raise ValueError("candidate thresholds must lie in [0, 1]")
    candidates = [
        ThresholdSelection(item, overlap_counts(labels, values >= item)) for item in thresholds
    ]
    return max(
        candidates, key=lambda item: (item.counts.f1, -abs(item.threshold - 0.5), -item.threshold)
    )


def decode_overlap_spans(
    primary_labels: tuple[str, ...],
    speaker_posteriors: ArrayLike,
    speakers: tuple[str, ...],
    overlap_mask: ArrayLike,
) -> tuple[MultiSpeakerFrames, ...]:
    """Add the strongest non-primary speaker on predicted overlap frames."""
    posteriors = np.asarray(speaker_posteriors, dtype=np.float64)
    mask = np.asarray(overlap_mask, dtype=np.bool_)
    frames = len(primary_labels)
    if (
        posteriors.shape != (frames, len(speakers))
        or mask.shape != (frames,)
        or len(set(speakers)) != len(speakers)
        or any(label not in speakers for label in primary_labels)
    ):
        raise ValueError("overlap decoding inputs have incompatible rosters or shapes")
    if not np.all(np.isfinite(posteriors)):
        raise ValueError("speaker posteriors must be finite")
    if np.any(mask) and len(speakers) < 2:
        raise ValueError("overlap decoding requires two candidate speakers")
    lookup = {speaker: index for index, speaker in enumerate(speakers)}
    activity = np.zeros((frames, len(speakers)), dtype=np.bool_)
    for frame, primary in enumerate(primary_labels):
        primary_index = lookup[primary]
        activity[frame, primary_index] = True
        if mask[frame]:
            scores = posteriors[frame].copy()
            scores[primary_index] = -np.inf
            activity[frame, int(np.argmax(scores))] = True
    return activity_to_spans(activity, speakers)


# -- extensions --
