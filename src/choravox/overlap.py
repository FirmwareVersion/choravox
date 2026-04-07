"""Deterministic frame-level overlap detection for synthetic meetings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def overlap_targets(activity: ArrayLike) -> BoolArray:
    """Mark frames containing at least two planted active speakers."""
    matrix = np.asarray(activity)
    if matrix.ndim != 2:
        raise ValueError("speaker activity must be a frame-by-speaker matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("speaker activity must be binary")
    return np.sum(matrix.astype(np.int64), axis=1) >= 2


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


# -- extensions --
