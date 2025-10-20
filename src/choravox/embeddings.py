"""Classical synthetic-trained speaker embedding mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SufficientStatistics:
    """Mixture occupancy, first-order, and second-order feature sums."""

    counts: FloatArray
    first_order: FloatArray
    second_order: FloatArray

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype=np.float64)
        first = np.asarray(self.first_order, dtype=np.float64)
        second = np.asarray(self.second_order, dtype=np.float64)
        if counts.ndim != 1 or first.ndim != 2 or second.shape != first.shape:
            raise ValueError("sufficient statistics shapes are invalid")
        if first.shape[0] != counts.size or not all(
            np.all(np.isfinite(value)) for value in (counts, first, second)
        ):
            raise ValueError("sufficient statistics must be aligned and finite")
        if np.any(counts < 0.0) or np.any(second < 0.0):
            raise ValueError("sufficient statistics counts and squares must be nonnegative")


def accumulate_statistics(features: ArrayLike, responsibilities: ArrayLike) -> SufficientStatistics:
    """Accumulate zero, first, and second order statistics for soft assignments."""
    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(responsibilities, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("features must be a finite matrix")
    if weights.ndim != 2 or weights.shape[0] != values.shape[0] or not np.all(np.isfinite(weights)):
        raise ValueError("responsibilities must align with feature rows")
    if np.any(weights < 0.0) or (
        weights.shape[0] and not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("responsibility rows must be nonnegative and sum to one")
    counts = np.sum(weights, axis=0)
    first = weights.T @ values
    second = weights.T @ np.square(values)
    return SufficientStatistics(np.round(counts, 9), np.round(first, 9), np.round(second, 9))


# -- extensions --
