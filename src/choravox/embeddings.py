"""Classical synthetic-trained speaker embedding mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.random import rng

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


@dataclass(frozen=True, slots=True)
class DiagonalGmm:
    """A normalized finite mixture with diagonal component covariance."""

    weights: FloatArray
    means: FloatArray
    variances: FloatArray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        means = np.asarray(self.means, dtype=np.float64)
        variances = np.asarray(self.variances, dtype=np.float64)
        if (
            weights.ndim != 1
            or weights.size == 0
            or means.ndim != 2
            or means.shape != variances.shape
            or means.shape[0] != weights.size
        ):
            raise ValueError("GMM parameter shapes are invalid")
        if not all(np.all(np.isfinite(value)) for value in (weights, means, variances)):
            raise ValueError("GMM parameters must be aligned and finite")
        if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-8):
            raise ValueError("GMM weights must be positive and sum to one")
        if np.any(variances <= 0.0):
            raise ValueError("GMM variances must be positive")

    @property
    def components(self) -> int:
        return int(self.weights.size)

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])


def initialize_gmm(
    features: ArrayLike,
    components: int,
    *,
    seed: int,
    variance_floor: float = 1e-4,
) -> DiagonalGmm:
    """Choose seeded observations as initial means with shared global variance."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("GMM initialization features must be a nonempty finite matrix")
    if components < 1 or components > values.shape[0]:
        raise ValueError("component count must fit the observation count")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive")
    indices = rng(seed, "gmm-initialize").choice(values.shape[0], components, replace=False)
    means = values[indices]
    global_variance = np.maximum(np.var(values, axis=0), variance_floor)
    variances = np.repeat(global_variance[None, :], components, axis=0)
    weights = np.full(components, 1.0 / components)
    return DiagonalGmm(np.round(weights, 12), np.round(means, 9), np.round(variances, 9))


def gmm_log_probabilities(features: ArrayLike, model: DiagonalGmm) -> FloatArray:
    """Evaluate weighted component log densities for every feature frame."""
    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1:] != (model.dimension,)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("features must be a finite matrix matching the GMM")
    differences = values[:, None, :] - model.means[None, :, :]
    log_density = -0.5 * np.sum(
        np.log(2.0 * np.pi * model.variances)[None, :, :]
        + np.square(differences) / model.variances[None, :, :],
        axis=2,
    )
    return np.asarray(log_density + np.log(model.weights)[None, :], dtype=np.float64)


def gmm_responsibilities(features: ArrayLike, model: DiagonalGmm) -> FloatArray:
    """Normalize component log densities with stable log-sum-exp."""
    scores = gmm_log_probabilities(features, model)
    if scores.shape[0] == 0:
        return np.empty((0, model.components), dtype=np.float64)
    maximum = np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores - maximum)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return np.round(probabilities, 12)


def estimate_gmm(statistics: SufficientStatistics, *, variance_floor: float = 1e-4) -> DiagonalGmm:
    """Estimate mixture parameters from sufficient statistics."""
    if variance_floor <= 0.0 or not np.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and positive")
    if statistics.counts.size == 0 or np.any(statistics.counts <= 0.0):
        raise ValueError("every GMM component must have positive occupancy")
    total = float(np.sum(statistics.counts))
    means = statistics.first_order / statistics.counts[:, None]
    raw_variances = statistics.second_order / statistics.counts[:, None] - np.square(means)
    variances = np.maximum(raw_variances, variance_floor)
    weights = statistics.counts / total
    return DiagonalGmm(np.round(weights, 12), np.round(means, 9), np.round(variances, 9))


def combine_statistics(items: list[SufficientStatistics]) -> SufficientStatistics:
    """Combine utterance statistics without revisiting their feature frames."""
    if not items:
        raise ValueError("at least one statistics item is required")
    shape = items[0].first_order.shape
    if any(item.first_order.shape != shape for item in items):
        raise ValueError("statistics items must share component and feature dimensions")
    return SufficientStatistics(
        np.round(np.sum([item.counts for item in items], axis=0), 9),
        np.round(np.sum([item.first_order for item in items], axis=0), 9),
        np.round(np.sum([item.second_order for item in items], axis=0), 9),
    )


def expectation_statistics(features: ArrayLike, model: DiagonalGmm) -> SufficientStatistics:
    """Run the GMM expectation step for a feature sequence."""
    values = np.asarray(features, dtype=np.float64)
    return accumulate_statistics(values, gmm_responsibilities(values, model))


def average_log_likelihood(features: ArrayLike, model: DiagonalGmm) -> float:
    """Return mean log likelihood under a mixture using stable reduction."""
    scores = gmm_log_probabilities(features, model)
    if scores.shape[0] == 0:
        raise ValueError("likelihood requires at least one feature frame")
    maximum = np.max(scores, axis=1)
    values = maximum + np.log(np.sum(np.exp(scores - maximum[:, None]), axis=1))
    return round(float(np.mean(values)), 9)


@dataclass(frozen=True, slots=True)
class GmmFit:
    """A fitted mixture and its expectation-maximization objective trace."""

    model: DiagonalGmm
    likelihoods: tuple[float, ...]


def fit_gmm(
    features: ArrayLike,
    components: int,
    *,
    seed: int,
    iterations: int = 8,
    variance_floor: float = 1e-4,
) -> GmmFit:
    """Fit a diagonal GMM with deterministic expectation-maximization."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    values = np.asarray(features, dtype=np.float64)
    model = initialize_gmm(values, components, seed=seed, variance_floor=variance_floor)
    trace: list[float] = []
    for _ in range(iterations):
        statistics = expectation_statistics(values, model)
        model = estimate_gmm(statistics, variance_floor=variance_floor)
        trace.append(average_log_likelihood(values, model))
    return GmmFit(model, tuple(trace))


def centered_statistics(statistics: SufficientStatistics, model: DiagonalGmm) -> FloatArray:
    """Center first-order statistics relative to UBM component means."""
    if statistics.first_order.shape != model.means.shape:
        raise ValueError("statistics must match the UBM geometry")
    centered = statistics.first_order - statistics.counts[:, None] * model.means
    return np.round(centered, 9)


def statistics_supervector(statistics: SufficientStatistics, model: DiagonalGmm) -> FloatArray:
    """Whiten and flatten UBM-centered statistics into a fixed-size vector."""
    centered = centered_statistics(statistics, model)
    scale = np.sqrt(np.maximum(statistics.counts[:, None], 1.0) * model.variances)
    return np.round((centered / scale).reshape(-1), 9)


def normalize_embedding(values: ArrayLike) -> FloatArray:
    """Length-normalize one finite speaker embedding."""
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("embedding must be a nonempty finite vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("embedding norm must be positive")
    return np.round(vector / norm, 12)


def embed_features(features: ArrayLike, model: DiagonalGmm) -> FloatArray:
    """Extract a length-normalized statistics embedding from one utterance."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("embedding extraction requires feature frames")
    return normalize_embedding(statistics_supervector(expectation_statistics(values, model), model))


# -- extensions --
