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


def cosine_similarity(left: ArrayLike, right: ArrayLike) -> float:
    """Score two speaker embeddings by their normalized dot product."""
    first = normalize_embedding(left)
    second = normalize_embedding(right)
    if first.shape != second.shape:
        raise ValueError("speaker embeddings must have matching dimensions")
    return round(float(first @ second), 12)


def pairwise_cosine(enrollment: ArrayLike, probes: ArrayLike) -> FloatArray:
    """Build an enrollment-by-probe cosine score matrix."""
    left = np.asarray(enrollment, dtype=np.float64)
    right = np.asarray(probes, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != right.shape[1:]:
        raise ValueError("embedding matrices must have matching dimensions")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("embedding matrices must be finite")
    left_norms = np.linalg.norm(left, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    if np.any(left_norms == 0.0) or np.any(right_norms == 0.0):
        raise ValueError("embedding rows must have positive norm")
    return np.round((left / left_norms) @ (right / right_norms).T, 12)


def center_cohort(embeddings: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Center an embedding cohort and return its reusable mean vector."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("cohort must be a nonempty finite matrix")
    mean = np.mean(matrix, axis=0)
    return np.round(matrix - mean, 12), np.round(mean, 12)


@dataclass(frozen=True, slots=True)
class PldaModel:
    """Synthetic-trained linear discriminant transform for speaker scoring."""

    mean: FloatArray
    projection: FloatArray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        projection = np.asarray(self.projection, dtype=np.float64)
        if (
            mean.ndim != 1
            or mean.size == 0
            or projection.ndim != 2
            or projection.shape[0] != mean.size
            or projection.shape[1] == 0
        ):
            raise ValueError("PLDA parameter shapes are invalid")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(projection)):
            raise ValueError("PLDA parameters must be finite")

    @property
    def input_dimension(self) -> int:
        return int(self.mean.size)

    @property
    def output_dimension(self) -> int:
        return int(self.projection.shape[1])


def _stable_eigen_signs(vectors: FloatArray) -> FloatArray:
    result = vectors.copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0.0:
            result[:, column] *= -1.0
    return result


def fit_plda(
    values: ArrayLike,
    labels: list[str] | tuple[str, ...],
    *,
    output_dimension: int | None = None,
    regularization: float = 1e-3,
) -> PldaModel:
    """Estimate a regularized PLDA-like discriminant projection."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels) or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings and labels must form an aligned finite matrix")
    speakers = tuple(dict.fromkeys(labels))
    if matrix.shape[0] < 2 or len(speakers) < 2 or any(not label for label in labels):
        raise ValueError("PLDA training requires at least two labeled speakers")
    dimension = matrix.shape[1]
    target = min(len(speakers) - 1, dimension) if output_dimension is None else output_dimension
    if target < 1 or target > dimension:
        raise ValueError("output_dimension must fit the embedding dimension")
    if regularization <= 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and positive")
    mean = np.mean(matrix, axis=0)
    within = np.zeros((dimension, dimension), dtype=np.float64)
    between = np.zeros_like(within)
    for speaker in speakers:
        group = matrix[np.asarray(labels) == speaker]
        group_mean = np.mean(group, axis=0)
        residual = group - group_mean
        within += residual.T @ residual
        offset = group_mean - mean
        between += group.shape[0] * np.outer(offset, offset)
    within = within / matrix.shape[0] + regularization * np.eye(dimension)
    between /= matrix.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(within)
    whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    discriminant = whitening.T @ between @ whitening
    values_between, vectors_between = np.linalg.eigh(discriminant)
    order = np.argsort(values_between)[::-1][:target]
    projection = _stable_eigen_signs(whitening @ vectors_between[:, order])
    return PldaModel(np.round(mean, 12), np.round(projection, 12))


def project_plda(values: ArrayLike, model: PldaModel, *, normalize: bool = True) -> FloatArray:
    """Project one embedding or a matrix into PLDA score space."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim not in {1, 2} or matrix.shape[-1] != model.input_dimension:
        raise ValueError("embeddings must match the PLDA input dimension")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be finite")
    projected = (matrix - model.mean) @ model.projection
    if normalize:
        norms = np.linalg.norm(projected, axis=-1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("centered PLDA embeddings must have positive norm")
        projected = projected / norms
    return np.round(np.asarray(projected, dtype=np.float64), 12)


def plda_scores(enrollment: ArrayLike, probes: ArrayLike, model: PldaModel) -> FloatArray:
    """Score enrollment and probe matrices after a learned projection."""
    left = project_plda(enrollment, model)
    right = project_plda(probes, model)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("PLDA scoring requires embedding matrices")
    return np.round(left @ right.T, 12)


@dataclass(frozen=True, slots=True)
class ScoreDistributions:
    """Genuine and impostor scores from planted speaker identities."""

    genuine: FloatArray
    impostor: FloatArray

    @property
    def genuine_mean(self) -> float:
        return round(float(np.mean(self.genuine)), 12)

    @property
    def impostor_mean(self) -> float:
        return round(float(np.mean(self.impostor)), 12)

    @property
    def separation(self) -> float:
        return round(self.genuine_mean - self.impostor_mean, 12)


def score_distributions(
    scores: ArrayLike, labels: list[str] | tuple[str, ...]
) -> ScoreDistributions:
    """Split the upper triangle of a score matrix by identity agreement."""
    matrix = np.asarray(scores, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] != len(labels)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("scores and labels must form an aligned finite square matrix")
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("trial score matrix must be symmetric")
    genuine: list[float] = []
    impostor: list[float] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            target = genuine if labels[left] == labels[right] else impostor
            target.append(round(float(matrix[left, right]), 12))
    if not genuine or not impostor:
        raise ValueError("trials require both genuine and impostor pairs")
    return ScoreDistributions(np.asarray(genuine), np.asarray(impostor))


@dataclass(frozen=True, slots=True)
class EerPoint:
    """Nearest equal-error operating point for synthetic verification trials."""

    rate: float
    threshold: float
    false_acceptance: float
    false_rejection: float


def eer_like(distributions: ScoreDistributions) -> EerPoint:
    """Find the deterministic threshold nearest equal false-accept/reject rates."""
    genuine = np.asarray(distributions.genuine, dtype=np.float64)
    impostor = np.asarray(distributions.impostor, dtype=np.float64)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("EER-like evaluation requires both trial classes")
    candidates = np.unique(np.concatenate((genuine, impostor)))
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        false_rejection = float(np.mean(genuine < threshold))
        false_acceptance = float(np.mean(impostor >= threshold))
        difference = abs(false_acceptance - false_rejection)
        candidate = (difference, float(threshold), false_acceptance, false_rejection)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    _, threshold, false_acceptance, false_rejection = best
    return EerPoint(
        round((false_acceptance + false_rejection) / 2.0, 12),
        round(threshold, 12),
        round(false_acceptance, 12),
        round(false_rejection, 12),
    )


@dataclass(frozen=True, slots=True)
class EmbeddingQuality:
    """Measured quality of embeddings on a planted synthetic speaker set."""

    trials: int
    genuine_mean: float
    impostor_mean: float
    separation: float
    eer: EerPoint


def evaluate_embeddings(values: ArrayLike, labels: list[str] | tuple[str, ...]) -> EmbeddingQuality:
    """Evaluate normalized embeddings against planted speaker labels."""
    matrix = np.asarray(values, dtype=np.float64)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("speaker labels must be nonempty strings")
    distributions = score_distributions(pairwise_cosine(matrix, matrix), labels)
    return EmbeddingQuality(
        int(distributions.genuine.size + distributions.impostor.size),
        distributions.genuine_mean,
        distributions.impostor_mean,
        distributions.separation,
        eer_like(distributions),
    )


# -- extensions --
