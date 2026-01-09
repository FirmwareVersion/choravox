"""Deterministic segmentation and speaker clustering mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True, order=True)
class FrameWindow:
    """A half-open analysis window on a frame index axis."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("frame window must have positive duration")

    @property
    def frame_count(self) -> int:
        return self.end - self.start


def sliding_windows(
    frame_count: int, window_size: int, shift: int, *, include_tail: bool = True
) -> tuple[FrameWindow, ...]:
    """Generate fixed windows and optionally anchor one final window at the tail."""
    if frame_count < 0 or window_size < 1 or shift < 1:
        raise ValueError("frame_count must be nonnegative and window geometry positive")
    if frame_count == 0:
        return ()
    if frame_count <= window_size:
        return (FrameWindow(0, frame_count),)
    starts = list(range(0, frame_count - window_size + 1, shift))
    tail_start = frame_count - window_size
    if include_tail and starts[-1] != tail_start:
        starts.append(tail_start)
    return tuple(FrameWindow(start, start + window_size) for start in starts)


def window_times(
    windows: tuple[FrameWindow, ...], *, hop_seconds: float
) -> tuple[tuple[float, float], ...]:
    """Convert frame windows to quantized wall-clock spans."""
    if hop_seconds <= 0.0 or not np.isfinite(hop_seconds):
        raise ValueError("hop_seconds must be finite and positive")
    return tuple(
        (round(window.start * hop_seconds, 9), round(window.end * hop_seconds, 9))
        for window in windows
    )


def pool_windows(features: ArrayLike, windows: tuple[FrameWindow, ...]) -> FloatArray:
    """Mean-pool frame features inside each diarization window."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a finite matrix")
    if any(window.end > matrix.shape[0] for window in windows):
        raise ValueError("window exceeds the feature frame count")
    if not windows:
        return np.empty((0, matrix.shape[1]), dtype=np.float64)
    return np.round(
        np.stack([np.mean(matrix[window.start : window.end], axis=0) for window in windows]), 12
    )


def adjacent_change_scores(embeddings: ArrayLike) -> FloatArray:
    """Measure cosine distance between consecutive window embeddings."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be a finite matrix")
    if matrix.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("embedding rows must have positive norm")
    normalized = matrix / norms[:, None]
    return np.round(1.0 - np.sum(normalized[:-1] * normalized[1:], axis=1), 12)


def detect_change_points(
    scores: ArrayLike, threshold: float, *, minimum_gap: int = 1
) -> tuple[int, ...]:
    """Select boundary indices from ordered adjacent-change scores."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("change scores must be a finite vector")
    if not 0.0 <= threshold <= 2.0 or minimum_gap < 1:
        raise ValueError("threshold and minimum_gap are invalid")
    candidates = [index + 1 for index, value in enumerate(values) if value >= threshold]
    selected: list[int] = []
    for candidate in candidates:
        if not selected or candidate - selected[-1] >= minimum_gap:
            selected.append(candidate)
        elif values[candidate - 1] > values[selected[-1] - 1]:
            selected[-1] = candidate
    return tuple(selected)


def partition_frames(frame_count: int, change_points: tuple[int, ...]) -> tuple[FrameWindow, ...]:
    """Partition a frame axis at strictly increasing interior boundaries."""
    if frame_count < 0:
        raise ValueError("frame_count must be nonnegative")
    if frame_count == 0:
        if change_points:
            raise ValueError("empty frame axis cannot have change points")
        return ()
    if tuple(sorted(set(change_points))) != change_points or any(
        point <= 0 or point >= frame_count for point in change_points
    ):
        raise ValueError("change points must be unique ordered interior indices")
    boundaries = (0, *change_points, frame_count)
    return tuple(FrameWindow(left, right) for left, right in pairwise(boundaries))


@dataclass(frozen=True, slots=True, order=True)
class LabeledFrames:
    """A contiguous frame span assigned to one speaker label."""

    start: int
    end: int
    speaker: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start or not self.speaker:
            raise ValueError("labeled frame span is invalid")


def labels_to_segments(labels: list[str] | tuple[str, ...]) -> tuple[LabeledFrames, ...]:
    """Run-length encode frame-level speaker labels."""
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("frame labels must be nonempty strings")
    if not labels:
        return ()
    segments: list[LabeledFrames] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            segments.append(LabeledFrames(start, index, labels[start]))
            start = index
    return tuple(segments)


def segments_to_labels(segments: tuple[LabeledFrames, ...], frame_count: int) -> tuple[str, ...]:
    """Decode contiguous segments while rejecting gaps and overlaps."""
    if frame_count < 0:
        raise ValueError("frame_count must be nonnegative")
    if not segments:
        if frame_count:
            raise ValueError("segments must cover every frame")
        return ()
    if segments[0].start != 0 or segments[-1].end != frame_count:
        raise ValueError("segments must cover every frame")
    for left, right in pairwise(segments):
        if left.end != right.start:
            raise ValueError("segments must be contiguous and nonoverlapping")
    labels: list[str] = []
    for segment in segments:
        labels.extend([segment.speaker] * (segment.end - segment.start))
    return tuple(labels)


def majority_smooth(labels: list[str] | tuple[str, ...], radius: int) -> tuple[str, ...]:
    """Smooth isolated label noise with a centered majority vote."""
    if radius < 0:
        raise ValueError("smoothing radius must be nonnegative")
    if any(not label for label in labels):
        raise ValueError("speaker labels must be nonempty")
    result: list[str] = []
    for index, current in enumerate(labels):
        neighborhood = labels[max(0, index - radius) : min(len(labels), index + radius + 1)]
        counts = {label: neighborhood.count(label) for label in set(neighborhood)}
        maximum = max(counts.values())
        winners = {label for label, count in counts.items() if count == maximum}
        result.append(current if current in winners else min(winners))
    return tuple(result)


def speaker_centroids(values: ArrayLike, labels: list[int] | tuple[int, ...]) -> FloatArray:
    """Compute centroids ordered by contiguous nonnegative cluster index."""
    matrix = np.asarray(values, dtype=np.float64)
    assignments = np.asarray(labels, dtype=np.int64)
    if (
        matrix.ndim != 2
        or assignments.shape != (matrix.shape[0],)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("values and labels must be aligned finite arrays")
    if assignments.size == 0 or np.any(assignments < 0):
        raise ValueError("cluster labels must be nonempty and nonnegative")
    unique = np.unique(assignments)
    if not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("cluster labels must be contiguous from zero")
    return np.round(
        np.stack([np.mean(matrix[assignments == label], axis=0) for label in unique]), 12
    )


def assign_centroids(values: ArrayLike, centroids: ArrayLike) -> tuple[IntArray, FloatArray]:
    """Assign vectors to nearest Euclidean centroid with deterministic ties."""
    matrix = np.asarray(values, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64)
    if (
        matrix.ndim != 2
        or centers.ndim != 2
        or centers.shape[0] == 0
        or matrix.shape[1:] != centers.shape[1:]
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(centers))
    ):
        raise ValueError("values and centroids must be compatible finite matrices")
    distances = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
    labels = np.argmin(distances, axis=1).astype(np.int64)
    selected = distances[np.arange(matrix.shape[0]), labels]
    return labels, np.round(selected, 12)


def cosine_distance_matrix(values: ArrayLike) -> FloatArray:
    """Return a symmetric cosine-distance matrix for nonzero embeddings."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be a nonempty finite matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("embedding rows must have positive norm")
    normalized = matrix / norms[:, None]
    distances = np.clip(1.0 - normalized @ normalized.T, 0.0, 2.0)
    np.fill_diagonal(distances, 0.0)
    return np.round(distances, 12)


@dataclass(frozen=True, slots=True)
class AhcResult:
    """Agglomerative clustering labels and accepted merge distances."""

    labels: IntArray
    merge_distances: tuple[float, ...]

    @property
    def speaker_count(self) -> int:
        return int(np.unique(self.labels).size)


def agglomerative_clustering(
    values: ArrayLike, *, threshold: float = 0.5, speaker_count: int | None = None
) -> AhcResult:
    """Cluster embeddings by deterministic centroid-linkage cosine distance."""
    matrix = np.asarray(values, dtype=np.float64)
    cosine_distance_matrix(matrix)
    if not 0.0 <= threshold <= 2.0:
        raise ValueError("AHC threshold must lie between zero and two")
    if speaker_count is not None and not 1 <= speaker_count <= matrix.shape[0]:
        raise ValueError("speaker_count must fit the observation count")
    clusters: list[list[int]] = [[index] for index in range(matrix.shape[0])]
    merges: list[float] = []
    while len(clusters) > (speaker_count or 1):
        centroids = np.stack([np.mean(matrix[cluster], axis=0) for cluster in clusters])
        distances = cosine_distance_matrix(centroids)
        np.fill_diagonal(distances, np.inf)
        flat = int(np.argmin(distances))
        left, right = np.unravel_index(flat, distances.shape)
        distance = float(distances[left, right])
        if speaker_count is None and distance > threshold:
            break
        clusters[left].extend(clusters[right])
        del clusters[right]
        merges.append(round(distance, 12))
    labels = np.empty(matrix.shape[0], dtype=np.int64)
    for label, cluster in enumerate(clusters):
        labels[cluster] = label
    return AhcResult(labels, tuple(merges))


def affinity_matrix(values: ArrayLike, *, scale: float = 0.5) -> FloatArray:
    """Convert cosine distances into a symmetric Gaussian affinity matrix."""
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError("affinity scale must be finite and positive")
    distances = cosine_distance_matrix(values)
    affinity = np.exp(-np.square(distances) / (2.0 * scale * scale))
    np.fill_diagonal(affinity, 0.0)
    return np.round(affinity, 12)


def normalized_laplacian(affinity: ArrayLike) -> FloatArray:
    """Construct the symmetric normalized graph Laplacian."""
    matrix = np.asarray(affinity, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        raise ValueError("affinity must be a finite square matrix")
    if np.any(matrix < 0.0) or not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("affinity must be nonnegative and symmetric")
    degrees = np.sum(matrix, axis=1)
    if np.any(degrees <= 0.0):
        raise ValueError("affinity graph must not contain isolated nodes")
    inverse = np.diag(1.0 / np.sqrt(degrees))
    return np.round(np.eye(matrix.shape[0]) - inverse @ matrix @ inverse, 12)


def estimate_speaker_count(affinity: ArrayLike, *, maximum: int | None = None) -> int:
    """Estimate speaker count from the largest low-spectrum eigengap."""
    laplacian = normalized_laplacian(affinity)
    count = laplacian.shape[0]
    upper = count if maximum is None else maximum
    if upper < 1 or upper > count:
        raise ValueError("maximum speaker count must fit the graph")
    if count == 1:
        return 1
    eigenvalues = np.linalg.eigvalsh(laplacian)
    gaps = np.diff(eigenvalues[: min(upper + 1, count)])
    return int(np.argmax(gaps) + 1)


def farthest_centroids(values: ArrayLike, count: int) -> FloatArray:
    """Choose deterministic farthest-first centroids for clustering."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("values must be a nonempty finite matrix")
    if count < 1 or count > matrix.shape[0]:
        raise ValueError("centroid count must fit observations")
    selected = [int(np.argmax(np.linalg.norm(matrix, axis=1)))]
    while len(selected) < count:
        distances = np.min(
            np.linalg.norm(matrix[:, None, :] - matrix[selected][None, :, :], axis=2), axis=1
        )
        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))
    return matrix[selected].copy()


@dataclass(frozen=True, slots=True)
class KMeansResult:
    labels: IntArray
    centroids: FloatArray
    inertia: float


def deterministic_kmeans(values: ArrayLike, count: int, *, iterations: int = 30) -> KMeansResult:
    """Cluster rows with farthest-first initialization and stable ties."""
    matrix = np.asarray(values, dtype=np.float64)
    centroids = farthest_centroids(matrix, count)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    labels = np.zeros(matrix.shape[0], dtype=np.int64)
    for _ in range(iterations):
        next_labels, distances = assign_centroids(matrix, centroids)
        if np.array_equal(next_labels, labels) and _:
            break
        labels = next_labels
        centroids = np.stack(
            [
                np.mean(matrix[labels == index], axis=0)
                if np.any(labels == index)
                else centroids[index]
                for index in range(count)
            ]
        )
    _, distances = assign_centroids(matrix, centroids)
    return KMeansResult(
        labels, np.round(centroids, 12), round(float(np.sum(np.square(distances))), 12)
    )


@dataclass(frozen=True, slots=True)
class SpectralResult:
    labels: IntArray
    speaker_count: int
    eigenvalues: FloatArray


def spectral_clustering(
    values: ArrayLike, *, speaker_count: int | None = None, maximum: int = 8
) -> SpectralResult:
    """Cluster normalized-Laplacian eigenvectors with deterministic k-means."""
    affinity = affinity_matrix(values)
    count = (
        estimate_speaker_count(affinity, maximum=min(maximum, affinity.shape[0]))
        if speaker_count is None
        else speaker_count
    )
    if count < 1 or count > affinity.shape[0]:
        raise ValueError("speaker_count must fit observations")
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian(affinity))
    embedding = eigenvectors[:, :count]
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    labels = deterministic_kmeans(embedding / np.maximum(norms, 1e-12), count).labels
    return SpectralResult(labels, count, np.round(eigenvalues, 12))


def speaker_transition_matrix(count: int, stay_probability: float = 0.95) -> FloatArray:
    """Create a row-stochastic speaker HMM transition matrix."""
    if count < 1 or not 0.0 < stay_probability <= 1.0:
        raise ValueError("speaker count and stay probability are invalid")
    if count == 1:
        return np.ones((1, 1), dtype=np.float64)
    change = (1.0 - stay_probability) / (count - 1)
    matrix = np.full((count, count), change)
    np.fill_diagonal(matrix, stay_probability)
    return np.round(matrix, 12)


def viterbi_speaker_path(posteriors: ArrayLike, transitions: ArrayLike) -> IntArray:
    """Decode the most likely speaker path from frame posteriors."""
    emissions = np.asarray(posteriors, dtype=np.float64)
    matrix = np.asarray(transitions, dtype=np.float64)
    if (
        emissions.ndim != 2
        or emissions.shape[0] == 0
        or matrix.shape != (emissions.shape[1], emissions.shape[1])
    ):
        raise ValueError("posteriors and transitions have incompatible shapes")
    if (
        np.any(emissions < 0.0)
        or np.any(matrix < 0.0)
        or not np.allclose(np.sum(emissions, axis=1), 1.0, atol=1e-8)
        or not np.allclose(np.sum(matrix, axis=1), 1.0, atol=1e-8)
    ):
        raise ValueError("posteriors and transitions must be stochastic")
    score = np.log(np.maximum(emissions[0], 1e-300)) - np.log(emissions.shape[1])
    back = np.zeros(emissions.shape, dtype=np.int64)
    log_transition = np.log(np.maximum(matrix, 1e-300))
    for frame in range(1, emissions.shape[0]):
        candidates = score[:, None] + log_transition
        back[frame] = np.argmax(candidates, axis=0)
        score = candidates[back[frame], np.arange(emissions.shape[1])] + np.log(
            np.maximum(emissions[frame], 1e-300)
        )
    path = np.empty(emissions.shape[0], dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for frame in range(emissions.shape[0] - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def enforce_label_duration(labels: ArrayLike, minimum_frames: int) -> IntArray:
    """Merge label runs shorter than a declared frame duration."""
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or minimum_frames < 1 or np.any(values < 0):
        raise ValueError("labels and minimum duration are invalid")
    result = values.copy()
    changed = True
    while changed and result.size:
        changed = False
        starts = np.r_[0, np.flatnonzero(result[1:] != result[:-1]) + 1]
        ends = np.r_[starts[1:], result.size]
        for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
            if end - start >= minimum_frames or len(starts) == 1:
                continue
            replacement = result[ends[index]] if index == 0 else result[starts[index] - 1]
            result[start:end] = replacement
            changed = True
            break
    return result


# -- extensions --
