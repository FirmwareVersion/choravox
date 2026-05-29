"""Online block processing, clustering, and hypothesis revision mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, order=True)
class BlockRange:
    """One half-open block on a sample axis."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("stream block must have positive duration")

    @property
    def sample_count(self) -> int:
        return self.end - self.start


def block_ranges(sample_count: int, block_size: int) -> tuple[BlockRange, ...]:
    """Partition samples into consecutive blocks while retaining the tail."""
    if sample_count < 0 or block_size < 1:
        raise ValueError("sample count and block size are invalid")
    return tuple(
        BlockRange(start, min(start + block_size, sample_count))
        for start in range(0, sample_count, block_size)
    )


@dataclass(frozen=True, slots=True)
class LatencyAccounting:
    """Separated online latency contributions in seconds."""

    buffering_s: float
    algorithmic_s: float
    processing_s: float = 0.0

    def __post_init__(self) -> None:
        values = (self.buffering_s, self.algorithmic_s, self.processing_s)
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("latency components must be finite and nonnegative")

    @property
    def total_s(self) -> float:
        return round(self.buffering_s + self.algorithmic_s + self.processing_s, 9)


def streaming_latency(
    *,
    block_size: int,
    sample_rate: int,
    lookahead_frames: int = 0,
    frame_shift_s: float = 0.01,
    processing_s: float = 0.0,
) -> LatencyAccounting:
    """Account for block buffering, feature lookahead, and measured processing."""
    if block_size < 1 or sample_rate < 1 or lookahead_frames < 0 or frame_shift_s <= 0.0:
        raise ValueError("streaming latency geometry is invalid")
    return LatencyAccounting(
        round(block_size / sample_rate, 9),
        round(lookahead_frames * frame_shift_s, 9),
        processing_s,
    )


@dataclass(frozen=True, slots=True)
class OnlineSegmentationState:
    previous: FloatArray | None = None
    frames_seen: int = 0
    last_boundary: int = 0


@dataclass(frozen=True, slots=True)
class OnlineBoundary:
    frame: int
    score: float


def online_segmentation_step(
    state: OnlineSegmentationState,
    embedding: ArrayLike,
    *,
    threshold: float = 0.5,
    minimum_gap: int = 1,
) -> tuple[OnlineSegmentationState, OnlineBoundary | None]:
    """Compare one incoming embedding to its predecessor and emit a boundary."""
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("online embedding must be a nonempty finite vector")
    if not 0.0 <= threshold <= 2.0 or minimum_gap < 1:
        raise ValueError("online segmentation controls are invalid")
    if state.previous is None:
        return OnlineSegmentationState(vector.copy(), 1, state.last_boundary), None
    if state.previous.shape != vector.shape:
        raise ValueError("online embeddings must keep a fixed dimension")
    denominator = np.linalg.norm(state.previous) * np.linalg.norm(vector)
    if denominator == 0.0:
        raise ValueError("online embeddings must have positive norm")
    score = round(1.0 - float(np.dot(state.previous, vector) / denominator), 12)
    frame = state.frames_seen
    event = None
    last_boundary = state.last_boundary
    if score >= threshold and frame - state.last_boundary >= minimum_gap:
        event = OnlineBoundary(frame, score)
        last_boundary = frame
    return OnlineSegmentationState(vector.copy(), frame + 1, last_boundary), event


@dataclass(frozen=True, slots=True)
class LeaderFollowerConfig:
    birth_threshold: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 <= self.birth_threshold <= 2.0:
            raise ValueError("cluster birth threshold must lie in [0, 2]")


@dataclass(frozen=True, slots=True)
class OnlineClusterState:
    centroids: FloatArray
    counts: NDArray[np.int64]
    labels: tuple[int, ...]


def leader_follower_step(
    state: OnlineClusterState | None,
    embedding: ArrayLike,
    config: LeaderFollowerConfig | None = None,
) -> tuple[OnlineClusterState, int]:
    """Assign an embedding to its nearest leader or create a new speaker."""
    config = LeaderFollowerConfig() if config is None else config
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("online cluster embedding must be a nonempty finite vector")
    if np.linalg.norm(vector) == 0.0:
        raise ValueError("online cluster embedding must have positive norm")
    if state is None:
        return OnlineClusterState(vector[None, :].copy(), np.ones(1, dtype=np.int64), (0,)), 0
    if state.centroids.ndim != 2 or state.centroids.shape[1] != vector.size:
        raise ValueError("online cluster state and embedding dimensions differ")
    centroid_norms = np.linalg.norm(state.centroids, axis=1)
    distances = 1.0 - (state.centroids @ vector) / (centroid_norms * np.linalg.norm(vector))
    nearest = int(np.argmin(distances))
    centroids = state.centroids.copy()
    counts = state.counts.copy()
    if distances[nearest] > config.birth_threshold:
        label = centroids.shape[0]
        centroids = np.vstack((centroids, vector))
        counts = np.append(counts, 1)
    else:
        label = nearest
        total = int(counts[label]) + 1
        centroids[label] = (centroids[label] * counts[label] + vector) / total
        counts[label] = total
    return OnlineClusterState(np.round(centroids, 12), counts, (*state.labels, label)), label


def merge_online_clusters(
    state: OnlineClusterState, *, threshold: float = 0.1
) -> OnlineClusterState:
    """Merge mutually similar online leaders and remap historical labels."""
    if not 0.0 <= threshold <= 2.0:
        raise ValueError("online merge threshold must lie in [0, 2]")
    centroids = [row.copy() for row in state.centroids]
    counts = [int(item) for item in state.counts]
    groups = [[index] for index in range(len(centroids))]
    while len(centroids) > 1:
        best: tuple[float, int, int] | None = None
        for left in range(len(centroids)):
            for right in range(left + 1, len(centroids)):
                denominator = np.linalg.norm(centroids[left]) * np.linalg.norm(centroids[right])
                distance = 1.0 - float(np.dot(centroids[left], centroids[right]) / denominator)
                candidate = (distance, left, right)
                if best is None or candidate < best:
                    best = candidate
        if best is None or best[0] > threshold:
            break
        _, left, right = best
        total = counts[left] + counts[right]
        centroids[left] = (
            centroids[left] * counts[left] + centroids[right] * counts[right]
        ) / total
        counts[left] = total
        groups[left].extend(groups[right])
        del centroids[right], counts[right], groups[right]
    remap = {old: new for new, group in enumerate(groups) for old in group}
    labels = tuple(remap[label] for label in state.labels)
    return OnlineClusterState(
        np.round(np.stack(centroids), 12), np.asarray(counts, dtype=np.int64), labels
    )


# -- extensions --
