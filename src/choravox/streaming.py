"""Online block processing, clustering, and hypothesis revision mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.metrics import CpWerResult, DiarizationError, edit_distance
from choravox.types import Transcript

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


@dataclass(frozen=True, slots=True)
class PartialHypothesis:
    words: tuple[str, ...]
    stable_count: int

    def __post_init__(self) -> None:
        if not 0 <= self.stable_count <= len(self.words) or any(not word for word in self.words):
            raise ValueError("partial hypothesis words and stable count are invalid")

    @property
    def stable_words(self) -> tuple[str, ...]:
        return self.words[: self.stable_count]

    @property
    def unstable_words(self) -> tuple[str, ...]:
        return self.words[self.stable_count :]


@dataclass(frozen=True, slots=True)
class HypothesisTracker:
    current: PartialHypothesis
    revisions: int = 0


def revise_hypothesis(
    tracker: HypothesisTracker, words: tuple[str, ...], stable_count: int
) -> HypothesisTracker:
    """Accept a partial result while counting edits to the previously unstable suffix."""
    previous = tracker.current
    if (
        stable_count < previous.stable_count
        or words[: previous.stable_count] != previous.stable_words
    ):
        raise ValueError("stable streaming words cannot be revised")
    current = PartialHypothesis(words, stable_count)
    changed = edit_distance(previous.unstable_words, words[previous.stable_count :])
    return HypothesisTracker(current, tracker.revisions + changed)


@dataclass(frozen=True, slots=True)
class OnlineOfflineComparison:
    """Measured quality and latency from the same synthetic meeting."""

    offline_der: float
    online_der: float
    offline_cpwer: float
    online_cpwer: float
    latency_s: float
    revisions: int

    def __post_init__(self) -> None:
        rates = (self.offline_der, self.online_der, self.offline_cpwer, self.online_cpwer)
        if any(not np.isfinite(value) or value < 0.0 for value in rates):
            raise ValueError("online/offline quality values must be finite and nonnegative")
        if not np.isfinite(self.latency_s) or self.latency_s < 0.0 or self.revisions < 0:
            raise ValueError("streaming latency and revisions must be nonnegative")

    @property
    def der_gap(self) -> float:
        return round(self.online_der - self.offline_der, 12)

    @property
    def cpwer_gap(self) -> float:
        return round(self.online_cpwer - self.offline_cpwer, 12)


def compare_online_offline(
    offline_diarization: DiarizationError,
    online_diarization: DiarizationError,
    offline_transcript: CpWerResult,
    online_transcript: CpWerResult,
    latency: LatencyAccounting,
    *,
    revisions: int,
) -> OnlineOfflineComparison:
    """Compare modes only when they share reference speaker frames and words."""
    if offline_diarization.reference_speaker_frames != online_diarization.reference_speaker_frames:
        raise ValueError("online and offline DER must share reference speaker time")
    if offline_transcript.reference_words != online_transcript.reference_words:
        raise ValueError("online and offline cpWER must share reference words")
    return OnlineOfflineComparison(
        offline_diarization.rate,
        online_diarization.rate,
        offline_transcript.rate,
        online_transcript.rate,
        latency.total_s,
        revisions,
    )


@dataclass(frozen=True, slots=True, order=True)
class LatencyQualityPoint:
    latency_s: float
    der: float
    cpwer: float
    block_s: float

    def __post_init__(self) -> None:
        values = (self.latency_s, self.der, self.cpwer, self.block_s)
        if any(not np.isfinite(value) or value < 0.0 for value in values) or self.block_s == 0.0:
            raise ValueError("latency-quality point values must be finite and nonnegative")


def latency_quality_curve(
    points: tuple[LatencyQualityPoint, ...],
) -> tuple[LatencyQualityPoint, ...]:
    """Canonicalize measured latency-quality samples by increasing latency."""
    if not points:
        raise ValueError("latency-quality curve requires measured points")
    selected: dict[float, LatencyQualityPoint] = {}
    for point in points:
        previous = selected.get(point.latency_s)
        if previous is None or (point.der + point.cpwer, point.block_s) < (
            previous.der + previous.cpwer,
            previous.block_s,
        ):
            selected[point.latency_s] = point
    return tuple(sorted(selected.values()))


def pareto_latency_quality(
    points: tuple[LatencyQualityPoint, ...],
) -> tuple[LatencyQualityPoint, ...]:
    """Keep operating points not dominated on latency, DER, and cpWER."""
    curve = latency_quality_curve(points)
    frontier = []
    for candidate in curve:
        dominated = any(
            other != candidate
            and other.latency_s <= candidate.latency_s
            and other.der <= candidate.der
            and other.cpwer <= candidate.cpwer
            and (
                other.latency_s < candidate.latency_s
                or other.der < candidate.der
                or other.cpwer < candidate.cpwer
            )
            for other in curve
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(frontier)


@dataclass(frozen=True, slots=True)
class EmissionLatencySummary:
    latencies_s: tuple[float, ...]
    mean_s: float
    maximum_s: float
    p95_s: float


def word_emission_latencies(
    transcript: Transcript, emission_times_s: tuple[float, ...]
) -> EmissionLatencySummary:
    """Measure delay from planted word end until its partial-hypothesis emission."""
    if len(transcript.words) != len(emission_times_s):
        raise ValueError("word and emission rosters must align")
    if any(
        not np.isfinite(emitted) or emitted < word.end
        for word, emitted in zip(transcript.words, emission_times_s, strict=True)
    ):
        raise ValueError("word emission cannot precede the planted word end")
    latencies = tuple(
        round(emitted - word.end, 9)
        for word, emitted in zip(transcript.words, emission_times_s, strict=True)
    )
    if not latencies:
        return EmissionLatencySummary((), 0.0, 0.0, 0.0)
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return EmissionLatencySummary(
        latencies,
        round(sum(latencies) / len(latencies), 9),
        max(latencies),
        ordered[p95_index],
    )


@dataclass(frozen=True, slots=True)
class StreamingTrial:
    meeting_id: str
    comparison: OnlineOfflineComparison

    def __post_init__(self) -> None:
        if not self.meeting_id:
            raise ValueError("streaming trial meeting identity is required")


@dataclass(frozen=True, slots=True)
class StreamingSummary:
    meetings: int
    mean_der_gap: float
    mean_cpwer_gap: float
    mean_latency_s: float
    total_revisions: int


def aggregate_streaming_trials(trials: tuple[StreamingTrial, ...]) -> StreamingSummary:
    """Macro-average measured online gaps across distinct synthetic meetings."""
    if not trials or len({trial.meeting_id for trial in trials}) != len(trials):
        raise ValueError("streaming trials require distinct nonempty meetings")
    count = len(trials)
    return StreamingSummary(
        count,
        round(sum(trial.comparison.der_gap for trial in trials) / count, 12),
        round(sum(trial.comparison.cpwer_gap for trial in trials) / count, 12),
        round(sum(trial.comparison.latency_s for trial in trials) / count, 9),
        sum(trial.comparison.revisions for trial in trials),
    )


# -- extensions --
