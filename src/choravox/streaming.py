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


# -- extensions --
