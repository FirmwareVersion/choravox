"""Deterministic segmentation and speaker clustering mechanics."""

from __future__ import annotations

from dataclasses import dataclass

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


# -- extensions --
