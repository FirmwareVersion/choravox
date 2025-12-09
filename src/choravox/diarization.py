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


# -- extensions --
