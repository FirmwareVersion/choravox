"""Deterministic segmentation and speaker clustering mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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


# -- extensions --
