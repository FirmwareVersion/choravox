"""Geometry-driven deterministic microphone-array simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from choravox.room import Point3D

FloatArray = NDArray[np.float64]


def _finite_point(point: Point3D) -> bool:
    return len(point) == 3 and all(math.isfinite(value) for value in point)


@dataclass(frozen=True, slots=True)
class MicrophoneArray:
    """Ordered microphone positions in metres."""

    positions_m: tuple[Point3D, ...]

    def __post_init__(self) -> None:
        if not self.positions_m:
            raise ValueError("microphone array must contain at least one position")
        if any(not _finite_point(point) for point in self.positions_m):
            raise ValueError("microphone positions must have three finite coordinates")
        if len(set(self.positions_m)) != len(self.positions_m):
            raise ValueError("microphone positions must be distinct")

    @property
    def channel_count(self) -> int:
        """Return the number of ordered output channels."""
        return len(self.positions_m)

    @property
    def centroid_m(self) -> Point3D:
        """Return the arithmetic centre of all microphone positions."""
        values = np.mean(np.asarray(self.positions_m, dtype=np.float64), axis=0)
        return tuple(round(float(value), 12) for value in values)


# -- extensions --
