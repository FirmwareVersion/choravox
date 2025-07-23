"""Geometry-driven deterministic microphone-array simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from choravox.room import Point3D, distance_m

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


def pairwise_baselines_m(array: MicrophoneArray) -> tuple[float, ...]:
    """Return unique microphone-pair distances in stable channel order."""
    return tuple(
        distance_m(array.positions_m[left], array.positions_m[right])
        for left in range(array.channel_count)
        for right in range(left + 1, array.channel_count)
    )


def aperture_m(array: MicrophoneArray) -> float:
    """Return the largest microphone baseline, or zero for one channel."""
    baselines = pairwise_baselines_m(array)
    return 0.0 if not baselines else max(baselines)


def linear_array(
    channel_count: int, spacing_m: float, *, center_m: Point3D = (0.0, 0.0, 0.0)
) -> MicrophoneArray:
    """Place an odd or even linear array symmetrically along the x axis."""
    if channel_count < 1 or not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("channel_count and spacing_m must be positive")
    if not _finite_point(center_m):
        raise ValueError("center_m must have three finite coordinates")
    offset = (channel_count - 1) / 2.0
    positions = tuple(
        (center_m[0] + (index - offset) * spacing_m, center_m[1], center_m[2])
        for index in range(channel_count)
    )
    return MicrophoneArray(positions)


def circular_array(
    channel_count: int, radius_m: float, *, center_m: Point3D = (0.0, 0.0, 0.0)
) -> MicrophoneArray:
    """Place microphones counter-clockwise in the centre's horizontal plane."""
    if channel_count < 2 or not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("circular arrays need at least two channels and a positive radius")
    if not _finite_point(center_m):
        raise ValueError("center_m must have three finite coordinates")
    positions = tuple(
        (
            center_m[0] + radius_m * math.cos(2.0 * math.pi * index / channel_count),
            center_m[1] + radius_m * math.sin(2.0 * math.pi * index / channel_count),
            center_m[2],
        )
        for index in range(channel_count)
    )
    return MicrophoneArray(positions)


# -- extensions --
