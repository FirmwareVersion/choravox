"""Geometry-driven deterministic microphone-array simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform
from choravox.room import Point3D, distance_m, fractional_delay_impulse, speed_of_sound

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


def look_direction(azimuth_deg: float, elevation_deg: float = 0.0) -> Point3D:
    """Convert source-bearing angles into a unit Cartesian direction."""
    if not math.isfinite(azimuth_deg) or not math.isfinite(elevation_deg):
        raise ValueError("look angles must be finite")
    if not -90.0 <= elevation_deg <= 90.0:
        raise ValueError("elevation_deg must be in [-90, 90]")
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = math.cos(elevation)
    return (
        round(horizontal * math.cos(azimuth), 12),
        round(horizontal * math.sin(azimuth), 12),
        round(math.sin(elevation), 12),
    )


def plane_wave_delays_s(
    array: MicrophoneArray,
    azimuth_deg: float,
    elevation_deg: float = 0.0,
    *,
    temperature_c: float = 20.0,
) -> FloatArray:
    """Return signed far-field arrival delays relative to the array centroid."""
    direction = np.asarray(look_direction(azimuth_deg, elevation_deg), dtype=np.float64)
    centered = np.asarray(array.positions_m, dtype=np.float64) - np.asarray(array.centroid_m)
    delays = -(centered @ direction) / speed_of_sound(temperature_c)
    return np.round(delays, 12)


def point_source_delays_s(
    array: MicrophoneArray, source_m: Point3D, *, temperature_c: float = 20.0
) -> FloatArray:
    """Return nonnegative spherical-wave delays relative to the first arrival."""
    if not _finite_point(source_m):
        raise ValueError("source_m must have three finite coordinates")
    distances = np.asarray([distance_m(source_m, point) for point in array.positions_m])
    return np.round((distances - np.min(distances)) / speed_of_sound(temperature_c), 12)


def capture_point_source(
    samples: ArrayLike,
    sample_rate: int,
    array: MicrophoneArray,
    source_m: Point3D,
    *,
    temperature_c: float = 20.0,
) -> FloatArray:
    """Render a mono signal at every microphone with delay and relative spreading loss."""
    waveform = as_waveform(samples)
    if waveform.ndim != 1:
        raise ValueError("point-source input must be mono")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    distances = np.asarray([distance_m(source_m, point) for point in array.positions_m])
    delays = point_source_delays_s(array, source_m, temperature_c=temperature_c)
    reference_distance = max(float(np.min(distances)), 0.1)
    channels = []
    for delay_s, distance in zip(delays, distances, strict=True):
        delay_samples = float(delay_s) * sample_rate
        impulse = fractional_delay_impulse(
            delay_samples,
            reference_distance / max(float(distance), 0.1),
            math.floor(delay_samples) + 2,
        )
        channels.append(np.convolve(waveform, impulse, mode="full")[: waveform.size])
    return np.asarray(np.column_stack(channels), dtype=np.float64)


def delay_and_sum(samples: ArrayLike, delays_s: ArrayLike, sample_rate: int) -> FloatArray:
    """Advance channels by known arrival delays and return their arithmetic mean."""
    waveform = as_waveform(samples)
    delays = np.asarray(delays_s, dtype=np.float64)
    if waveform.ndim != 2:
        raise ValueError("beamformer input must be multichannel")
    if delays.shape != (waveform.shape[1],) or not np.all(np.isfinite(delays)):
        raise ValueError("delays_s must contain one finite value per channel")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    positions = np.arange(waveform.shape[0], dtype=np.float64)
    aligned = np.column_stack(
        [
            np.interp(
                positions + delay * sample_rate,
                positions,
                waveform[:, channel],
                left=0.0,
                right=0.0,
            )
            for channel, delay in enumerate(delays)
        ]
    )
    return np.asarray(np.mean(aligned, axis=1), dtype=np.float64)


# -- extensions --
