"""Deterministic rectangular-room image-method acoustics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Point3D = tuple[float, float, float]


def _validate_point(point: Point3D, name: str) -> None:
    if len(point) != 3 or any(not math.isfinite(value) for value in point):
        raise ValueError(f"{name} must contain three finite coordinates")


@dataclass(frozen=True, slots=True)
class RoomGeometry:
    """Axis-aligned room with one source and one microphone position."""

    size_m: Point3D
    source_m: Point3D
    microphone_m: Point3D

    def __post_init__(self) -> None:
        _validate_point(self.size_m, "size_m")
        _validate_point(self.source_m, "source_m")
        _validate_point(self.microphone_m, "microphone_m")
        if any(size <= 0.0 for size in self.size_m):
            raise ValueError("room dimensions must be positive")
        for point_name, point in (("source", self.source_m), ("microphone", self.microphone_m)):
            if any(
                coordinate < 0.0 or coordinate > size
                for coordinate, size in zip(point, self.size_m, strict=True)
            ):
                raise ValueError(f"{point_name} must lie inside the room")


def distance_m(left: Point3D, right: Point3D) -> float:
    """Return Euclidean distance between finite three-dimensional points."""
    _validate_point(left, "left")
    _validate_point(right, "right")
    return round(math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True))), 9)


def speed_of_sound(temperature_c: float = 20.0) -> float:
    """Approximate dry-air sound speed from ambient Celsius temperature."""
    if not math.isfinite(temperature_c) or not -40.0 <= temperature_c <= 60.0:
        raise ValueError("temperature_c must be finite and in [-40, 60]")
    return round(331.3 + 0.606 * temperature_c, 6)


def travel_time_s(
    source_m: Point3D, microphone_m: Point3D, *, temperature_c: float = 20.0
) -> float:
    """Return direct-path acoustic travel time in seconds."""
    return round(distance_m(source_m, microphone_m) / speed_of_sound(temperature_c), 12)


@dataclass(frozen=True, slots=True)
class ImageSource:
    """Virtual source position and wall-reflection count."""

    position_m: Point3D
    reflections: int


def first_order_images(geometry: RoomGeometry) -> tuple[ImageSource, ...]:
    """Return direct and six single-wall image sources in stable axis order."""
    images = [ImageSource(geometry.source_m, 0)]
    for axis, size in enumerate(geometry.size_m):
        near = list(geometry.source_m)
        far = list(geometry.source_m)
        near[axis] = -geometry.source_m[axis]
        far[axis] = 2.0 * size - geometry.source_m[axis]
        images.extend((ImageSource(tuple(near), 1), ImageSource(tuple(far), 1)))
    return tuple(images)


FloatArray = NDArray[np.float64]


def fractional_delay_impulse(delay_samples: float, gain: float, length: int) -> FloatArray:
    """Place a linearly interpolated impulse without quantizing its delay."""
    if not math.isfinite(delay_samples) or delay_samples < 0.0:
        raise ValueError("delay_samples must be finite and nonnegative")
    if not math.isfinite(gain) or length <= 0:
        raise ValueError("gain must be finite and length positive")
    lower = math.floor(delay_samples)
    if lower >= length:
        raise ValueError("delay must fit the impulse response")
    fraction = delay_samples - lower
    impulse = np.zeros(length, dtype=np.float64)
    impulse[lower] += gain * (1.0 - fraction)
    if fraction and lower + 1 < length:
        impulse[lower + 1] += gain * fraction
    return impulse


def direct_path_rir(
    geometry: RoomGeometry,
    sample_rate: int,
    *,
    temperature_c: float = 20.0,
    length: int | None = None,
) -> FloatArray:
    """Render inverse-distance direct-path propagation as an impulse response."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    distance = distance_m(geometry.source_m, geometry.microphone_m)
    delay = distance / speed_of_sound(temperature_c) * sample_rate
    required = math.floor(delay) + 2
    output_length = required if length is None else length
    gain = 1.0 / max(distance, 0.1)
    return fractional_delay_impulse(delay, gain, output_length)


def wall_absorption(size_m: Point3D, rt60_s: float) -> float:
    """Estimate uniform Sabine wall absorption from room geometry and RT60."""
    _validate_point(size_m, "size_m")
    if any(size <= 0.0 for size in size_m) or not math.isfinite(rt60_s) or rt60_s <= 0.0:
        raise ValueError("room dimensions and rt60_s must be positive")
    width, depth, height = size_m
    volume = width * depth * height
    surface = 2.0 * (width * depth + width * height + depth * height)
    absorption = 0.161 * volume / (surface * rt60_s)
    if absorption >= 1.0:
        raise ValueError("requested RT60 requires impossible wall absorption")
    return round(absorption, 9)


def reflection_coefficient(size_m: Point3D, rt60_s: float) -> float:
    """Return pressure reflection magnitude implied by uniform absorption."""
    return round(math.sqrt(1.0 - wall_absorption(size_m, rt60_s)), 9)


def image_method_rir(
    geometry: RoomGeometry,
    sample_rate: int,
    rt60_s: float,
    *,
    order: int = 4,
    temperature_c: float = 20.0,
) -> FloatArray:
    """Sum rectangular-room image paths through a bounded reflection order."""
    if sample_rate <= 0 or order < 0:
        raise ValueError("sample_rate and image order must be nonnegative")
    reflection = reflection_coefficient(geometry.size_m, rt60_s)
    length = max(2, math.ceil(rt60_s * sample_rate))
    response = np.zeros(length, dtype=np.float64)
    source = geometry.source_m
    for x_order in range(-order, order + 1):
        for y_order in range(-order, order + 1):
            for z_order in range(-order, order + 1):
                reflection_count = abs(x_order) + abs(y_order) + abs(z_order)
                if reflection_count > order:
                    continue
                orders = (x_order, y_order, z_order)
                image = tuple(
                    2.0 * axis_order * size + ((-1.0) ** axis_order) * coordinate
                    for axis_order, size, coordinate in zip(
                        orders, geometry.size_m, source, strict=True
                    )
                )
                distance = distance_m(image, geometry.microphone_m)
                delay = distance / speed_of_sound(temperature_c) * sample_rate
                if math.floor(delay) + 1 >= length:
                    continue
                gain = reflection**reflection_count / max(distance, 0.1)
                response += fractional_delay_impulse(delay, gain, length)
    peak = np.max(np.abs(response))
    return np.round(response / peak if peak else response, 9)


# -- extensions --
