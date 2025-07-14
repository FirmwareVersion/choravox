"""Deterministic rectangular-room image-method acoustics."""

from __future__ import annotations

import math
from dataclasses import dataclass

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


# -- extensions --
