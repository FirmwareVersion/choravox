from __future__ import annotations

import pytest

from choravox import room


def test_room_geometry_accepts_points_on_boundaries() -> None:
    geometry = room.RoomGeometry((5.0, 4.0, 3.0), (0.0, 2.0, 1.0), (5.0, 2.0, 1.0))
    assert geometry.size_m == (5.0, 4.0, 3.0)


def test_distance_has_known_three_four_five_value() -> None:
    assert room.distance_m((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == 5.0


def test_room_rejects_source_outside_any_axis() -> None:
    with pytest.raises(ValueError, match="source"):
        room.RoomGeometry((5.0, 4.0, 3.0), (5.1, 2.0, 1.0), (1.0, 1.0, 1.0))


def test_room_rejects_nonpositive_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        room.RoomGeometry((5.0, 0.0, 3.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0))


def test_distance_is_symmetric_and_zero_on_identity() -> None:
    left = (1.5, 2.0, 0.5)
    right = (4.0, 1.0, 2.5)
    assert room.distance_m(left, right) == room.distance_m(right, left)
    assert room.distance_m(left, left) == 0.0


def test_sound_speed_matches_room_temperature_approximation() -> None:
    assert room.speed_of_sound(20.0) == 343.42
    assert room.speed_of_sound(30.0) > room.speed_of_sound(0.0)


def test_travel_time_scales_with_path_length() -> None:
    short = room.travel_time_s((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    long = room.travel_time_s((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert long == pytest.approx(2.0 * short, abs=1e-9)


# -- extensions --
