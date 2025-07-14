from __future__ import annotations

from choravox import room


def test_room_geometry_accepts_points_on_boundaries() -> None:
    geometry = room.RoomGeometry((5.0, 4.0, 3.0), (0.0, 2.0, 1.0), (5.0, 2.0, 1.0))
    assert geometry.size_m == (5.0, 4.0, 3.0)


def test_distance_has_known_three_four_five_value() -> None:
    assert room.distance_m((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == 5.0


# -- extensions --
