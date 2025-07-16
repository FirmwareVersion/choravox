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


def test_sound_speed_rejects_out_of_range_temperature() -> None:
    with pytest.raises(ValueError, match=r"\[-40, 60\]"):
        room.speed_of_sound(-41.0)
    with pytest.raises(ValueError, match="finite"):
        room.speed_of_sound(float("nan"))


def test_colocated_source_has_zero_travel_time() -> None:
    point = (1.0, 2.0, 3.0)
    assert room.travel_time_s(point, point) == 0.0


def test_one_meter_path_is_about_three_milliseconds() -> None:
    delay = room.travel_time_s((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert delay == pytest.approx(0.002912, abs=1e-6)


def test_first_order_images_include_direct_and_six_walls() -> None:
    geometry = room.RoomGeometry((5.0, 4.0, 3.0), (1.0, 1.5, 2.0), (2.0, 2.0, 1.0))
    images = room.first_order_images(geometry)
    assert len(images) == 7
    assert images[0] == room.ImageSource((1.0, 1.5, 2.0), 0)
    assert [image.reflections for image in images] == [0, 1, 1, 1, 1, 1, 1]


def test_x_wall_images_follow_mirror_geometry() -> None:
    geometry = room.RoomGeometry((5.0, 4.0, 3.0), (1.0, 1.5, 2.0), (2.0, 2.0, 1.0))
    images = room.first_order_images(geometry)
    assert images[1].position_m == (-1.0, 1.5, 2.0)
    assert images[2].position_m == (9.0, 1.5, 2.0)


# -- extensions --
