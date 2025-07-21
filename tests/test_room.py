from __future__ import annotations

import numpy as np
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


def test_centered_source_has_paired_equal_wall_paths() -> None:
    geometry = room.RoomGeometry((6.0, 4.0, 2.0), (3.0, 2.0, 1.0), (3.0, 2.0, 1.0))
    images = room.first_order_images(geometry)
    distances = [room.distance_m(image.position_m, geometry.microphone_m) for image in images]
    assert distances[1] == distances[2]
    assert distances[3] == distances[4]
    assert distances[5] == distances[6]


def test_image_source_order_is_axis_then_near_far_wall() -> None:
    geometry = room.RoomGeometry((5.0, 4.0, 3.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    positions = [image.position_m for image in room.first_order_images(geometry)]
    assert positions[1][0] < 0 < positions[2][0]
    assert positions[3][1] < 0 < positions[4][1]
    assert positions[5][2] < 0 < positions[6][2]


def test_first_order_image_enumeration_is_deterministic() -> None:
    geometry = room.RoomGeometry((5.0, 4.0, 3.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0))
    assert room.first_order_images(geometry) == room.first_order_images(geometry)


def test_fractional_delay_splits_gain_between_two_samples() -> None:
    impulse = room.fractional_delay_impulse(2.25, 2.0, 6)
    np.testing.assert_array_equal(impulse, [0.0, 0.0, 1.5, 0.5, 0.0, 0.0])
    assert np.sum(impulse) == 2.0


def test_direct_rir_delay_matches_propagation_time() -> None:
    geometry = room.RoomGeometry((5.0, 4.0, 3.0), (1.0, 1.0, 1.0), (2.0, 1.0, 1.0))
    impulse = room.direct_path_rir(geometry, 16_000)
    expected = room.travel_time_s(geometry.source_m, geometry.microphone_m) * 16_000
    center = np.sum(np.arange(impulse.size) * impulse) / np.sum(impulse)
    assert center == pytest.approx(expected, abs=1e-5)


# -- extensions --
