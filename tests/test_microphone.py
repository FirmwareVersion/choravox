from __future__ import annotations

import numpy as np
import pytest

from choravox import microphone


def test_microphone_array_preserves_channel_order_and_centroid() -> None:
    array = microphone.MicrophoneArray(((-0.1, 0.0, 1.0), (0.1, 0.0, 1.0)))
    assert array.channel_count == 2
    assert array.centroid_m == (0.0, 0.0, 1.0)
    assert array.positions_m[0][0] < array.positions_m[1][0]


def test_microphone_array_rejects_empty_duplicate_or_nonfinite_positions() -> None:
    with pytest.raises(ValueError, match="at least one"):
        microphone.MicrophoneArray(())
    with pytest.raises(ValueError, match="distinct"):
        microphone.MicrophoneArray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="finite"):
        microphone.MicrophoneArray(((float("nan"), 0.0, 0.0),))


def test_pairwise_baselines_have_n_choose_two_order() -> None:
    array = microphone.MicrophoneArray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.2, 0.0)))
    assert microphone.pairwise_baselines_m(array) == pytest.approx((0.1, 0.2, 5**0.5 / 10.0))


def test_aperture_is_zero_for_one_channel_and_maximum_for_many() -> None:
    mono = microphone.MicrophoneArray(((1.0, 2.0, 3.0),))
    array = microphone.MicrophoneArray(((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.1, 0.0, 0.0)))
    assert microphone.aperture_m(mono) == 0.0
    assert microphone.aperture_m(array) == 0.3


def test_linear_array_is_centered_for_even_and_odd_channel_counts() -> None:
    for count in (3, 4):
        array = microphone.linear_array(count, 0.05, center_m=(1.0, 2.0, 1.5))
        assert array.channel_count == count
        assert array.centroid_m == pytest.approx((1.0, 2.0, 1.5))
        assert microphone.aperture_m(array) == pytest.approx((count - 1) * 0.05)


def test_four_channel_circle_starts_on_positive_x_and_turns_counter_clockwise() -> None:
    array = microphone.circular_array(4, 0.1, center_m=(1.0, 1.0, 1.2))
    np.testing.assert_allclose(
        array.positions_m,
        ((1.1, 1.0, 1.2), (1.0, 1.1, 1.2), (0.9, 1.0, 1.2), (1.0, 0.9, 1.2)),
        atol=1e-12,
    )


def test_layout_constructors_reject_degenerate_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        microphone.linear_array(2, 0.0)
    with pytest.raises(ValueError, match="at least two"):
        microphone.circular_array(1, 0.1)
    with pytest.raises(ValueError, match="positive radius"):
        microphone.circular_array(3, -0.1)


def test_circular_layout_preserves_radius_and_height() -> None:
    array = microphone.circular_array(7, 0.12, center_m=(0.5, -0.3, 1.1))
    for point in array.positions_m:
        horizontal_radius = np.hypot(point[0] - 0.5, point[1] + 0.3)
        assert horizontal_radius == pytest.approx(0.12)
        assert point[2] == 1.1


def test_look_direction_follows_compass_angles_in_horizontal_plane() -> None:
    assert microphone.look_direction(0.0) == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert microphone.look_direction(90.0) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert microphone.look_direction(0.0, 90.0) == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)


def test_endfire_plane_wave_reaches_near_linear_microphone_first() -> None:
    array = microphone.linear_array(3, 0.1)
    delays = microphone.plane_wave_delays_s(array, 0.0)
    assert delays[2] < delays[1] < delays[0]
    assert np.mean(delays) == pytest.approx(0.0, abs=1e-12)


def test_broadside_wave_has_equal_delays_on_x_axis_array() -> None:
    delays = microphone.plane_wave_delays_s(microphone.linear_array(4, 0.05), 90.0)
    np.testing.assert_allclose(delays, 0.0, atol=1e-12)


def test_look_direction_rejects_invalid_elevation_and_nonfinite_angles() -> None:
    with pytest.raises(ValueError, match="elevation"):
        microphone.look_direction(0.0, 91.0)
    with pytest.raises(ValueError, match="finite"):
        microphone.look_direction(float("inf"))


def test_warmer_air_reduces_interchannel_delay_magnitude() -> None:
    array = microphone.linear_array(2, 0.2)
    cold = microphone.plane_wave_delays_s(array, 0.0, temperature_c=0.0)
    warm = microphone.plane_wave_delays_s(array, 0.0, temperature_c=40.0)
    assert np.ptp(warm) < np.ptp(cold)


def test_opposite_bearings_reverse_signed_delays() -> None:
    array = microphone.circular_array(6, 0.08)
    forward = microphone.plane_wave_delays_s(array, 35.0)
    reverse = microphone.plane_wave_delays_s(array, 215.0)
    np.testing.assert_allclose(forward, -reverse, atol=1e-12)


def test_point_source_delays_start_at_zero_and_follow_distance() -> None:
    array = microphone.linear_array(3, 0.1)
    delays = microphone.point_source_delays_s(array, (1.0, 0.0, 0.0))
    assert delays[2] == 0.0
    assert delays[0] > delays[1] > delays[2]


def test_point_source_capture_emits_samples_by_channels_audio() -> None:
    signal = np.zeros(64)
    signal[4] = 1.0
    array = microphone.linear_array(3, 0.1)
    capture = microphone.capture_point_source(signal, 8000, array, (1.0, 0.0, 0.0))
    assert capture.shape == (64, 3)
    assert np.argmax(capture[:, 2]) <= np.argmax(capture[:, 1]) <= np.argmax(capture[:, 0])


def test_nearest_microphone_has_largest_point_source_peak() -> None:
    signal = np.zeros(128)
    signal[10] = 1.0
    array = microphone.linear_array(4, 0.08)
    capture = microphone.capture_point_source(signal, 16_000, array, (2.0, 0.0, 0.0))
    peaks = np.max(np.abs(capture), axis=0)
    assert peaks[-1] > peaks[0]
    assert peaks[-1] <= 1.0


def test_point_source_capture_is_byte_deterministic() -> None:
    signal = np.linspace(-0.5, 0.5, 100)
    array = microphone.circular_array(5, 0.07)
    first = microphone.capture_point_source(signal, 8000, array, (1.2, -0.5, 0.3))
    second = microphone.capture_point_source(signal, 8000, array, (1.2, -0.5, 0.3))
    assert first.tobytes() == second.tobytes()


def test_point_source_capture_rejects_multichannel_input() -> None:
    with pytest.raises(ValueError, match="mono"):
        microphone.capture_point_source(
            np.ones((10, 2)), 8000, microphone.linear_array(2, 0.1), (1.0, 0.0, 0.0)
        )


# -- extensions --
