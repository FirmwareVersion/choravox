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


# -- extensions --
