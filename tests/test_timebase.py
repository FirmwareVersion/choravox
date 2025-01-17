from __future__ import annotations

import pytest

from choravox import timebase


def test_sample_second_conversion_roundtrips_boundaries() -> None:
    for sample in (0, 1, 159, 160, 16_001):
        assert (
            timebase.seconds_to_samples(timebase.samples_to_seconds(sample, 16_000), 16_000)
            == sample
        )


def test_time_conversion_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        timebase.samples_to_seconds(-1, 16_000)
    with pytest.raises(ValueError, match="nonnegative"):
        timebase.seconds_to_samples(-0.1, 16_000)


def test_frame_spans_distinguish_drop_and_pad_policies() -> None:
    assert timebase.frame_spans(10, 4, 3) == ((0, 4), (3, 7), (6, 10))
    assert timebase.frame_spans(11, 4, 3, pad_end=True) == ((0, 4), (3, 7), (6, 10), (9, 11))


def test_empty_signal_has_no_frames() -> None:
    assert timebase.frame_spans(0, 4, 2) == ()
    with pytest.raises(ValueError, match="geometry"):
        timebase.frame_spans(10, 0, 2)


# -- extensions --
