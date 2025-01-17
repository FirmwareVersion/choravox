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


# -- extensions --
