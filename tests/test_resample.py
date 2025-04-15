from __future__ import annotations

from choravox import resample


def test_sampling_ratio_is_reduced() -> None:
    assert resample.sampling_ratio(48_000, 16_000) == resample.SamplingRatio(1, 3)
    assert resample.sampling_ratio(44_100, 16_000) == resample.SamplingRatio(160, 441)


def test_resampled_length_uses_half_up_duration_rounding() -> None:
    assert resample.resampled_length(3, 4, 2) == 2
    assert resample.resampled_length(5, 2, 4) == 10
    assert resample.resampled_length(0, 16_000, 8_000) == 0


# -- extensions --
