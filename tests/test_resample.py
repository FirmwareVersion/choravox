from __future__ import annotations

import numpy as np

from choravox import resample


def test_sampling_ratio_is_reduced() -> None:
    assert resample.sampling_ratio(48_000, 16_000) == resample.SamplingRatio(1, 3)
    assert resample.sampling_ratio(44_100, 16_000) == resample.SamplingRatio(160, 441)


def test_resampled_length_uses_half_up_duration_rounding() -> None:
    assert resample.resampled_length(3, 4, 2) == 2
    assert resample.resampled_length(5, 2, 4) == 10
    assert resample.resampled_length(0, 16_000, 8_000) == 0


def test_polyphase_kernel_has_one_unit_sum_per_phase() -> None:
    kernel = resample.polyphase_kernel(16_000, 22_050, half_width=12)
    assert kernel.shape == (441, 24)
    np.testing.assert_allclose(np.sum(kernel, axis=1), np.ones(441), atol=1e-14)


def test_downsampling_kernel_has_lower_cutoff_than_upsampling() -> None:
    down = resample.polyphase_kernel(16_000, 8_000)[0]
    up = resample.polyphase_kernel(8_000, 16_000)[0]
    assert np.sum(np.abs(np.diff(down))) < np.sum(np.abs(np.diff(up)))


# -- extensions --
