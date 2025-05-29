from __future__ import annotations

import numpy as np

from choravox.config import FeatureConfig
from choravox.frontend import timed_log_mel


def test_timed_features_use_sample_accurate_geometry() -> None:
    config = FeatureConfig(sample_rate=8000, frame_ms=20.0, hop_ms=10.0, n_fft=256, max_hz=3800)
    result = timed_log_mel(np.ones(800), config)
    assert result.values.shape[0] == 9
    np.testing.assert_allclose(result.starts_s, np.arange(9) * 0.01)
    np.testing.assert_allclose(result.centers_s, np.arange(9) * 0.01 + 0.01)
    assert result.frame_duration_s == 0.02


def test_empty_timed_features_have_empty_time_axes() -> None:
    result = timed_log_mel([])
    assert result.values.shape[0] == result.starts_s.size == result.centers_s.size == 0


def test_feature_starts_advance_by_exact_configured_hop() -> None:
    config = FeatureConfig(sample_rate=16_000, hop_ms=7.5)
    result = timed_log_mel(np.linspace(-1.0, 1.0, 2000), config)
    np.testing.assert_allclose(np.diff(result.starts_s), config.hop_length / config.sample_rate)


def test_frame_centers_are_offset_from_starts_by_half_duration() -> None:
    result = timed_log_mel(np.ones(1000))
    np.testing.assert_allclose(result.centers_s - result.starts_s, result.frame_duration_s / 2)


def test_timed_feature_extraction_is_byte_deterministic() -> None:
    signal = np.random.default_rng(88).normal(size=1200)
    first = timed_log_mel(signal)
    second = timed_log_mel(signal)
    assert first.values.tobytes() == second.values.tobytes()
    assert first.starts_s.tobytes() == second.starts_s.tobytes()


# -- extensions --
