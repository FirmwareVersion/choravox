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


# -- extensions --
