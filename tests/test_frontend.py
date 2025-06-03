from __future__ import annotations

import numpy as np

from choravox.config import FeatureConfig
from choravox.frontend import LogMelExtractor, extract_batch, timed_log_mel


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


def test_cached_extractor_can_be_reused_without_state_drift() -> None:
    extractor = LogMelExtractor(FeatureConfig(n_mels=10))
    first_signal = np.sin(np.linspace(0.0, 10.0, 1200))
    second_signal = np.cos(np.linspace(0.0, 10.0, 1200))
    first = extractor.transform(first_signal)
    extractor.transform(second_signal)
    np.testing.assert_array_equal(extractor.transform(first_signal), first)


def test_cached_extractor_downmixes_identical_stereo_exactly() -> None:
    extractor = LogMelExtractor()
    mono = np.sin(np.linspace(0.0, 30.0, 1600))
    np.testing.assert_array_equal(
        extractor.transform(mono), extractor.transform(np.column_stack((mono, mono)))
    )


def test_cached_extractor_exposes_its_validated_config() -> None:
    config = FeatureConfig(n_mels=13)
    assert LogMelExtractor(config).config is config


def test_batch_results_equal_independent_extraction() -> None:
    recordings = {
        "first": np.random.default_rng(1).normal(size=900),
        "second": np.random.default_rng(2).normal(size=1300),
    }
    batch = extract_batch(recordings)
    for name, signal in recordings.items():
        np.testing.assert_array_equal(batch[name].values, timed_log_mel(signal).values)


def test_batch_rejects_empty_and_nonstring_names() -> None:
    import pytest

    with pytest.raises(ValueError, match="names"):
        extract_batch({"": np.ones(20)})
    with pytest.raises(ValueError, match="names"):
        extract_batch({1: np.ones(20)})  # type: ignore[dict-item]


def test_batch_input_mapping_is_not_mutated() -> None:
    recordings = {"only": np.ones(500)}
    original = recordings["only"].copy()
    extract_batch(recordings)
    np.testing.assert_array_equal(recordings["only"], original)


# -- extensions --
def test_feature_batch_uses_sorted_names_and_shared_geometry() -> None:
    recordings = {"zeta": np.ones(800), "alpha": np.ones(1200)}
    batch = extract_batch(recordings, FeatureConfig(n_mels=8))
    assert list(batch) == ["alpha", "zeta"]
    assert batch["alpha"].values.shape[1] == batch["zeta"].values.shape[1] == 8


def test_empty_feature_batch_is_well_defined() -> None:
    assert extract_batch({}) == {}


def test_cached_extractor_matches_functional_front_end() -> None:
    from choravox.features import extract_log_mel

    config = FeatureConfig(n_mels=18, delta_order=1)
    signal = np.random.default_rng(14).normal(size=2400)
    np.testing.assert_array_equal(
        LogMelExtractor(config).transform(signal), extract_log_mel(signal, config)
    )


def test_cached_analysis_arrays_are_read_only() -> None:
    extractor = LogMelExtractor()
    assert not extractor._window.flags.writeable
    assert not extractor._bank.flags.writeable
