from __future__ import annotations

import pytest

from choravox import config as config_mod


def test_audio_config_defaults_are_speech_friendly() -> None:
    config = config_mod.AudioConfig()
    assert (config.sample_rate, config.frame_ms, config.hop_ms) == (16_000, 25.0, 10.0)


def test_audio_config_rejects_invalid_time_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        config_mod.AudioConfig(sample_rate=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        config_mod.AudioConfig(frame_ms=10.0, hop_ms=20.0)


def test_meeting_config_accepts_boundary_ratios() -> None:
    assert config_mod.MeetingConfig(overlap_ratio=0.0).overlap_ratio == 0.0
    assert config_mod.MeetingConfig(overlap_ratio=1.0).overlap_ratio == 1.0


def test_meeting_config_rejects_impossible_population() -> None:
    with pytest.raises(ValueError, match="between 1 and 16"):
        config_mod.MeetingConfig(speakers=17)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        config_mod.MeetingConfig(silence_ratio=1.0)


@pytest.mark.parametrize(
    "value",
    [config_mod.AudioConfig(sample_rate=8_000), config_mod.MeetingConfig(seed=19)],
)
def test_configuration_json_roundtrip(value: object) -> None:
    encoded = config_mod.config_to_json(value)  # type: ignore[arg-type]
    assert config_mod.config_from_json(encoded) == value
    assert " " not in encoded


def test_configuration_json_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown"):
        config_mod.config_from_json('{"kind":"RoomConfig","values":{}}')


def test_audio_configuration_has_stable_golden_encoding() -> None:
    assert config_mod.config_to_json(config_mod.AudioConfig()) == (
        '{"kind":"AudioConfig","values":{"frame_ms":25.0,"hop_ms":10.0,"sample_rate":16000}}'
    )


def test_meeting_configuration_has_stable_golden_encoding() -> None:
    assert config_mod.config_to_json(config_mod.MeetingConfig()) == (
        '{"kind":"MeetingConfig","values":{"duration_s":8.0,"overlap_ratio":0.1,'
        '"seed":0,"silence_ratio":0.15,"speakers":2}}'
    )


def test_configuration_decoder_reports_document_errors() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        config_mod.config_from_json("{")
    with pytest.raises(ValueError, match="kind and values"):
        config_mod.config_from_json("[]")


def test_feature_config_derives_integer_sample_geometry() -> None:
    config = config_mod.FeatureConfig(sample_rate=16_000, frame_ms=25.0, hop_ms=10.0)
    assert (config.frame_length, config.hop_length) == (400, 160)


def test_feature_config_roundtrips_through_public_codec() -> None:
    config = config_mod.FeatureConfig(n_mels=24, delta_order=2)
    assert config_mod.config_from_json(config_mod.config_to_json(config)) == config


def test_feature_config_rejects_fft_and_frequency_mismatch() -> None:
    with pytest.raises(ValueError, match="accommodate"):
        config_mod.FeatureConfig(n_fft=128)
    with pytest.raises(ValueError, match="frequency"):
        config_mod.FeatureConfig(max_hz=9000.0)


# -- extensions --
