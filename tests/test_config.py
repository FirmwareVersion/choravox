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


# -- extensions --
