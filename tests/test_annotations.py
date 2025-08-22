from __future__ import annotations

import pytest

from choravox import annotations


def test_rttm_roundtrip_has_stable_field_layout() -> None:
    entries = (
        annotations.RttmEntry("meeting-01", 1, 0.25, 1.5, "speaker-00"),
        annotations.RttmEntry("meeting-01", 1, 1.8, 0.2, "speaker-01"),
    )
    text = annotations.rttm_to_text(entries)
    assert (
        text.splitlines()[0]
        == "SPEAKER meeting-01 1 0.250000 1.500000 <NA> <NA> speaker-00 <NA> <NA>"
    )
    assert annotations.rttm_from_text(text) == entries


def test_rttm_entry_converts_duration_to_segment_end() -> None:
    segment = annotations.RttmEntry("rec", 1, 0.125, 0.375, "alice").segment()
    assert (segment.start, segment.end, segment.speaker) == (0.125, 0.5, "alice")


def test_rttm_rejects_wrong_record_type_or_field_count() -> None:
    with pytest.raises(ValueError, match="line 1"):
        annotations.rttm_from_text("LEXEME rec 1 0.0 1.0 <NA> <NA> spk <NA> <NA>\n")
    with pytest.raises(ValueError, match="line 1"):
        annotations.rttm_from_text("SPEAKER too short\n")


def test_rttm_rejects_invalid_times_channel_and_tokens() -> None:
    with pytest.raises(ValueError, match="channel"):
        annotations.RttmEntry("rec", 0, 0.0, 1.0, "spk")
    with pytest.raises(ValueError, match="positive duration"):
        annotations.RttmEntry("rec", 1, 0.0, 0.0, "spk")
    with pytest.raises(ValueError, match="speaker"):
        annotations.RttmEntry("rec", 1, 0.0, 1.0, "two words")


def test_empty_rttm_is_a_valid_empty_annotation() -> None:
    assert annotations.rttm_to_text(()) == ""
    assert annotations.rttm_from_text("\n") == ()


# -- extensions --
