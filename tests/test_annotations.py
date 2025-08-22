from __future__ import annotations

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


# -- extensions --
