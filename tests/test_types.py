from __future__ import annotations

import pytest

from choravox import types as types_mod


def test_segment_uses_half_open_duration() -> None:
    segment = types_mod.Segment(0.25, 1.0, "speaker-a")
    assert segment.duration == 0.75


def test_segment_rejects_empty_or_reversed_values() -> None:
    with pytest.raises(ValueError, match="start < end"):
        types_mod.Segment(1.0, 1.0, "a")
    with pytest.raises(ValueError, match="nonempty"):
        types_mod.Segment(0.0, 1.0, " ")


def test_intersection_is_symmetric_and_zero_when_disjoint() -> None:
    left = types_mod.Segment(0.0, 1.0, "a")
    right = types_mod.Segment(0.75, 2.0, "b")
    assert types_mod.intersection(left, right) == types_mod.intersection(right, left) == 0.25
    assert types_mod.intersection(left, types_mod.Segment(2.0, 3.0, "c")) == 0.0


def test_clip_segment_preserves_label_and_handles_empty_result() -> None:
    segment = types_mod.Segment(0.5, 2.0, "a")
    assert types_mod.clip_segment(segment, 1.0, 1.5) == types_mod.Segment(1.0, 1.5, "a")
    assert types_mod.clip_segment(segment, 3.0, 4.0) is None


def test_word_accepts_confidence_boundaries() -> None:
    assert types_mod.Word("yes", 0.0, 0.1, "a", 0.0).confidence == 0.0
    assert types_mod.Word("no", 0.1, 0.2, "b", 1.0).confidence == 1.0


def test_word_rejects_whitespace_and_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="token"):
        types_mod.Word("two words", 0.0, 1.0, "a")
    with pytest.raises(ValueError, match="confidence"):
        types_mod.Word("word", 0.0, 1.0, "a", 1.01)


# -- extensions --
