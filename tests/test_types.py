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


# -- extensions --
