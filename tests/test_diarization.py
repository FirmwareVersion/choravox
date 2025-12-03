from __future__ import annotations

import pytest

from choravox import diarization


def test_frame_window_uses_half_open_duration() -> None:
    window = diarization.FrameWindow(3, 8)
    assert window.frame_count == 5


def test_frame_window_rejects_negative_or_empty_span() -> None:
    with pytest.raises(ValueError, match="positive duration"):
        diarization.FrameWindow(-1, 2)
    with pytest.raises(ValueError, match="positive duration"):
        diarization.FrameWindow(3, 3)


# -- extensions --
