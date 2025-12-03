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


def test_sliding_windows_anchor_tail_without_duplicates() -> None:
    windows = diarization.sliding_windows(11, 4, 3)
    assert windows == (
        diarization.FrameWindow(0, 4),
        diarization.FrameWindow(3, 7),
        diarization.FrameWindow(6, 10),
        diarization.FrameWindow(7, 11),
    )


def test_sliding_windows_handle_empty_and_short_recordings() -> None:
    assert diarization.sliding_windows(0, 4, 2) == ()
    assert diarization.sliding_windows(3, 4, 2) == (diarization.FrameWindow(0, 3),)


# -- extensions --
