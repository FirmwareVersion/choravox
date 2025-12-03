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


def test_sliding_windows_cover_every_frame_when_shift_fits_window() -> None:
    for frame_count in range(1, 24):
        windows = diarization.sliding_windows(frame_count, 5, 3)
        covered = {frame for window in windows for frame in range(window.start, window.end)}
        assert covered == set(range(frame_count))


def test_sliding_windows_can_omit_irregular_tail_anchor() -> None:
    windows = diarization.sliding_windows(11, 4, 3, include_tail=False)
    assert windows[-1] == diarization.FrameWindow(6, 10)


def test_sliding_windows_validate_all_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        diarization.sliding_windows(-1, 4, 2)
    with pytest.raises(ValueError, match="geometry"):
        diarization.sliding_windows(5, 0, 2)
    with pytest.raises(ValueError, match="geometry"):
        diarization.sliding_windows(5, 2, 0)


# -- extensions --
