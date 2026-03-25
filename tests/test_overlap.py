from __future__ import annotations

import numpy as np
import pytest

from choravox import overlap


def test_overlap_targets_distinguish_silence_single_and_multiple_speakers() -> None:
    activity = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    assert overlap.overlap_targets(activity).tolist() == [False, False, True, True]


def test_overlap_targets_accept_empty_frame_and_speaker_axes() -> None:
    assert overlap.overlap_targets(np.zeros((0, 2))).shape == (0,)
    assert overlap.overlap_targets(np.zeros((3, 0))).tolist() == [False] * 3


def test_overlap_targets_reject_nonbinary_or_rank_one_activity() -> None:
    with pytest.raises(ValueError, match="matrix"):
        overlap.overlap_targets([0, 1])
    with pytest.raises(ValueError, match="binary"):
        overlap.overlap_targets([[0.0, 0.5]])


# -- extensions --
