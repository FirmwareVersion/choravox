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


def test_overlap_targets_equal_a_direct_active_count_oracle() -> None:
    generator = np.random.default_rng(12)
    for speaker_count in range(1, 6):
        activity = generator.integers(0, 2, size=(20, speaker_count), dtype=np.int64)
        expected = [sum(row) >= 2 for row in activity.tolist()]
        assert overlap.overlap_targets(activity).tolist() == expected


def test_permuting_speaker_columns_does_not_change_overlap_truth() -> None:
    activity = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool)
    expected = overlap.overlap_targets(activity)
    np.testing.assert_array_equal(overlap.overlap_targets(activity[:, [2, 0, 1]]), expected)


def test_boolean_and_integer_activity_have_identical_targets() -> None:
    activity = np.array([[1, 1], [1, 0], [0, 0]], dtype=np.int64)
    np.testing.assert_array_equal(
        overlap.overlap_targets(activity), overlap.overlap_targets(activity.astype(bool))
    )


# -- extensions --
# -- extensions --
