from __future__ import annotations

import numpy as np
import pytest

from choravox import features


def test_frame_signal_uses_declared_hop_and_drops_incomplete_tail() -> None:
    framed = features.frame_signal(np.arange(7), frame_length=4, hop_length=2)
    np.testing.assert_array_equal(framed, [[0, 1, 2, 3], [2, 3, 4, 5]])


def test_frame_signal_can_pad_the_last_partial_frame() -> None:
    framed = features.frame_signal([1.0, 2.0, 3.0], 2, 2, end="pad", pad_value=-1.0)
    np.testing.assert_array_equal(framed, [[1.0, 2.0], [3.0, -1.0]])


def test_frame_signal_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        features.frame_signal([1.0], 0, 1)
    with pytest.raises(ValueError, match="end"):
        features.frame_signal([1.0], 1, 1, end="mirror")  # type: ignore[arg-type]


def test_overlap_add_inverts_complete_overlapping_frames() -> None:
    signal = np.arange(8, dtype=np.float64)
    frames = features.frame_signal(signal, 4, 2)
    np.testing.assert_array_equal(features.overlap_add(frames, 2), signal)


def test_overlap_add_respects_requested_trim_and_padding() -> None:
    frames = features.frame_signal([1.0, 2.0, 3.0], 2, 1)
    np.testing.assert_array_equal(features.overlap_add(frames, 1, length=2), [1.0, 2.0])
    np.testing.assert_array_equal(features.overlap_add(frames, 1, length=5), [1, 2, 3, 0, 0])


# -- extensions --
