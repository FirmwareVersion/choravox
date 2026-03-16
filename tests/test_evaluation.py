from __future__ import annotations

import pytest

from choravox import evaluation


def test_interval_bins_are_left_closed_and_right_open() -> None:
    interval = evaluation.IntervalBin("medium", 0.2, 0.5)
    assert interval.contains(0.2)
    assert interval.contains(0.499)
    assert not interval.contains(0.5)


def test_open_tail_accepts_finite_values_only() -> None:
    interval = evaluation.IntervalBin("high", 10.0, None)
    assert interval.contains(10.0) and interval.contains(100.0)
    assert not interval.contains(float("inf"))


def test_bin_validation_rejects_empty_and_reversed_definitions() -> None:
    with pytest.raises(ValueError, match="label"):
        evaluation.IntervalBin("", 0.0, 1.0)
    with pytest.raises(ValueError, match="upper"):
        evaluation.IntervalBin("bad", 1.0, 1.0)


# -- extensions --
