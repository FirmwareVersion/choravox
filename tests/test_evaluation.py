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


def test_adjacent_bins_assign_every_boundary_once() -> None:
    bins = (
        evaluation.IntervalBin("low", -20.0, 0.0),
        evaluation.IntervalBin("mid", 0.0, 10.0),
        evaluation.IntervalBin("high", 10.0, None),
    )
    assert [evaluation.locate_bin(value, bins) for value in (-20.0, -0.1, 0.0, 9.9, 10.0)] == [
        0,
        0,
        1,
        1,
        2,
    ]


def test_values_outside_declared_coverage_are_unassigned() -> None:
    bins = (evaluation.IntervalBin("audible", 0.0, 1.0),)
    assert evaluation.locate_bin(-0.1, bins) is None
    assert evaluation.locate_bin(1.0, bins) is None


def test_overlapping_bins_are_rejected_when_they_match() -> None:
    bins = (
        evaluation.IntervalBin("first", 0.0, 2.0),
        evaluation.IntervalBin("second", 1.0, 3.0),
    )
    with pytest.raises(ValueError, match="overlap"):
        evaluation.locate_bin(1.5, bins)


def test_nonfinite_values_never_enter_a_condition_bin() -> None:
    bins = (evaluation.IntervalBin("all", -10.0, None),)
    assert evaluation.locate_bin(float("nan"), bins) is None
    assert evaluation.locate_bin(float("inf"), bins) is None


# -- extensions --
# -- extensions --
