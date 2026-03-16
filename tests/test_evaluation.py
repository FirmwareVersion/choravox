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


def test_condition_slicer_uses_planted_snr_metadata_and_weights() -> None:
    observations = (
        evaluation.MetricObservation("a", 0.2, 2.0, 0.0, 0.1, 1.0),
        evaluation.MetricObservation("b", 0.8, 1.0, 5.0, 0.3, 2.0),
    )
    bins = (
        evaluation.IntervalBin("low", -10.0, 3.0),
        evaluation.IntervalBin("high", 3.0, None),
    )
    result = evaluation.slice_observations(observations, "snr_db", bins)
    assert result == (
        evaluation.SliceSummary("low", 1, 2.0, 0.2),
        evaluation.SliceSummary("high", 1, 1.0, 0.8),
    )


def test_empty_condition_slice_has_no_invented_score() -> None:
    result = evaluation.slice_observations(
        (), "overlap_ratio", (evaluation.IntervalBin("none", 0.0, 0.01),)
    )
    assert result == (evaluation.SliceSummary("none", 0, 0.0, None),)


def test_all_supported_condition_axes_select_the_expected_recording() -> None:
    observations = (
        evaluation.MetricObservation("short-clean", 0.1, 1.0, 20.0, 0.0, 0.4),
        evaluation.MetricObservation("long-overlap", 0.6, 1.0, 2.0, 0.5, 3.0),
    )
    cases = (
        ("snr_db", evaluation.IntervalBin("selected", 10.0, None), "short-clean"),
        ("overlap_ratio", evaluation.IntervalBin("selected", 0.25, None), "long-overlap"),
        ("turn_duration_s", evaluation.IntervalBin("selected", 2.0, None), "long-overlap"),
    )
    for condition, interval, selected in cases:
        sliced = evaluation.slice_observations(observations, condition, (interval,))
        expected = next(item for item in observations if item.recording_id == selected)
        assert sliced[0].count == 1 and sliced[0].mean == expected.value


def test_slice_counts_equal_covered_observations_without_duplication() -> None:
    observations = tuple(
        evaluation.MetricObservation(str(index), index / 10, 1.0, index, index / 10, index + 1)
        for index in range(6)
    )
    bins = (
        evaluation.IntervalBin("first", 0.0, 2.0),
        evaluation.IntervalBin("second", 2.0, 4.0),
        evaluation.IntervalBin("third", 4.0, 6.0),
    )
    result = evaluation.slice_observations(observations, "snr_db", bins)
    assert sum(item.count for item in result) == len(observations)
    assert [item.count for item in result] == [2, 2, 2]


def test_observation_validation_rejects_impossible_metadata() -> None:
    with pytest.raises(ValueError, match="domains"):
        evaluation.MetricObservation("bad", 0.1, 0.0, 5.0, 0.2, 1.0)
    with pytest.raises(ValueError, match="duration"):
        evaluation.MetricObservation("bad", 0.1, 1.0, 5.0, 0.2, 0.0)


# -- extensions --
# -- extensions --
