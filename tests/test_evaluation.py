from __future__ import annotations

import pytest

from choravox import evaluation
from choravox.metrics import DiarizationError, ErrorRate


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


def test_error_rate_aggregation_preserves_micro_counts() -> None:
    result = evaluation.aggregate_error_rates((ErrorRate(1, 2), ErrorRate(1, 8)))
    assert result.errors == 2 and result.reference_units == 10
    assert result.micro == 0.2
    assert result.macro == 0.3125


def test_empty_rate_aggregate_is_exactly_zero() -> None:
    result = evaluation.aggregate_error_rates(())
    assert result == evaluation.RateAggregate(0, 0, 0, 0.0, 0.0)


def test_micro_average_weights_large_recordings_by_reference_units() -> None:
    rates = (ErrorRate(1, 1), ErrorRate(0, 9))
    result = evaluation.aggregate_error_rates(rates)
    assert result.micro == 0.1
    assert result.macro == 0.5


def test_zero_reference_with_errors_is_excluded_from_finite_macro() -> None:
    rates = (ErrorRate(1, 0), ErrorRate(1, 2))
    result = evaluation.aggregate_error_rates(rates)
    assert result.micro == 1.0
    assert result.macro == 0.5


def test_appending_identity_counts_cannot_increase_micro_error() -> None:
    original = evaluation.aggregate_error_rates((ErrorRate(2, 4),))
    expanded = evaluation.aggregate_error_rates((ErrorRate(2, 4), ErrorRate(0, 4)))
    assert expanded.micro <= original.micro


def test_aggregate_rounds_repeating_fractions_deterministically() -> None:
    result = evaluation.aggregate_error_rates((ErrorRate(1, 3), ErrorRate(2, 3)))
    assert result.micro == 0.5
    assert result.macro == 0.5


def test_diarization_aggregation_sums_error_components() -> None:
    recordings = (
        DiarizationError(1, 2, 3, 10, 8),
        DiarizationError(2, 1, 0, 20, 15),
    )
    result = evaluation.aggregate_diarization(recordings)
    assert result == DiarizationError(3, 3, 3, 30, 23)
    assert result.rate == 0.3


def test_empty_diarization_corpus_has_zero_counts() -> None:
    result = evaluation.aggregate_diarization(())
    assert result.errors == 0 and result.rate == 0.0


def test_corpus_der_is_weighted_by_reference_speaker_time() -> None:
    short_bad = DiarizationError(1, 0, 0, 1, 1)
    long_clean = DiarizationError(0, 0, 0, 9, 9)
    result = evaluation.aggregate_diarization((short_bad, long_clean))
    assert result.rate == 0.1


def test_appending_clean_diarization_cannot_raise_corpus_der() -> None:
    noisy = DiarizationError(3, 1, 1, 10, 10)
    clean = DiarizationError(0, 0, 0, 10, 10)
    assert evaluation.aggregate_diarization((noisy, clean)).rate <= noisy.rate


def test_every_component_contributes_once_to_aggregate_errors() -> None:
    items = tuple(DiarizationError(index, index + 1, index + 2, 20, 10) for index in range(4))
    result = evaluation.aggregate_diarization(items)
    assert result.errors == sum(item.errors for item in items)
    assert result.reference_speaker_frames == 80


def test_false_alarm_only_empty_reference_remains_infinite() -> None:
    item = DiarizationError(0, 2, 0, 0, 3)
    assert evaluation.aggregate_diarization((item,)).rate == float("inf")


def test_paired_comparison_counts_wins_ties_and_losses() -> None:
    result = evaluation.compare_scores(
        {"a": 0.4, "b": 0.2, "c": 0.1}, {"a": 0.3, "b": 0.2, "c": 0.2}
    )
    assert (result.improvements, result.ties, result.regressions) == (1, 1, 1)
    assert result.mean_difference == 0.0
    assert [pair.recording_id for pair in result.pairs] == ["a", "b", "c"]


def test_comparison_requires_the_same_recordings() -> None:
    with pytest.raises(ValueError, match="identical"):
        evaluation.compare_scores({"a": 0.1}, {"b": 0.1})


def test_comparison_tolerance_groups_numerical_ties() -> None:
    result = evaluation.compare_scores(
        {"a": 0.1, "b": 0.2}, {"a": 0.100001, "b": 0.199999}, tolerance=0.00001
    )
    assert result.ties == 2 and result.improvements == result.regressions == 0


def test_reversing_comparison_negates_every_delta() -> None:
    left = {"a": 0.1, "b": 0.4}
    right = {"a": 0.3, "b": 0.2}
    forward = evaluation.compare_scores(left, right)
    reverse = evaluation.compare_scores(right, left)
    assert [item.difference for item in forward.pairs] == [
        -item.difference for item in reverse.pairs
    ]
    assert forward.improvements == reverse.regressions


def test_comparison_rejects_nonfinite_scores_and_tolerance() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluation.compare_scores({"a": 0.1}, {"a": float("nan")})
    with pytest.raises(ValueError, match="tolerance"):
        evaluation.compare_scores({"a": 0.1}, {"a": 0.2}, tolerance=-1.0)


def test_pair_difference_is_quantized_at_production() -> None:
    pair = evaluation.PairedDelta("meeting", 0.1, 0.3)
    assert pair.difference == 0.2


# -- extensions --
# -- extensions --
