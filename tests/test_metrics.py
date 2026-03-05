from __future__ import annotations

import numpy as np
import pytest

from choravox import metrics
from choravox.types import Segment


def test_edit_distance_covers_identity_insertion_deletion_and_substitution() -> None:
    assert metrics.edit_distance(["a", "b"], ["a", "b"]) == 0
    assert metrics.edit_distance(["a"], ["a", "b"]) == 1
    assert metrics.edit_distance(["a", "b"], ["a"]) == 1
    assert metrics.edit_distance(["a"], ["b"]) == 1


def test_edit_distance_is_symmetric_and_handles_empty_sequences() -> None:
    assert metrics.edit_distance([], ["a", "b"]) == 2
    assert metrics.edit_distance(["a", "b"], []) == 2
    assert metrics.edit_distance(["a", "b"], ["a", "c", "d"]) == metrics.edit_distance(
        ["a", "c", "d"], ["a", "b"]
    )


def test_edit_alignment_backtrace_reconstructs_both_sequences() -> None:
    alignment = metrics.edit_alignment(["we", "meet", "now"], ["we", "met", "today"])
    assert [item.reference for item in alignment if item.reference is not None] == [
        "we",
        "meet",
        "now",
    ]
    assert [item.hypothesis for item in alignment if item.hypothesis is not None] == [
        "we",
        "met",
        "today",
    ]
    assert sum(item.kind != "equal" for item in alignment) == 2


def test_alignment_error_count_matches_edit_distance() -> None:
    reference = ["a", "b", "c"]
    hypothesis = ["a", "x", "c", "d"]
    alignment = metrics.edit_alignment(reference, hypothesis)
    assert sum(item.kind != "equal" for item in alignment) == metrics.edit_distance(
        reference, hypothesis
    )


def test_edit_distance_matches_alignment_for_small_binary_sequences() -> None:
    from itertools import product

    sequences = [tuple(item) for length in range(4) for item in product(("a", "b"), repeat=length)]
    for reference in sequences:
        for hypothesis in sequences:
            errors = sum(
                item.kind != "equal" for item in metrics.edit_alignment(reference, hypothesis)
            )
            assert errors == metrics.edit_distance(reference, hypothesis)


def test_wer_reports_hand_computed_rate_and_counts() -> None:
    result = metrics.wer("we meet now", "we met today")
    assert result.errors == 2
    assert result.reference_units == 3
    assert result.rate == pytest.approx(2.0 / 3.0)


def test_wer_defines_empty_reference_boundaries() -> None:
    assert metrics.wer("", "").rate == 0.0
    assert metrics.wer("", "extra").rate == float("inf")


def test_cer_counts_unicode_codepoints() -> None:
    result = metrics.cer("你好世界", "你好世")
    assert result.errors == 1
    assert result.reference_units == 4
    assert result.rate == 0.25


def test_cer_can_include_or_ignore_spaces() -> None:
    assert metrics.cer("a b", "ab").errors == 0
    assert metrics.cer("a b", "ab", ignore_spaces=False).errors == 1


def test_error_rates_are_zero_exactly_on_identity() -> None:
    samples = ("", "one", "two words", "合成 会议")
    for text in samples:
        assert metrics.wer(text, text).rate == 0.0
        assert metrics.cer(text, text).rate == 0.0


def test_added_word_cannot_reduce_identity_wer() -> None:
    reference = "a short meeting"
    assert (
        metrics.wer(reference, reference + " extra").rate > metrics.wer(reference, reference).rate
    )


def test_speaker_error_matrix_contains_pairwise_word_costs() -> None:
    matrix, reference_labels, hypothesis_labels = metrics.speaker_error_matrix(
        {"a": "one two", "b": "three"}, {"x": "three", "y": "one two"}
    )
    assert reference_labels == ("a", "b") and hypothesis_labels == ("x", "y")
    assert matrix.tolist() == [[2, 0], [0, 2]]


def test_speaker_error_matrix_pads_missing_speaker_with_empty_text() -> None:
    matrix, _, _ = metrics.speaker_error_matrix({"a": "one", "b": "two three"}, {"x": "one"})
    assert matrix.shape == (2, 2)
    assert matrix[1, 1] == 2


def test_optimal_assignment_finds_crossed_speaker_mapping() -> None:
    result = metrics.optimal_assignment(np.array([[4, 0], [0, 3]], dtype=np.int64))
    assert result.columns == (1, 0)
    assert result.cost == 0


def test_optimal_assignment_breaks_equal_cost_by_lexical_permutation() -> None:
    result = metrics.optimal_assignment(np.ones((3, 3), dtype=np.int64))
    assert result.columns == (0, 1, 2)


def test_optimal_assignment_matches_brute_force_cost_on_small_matrices() -> None:
    from itertools import permutations

    import numpy as np

    generator = np.random.default_rng(3)
    for size in range(1, 5):
        costs = generator.integers(0, 8, size=(size, size), dtype=np.int64)
        observed = metrics.optimal_assignment(costs).cost
        expected = min(
            sum(int(costs[row, column]) for row, column in enumerate(order))
            for order in permutations(range(size))
        )
        assert observed == expected


def test_assignment_rejects_rectangular_or_negative_costs() -> None:
    import numpy as np

    with pytest.raises(ValueError, match="square"):
        metrics.optimal_assignment(np.ones((2, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="nonnegative"):
        metrics.optimal_assignment(np.array([[0, -1], [1, 0]], dtype=np.int64))


def test_cpwer_is_zero_under_speaker_permutation() -> None:
    reference = {"a": "hello there", "b": "meeting now"}
    hypothesis = {"x": "meeting now", "y": "hello there"}
    result = metrics.cpwer(reference, hypothesis)
    assert result.errors == 0 and result.rate == 0.0
    assert result.mapping == (("a", "y"), ("b", "x"))


def test_cpwer_accounts_for_missing_hypothesis_speaker() -> None:
    result = metrics.cpwer({"a": "one", "b": "two three"}, {"x": "one"})
    assert result.errors == 2
    assert result.rate == pytest.approx(2.0 / 3.0)


def test_segment_rasterization_preserves_silence_and_overlap() -> None:
    segments = (Segment(0.0, 0.2, "a"), Segment(0.1, 0.3, "b"))
    result = metrics.rasterize_segments(segments, 4, 0.1)
    assert result.speakers == ("a", "b")
    assert result.values.tolist() == [[True, False], [True, True], [False, True], [False, False]]


def test_segment_rasterization_validates_scoring_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        metrics.rasterize_segments((), 2, 0.0)
    with pytest.raises(ValueError, match="exceeds"):
        metrics.rasterize_segments((Segment(0.0, 0.3, "a"),), 2, 0.1)


def test_rasterization_uses_half_open_midpoint_boundaries() -> None:
    segments = (Segment(0.05, 0.15, "a"), Segment(0.15, 0.25, "b"))
    result = metrics.rasterize_segments(segments, 3, 0.1)
    assert result.values.tolist() == [[True, False], [False, True], [False, False]]


def test_rasterization_is_invariant_to_segment_order() -> None:
    segments = (Segment(0.0, 0.2, "b"), Segment(0.1, 0.3, "a"))
    forward = metrics.rasterize_segments(segments, 3, 0.1)
    reverse = metrics.rasterize_segments(tuple(reversed(segments)), 3, 0.1)
    assert forward.speakers == reverse.speakers
    np.testing.assert_array_equal(forward.values, reverse.values)


def test_empty_activity_retains_requested_frame_axis() -> None:
    result = metrics.rasterize_segments((), 5, 0.02)
    assert result.values.shape == (5, 0)
    assert result.speakers == ()


def test_speaker_mapping_recovers_permuted_frame_labels() -> None:
    reference = metrics.rasterize_segments((Segment(0.0, 0.2, "a"), Segment(0.2, 0.4, "b")), 4, 0.1)
    hypothesis = metrics.rasterize_segments(
        (Segment(0.0, 0.2, "y"), Segment(0.2, 0.4, "x")), 4, 0.1
    )
    result = metrics.optimal_speaker_mapping(reference, hypothesis)
    assert result.pairs == (("a", "y"), ("b", "x"))
    assert result.matched_frames == 4


def test_speaker_mapping_pads_missing_hypothesis_labels() -> None:
    reference = metrics.rasterize_segments((Segment(0.0, 0.1, "a"), Segment(0.1, 0.2, "b")), 2, 0.1)
    hypothesis = metrics.rasterize_segments((Segment(0.0, 0.1, "x"),), 2, 0.1)
    result = metrics.optimal_speaker_mapping(reference, hypothesis)
    assert result.pairs == (("a", "x"), ("b", None))
    assert result.matched_frames == 1


def test_speaker_mapping_matches_brute_force_overlap_maximum() -> None:
    from itertools import permutations

    generator = np.random.default_rng(8)
    for speaker_count in range(1, 5):
        reference_values = generator.random((7, speaker_count)) > 0.5
        hypothesis_values = generator.random((7, speaker_count)) > 0.5
        reference = metrics.FrameActivity(
            reference_values, tuple(f"r{i}" for i in range(speaker_count))
        )
        hypothesis = metrics.FrameActivity(
            hypothesis_values, tuple(f"h{i}" for i in range(speaker_count))
        )
        overlap = reference_values.astype(int).T @ hypothesis_values.astype(int)
        expected = max(
            sum(int(overlap[row, column]) for row, column in enumerate(order))
            for order in permutations(range(speaker_count))
        )
        assert metrics.optimal_speaker_mapping(reference, hypothesis).matched_frames == expected


def test_empty_speaker_mapping_is_well_defined() -> None:
    empty = metrics.FrameActivity(np.zeros((3, 0), dtype=bool), ())
    assert metrics.optimal_speaker_mapping(empty, empty) == metrics.SpeakerFrameMapping((), 0)


def test_speaker_mapping_rejects_mismatched_frame_axes() -> None:
    short = metrics.FrameActivity(np.zeros((2, 0), dtype=bool), ())
    long = metrics.FrameActivity(np.zeros((3, 0), dtype=bool), ())
    with pytest.raises(ValueError, match="axes"):
        metrics.optimal_speaker_mapping(short, long)


def test_diarization_score_decomposes_frame_errors() -> None:
    reference = metrics.FrameActivity(
        np.array([[1, 0], [1, 0], [0, 1], [0, 0]], dtype=bool), ("a", "b")
    )
    hypothesis = metrics.FrameActivity(
        np.array([[1, 0], [0, 0], [1, 0], [0, 1]], dtype=bool), ("x", "y")
    )
    result = metrics.score_diarization_frames(reference, hypothesis)
    assert (result.miss, result.false_alarm, result.confusion) == (1, 1, 1)
    assert result.errors == 3 and result.reference_speaker_frames == 3
    assert result.rate == 1.0


def test_diarization_score_is_zero_under_label_permutation() -> None:
    reference = metrics.FrameActivity(np.eye(2, dtype=bool), ("a", "b"))
    hypothesis = metrics.FrameActivity(np.fliplr(np.eye(2, dtype=bool)), ("x", "y"))
    result = metrics.score_diarization_frames(reference, hypothesis)
    assert result.errors == 0 and result.rate == 0.0


def test_diarization_miss_false_alarm_and_confusion_are_independent() -> None:
    one = metrics.FrameActivity(np.ones((1, 1), dtype=bool), ("a",))
    zero = metrics.FrameActivity(np.zeros((1, 0), dtype=bool), ())
    assert metrics.score_diarization_frames(one, zero).miss == 1
    assert metrics.score_diarization_frames(zero, one).false_alarm == 1
    reference = metrics.FrameActivity(np.array([[1, 0], [0, 1]], dtype=bool), ("a", "b"))
    hypothesis = metrics.FrameActivity(np.array([[1], [1]], dtype=bool), ("x",))
    result = metrics.score_diarization_frames(reference, hypothesis)
    assert result.confusion == 1 and result.miss == result.false_alarm == 0


def test_diarization_score_mask_excludes_unscored_frames() -> None:
    reference = metrics.FrameActivity(np.ones((3, 1), dtype=bool), ("a",))
    hypothesis = metrics.FrameActivity(np.zeros((3, 0), dtype=bool), ())
    result = metrics.score_diarization_frames(reference, hypothesis, score_mask=[1, 0, 1])
    assert result.miss == 2 and result.evaluated_frames == 2
    with pytest.raises(ValueError, match="mask"):
        metrics.score_diarization_frames(reference, hypothesis, score_mask=[True])


def test_empty_reference_defines_false_alarm_rate_as_infinite() -> None:
    reference = metrics.FrameActivity(np.zeros((2, 0), dtype=bool), ())
    hypothesis = metrics.FrameActivity(np.ones((2, 1), dtype=bool), ("x",))
    result = metrics.score_diarization_frames(reference, hypothesis)
    assert result.false_alarm == 2 and result.rate == float("inf")


def test_reference_collar_excludes_frames_around_turn_boundaries() -> None:
    reference = (Segment(0.1, 0.4, "a"),)
    mask = metrics.reference_collar_mask(reference, 5, 0.1, 0.11)
    assert mask.tolist() == [False, False, True, False, False]


def test_diarization_error_ignores_hypothesis_shift_inside_collar() -> None:
    reference = (Segment(0.1, 0.4, "a"),)
    hypothesis = (Segment(0.2, 0.5, "x"),)
    raw = metrics.diarization_error(reference, hypothesis, frame_count=5, frame_shift_s=0.1)
    collared = metrics.diarization_error(
        reference, hypothesis, frame_count=5, frame_shift_s=0.1, collar_s=0.11
    )
    assert raw.errors == 2
    assert collared.errors == 0 and collared.evaluated_frames == 1


def test_zero_collar_scores_every_frame() -> None:
    reference = (Segment(0.0, 0.2, "a"),)
    assert metrics.reference_collar_mask(reference, 4, 0.1, 0.0).tolist() == [True] * 4


def test_collar_is_centered_on_start_and_end_independently() -> None:
    reference = (Segment(0.2, 0.5, "a"),)
    mask = metrics.reference_collar_mask(reference, 7, 0.1, 0.06)
    assert mask.tolist() == [True, False, False, True, False, False, True]


def test_touching_turns_share_a_single_collared_boundary_region() -> None:
    reference = (Segment(0.0, 0.3, "a"), Segment(0.3, 0.6, "b"))
    mask = metrics.reference_collar_mask(reference, 6, 0.1, 0.11)
    assert mask.tolist() == [False, True, False, False, True, False]


def test_collar_geometry_rejects_negative_width() -> None:
    with pytest.raises(ValueError, match="geometry"):
        metrics.reference_collar_mask((), 3, 0.1, -0.01)


def test_skip_overlap_removes_multi_speaker_reference_frames() -> None:
    reference = (Segment(0.0, 0.3, "a"), Segment(0.1, 0.2, "b"))
    hypothesis = (Segment(0.0, 0.3, "x"),)
    aware = metrics.diarization_error(reference, hypothesis, frame_count=3, frame_shift_s=0.1)
    skipped = metrics.diarization_error(
        reference, hypothesis, frame_count=3, frame_shift_s=0.1, skip_overlap=True
    )
    assert aware.miss == 1 and aware.reference_speaker_frames == 4
    assert skipped.errors == 0 and skipped.reference_speaker_frames == 2


def test_overlap_aware_score_rewards_two_matching_speakers() -> None:
    reference = (Segment(0.0, 0.2, "a"), Segment(0.0, 0.2, "b"))
    hypothesis = (Segment(0.0, 0.2, "x"), Segment(0.0, 0.2, "y"))
    result = metrics.diarization_error(reference, hypothesis, frame_count=2, frame_shift_s=0.1)
    assert result.errors == 0 and result.reference_speaker_frames == 4


def test_overlap_policy_changes_only_overlapped_reference_frames() -> None:
    reference = (Segment(0.0, 0.4, "a"), Segment(0.1, 0.3, "b"))
    hypothesis = (Segment(0.0, 0.2, "x"), Segment(0.2, 0.4, "y"))
    aware = metrics.diarization_error(reference, hypothesis, frame_count=4, frame_shift_s=0.1)
    skipped = metrics.diarization_error(
        reference, hypothesis, frame_count=4, frame_shift_s=0.1, skip_overlap=True
    )
    assert aware.evaluated_frames == 4 and skipped.evaluated_frames == 2
    assert skipped.reference_speaker_frames == 2


def test_all_overlap_can_produce_an_empty_scoring_region() -> None:
    reference = (Segment(0.0, 0.2, "a"), Segment(0.0, 0.2, "b"))
    result = metrics.diarization_error(
        reference, (), frame_count=2, frame_shift_s=0.1, skip_overlap=True
    )
    assert result.evaluated_frames == 0
    assert result.reference_speaker_frames == 0
    assert result.rate == 0.0


def test_overlap_and_collar_masks_compose_by_intersection() -> None:
    reference = (Segment(0.0, 0.4, "a"), Segment(0.1, 0.3, "b"))
    result = metrics.diarization_error(
        reference,
        (),
        frame_count=4,
        frame_shift_s=0.1,
        collar_s=0.06,
        skip_overlap=True,
    )
    assert result.evaluated_frames == 0


# -- extensions --
