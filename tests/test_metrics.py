from __future__ import annotations

import pytest

from choravox import metrics


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


# -- extensions --
