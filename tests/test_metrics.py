from __future__ import annotations

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


# -- extensions --
