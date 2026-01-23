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


# -- extensions --
