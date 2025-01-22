from __future__ import annotations

import math

import pytest

from choravox import hashing


def test_canonical_json_ignores_mapping_insertion_order() -> None:
    left = {"seed": 3, "speakers": ["甲", "b"]}
    right = {"speakers": ["甲", "b"], "seed": 3}
    assert hashing.canonical_json(left) == hashing.canonical_json(right)
    assert hashing.content_hash(left) == hashing.content_hash(right)


def test_canonical_json_rejects_nonfinite_numbers() -> None:
    with pytest.raises(ValueError):
        hashing.canonical_json({"score": math.nan})


# -- extensions --
