from __future__ import annotations

import numpy as np
import pytest

from choravox import random as random_mod


def test_derived_generators_repeat_by_namespace() -> None:
    first = random_mod.rng(7, "noise").normal(size=8)
    second = random_mod.rng(7, "noise").normal(size=8)
    np.testing.assert_array_equal(first, second)


def test_derived_generators_separate_components() -> None:
    noise = random_mod.rng(7, "noise").integers(0, 1_000, size=6)
    speech = random_mod.rng(7, "speech").integers(0, 1_000, size=6)
    assert not np.array_equal(noise, speech)
    with pytest.raises(ValueError, match="namespace"):
        random_mod.derive_seed(0, "")


def test_deterministic_split_covers_every_index_once() -> None:
    parts = random_mod.deterministic_split(23, (0.6, 0.2, 0.2), seed=5)
    combined = np.concatenate(parts)
    assert sorted(combined.tolist()) == list(range(23))
    assert [len(part) for part in parts] == [14, 5, 4]


def test_deterministic_split_handles_empty_population() -> None:
    parts = random_mod.deterministic_split(0, (1.0, 1.0), seed=0)
    assert tuple(map(len, parts)) == (0, 0)
    with pytest.raises(ValueError, match="positive"):
        random_mod.deterministic_split(3, (0.0, 0.0), seed=0)


# -- extensions --
