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


# -- extensions --
