"""Permutation-aware transcript and diarization metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]


def edit_distance(
    reference: list[str] | tuple[str, ...], hypothesis: list[str] | tuple[str, ...]
) -> int:
    """Compute Levenshtein distance with unit insertion, deletion, and substitution costs."""
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, observed in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


# -- extensions --
