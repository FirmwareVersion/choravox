"""Permutation-aware transcript and diarization metrics."""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class EditOperation:
    kind: str
    reference: str | None
    hypothesis: str | None


def edit_alignment(
    reference: list[str] | tuple[str, ...], hypothesis: list[str] | tuple[str, ...]
) -> tuple[EditOperation, ...]:
    """Backtrace one deterministic optimal token alignment."""
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    costs = np.zeros((rows, columns), dtype=np.int64)
    costs[:, 0] = np.arange(rows)
    costs[0, :] = np.arange(columns)
    for row in range(1, rows):
        for column in range(1, columns):
            costs[row, column] = min(
                costs[row - 1, column] + 1,
                costs[row, column - 1] + 1,
                costs[row - 1, column - 1] + (reference[row - 1] != hypothesis[column - 1]),
            )
    operations = []
    row, column = rows - 1, columns - 1
    while row or column:
        if (
            row
            and column
            and costs[row, column]
            == costs[row - 1, column - 1] + (reference[row - 1] != hypothesis[column - 1])
        ):
            kind = "equal" if reference[row - 1] == hypothesis[column - 1] else "substitute"
            operations.append(EditOperation(kind, reference[row - 1], hypothesis[column - 1]))
            row -= 1
            column -= 1
        elif row and costs[row, column] == costs[row - 1, column] + 1:
            operations.append(EditOperation("delete", reference[row - 1], None))
            row -= 1
        else:
            operations.append(EditOperation("insert", None, hypothesis[column - 1]))
            column -= 1
    return tuple(reversed(operations))


@dataclass(frozen=True, slots=True)
class ErrorRate:
    errors: int
    reference_units: int

    @property
    def rate(self) -> float:
        if self.reference_units == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return round(self.errors / self.reference_units, 12)


def wer(reference: str, hypothesis: str) -> ErrorRate:
    """Measure whitespace-tokenized word error rate."""
    expected, observed = reference.split(), hypothesis.split()
    return ErrorRate(edit_distance(expected, observed), len(expected))


# -- extensions --
