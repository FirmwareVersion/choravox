"""Permutation-aware transcript and diarization metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

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


def cer(reference: str, hypothesis: str, *, ignore_spaces: bool = True) -> ErrorRate:
    """Measure Unicode code-point error rate with optional space removal."""
    expected = reference.replace(" ", "") if ignore_spaces else reference
    observed = hypothesis.replace(" ", "") if ignore_spaces else hypothesis
    return ErrorRate(edit_distance(tuple(expected), tuple(observed)), len(expected))


def speaker_error_matrix(
    reference: dict[str, str], hypothesis: dict[str, str]
) -> tuple[IntArray, tuple[str, ...], tuple[str, ...]]:
    """Build pairwise word-error costs for speaker permutation search."""
    reference_labels = tuple(sorted(reference))
    hypothesis_labels = tuple(sorted(hypothesis))
    size = max(len(reference_labels), len(hypothesis_labels))
    matrix = np.zeros((size, size), dtype=np.int64)
    for row in range(size):
        expected = reference.get(reference_labels[row], "") if row < len(reference_labels) else ""
        for column in range(size):
            observed = (
                hypothesis.get(hypothesis_labels[column], "")
                if column < len(hypothesis_labels)
                else ""
            )
            matrix[row, column] = edit_distance(expected.split(), observed.split())
    return matrix, reference_labels, hypothesis_labels


@dataclass(frozen=True, slots=True)
class Assignment:
    columns: tuple[int, ...]
    cost: int


def optimal_assignment(costs: IntArray) -> Assignment:
    """Find an exact square assignment with deterministic lexical ties."""
    matrix = np.asarray(costs, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or np.any(matrix < 0):
        raise ValueError("assignment costs must be a nonnegative square matrix")
    candidates = permutations(range(matrix.shape[0]))
    columns = min(
        candidates,
        key=lambda item: (sum(int(matrix[row, column]) for row, column in enumerate(item)), item),
    )
    return Assignment(columns, sum(int(matrix[row, column]) for row, column in enumerate(columns)))


@dataclass(frozen=True, slots=True)
class CpWerResult:
    errors: int
    reference_words: int
    mapping: tuple[tuple[str | None, str | None], ...]

    @property
    def rate(self) -> float:
        return ErrorRate(self.errors, self.reference_words).rate


def cpwer(reference: dict[str, str], hypothesis: dict[str, str]) -> CpWerResult:
    """Compute concatenated minimum-permutation word error rate."""
    costs, reference_labels, hypothesis_labels = speaker_error_matrix(reference, hypothesis)
    assignment = optimal_assignment(costs)
    mapping = tuple(
        (
            reference_labels[row] if row < len(reference_labels) else None,
            hypothesis_labels[column] if column < len(hypothesis_labels) else None,
        )
        for row, column in enumerate(assignment.columns)
    )
    words = sum(len(text.split()) for text in reference.values())
    return CpWerResult(assignment.cost, words, mapping)


# -- extensions --
