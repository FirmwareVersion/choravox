"""CTC decoding and forced-alignment mechanics for synthetic speech."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class CtcVocabulary:
    """Ordered CTC tokens with blank at zero and an explicit speaker change."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.tokens) < 3
            or self.tokens[0] != "<blank>"
            or "<spk>" not in self.tokens
            or len(set(self.tokens)) != len(self.tokens)
            or any(not token or token.isspace() for token in self.tokens)
        ):
            raise ValueError(
                "CTC vocabulary requires unique blank, speaker-change, and text tokens"
            )

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def speaker_change_id(self) -> int:
        return self.tokens.index("<spk>")

    def encode(self, tokens: tuple[str, ...]) -> tuple[int, ...]:
        """Map known token strings to integer identifiers."""
        lookup = {token: index for index, token in enumerate(self.tokens)}
        try:
            return tuple(lookup[token] for token in tokens)
        except KeyError as error:
            raise ValueError(f"unknown CTC token: {error.args[0]}") from error

    def decode(self, token_ids: tuple[int, ...]) -> tuple[str, ...]:
        """Map valid integer identifiers back to token strings."""
        if any(token_id < 0 or token_id >= len(self.tokens) for token_id in token_ids):
            raise ValueError("CTC token identifier is outside the vocabulary")
        return tuple(self.tokens[token_id] for token_id in token_ids)


def collapse_ctc_path(path: tuple[int, ...], *, blank_id: int = 0) -> tuple[int, ...]:
    """Collapse adjacent repeats and then remove CTC blanks."""
    if blank_id < 0 or any(token_id < 0 for token_id in path):
        raise ValueError("CTC path identifiers must be nonnegative")
    output = []
    previous: int | None = None
    for token_id in path:
        if token_id != blank_id and token_id != previous:
            output.append(token_id)
        previous = token_id
    return tuple(output)


@dataclass(frozen=True, slots=True)
class GreedyCtcResult:
    token_ids: tuple[int, ...]
    path: tuple[int, ...]
    score: float


def greedy_decode(log_probabilities: ArrayLike, *, blank_id: int = 0) -> GreedyCtcResult:
    """Take the best token per frame and apply standard CTC collapse."""
    matrix = np.asarray(log_probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("CTC scores must be a nonempty frame-by-token matrix")
    if not np.all(np.isfinite(matrix)) or not 0 <= blank_id < matrix.shape[1]:
        raise ValueError("CTC scores and blank identifier are invalid")
    path_array = np.argmax(matrix, axis=1)
    path = tuple(int(item) for item in path_array)
    score = round(float(np.sum(matrix[np.arange(matrix.shape[0]), path_array])), 12)
    return GreedyCtcResult(collapse_ctc_path(path, blank_id=blank_id), path, score)


# -- extensions --
