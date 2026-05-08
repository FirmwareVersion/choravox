"""CTC decoding and forced-alignment mechanics for synthetic speech."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.types import Word

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


@dataclass(frozen=True, slots=True)
class CtcCandidate:
    token_ids: tuple[int, ...]
    score: float


def nbest_decode(
    log_probabilities: ArrayLike, *, beam_size: int = 8, blank_id: int = 0
) -> tuple[CtcCandidate, ...]:
    """Generate deterministic max-path CTC candidates with frame-wise beam pruning."""
    matrix = np.asarray(log_probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("CTC scores must be a nonempty frame-by-token matrix")
    if not np.all(np.isfinite(matrix)) or not 0 <= blank_id < matrix.shape[1] or beam_size < 1:
        raise ValueError("CTC beam controls are invalid")
    beams: dict[tuple[tuple[int, ...], int | None], float] = {((), None): 0.0}
    for frame in matrix:
        next_scores: dict[tuple[tuple[int, ...], int], float] = {}
        for (prefix, previous_token), score in beams.items():
            for token_id, token_score in enumerate(frame):
                collapsed = prefix
                if token_id != blank_id and token_id != previous_token:
                    collapsed = (*prefix, token_id)
                state = (collapsed, token_id)
                candidate_score = score + float(token_score)
                previous = next_scores.get(state)
                if previous is None or candidate_score > previous:
                    next_scores[state] = candidate_score
        state_limit = beam_size * matrix.shape[1]
        beams = dict(
            sorted(next_scores.items(), key=lambda item: (-item[1], item[0]))[:state_limit]
        )
    transcript_scores: dict[tuple[int, ...], float] = {}
    for (prefix, _), score in beams.items():
        transcript_scores[prefix] = max(transcript_scores.get(prefix, -np.inf), score)
    ranked = sorted(transcript_scores.items(), key=lambda item: (-item[1], item[0]))[:beam_size]
    return tuple(CtcCandidate(path, round(score, 12)) for path, score in ranked)


@dataclass(frozen=True, slots=True)
class AlignedToken:
    token_id: int
    start_frame: int
    end_frame: int
    confidence: float


@dataclass(frozen=True, slots=True)
class ForcedAlignment:
    tokens: tuple[AlignedToken, ...]
    state_path: tuple[int, ...]
    score: float


def forced_align(
    log_probabilities: ArrayLike, target_ids: tuple[int, ...], *, blank_id: int = 0
) -> ForcedAlignment:
    """Viterbi-align a known token sequence through the CTC state graph."""
    scores = np.asarray(log_probabilities, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("alignment scores must be a nonempty matrix")
    if not np.all(np.isfinite(scores)) or not 0 <= blank_id < scores.shape[1]:
        raise ValueError("alignment scores or blank identifier are invalid")
    if not target_ids or any(
        token_id == blank_id or token_id < 0 or token_id >= scores.shape[1]
        for token_id in target_ids
    ):
        raise ValueError("alignment target must contain valid nonblank tokens")
    states = tuple(
        blank_id if index % 2 == 0 else target_ids[index // 2]
        for index in range(2 * len(target_ids) + 1)
    )
    dynamic = np.full((scores.shape[0], len(states)), -np.inf)
    back = np.full((scores.shape[0], len(states)), -1, dtype=np.int64)
    dynamic[0, 0] = scores[0, blank_id]
    dynamic[0, 1] = scores[0, states[1]]
    for frame in range(1, scores.shape[0]):
        for state, token_id in enumerate(states):
            previous_states = [state]
            if state > 0:
                previous_states.append(state - 1)
            if state > 1 and token_id != blank_id and token_id != states[state - 2]:
                previous_states.append(state - 2)
            previous = max(previous_states, key=lambda item: (dynamic[frame - 1, item], -item))
            if np.isfinite(dynamic[frame - 1, previous]):
                dynamic[frame, state] = dynamic[frame - 1, previous] + scores[frame, token_id]
                back[frame, state] = previous
    final_states = (len(states) - 1, len(states) - 2)
    final = max(final_states, key=lambda item: (dynamic[-1, item], -item))
    if not np.isfinite(dynamic[-1, final]):
        raise ValueError("target cannot be aligned to the available frames")
    state_path = [final]
    for frame in range(scores.shape[0] - 1, 0, -1):
        state_path.append(int(back[frame, state_path[-1]]))
    state_path.reverse()
    aligned = []
    for target_index, token_id in enumerate(target_ids):
        state = target_index * 2 + 1
        frames = np.flatnonzero(np.asarray(state_path) == state)
        confidence = float(np.exp(np.mean(scores[frames, token_id])))
        aligned.append(
            AlignedToken(token_id, int(frames[0]), int(frames[-1] + 1), round(confidence, 12))
        )
    return ForcedAlignment(tuple(aligned), tuple(state_path), round(float(dynamic[-1, final]), 12))


def alignment_words(
    token_texts: tuple[str, ...],
    alignment: ForcedAlignment,
    *,
    frame_shift_s: float,
    speaker: str,
    offset_s: float = 0.0,
) -> tuple[Word, ...]:
    """Convert aligned token frames into public confidence-bearing words."""
    if len(token_texts) != len(alignment.tokens):
        raise ValueError("token texts must align one-to-one with CTC tokens")
    if frame_shift_s <= 0.0 or offset_s < 0.0 or not speaker:
        raise ValueError("word timing geometry and speaker must be valid")
    return tuple(
        Word(
            text,
            round(offset_s + token.start_frame * frame_shift_s, 6),
            round(offset_s + token.end_frame * frame_shift_s, 6),
            speaker,
            token.confidence,
        )
        for text, token in zip(token_texts, alignment.tokens, strict=True)
    )


def repair_alignment_boundaries(boundaries: tuple[int, ...], frame_count: int) -> tuple[int, ...]:
    """Repair clipped or duplicate token boundaries into positive ordered spans."""
    if len(boundaries) < 2 or frame_count < len(boundaries) - 1:
        raise ValueError("alignment repair requires at least one frame per token")
    repaired = [0, *(min(max(value, 0), frame_count) for value in boundaries[1:-1]), frame_count]
    for index in range(1, len(repaired) - 1):
        repaired[index] = max(repaired[index], repaired[index - 1] + 1)
    for index in range(len(repaired) - 2, 0, -1):
        repaired[index] = min(repaired[index], repaired[index + 1] - 1)
    return tuple(repaired)


# -- extensions --
