"""Speaker-attributed decoding mechanics for synthetic CTC output."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.ctc import CtcCandidate, CtcVocabulary
from choravox.embeddings import cosine_similarity
from choravox.types import Transcript, Word

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AttributedToken:
    text: str
    speaker: str

    def __post_init__(self) -> None:
        if not self.text or not self.speaker:
            raise ValueError("attributed token text and speaker must be nonempty")


def serialize_sot(
    tokens: tuple[AttributedToken, ...], vocabulary: CtcVocabulary
) -> tuple[int, ...]:
    """Insert a speaker-change token whenever the attributed speaker changes."""
    if not tokens:
        return ()
    output: list[str] = []
    previous = tokens[0].speaker
    for token in tokens:
        if token.speaker != previous:
            output.append("<spk>")
        output.append(token.text)
        previous = token.speaker
    return vocabulary.encode(tuple(output))


def parse_sot(
    token_ids: tuple[int, ...], vocabulary: CtcVocabulary, speaker_order: tuple[str, ...]
) -> tuple[AttributedToken, ...]:
    """Assign serialized tokens to successive speakers in a declared order."""
    if not speaker_order or any(not speaker for speaker in speaker_order):
        raise ValueError("SOT parsing requires a nonempty speaker order")
    texts = vocabulary.decode(token_ids)
    speaker_index = 0
    output = []
    for text in texts:
        if text == "<blank>":
            raise ValueError("collapsed SOT tokens must not contain blank")
        if text == "<spk>":
            speaker_index += 1
            if speaker_index >= len(speaker_order):
                raise ValueError("SOT stream contains more speaker changes than declared speakers")
        else:
            output.append(AttributedToken(text, speaker_order[speaker_index]))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class SpeakerDecodeCandidate:
    speaker: str
    token_ids: tuple[int, ...]
    acoustic_score: float
    embedding_similarity: float
    combined_score: float


def attribute_nbest(
    candidates: tuple[CtcCandidate, ...],
    speaker_similarities: dict[str, float],
    *,
    embedding_weight: float = 1.0,
) -> tuple[SpeakerDecodeCandidate, ...]:
    """Cross-rescore acoustic candidates against synthetic speaker embeddings."""
    if not candidates or not speaker_similarities:
        raise ValueError("attribution requires acoustic and speaker candidates")
    if embedding_weight < 0.0 or not np.isfinite(embedding_weight):
        raise ValueError("embedding weight must be finite and nonnegative")
    if any(
        not speaker or not np.isfinite(score) for speaker, score in speaker_similarities.items()
    ):
        raise ValueError("speaker similarity roster must be finite and identified")
    output = []
    for candidate in candidates:
        for speaker, similarity in sorted(speaker_similarities.items()):
            combined = candidate.score + embedding_weight * similarity
            output.append(
                SpeakerDecodeCandidate(
                    speaker,
                    candidate.token_ids,
                    candidate.score,
                    similarity,
                    round(combined, 12),
                )
            )
    return tuple(
        sorted(output, key=lambda item: (-item.combined_score, item.speaker, item.token_ids))
    )


def enrollment_similarities(
    enrollment: ArrayLike, probes: ArrayLike, speakers: tuple[str, ...]
) -> dict[str, float]:
    """Score one target enrollment vector against named probe speakers."""
    reference = np.asarray(enrollment, dtype=np.float64)
    matrix = np.asarray(probes, dtype=np.float64)
    if (
        reference.ndim != 1
        or matrix.ndim != 2
        or matrix.shape != (len(speakers), reference.size)
        or len(set(speakers)) != len(speakers)
        or any(not speaker for speaker in speakers)
    ):
        raise ValueError("enrollment probes and speaker roster are incompatible")
    return {
        speaker: cosine_similarity(reference, matrix[index])
        for index, speaker in enumerate(speakers)
    }


def select_target_speaker(
    enrollment: ArrayLike, probes: ArrayLike, speakers: tuple[str, ...]
) -> tuple[str, float]:
    """Choose the most similar speaker with lexical tie breaking."""
    scores = enrollment_similarities(enrollment, probes, speakers)
    return min(scores.items(), key=lambda item: (-item[1], item[0]))


def target_speaker_mask(
    frame_embeddings: ArrayLike, enrollment: ArrayLike, *, threshold: float = 0.5
) -> NDArray[np.bool_]:
    """Select frames similar enough to a target enrollment utterance."""
    matrix = np.asarray(frame_embeddings, dtype=np.float64)
    reference = np.asarray(enrollment, dtype=np.float64)
    if matrix.ndim != 2 or reference.ndim != 1 or matrix.shape[1] != reference.size:
        raise ValueError("target-speaker frame and enrollment dimensions must match")
    if not -1.0 <= threshold <= 1.0 or not np.isfinite(threshold):
        raise ValueError("target-speaker threshold must lie in [-1, 1]")
    scores = np.asarray([cosine_similarity(reference, row) for row in matrix])
    return scores >= threshold


def attributed_words(
    tokens: tuple[AttributedToken, ...],
    frame_spans: tuple[tuple[int, int, float], ...],
    *,
    frame_shift_s: float,
    offset_s: float = 0.0,
) -> Transcript:
    """Build a public transcript from aligned speaker-attributed tokens."""
    if len(tokens) != len(frame_spans):
        raise ValueError("attributed tokens and alignment spans must match")
    if frame_shift_s <= 0.0 or offset_s < 0.0:
        raise ValueError("attributed word timing geometry is invalid")
    words = []
    for token, (start, end, confidence) in zip(tokens, frame_spans, strict=True):
        if start < 0 or end <= start:
            raise ValueError("attributed alignment span is invalid")
        words.append(
            Word(
                token.text,
                round(offset_s + start * frame_shift_s, 6),
                round(offset_s + end * frame_shift_s, 6),
                token.speaker,
                confidence,
            )
        )
    return Transcript(tuple(words))


# -- extensions --
