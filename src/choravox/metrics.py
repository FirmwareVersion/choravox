"""Permutation-aware transcript and diarization metrics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.types import Segment

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


@dataclass(frozen=True, slots=True)
class FrameActivity:
    """Boolean frame activity with columns named by sorted speaker labels."""

    values: NDArray[np.bool_]
    speakers: tuple[str, ...]


def rasterize_segments(
    segments: tuple[Segment, ...], frame_count: int, frame_shift_s: float
) -> FrameActivity:
    """Sample half-open speaker segments at deterministic frame midpoints."""
    if frame_count < 0 or frame_shift_s <= 0.0 or not np.isfinite(frame_shift_s):
        raise ValueError("frame geometry must be finite and positive")
    duration = frame_count * frame_shift_s
    if any(segment.end > duration + 1e-12 for segment in segments):
        raise ValueError("segment exceeds the scoring duration")
    speakers = tuple(sorted({segment.speaker for segment in segments}))
    values = np.zeros((frame_count, len(speakers)), dtype=np.bool_)
    lookup = {speaker: index for index, speaker in enumerate(speakers)}
    midpoints = (np.arange(frame_count, dtype=np.float64) + 0.5) * frame_shift_s
    for segment in segments:
        active = (midpoints >= segment.start) & (midpoints < segment.end)
        values[active, lookup[segment.speaker]] = True
    return FrameActivity(values, speakers)


@dataclass(frozen=True, slots=True)
class SpeakerFrameMapping:
    """Optimal reference-to-hypothesis pairing by coactive frame count."""

    pairs: tuple[tuple[str | None, str | None], ...]
    matched_frames: int


def optimal_speaker_mapping(
    reference: FrameActivity, hypothesis: FrameActivity
) -> SpeakerFrameMapping:
    """Maximize jointly active frames under a one-to-one speaker mapping."""
    if reference.values.shape[0] != hypothesis.values.shape[0]:
        raise ValueError("reference and hypothesis frame axes must match")
    reference_count = len(reference.speakers)
    hypothesis_count = len(hypothesis.speakers)
    size = max(reference_count, hypothesis_count)
    if size == 0:
        return SpeakerFrameMapping((), 0)
    overlaps = np.zeros((size, size), dtype=np.int64)
    if reference_count and hypothesis_count:
        overlaps[:reference_count, :hypothesis_count] = reference.values.astype(
            np.int64
        ).T @ hypothesis.values.astype(np.int64)
    maximum = int(np.max(overlaps, initial=0))
    assignment = optimal_assignment(maximum - overlaps)
    pairs = tuple(
        (
            reference.speakers[row] if row < reference_count else None,
            hypothesis.speakers[column] if column < hypothesis_count else None,
        )
        for row, column in enumerate(assignment.columns)
    )
    matched = sum(int(overlaps[row, column]) for row, column in enumerate(assignment.columns))
    return SpeakerFrameMapping(pairs, matched)


@dataclass(frozen=True, slots=True)
class DiarizationError:
    """Frame-count decomposition of diarization error."""

    miss: int
    false_alarm: int
    confusion: int
    reference_speaker_frames: int
    evaluated_frames: int

    @property
    def errors(self) -> int:
        return self.miss + self.false_alarm + self.confusion

    @property
    def rate(self) -> float:
        return ErrorRate(self.errors, self.reference_speaker_frames).rate


def score_diarization_frames(
    reference: FrameActivity,
    hypothesis: FrameActivity,
    *,
    score_mask: ArrayLike | None = None,
) -> DiarizationError:
    """Score miss, false alarm, and mapped speaker confusion per frame."""
    if reference.values.shape[0] != hypothesis.values.shape[0]:
        raise ValueError("reference and hypothesis frame axes must match")
    frames = reference.values.shape[0]
    mask = np.ones(frames, dtype=np.bool_)
    if score_mask is not None:
        mask = np.asarray(score_mask, dtype=np.bool_)
        if mask.shape != (frames,):
            raise ValueError("score mask must align with frame axes")
    mapping = optimal_speaker_mapping(
        FrameActivity(reference.values[mask], reference.speakers),
        FrameActivity(hypothesis.values[mask], hypothesis.speakers),
    )
    ref_counts = np.sum(reference.values[mask], axis=1, dtype=np.int64)
    hyp_counts = np.sum(hypothesis.values[mask], axis=1, dtype=np.int64)
    correct = np.zeros(ref_counts.shape, dtype=np.int64)
    ref_lookup = {speaker: index for index, speaker in enumerate(reference.speakers)}
    hyp_lookup = {speaker: index for index, speaker in enumerate(hypothesis.speakers)}
    for ref_speaker, hyp_speaker in mapping.pairs:
        if ref_speaker is not None and hyp_speaker is not None:
            correct += (
                reference.values[mask, ref_lookup[ref_speaker]]
                & hypothesis.values[mask, hyp_lookup[hyp_speaker]]
            )
    miss = np.maximum(ref_counts - hyp_counts, 0)
    false_alarm = np.maximum(hyp_counts - ref_counts, 0)
    confusion = np.minimum(ref_counts, hyp_counts) - correct
    return DiarizationError(
        int(np.sum(miss)),
        int(np.sum(false_alarm)),
        int(np.sum(confusion)),
        int(np.sum(ref_counts)),
        int(np.sum(mask)),
    )


def reference_collar_mask(
    reference: tuple[Segment, ...],
    frame_count: int,
    frame_shift_s: float,
    collar_s: float,
) -> NDArray[np.bool_]:
    """Mask frame midpoints within a symmetric collar of reference boundaries."""
    if frame_count < 0 or frame_shift_s <= 0.0 or collar_s < 0.0:
        raise ValueError("collar scoring geometry is invalid")
    midpoints = (np.arange(frame_count, dtype=np.float64) + 0.5) * frame_shift_s
    mask = np.ones(frame_count, dtype=np.bool_)
    if collar_s == 0.0:
        return mask
    for segment in reference:
        for boundary in (segment.start, segment.end):
            mask &= np.abs(midpoints - boundary) >= collar_s
    return mask


def diarization_error(
    reference: tuple[Segment, ...],
    hypothesis: tuple[Segment, ...],
    *,
    frame_count: int,
    frame_shift_s: float,
    collar_s: float = 0.0,
    skip_overlap: bool = False,
) -> DiarizationError:
    """Measure frame DER with optional reference collars and overlap exclusion."""
    reference_activity = rasterize_segments(reference, frame_count, frame_shift_s)
    hypothesis_activity = rasterize_segments(hypothesis, frame_count, frame_shift_s)
    mask = reference_collar_mask(reference, frame_count, frame_shift_s, collar_s)
    if skip_overlap:
        mask &= np.sum(reference_activity.values, axis=1) <= 1
    return score_diarization_frames(reference_activity, hypothesis_activity, score_mask=mask)


@dataclass(frozen=True, slots=True)
class SpeakerJaccard:
    speaker: str
    intersection: int
    union: int

    @property
    def error(self) -> float:
        return 0.0 if self.union == 0 else round(1.0 - self.intersection / self.union, 12)


@dataclass(frozen=True, slots=True)
class JerResult:
    speakers: tuple[SpeakerJaccard, ...]

    @property
    def rate(self) -> float:
        if not self.speakers:
            return 0.0
        return round(sum(item.error for item in self.speakers) / len(self.speakers), 12)


def jaccard_error_rate(
    reference: tuple[Segment, ...],
    hypothesis: tuple[Segment, ...],
    *,
    frame_count: int,
    frame_shift_s: float,
) -> JerResult:
    """Average mapped per-reference-speaker Jaccard error."""
    ref = rasterize_segments(reference, frame_count, frame_shift_s)
    hyp = rasterize_segments(hypothesis, frame_count, frame_shift_s)
    mapping = optimal_speaker_mapping(ref, hyp)
    hyp_lookup = {speaker: index for index, speaker in enumerate(hyp.speakers)}
    paired = {left: right for left, right in mapping.pairs if left is not None}
    scores = []
    for index, speaker in enumerate(ref.speakers):
        ref_active = ref.values[:, index]
        partner = paired[speaker]
        hyp_active = (
            np.zeros(frame_count, dtype=np.bool_)
            if partner is None
            else hyp.values[:, hyp_lookup[partner]]
        )
        scores.append(
            SpeakerJaccard(
                speaker,
                int(np.sum(ref_active & hyp_active)),
                int(np.sum(ref_active | hyp_active)),
            )
        )
    return JerResult(tuple(scores))


# -- extensions --
