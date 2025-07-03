"""Planted dialogue scripts for synthetic multi-speaker meetings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Literal

from choravox.phonetics import get_phone


@dataclass(frozen=True, slots=True)
class Lexeme:
    """Synthetic transcript token and its phone pronunciation."""

    text: str
    phones: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text or any(char.isspace() for char in self.text):
            raise ValueError("lexeme text must be a nonempty token")
        if not self.phones:
            raise ValueError("lexeme pronunciation must be nonempty")
        for symbol in self.phones:
            get_phone(symbol)


_LEXEMES = (
    Lexeme("mava", ("aa", "f", "aa")),
    Lexeme("sito", ("s", "iy", "t", "uw")),
    Lexeme("nena", ("n", "eh", "n", "aa")),
    Lexeme("fisu", ("f", "iy", "s", "uw")),
    Lexeme("shana", ("sh", "aa", "n", "aa")),
    Lexeme("teti", ("t", "eh", "t", "iy")),
    Lexeme("nufi", ("n", "uw", "f", "iy")),
    Lexeme("sena", ("s", "eh", "n", "aa")),
)
LEXICON = MappingProxyType({lexeme.text: lexeme for lexeme in _LEXEMES})


def pronunciation(text: str) -> tuple[str, ...]:
    """Return the planted pronunciation of one synthetic token."""
    try:
        return LEXICON[text].phones
    except KeyError as error:
        raise ValueError(f"unknown synthetic word: {text}") from error


TurnKind = Literal["speech", "backchannel"]


@dataclass(frozen=True, slots=True)
class DialogueTurn:
    """Half-open planted speaker turn with one or more transcript tokens."""

    start_s: float
    end_s: float
    speaker: str
    words: tuple[str, ...]
    kind: TurnKind = "speech"

    def __post_init__(self) -> None:
        if self.start_s < 0.0 or self.end_s <= self.start_s:
            raise ValueError("dialogue turn requires 0 <= start < end")
        if not self.speaker:
            raise ValueError("turn speaker must be nonempty")
        if not self.words:
            raise ValueError("turn words must be nonempty")
        for word in self.words:
            pronunciation(word)


@dataclass(frozen=True, slots=True)
class DialogueScript:
    """Chronological planted turns within a declared meeting duration."""

    duration_s: float
    turns: tuple[DialogueTurn, ...]

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("dialogue duration must be positive")
        if any(turn.end_s > self.duration_s for turn in self.turns):
            raise ValueError("dialogue turn exceeds declared duration")
        if any(
            right.start_s < left.start_s
            for left, right in zip(self.turns, self.turns[1:], strict=False)
        ):
            raise ValueError("dialogue turns must be ordered by start time")

    @property
    def speakers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(turn.speaker for turn in self.turns))


@dataclass(frozen=True, slots=True)
class ActivityDurations:
    """Disjoint wall-clock speech, overlap, and silence durations."""

    speech_s: float
    overlap_s: float
    silence_s: float


def activity_durations(script: DialogueScript) -> ActivityDurations:
    """Integrate unique active speakers between every planted boundary."""
    boundaries = sorted(
        {
            0.0,
            script.duration_s,
            *(turn.start_s for turn in script.turns),
            *(turn.end_s for turn in script.turns),
        }
    )
    speech = overlap = 0.0
    for start, end in pairwise(boundaries):
        active = {
            turn.speaker for turn in script.turns if turn.start_s < end and turn.end_s > start
        }
        if active:
            speech += end - start
        if len(active) >= 2:
            overlap += end - start
    return ActivityDurations(
        round(speech, 6), round(overlap, 6), round(script.duration_s - speech, 6)
    )


def monologue_script(
    words: tuple[str, ...],
    *,
    speaker: str = "speaker-00",
    word_duration_s: float = 0.35,
    pause_s: float = 0.1,
) -> DialogueScript:
    """Place one synthetic word per separated monologue turn."""
    if not words or word_duration_s <= 0.0 or pause_s < 0.0:
        raise ValueError("monologue words and timing must be valid")
    turns: list[DialogueTurn] = []
    cursor = 0.0
    for word in words:
        end = cursor + word_duration_s
        turns.append(DialogueTurn(round(cursor, 6), round(end, 6), speaker, (word,)))
        cursor = end + pause_s
    duration = cursor - pause_s
    return DialogueScript(round(duration, 6), tuple(turns))


# -- extensions --
