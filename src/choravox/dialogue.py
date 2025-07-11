"""Planted dialogue scripts for synthetic multi-speaker meetings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Any, Literal

from choravox.hashing import canonical_json, content_hash
from choravox.phonetics import get_phone
from choravox.random import rng


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


def rapid_exchange_script(
    speakers: tuple[str, ...],
    rounds: int,
    *,
    seed: int,
    turn_duration_s: float = 0.45,
    gap_s: float = 0.05,
) -> DialogueScript:
    """Alternate short deterministic turns across a speaker roster."""
    if len(set(speakers)) < 2 or any(not speaker for speaker in speakers):
        raise ValueError("rapid exchange requires at least two distinct speakers")
    if rounds <= 0 or turn_duration_s <= 0.0 or gap_s < 0.0:
        raise ValueError("rapid exchange timing and rounds must be valid")
    generator = rng(seed, "rapid-exchange")
    vocabulary = tuple(LEXICON)
    turns: list[DialogueTurn] = []
    cursor = 0.0
    for index in range(rounds * len(speakers)):
        word = vocabulary[int(generator.integers(len(vocabulary)))]
        end = cursor + turn_duration_s
        turns.append(
            DialogueTurn(round(cursor, 6), round(end, 6), speakers[index % len(speakers)], (word,))
        )
        cursor = end + gap_s
    return DialogueScript(round(cursor - gap_s, 6), tuple(turns))


def add_backchannels(
    script: DialogueScript,
    listener: str,
    *,
    word: str = "nena",
    duration_s: float = 0.18,
    minimum_host_s: float = 0.5,
) -> DialogueScript:
    """Place one listener backchannel near the final third of each eligible host turn."""
    pronunciation(word)
    if not listener or duration_s <= 0.0 or minimum_host_s <= 0.0:
        raise ValueError("backchannel controls must be positive and named")
    additions: list[DialogueTurn] = []
    for turn in script.turns:
        if turn.speaker == listener or turn.end_s - turn.start_s < minimum_host_s:
            continue
        center = turn.start_s + 0.7 * (turn.end_s - turn.start_s)
        start = min(max(turn.start_s, center - duration_s / 2.0), turn.end_s - duration_s)
        additions.append(
            DialogueTurn(
                round(start, 6), round(start + duration_s, 6), listener, (word,), "backchannel"
            )
        )
    turns = tuple(
        sorted((*script.turns, *additions), key=lambda turn: (turn.start_s, turn.speaker))
    )
    return DialogueScript(script.duration_s, turns)


def controlled_dialogue_script(
    speakers: tuple[str, ...],
    duration_s: float,
    overlap_ratio: float,
    silence_ratio: float,
    *,
    seed: int,
    turn_duration_s: float = 0.8,
) -> DialogueScript:
    """Construct a script whose planted wall-clock overlap and silence match controls."""
    if len(set(speakers)) < 2 or any(not speaker for speaker in speakers):
        raise ValueError("controlled dialogue requires two distinct speakers")
    if duration_s <= 0.0 or turn_duration_s <= 0.0:
        raise ValueError("dialogue durations must be positive")
    if overlap_ratio < 0.0 or silence_ratio < 0.0 or overlap_ratio + silence_ratio > 1.0:
        raise ValueError("overlap and silence ratios must be nonnegative and sum to at most one")
    generator = rng(seed, "controlled-dialogue")
    vocabulary = tuple(LEXICON)
    speech_end = duration_s * (1.0 - silence_ratio)
    base: list[DialogueTurn] = []
    cursor = 0.0
    index = 0
    while cursor < speech_end - 1e-12:
        end = min(speech_end, cursor + turn_duration_s)
        word = vocabulary[int(generator.integers(len(vocabulary)))]
        base.append(
            DialogueTurn(round(cursor, 6), round(end, 6), speakers[index % len(speakers)], (word,))
        )
        cursor = end
        index += 1
    remaining = duration_s * overlap_ratio
    overlays: list[DialogueTurn] = []
    for turn in base:
        if remaining <= 1e-12:
            break
        span = min(turn.end_s - turn.start_s, remaining)
        listener_index = (speakers.index(turn.speaker) + 1) % len(speakers)
        word = vocabulary[int(generator.integers(len(vocabulary)))]
        overlays.append(
            DialogueTurn(
                turn.start_s,
                round(turn.start_s + span, 6),
                speakers[listener_index],
                (word,),
                "backchannel",
            )
        )
        remaining -= span
    turns = tuple(sorted((*base, *overlays), key=lambda turn: (turn.start_s, turn.speaker)))
    return DialogueScript(round(duration_s, 6), turns)


def dialogue_to_json(script: DialogueScript) -> str:
    """Encode a dialogue script in an integrity-protected versioned document."""
    payload = {"duration_s": script.duration_s, "turns": [asdict(turn) for turn in script.turns]}
    document = {
        "digest": content_hash(payload),
        "payload": payload,
        "schema": "choravox.dialogue",
        "version": 1,
    }
    return canonical_json(document).decode("utf-8")


def dialogue_from_json(text: str) -> DialogueScript:
    """Decode and validate a version-one planted dialogue document."""
    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("dialogue must be valid JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != "choravox.dialogue"
        or document.get("version") != 1
    ):
        raise ValueError("unsupported dialogue schema or version")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("digest") != content_hash(payload):
        raise ValueError("dialogue digest mismatch")
    try:
        turns = tuple(
            DialogueTurn(
                start_s=float(turn["start_s"]),
                end_s=float(turn["end_s"]),
                speaker=str(turn["speaker"]),
                words=tuple(turn["words"]),
                kind=turn["kind"],
            )
            for turn in payload["turns"]
        )
        return DialogueScript(float(payload["duration_s"]), turns)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("dialogue payload is malformed") from error


# -- extensions --
