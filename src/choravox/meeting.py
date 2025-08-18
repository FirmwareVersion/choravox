"""Clean synthetic meeting rendering with planted timing truth."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from choravox.config import MeetingConfig
from choravox.dialogue import (
    LEXICON,
    DialogueScript,
    DialogueTurn,
    controlled_dialogue_script,
    pronunciation,
)
from choravox.phonetics import SpeakerProfile, speaker_profile
from choravox.random import derive_seed, rng
from choravox.synthesis import PhoneToken, synthesize_sequence

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class PlantedWord:
    """One speaker-attributed word on the meeting clock."""

    text: str
    speaker: str
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if not self.text or not self.speaker:
            raise ValueError("planted word text and speaker must be nonempty")
        if not 0.0 <= self.start_s < self.end_s or not all(
            math.isfinite(value) for value in (self.start_s, self.end_s)
        ):
            raise ValueError("planted word requires finite 0 <= start < end")


@dataclass(frozen=True, slots=True)
class SyntheticMeeting:
    """Rendered waveform paired with its complete planted annotations."""

    samples: FloatArray
    sample_rate: int
    script: DialogueScript
    words: tuple[PlantedWord, ...]
    seed: int

    def __post_init__(self) -> None:
        waveform = np.asarray(self.samples)
        if waveform.ndim not in (1, 2) or not np.all(np.isfinite(waveform)):
            raise ValueError("meeting samples must be finite mono or multichannel audio")
        if self.sample_rate <= 0 or self.seed < 0:
            raise ValueError("meeting sample_rate must be positive and seed nonnegative")
        expected = round(self.script.duration_s * self.sample_rate)
        if waveform.shape[0] != expected:
            raise ValueError("meeting audio length must match the dialogue duration")
        if any(word.end_s > self.script.duration_s for word in self.words):
            raise ValueError("planted word exceeds the meeting duration")


def word_phone_tokens(word: str, sample_count: int, sample_rate: int) -> tuple[PhoneToken, ...]:
    """Allocate integer samples across a synthetic word's planted pronunciation."""
    phones = pronunciation(word)
    if sample_rate <= 0 or sample_count < len(phones):
        raise ValueError("word span must provide at least one sample per phone")
    base, remainder = divmod(sample_count, len(phones))
    return tuple(
        PhoneToken(symbol, (base + (index < remainder)) / sample_rate)
        for index, symbol in enumerate(phones)
    )


@dataclass(frozen=True, slots=True)
class RenderedTurn:
    """One rendered turn and its word annotations on the meeting clock."""

    samples: FloatArray
    words: tuple[PlantedWord, ...]


def render_turn(
    turn: DialogueTurn, profile: SpeakerProfile, sample_rate: int, *, seed: int
) -> RenderedTurn:
    """Render each word independently while preserving exact turn sample bounds."""
    if profile.speaker_id != turn.speaker:
        raise ValueError("speaker profile must match the dialogue turn")
    start_sample = round(turn.start_s * sample_rate)
    end_sample = round(turn.end_s * sample_rate)
    total = end_sample - start_sample
    if sample_rate <= 0 or total < len(turn.words) * 3:
        raise ValueError("turn is too short for its planted words")
    base, remainder = divmod(total, len(turn.words))
    pieces: list[FloatArray] = []
    words: list[PlantedWord] = []
    cursor = 0
    for index, text in enumerate(turn.words):
        word_samples = base + (index < remainder)
        tokens = word_phone_tokens(text, word_samples, sample_rate)
        sequence = synthesize_sequence(
            tokens, profile, sample_rate, seed=derive_seed(seed, f"turn-word:{index}:{text}")
        )
        pieces.append(sequence.samples)
        word_start = start_sample + cursor
        cursor += sequence.samples.size
        words.append(
            PlantedWord(
                text,
                turn.speaker,
                round(word_start / sample_rate, 9),
                round((start_sample + cursor) / sample_rate, 9),
            )
        )
    return RenderedTurn(np.concatenate(pieces), tuple(words))


def render_clean_meeting(
    script: DialogueScript, sample_rate: int, *, seed: int
) -> SyntheticMeeting:
    """Render and add every planted turn using stable per-speaker voice profiles."""
    if sample_rate <= 0 or seed < 0:
        raise ValueError("sample_rate must be positive and seed nonnegative")
    output = np.zeros(round(script.duration_s * sample_rate), dtype=np.float64)
    words: list[PlantedWord] = []
    profiles = {
        speaker: speaker_profile(speaker, derive_seed(seed, f"profile:{speaker}"))
        for speaker in script.speakers
    }
    for index, turn in enumerate(script.turns):
        rendered = render_turn(
            turn,
            profiles[turn.speaker],
            sample_rate,
            seed=derive_seed(seed, f"turn:{index}"),
        )
        start = round(turn.start_s * sample_rate)
        output[start : start + rendered.samples.size] += rendered.samples
        words.extend(rendered.words)
    ordered_words = tuple(sorted(words, key=lambda word: (word.start_s, word.speaker)))
    return SyntheticMeeting(np.round(output, 9), sample_rate, script, ordered_words, seed)


def speaker_activity(script: DialogueScript, sample_rate: int) -> tuple[tuple[str, ...], BoolArray]:
    """Return planted samples-by-speakers activity using half-open turn bounds."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    speakers = script.speakers
    indices = {speaker: index for index, speaker in enumerate(speakers)}
    activity = np.zeros((round(script.duration_s * sample_rate), len(speakers)), dtype=np.bool_)
    for turn in script.turns:
        start = round(turn.start_s * sample_rate)
        end = round(turn.end_s * sample_rate)
        activity[start:end, indices[turn.speaker]] = True
    return speakers, activity


def speaker_roster(count: int) -> tuple[str, ...]:
    """Return stable synthetic speaker identifiers for a configured count."""
    if not 1 <= count <= 16:
        raise ValueError("speaker count must be between 1 and 16")
    return tuple(f"speaker-{index:02d}" for index in range(count))


def script_from_config(config: MeetingConfig) -> DialogueScript:
    """Generate a planted dialogue script from high-level meeting controls."""
    speakers = speaker_roster(config.speakers)
    if config.overlap_ratio + config.silence_ratio > 1.0:
        raise ValueError("overlap_ratio and silence_ratio must sum to at most one")
    if config.speakers == 1:
        if config.overlap_ratio != 0.0:
            raise ValueError("a one-speaker meeting cannot contain overlap")
        end = round(config.duration_s * (1.0 - config.silence_ratio), 6)
        generator = rng(config.seed, "single-speaker-script")
        vocabulary = tuple(LEXICON)
        word_count = max(1, round(end / 0.45))
        words = tuple(
            vocabulary[int(generator.integers(len(vocabulary)))] for _ in range(word_count)
        )
        return DialogueScript(config.duration_s, (DialogueTurn(0.0, end, speakers[0], words),))
    return controlled_dialogue_script(
        speakers,
        config.duration_s,
        config.overlap_ratio,
        config.silence_ratio,
        seed=config.seed,
    )


@dataclass(frozen=True, slots=True)
class MeetingFactory:
    """Render clean synthetic meetings from validated high-level controls."""

    config: MeetingConfig
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("factory sample_rate must be positive")

    def build(self) -> SyntheticMeeting:
        """Generate script, waveform, and planted annotations in one operation."""
        return render_clean_meeting(
            script_from_config(self.config), self.sample_rate, seed=self.config.seed
        )


# -- extensions --
