"""Clean synthetic meeting rendering with planted timing truth."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from choravox.dialogue import DialogueScript

FloatArray = NDArray[np.float64]


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


# -- extensions --
