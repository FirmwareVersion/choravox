from __future__ import annotations

import numpy as np
import pytest

from choravox import dialogue, meeting


def _script() -> dialogue.DialogueScript:
    return dialogue.DialogueScript(
        1.0,
        (dialogue.DialogueTurn(0.0, 0.5, "speaker-00", ("mava",)),),
    )


def test_synthetic_meeting_accepts_matching_planted_artifacts() -> None:
    word = meeting.PlantedWord("mava", "speaker-00", 0.0, 0.5)
    artifact = meeting.SyntheticMeeting(np.zeros(8000), 8000, _script(), (word,), 7)
    assert artifact.samples.shape == (8000,)
    assert artifact.words == (word,)


def test_planted_word_rejects_empty_identity_or_reversed_time() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        meeting.PlantedWord("", "speaker-00", 0.0, 0.1)
    with pytest.raises(ValueError, match="start"):
        meeting.PlantedWord("mava", "speaker-00", 0.2, 0.1)


def test_meeting_rejects_audio_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        meeting.SyntheticMeeting(np.zeros(7999), 8000, _script(), (), 0)


def test_meeting_rejects_nonfinite_multichannel_audio() -> None:
    samples = np.zeros((8000, 2))
    samples[10, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        meeting.SyntheticMeeting(samples, 8000, _script(), (), 0)


def test_meeting_rejects_word_beyond_script() -> None:
    word = meeting.PlantedWord("mava", "speaker-00", 0.9, 1.1)
    with pytest.raises(ValueError, match="exceeds"):
        meeting.SyntheticMeeting(np.zeros(8000), 8000, _script(), (word,), 0)


def test_word_phone_allocation_preserves_pronunciation_and_samples() -> None:
    tokens = meeting.word_phone_tokens("sito", 401, 8000)
    assert tuple(token.symbol for token in tokens) == dialogue.pronunciation("sito")
    assert sum(round(token.duration_s * 8000) for token in tokens) == 401


def test_word_phone_allocation_distributes_remainder_from_start() -> None:
    counts = [
        round(token.duration_s * 1000) for token in meeting.word_phone_tokens("mava", 11, 1000)
    ]
    assert counts == [4, 4, 3]


# -- extensions --
