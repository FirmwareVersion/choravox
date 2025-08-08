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


# -- extensions --
