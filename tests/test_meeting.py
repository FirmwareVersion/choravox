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


def test_word_phone_allocation_rejects_unknown_word() -> None:
    with pytest.raises(ValueError, match="unknown synthetic word"):
        meeting.word_phone_tokens("hello", 100, 8000)


def test_word_phone_allocation_requires_one_sample_per_phone() -> None:
    with pytest.raises(ValueError, match="one sample per phone"):
        meeting.word_phone_tokens("sito", 3, 8000)


def test_word_phone_allocation_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValueError, match="word span"):
        meeting.word_phone_tokens("mava", 30, 0)


def test_rendered_turn_matches_exact_sample_span() -> None:
    from choravox.phonetics import speaker_profile

    turn = dialogue.DialogueTurn(0.1, 0.601, "speaker-00", ("mava", "sito"))
    rendered = meeting.render_turn(turn, speaker_profile("speaker-00", 4), 8000, seed=5)
    assert rendered.samples.size == round(0.601 * 8000) - round(0.1 * 8000)
    assert rendered.words[0].start_s == 0.1
    assert rendered.words[-1].end_s == round(0.601 * 8000) / 8000


def test_rendered_turn_word_spans_are_contiguous() -> None:
    from choravox.phonetics import speaker_profile

    turn = dialogue.DialogueTurn(0.0, 0.9, "speaker-01", ("mava", "sito", "nena"))
    rendered = meeting.render_turn(turn, speaker_profile("speaker-01", 1), 4000, seed=2)
    assert [word.text for word in rendered.words] == list(turn.words)
    assert all(
        left.end_s == right.start_s
        for left, right in zip(rendered.words, rendered.words[1:], strict=False)
    )


def test_render_turn_rejects_mismatched_speaker_profile() -> None:
    from choravox.phonetics import speaker_profile

    turn = dialogue.DialogueTurn(0.0, 0.3, "speaker-00", ("mava",))
    with pytest.raises(ValueError, match="match"):
        meeting.render_turn(turn, speaker_profile("speaker-01", 0), 8000, seed=0)


def test_render_turn_repeats_for_same_seed_and_changes_for_another() -> None:
    from choravox.phonetics import speaker_profile

    turn = dialogue.DialogueTurn(0.0, 0.4, "speaker-00", ("fisu",))
    profile = speaker_profile("speaker-00", 3)
    first = meeting.render_turn(turn, profile, 8000, seed=11).samples
    assert first.tobytes() == meeting.render_turn(turn, profile, 8000, seed=11).samples.tobytes()
    assert not np.array_equal(first, meeting.render_turn(turn, profile, 8000, seed=12).samples)


def test_render_turn_rejects_span_too_short_for_phones() -> None:
    from choravox.phonetics import speaker_profile

    turn = dialogue.DialogueTurn(0.0, 0.0001, "speaker-00", ("sito",))
    with pytest.raises(ValueError, match="too short"):
        meeting.render_turn(turn, speaker_profile("speaker-00", 0), 8000, seed=0)


def test_clean_meeting_places_turns_on_shared_clock() -> None:
    script = dialogue.DialogueScript(
        1.0,
        (
            dialogue.DialogueTurn(0.0, 0.4, "speaker-00", ("mava",)),
            dialogue.DialogueTurn(0.5, 0.9, "speaker-01", ("sito",)),
        ),
    )
    rendered = meeting.render_clean_meeting(script, 4000, seed=8)
    assert rendered.samples.shape == (4000,)
    assert np.all(rendered.samples[1600:2000] == 0.0)
    assert [word.speaker for word in rendered.words] == ["speaker-00", "speaker-01"]


def test_clean_meeting_adds_overlapping_turn_waveforms() -> None:
    host = dialogue.DialogueTurn(0.0, 0.5, "speaker-00", ("mava",))
    guest = dialogue.DialogueTurn(0.0, 0.5, "speaker-01", ("sito",))
    both = meeting.render_clean_meeting(dialogue.DialogueScript(0.5, (host, guest)), 4000, seed=2)
    host_only = meeting.render_clean_meeting(dialogue.DialogueScript(0.5, (host,)), 4000, seed=2)
    assert not np.array_equal(both.samples, host_only.samples)
    assert {word.speaker for word in both.words} == {"speaker-00", "speaker-01"}


def test_clean_meeting_repeats_audio_and_annotations_by_seed() -> None:
    script = dialogue.rapid_exchange_script(("speaker-00", "speaker-01"), 2, seed=3)
    first = meeting.render_clean_meeting(script, 4000, seed=20)
    second = meeting.render_clean_meeting(script, 4000, seed=20)
    assert first.samples.tobytes() == second.samples.tobytes()
    assert first.words == second.words


def test_clean_meeting_seed_changes_voice_but_not_planted_timing() -> None:
    script = dialogue.monologue_script(("mava", "sito"), word_duration_s=0.25)
    first = meeting.render_clean_meeting(script, 4000, seed=20)
    second = meeting.render_clean_meeting(script, 4000, seed=21)
    assert not np.array_equal(first.samples, second.samples)
    assert first.words == second.words


def test_clean_meeting_rejects_invalid_render_controls() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        meeting.render_clean_meeting(_script(), 0, seed=0)
    with pytest.raises(ValueError, match="seed"):
        meeting.render_clean_meeting(_script(), 8000, seed=-1)


def test_speaker_activity_uses_half_open_sample_intervals() -> None:
    script = dialogue.DialogueScript(
        1.0,
        (
            dialogue.DialogueTurn(0.0, 0.5, "a", ("mava",)),
            dialogue.DialogueTurn(0.25, 0.75, "b", ("sito",)),
        ),
    )
    speakers, activity = meeting.speaker_activity(script, 100)
    assert speakers == ("a", "b")
    assert np.sum(activity[:, 0]) == 50
    assert np.sum(activity[:, 1]) == 50
    assert np.sum(np.sum(activity, axis=1) == 2) == 25
    assert not np.any(activity[75:])


# -- extensions --
