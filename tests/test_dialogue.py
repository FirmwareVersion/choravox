from __future__ import annotations

import pytest

from choravox import dialogue


def test_synthetic_lexicon_has_stable_token_order() -> None:
    assert tuple(dialogue.LEXICON) == (
        "mava",
        "sito",
        "nena",
        "fisu",
        "shana",
        "teti",
        "nufi",
        "sena",
    )


def test_every_pronunciation_resolves_inventory_phones() -> None:
    from choravox.phonetics import PHONE_INVENTORY

    for lexeme in dialogue.LEXICON.values():
        assert all(symbol in PHONE_INVENTORY for symbol in lexeme.phones)


def test_pronunciation_returns_stable_phone_tuple() -> None:
    assert dialogue.pronunciation("sito") == ("s", "iy", "t", "uw")
    assert dialogue.pronunciation("nena") == ("n", "eh", "n", "aa")


def test_pronunciation_rejects_out_of_vocabulary_text() -> None:
    with pytest.raises(ValueError, match="unknown synthetic word"):
        dialogue.pronunciation("hello")


def test_lexeme_rejects_unknown_phone_and_empty_text() -> None:
    with pytest.raises(ValueError, match="nonempty token"):
        dialogue.Lexeme("", ("aa",))
    with pytest.raises(ValueError, match="unknown phone"):
        dialogue.Lexeme("bad", ("zz",))


def test_dialogue_script_allows_ordered_overlap() -> None:
    turns = (
        dialogue.DialogueTurn(0.0, 1.0, "speaker-a", ("mava",)),
        dialogue.DialogueTurn(0.5, 0.8, "speaker-b", ("sito",), "backchannel"),
    )
    script = dialogue.DialogueScript(1.0, turns)
    assert script.speakers == ("speaker-a", "speaker-b")


def test_dialogue_turn_rejects_unknown_transcript_token() -> None:
    with pytest.raises(ValueError, match="unknown synthetic word"):
        dialogue.DialogueTurn(0.0, 1.0, "speaker-a", ("unknown",))


def test_dialogue_rejects_turn_beyond_duration() -> None:
    turn = dialogue.DialogueTurn(0.0, 1.1, "speaker-a", ("mava",))
    with pytest.raises(ValueError, match="exceeds"):
        dialogue.DialogueScript(1.0, (turn,))


def test_dialogue_requires_nondecreasing_turn_starts() -> None:
    later = dialogue.DialogueTurn(0.5, 0.9, "speaker-a", ("mava",))
    earlier = dialogue.DialogueTurn(0.2, 0.4, "speaker-b", ("sito",))
    with pytest.raises(ValueError, match="ordered"):
        dialogue.DialogueScript(1.0, (later, earlier))


def test_dialogue_turn_validates_time_and_speaker() -> None:
    with pytest.raises(ValueError, match="start < end"):
        dialogue.DialogueTurn(1.0, 1.0, "speaker-a", ("mava",))
    with pytest.raises(ValueError, match="speaker"):
        dialogue.DialogueTurn(0.0, 1.0, "", ("mava",))


def test_activity_durations_partition_wall_clock_time() -> None:
    turns = (
        dialogue.DialogueTurn(0.2, 1.0, "a", ("mava",)),
        dialogue.DialogueTurn(0.7, 1.2, "b", ("sito",)),
    )
    durations = dialogue.activity_durations(dialogue.DialogueScript(1.5, turns))
    assert durations == dialogue.ActivityDurations(speech_s=1.0, overlap_s=0.3, silence_s=0.5)


def test_same_speaker_self_overlap_is_not_cross_speaker_overlap() -> None:
    turns = (
        dialogue.DialogueTurn(0.0, 1.0, "a", ("mava",)),
        dialogue.DialogueTurn(0.5, 1.5, "a", ("sito",)),
    )
    durations = dialogue.activity_durations(dialogue.DialogueScript(2.0, turns))
    assert durations.overlap_s == 0.0 and durations.speech_s == 1.5


def test_empty_script_is_all_silence() -> None:
    durations = dialogue.activity_durations(dialogue.DialogueScript(2.5, ()))
    assert durations == dialogue.ActivityDurations(0.0, 0.0, 2.5)


def test_full_two_speaker_overlap_counts_speech_once() -> None:
    turns = (
        dialogue.DialogueTurn(0.0, 1.0, "a", ("mava",)),
        dialogue.DialogueTurn(0.0, 1.0, "b", ("sito",)),
    )
    durations = dialogue.activity_durations(dialogue.DialogueScript(1.0, turns))
    assert durations == dialogue.ActivityDurations(1.0, 1.0, 0.0)


def test_touching_turns_do_not_overlap() -> None:
    turns = (
        dialogue.DialogueTurn(0.0, 0.5, "a", ("mava",)),
        dialogue.DialogueTurn(0.5, 1.0, "b", ("sito",)),
    )
    assert dialogue.activity_durations(dialogue.DialogueScript(1.0, turns)).overlap_s == 0.0


def test_monologue_generator_places_declared_pauses() -> None:
    script = dialogue.monologue_script(("mava", "sito", "nena"), word_duration_s=0.4, pause_s=0.2)
    assert script.duration_s == 1.6
    assert [turn.start_s for turn in script.turns] == [0.0, 0.6, 1.2]
    assert dialogue.activity_durations(script) == dialogue.ActivityDurations(1.2, 0.0, 0.4)


def test_monologue_preserves_word_order() -> None:
    words = ("sito", "mava", "fisu")
    script = dialogue.monologue_script(words)
    assert tuple(turn.words[0] for turn in script.turns) == words


def test_zero_pause_monologue_has_contiguous_turns() -> None:
    script = dialogue.monologue_script(("mava", "sito"), word_duration_s=0.5, pause_s=0.0)
    assert script.turns[0].end_s == script.turns[1].start_s
    assert dialogue.activity_durations(script).silence_s == 0.0


def test_monologue_uses_requested_speaker_for_every_turn() -> None:
    script = dialogue.monologue_script(("mava", "sito"), speaker="narrator")
    assert script.speakers == ("narrator",)


def test_monologue_rejects_empty_words_and_negative_pause() -> None:
    with pytest.raises(ValueError, match="valid"):
        dialogue.monologue_script(())
    with pytest.raises(ValueError, match="valid"):
        dialogue.monologue_script(("mava",), pause_s=-0.1)


def test_rapid_exchange_alternates_the_roster() -> None:
    script = dialogue.rapid_exchange_script(("a", "b", "c"), 2, seed=1)
    assert [turn.speaker for turn in script.turns] == ["a", "b", "c", "a", "b", "c"]
    assert len(script.turns) == 6


def test_rapid_exchange_repeats_transcript_by_seed() -> None:
    first = dialogue.rapid_exchange_script(("a", "b"), 4, seed=9)
    second = dialogue.rapid_exchange_script(("a", "b"), 4, seed=9)
    assert first == second


def test_rapid_exchange_has_declared_gap_and_no_overlap() -> None:
    script = dialogue.rapid_exchange_script(("a", "b"), 3, seed=2, turn_duration_s=0.3, gap_s=0.1)
    assert all(
        right.start_s - left.end_s == pytest.approx(0.1)
        for left, right in zip(script.turns, script.turns[1:], strict=False)
    )
    assert dialogue.activity_durations(script).overlap_s == 0.0


def test_rapid_exchange_requires_distinct_speakers() -> None:
    with pytest.raises(ValueError, match="distinct"):
        dialogue.rapid_exchange_script(("a", "a"), 2, seed=0)


def test_rapid_exchange_seed_changes_word_selection() -> None:
    first = dialogue.rapid_exchange_script(("a", "b"), 10, seed=1)
    second = dialogue.rapid_exchange_script(("a", "b"), 10, seed=2)
    assert tuple(turn.words for turn in first.turns) != tuple(turn.words for turn in second.turns)


def test_backchannels_overlap_eligible_host_turns() -> None:
    host = dialogue.DialogueScript(2.0, (dialogue.DialogueTurn(0.0, 2.0, "a", ("mava", "sito")),))
    result = dialogue.add_backchannels(host, "b", duration_s=0.2)
    backchannel = next(turn for turn in result.turns if turn.kind == "backchannel")
    assert backchannel.speaker == "b" and backchannel.words == ("nena",)
    assert dialogue.activity_durations(result).overlap_s == 0.2


def test_backchannels_skip_listener_and_short_turns() -> None:
    script = dialogue.DialogueScript(
        1.0,
        (
            dialogue.DialogueTurn(0.0, 0.4, "a", ("mava",)),
            dialogue.DialogueTurn(0.5, 1.0, "b", ("sito",)),
        ),
    )
    assert dialogue.add_backchannels(script, "b").turns == script.turns


# -- extensions --
