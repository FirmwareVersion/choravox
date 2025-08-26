from __future__ import annotations

import json

import pytest

from choravox import annotations


def test_rttm_roundtrip_has_stable_field_layout() -> None:
    entries = (
        annotations.RttmEntry("meeting-01", 1, 0.25, 1.5, "speaker-00"),
        annotations.RttmEntry("meeting-01", 1, 1.8, 0.2, "speaker-01"),
    )
    text = annotations.rttm_to_text(entries)
    assert (
        text.splitlines()[0]
        == "SPEAKER meeting-01 1 0.250000 1.500000 <NA> <NA> speaker-00 <NA> <NA>"
    )
    assert annotations.rttm_from_text(text) == entries


def test_rttm_entry_converts_duration_to_segment_end() -> None:
    segment = annotations.RttmEntry("rec", 1, 0.125, 0.375, "alice").segment()
    assert (segment.start, segment.end, segment.speaker) == (0.125, 0.5, "alice")


def test_rttm_rejects_wrong_record_type_or_field_count() -> None:
    with pytest.raises(ValueError, match="line 1"):
        annotations.rttm_from_text("LEXEME rec 1 0.0 1.0 <NA> <NA> spk <NA> <NA>\n")
    with pytest.raises(ValueError, match="line 1"):
        annotations.rttm_from_text("SPEAKER too short\n")


def test_rttm_rejects_invalid_times_channel_and_tokens() -> None:
    with pytest.raises(ValueError, match="channel"):
        annotations.RttmEntry("rec", 0, 0.0, 1.0, "spk")
    with pytest.raises(ValueError, match="positive duration"):
        annotations.RttmEntry("rec", 1, 0.0, 0.0, "spk")
    with pytest.raises(ValueError, match="speaker"):
        annotations.RttmEntry("rec", 1, 0.0, 1.0, "two words")


def test_empty_rttm_is_a_valid_empty_annotation() -> None:
    assert annotations.rttm_to_text(()) == ""
    assert annotations.rttm_from_text("\n") == ()


def test_stm_roundtrip_preserves_multiword_transcript() -> None:
    entries = (
        annotations.StmEntry("meeting-02", 1, "speaker-00", 0.0, 1.25, "mava sito"),
        annotations.StmEntry("meeting-02", 1, "speaker-01", 1.3, 1.8, "nena"),
    )
    text = annotations.stm_to_text(entries)
    assert text.splitlines()[0] == "meeting-02 1 speaker-00 0.000000 1.250000 <NA> mava sito"
    assert annotations.stm_from_text(text) == entries


def test_stm_empty_document_roundtrips() -> None:
    assert annotations.stm_to_text(()) == ""
    assert annotations.stm_from_text("") == ()


def test_stm_rejects_reversed_interval_and_empty_transcript() -> None:
    with pytest.raises(ValueError, match="interval"):
        annotations.StmEntry("rec", 1, "spk", 1.0, 1.0, "mava")
    with pytest.raises(ValueError, match="transcript"):
        annotations.StmEntry("rec", 1, "spk", 0.0, 1.0, "  ")


def test_stm_parser_reports_line_of_malformed_record() -> None:
    good = "rec 1 spk 0.000000 1.000000 <NA> mava"
    with pytest.raises(ValueError, match="line 2"):
        annotations.stm_from_text(good + "\nrec 1 broken\n")


def test_stm_rejects_embedded_newline_in_transcript() -> None:
    with pytest.raises(ValueError, match="single-line"):
        annotations.StmEntry("rec", 1, "spk", 0.0, 1.0, "mava\nsito")


def test_transcript_jsonl_roundtrip_preserves_word_fields() -> None:
    from choravox.types import Transcript, Word

    transcript = Transcript(
        (
            Word("mava", 0.0, 0.3, "speaker-00", 0.9),
            Word("sito", 0.3, 0.7, "speaker-00", 0.8),
        )
    )
    encoded = annotations.transcript_to_jsonl(transcript)
    assert json.loads(encoded.splitlines()[0])["schema"] == "choravox.transcript"
    assert annotations.transcript_from_jsonl(encoded) == transcript


def test_empty_transcript_still_has_integrity_header() -> None:
    from choravox.types import Transcript

    encoded = annotations.transcript_to_jsonl(Transcript(()))
    assert len(encoded.splitlines()) == 1
    assert annotations.transcript_from_jsonl(encoded).words == ()


def test_transcript_jsonl_rejects_missing_or_unknown_header() -> None:
    with pytest.raises(ValueError, match="version header"):
        annotations.transcript_from_jsonl("")
    bad = '{"schema":"choravox.transcript","version":2,"digest":"x"}\n'
    with pytest.raises(ValueError, match="unsupported"):
        annotations.transcript_from_jsonl(bad)


def test_transcript_jsonl_detects_changed_word_line() -> None:
    from choravox.types import Transcript, Word

    encoded = annotations.transcript_to_jsonl(Transcript((Word("mava", 0.0, 0.2, "spk"),)))
    changed = encoded.replace('"mava"', '"sito"')
    with pytest.raises(ValueError, match="digest mismatch"):
        annotations.transcript_from_jsonl(changed)


def test_transcript_jsonl_rejects_malformed_word_payload() -> None:
    header = {"schema": "choravox.transcript", "version": 1, "digest": "invalid"}
    with pytest.raises(ValueError, match="digest mismatch"):
        annotations.transcript_from_jsonl(json.dumps(header) + "\n{}\n")


def test_planted_script_converts_to_rttm_without_time_loss() -> None:
    from choravox.dialogue import DialogueScript, DialogueTurn

    script = DialogueScript(1.0, (DialogueTurn(0.1, 0.6, "spk", ("mava",)),))
    entries = annotations.script_to_rttm(script, "rec")
    assert entries == (annotations.RttmEntry("rec", 1, 0.1, 0.5, "spk"),)
    assert annotations.rttm_from_text(annotations.rttm_to_text(entries)) == entries


def test_planted_words_convert_to_transcript_and_stm() -> None:
    from choravox.meeting import PlantedWord

    words = (PlantedWord("mava", "spk", 0.0, 0.3), PlantedWord("sito", "spk", 0.3, 0.7))
    transcript = annotations.words_to_transcript(words)
    stm = annotations.words_to_stm(words, "rec")
    assert transcript.text() == "mava sito"
    assert [entry.transcript for entry in stm] == ["mava", "sito"]


def test_rttm_export_retains_simultaneous_speaker_turns() -> None:
    from choravox.dialogue import DialogueScript, DialogueTurn

    script = DialogueScript(
        1.0,
        (
            DialogueTurn(0.0, 0.7, "a", ("mava",)),
            DialogueTurn(0.4, 0.9, "b", ("sito",)),
        ),
    )
    entries = annotations.script_to_rttm(script, "overlap")
    assert len(entries) == 2
    assert entries[0].segment().end > entries[1].segment().start
    assert {entry.speaker for entry in entries} == {"a", "b"}


def test_annotation_conversion_rejects_unsafe_recording_id() -> None:
    from choravox.dialogue import DialogueScript

    with pytest.raises(ValueError, match="recording_id"):
        annotations.script_to_rttm(DialogueScript(1.0, ()), "two words")


def test_word_conversion_keeps_chronological_order() -> None:
    from choravox.meeting import PlantedWord

    words = (PlantedWord("mava", "a", 0.0, 0.4), PlantedWord("sito", "b", 0.2, 0.5))
    assert [word.text for word in annotations.words_to_transcript(words).words] == ["mava", "sito"]


def test_phone_expansion_matches_synthetic_pronunciation() -> None:
    from choravox.dialogue import pronunciation
    from choravox.meeting import PlantedWord

    word = PlantedWord("sito", "speaker-00", 0.2, 0.6)
    phones = annotations.words_to_phones((word,))
    assert tuple(phone.symbol for phone in phones) == pronunciation("sito")
    assert phones[0].start_s == word.start_s
    assert phones[-1].end_s == word.end_s


def test_phone_expansion_preserves_word_and_speaker_identity() -> None:
    from choravox.meeting import PlantedWord

    phones = annotations.words_to_phones((PlantedWord("mava", "spk", 0.0, 0.3),))
    assert {phone.word for phone in phones} == {"mava"}
    assert {phone.speaker for phone in phones} == {"spk"}


# -- extensions --
