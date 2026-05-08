from __future__ import annotations

import pytest

from choravox import asr
from choravox.ctc import CtcVocabulary


def vocabulary() -> CtcVocabulary:
    return CtcVocabulary(("<blank>", "sito", "mava", "noki", "<spk>"))


def test_sot_roundtrip_preserves_speaker_turn_sequence() -> None:
    tokens = (
        asr.AttributedToken("sito", "a"),
        asr.AttributedToken("mava", "a"),
        asr.AttributedToken("noki", "b"),
    )
    encoded = asr.serialize_sot(tokens, vocabulary())
    assert encoded == (1, 2, 4, 3)
    assert asr.parse_sot(encoded, vocabulary(), ("a", "b")) == tokens


def test_empty_sot_serialization_is_explicit() -> None:
    assert asr.serialize_sot((), vocabulary()) == ()


def test_sot_parser_rejects_blank_and_excess_speaker_changes() -> None:
    with pytest.raises(ValueError, match="blank"):
        asr.parse_sot((0,), vocabulary(), ("a",))
    with pytest.raises(ValueError, match="more speaker"):
        asr.parse_sot((1, 4, 2), vocabulary(), ("a",))


def test_sot_roundtrip_handles_rapid_alternating_speakers() -> None:
    tokens = tuple(
        asr.AttributedToken(text, speaker)
        for text, speaker in zip(("sito", "mava", "noki"), ("a", "b", "c"), strict=True)
    )
    encoded = asr.serialize_sot(tokens, vocabulary())
    assert encoded == (1, 4, 2, 4, 3)
    assert asr.parse_sot(encoded, vocabulary(), ("a", "b", "c")) == tokens


def test_sot_encoding_does_not_insert_change_within_a_turn() -> None:
    tokens = tuple(asr.AttributedToken(text, "a") for text in ("sito", "mava", "noki"))
    assert asr.serialize_sot(tokens, vocabulary()) == (1, 2, 3)


def test_sot_parse_requires_a_valid_speaker_roster() -> None:
    with pytest.raises(ValueError, match="speaker order"):
        asr.parse_sot((1,), vocabulary(), ())
    with pytest.raises(ValueError, match="speaker order"):
        asr.parse_sot((1,), vocabulary(), ("",))


def test_attributed_tokens_validate_both_labels() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        asr.AttributedToken("", "a")
    with pytest.raises(ValueError, match="nonempty"):
        asr.AttributedToken("sito", "")


# -- extensions --
# -- extensions --
