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


# -- extensions --
