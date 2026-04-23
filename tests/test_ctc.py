from __future__ import annotations

import pytest

from choravox import ctc


def vocabulary() -> ctc.CtcVocabulary:
    return ctc.CtcVocabulary(("<blank>", "sito", "mava", "<spk>"))


def test_ctc_vocabulary_roundtrips_words_and_speaker_changes() -> None:
    vocab = vocabulary()
    tokens = ("sito", "<spk>", "mava")
    assert vocab.decode(vocab.encode(tokens)) == tokens
    assert vocab.blank_id == 0 and vocab.speaker_change_id == 3


def test_ctc_vocabulary_rejects_unknown_tokens_and_identifiers() -> None:
    vocab = vocabulary()
    with pytest.raises(ValueError, match="unknown"):
        vocab.encode(("outside",))
    with pytest.raises(ValueError, match="outside"):
        vocab.decode((4,))


# -- extensions --
