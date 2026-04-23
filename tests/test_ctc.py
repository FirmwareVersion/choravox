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


def test_ctc_vocabulary_requires_blank_at_zero_and_speaker_change() -> None:
    invalid = (
        ("word", "<blank>", "<spk>"),
        ("<blank>", "word", "other"),
        ("<blank>", "<spk>", "<spk>"),
        ("<blank>", "<spk>"),
    )
    for tokens in invalid:
        with pytest.raises(ValueError, match="vocabulary"):
            ctc.CtcVocabulary(tokens)


def test_vocabulary_encoding_is_stable_under_repeated_use() -> None:
    vocab = vocabulary()
    first = vocab.encode(("mava", "sito", "mava"))
    second = vocab.encode(("mava", "sito", "mava"))
    assert first == second == (2, 1, 2)


def test_vocabulary_decodes_empty_sequence_without_special_cases() -> None:
    assert vocabulary().decode(()) == ()


# -- extensions --
# -- extensions --
