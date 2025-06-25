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


# -- extensions --
