from __future__ import annotations

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


# -- extensions --
