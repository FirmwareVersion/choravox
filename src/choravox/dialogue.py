"""Planted dialogue scripts for synthetic multi-speaker meetings."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from choravox.phonetics import get_phone


@dataclass(frozen=True, slots=True)
class Lexeme:
    """Synthetic transcript token and its phone pronunciation."""

    text: str
    phones: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.text or any(char.isspace() for char in self.text):
            raise ValueError("lexeme text must be a nonempty token")
        if not self.phones:
            raise ValueError("lexeme pronunciation must be nonempty")
        for symbol in self.phones:
            get_phone(symbol)


_LEXEMES = (
    Lexeme("mava", ("aa", "f", "aa")),
    Lexeme("sito", ("s", "iy", "t", "uw")),
    Lexeme("nena", ("n", "eh", "n", "aa")),
    Lexeme("fisu", ("f", "iy", "s", "uw")),
    Lexeme("shana", ("sh", "aa", "n", "aa")),
    Lexeme("teti", ("t", "eh", "t", "iy")),
    Lexeme("nufi", ("n", "uw", "f", "iy")),
    Lexeme("sena", ("s", "eh", "n", "aa")),
)
LEXICON = MappingProxyType({lexeme.text: lexeme for lexeme in _LEXEMES})


def pronunciation(text: str) -> tuple[str, ...]:
    """Return the planted pronunciation of one synthetic token."""
    try:
        return LEXICON[text].phones
    except KeyError as error:
        raise ValueError(f"unknown synthetic word: {text}") from error


# -- extensions --
