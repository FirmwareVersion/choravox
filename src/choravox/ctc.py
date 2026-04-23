"""CTC decoding and forced-alignment mechanics for synthetic speech."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CtcVocabulary:
    """Ordered CTC tokens with blank at zero and an explicit speaker change."""

    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.tokens) < 3
            or self.tokens[0] != "<blank>"
            or "<spk>" not in self.tokens
            or len(set(self.tokens)) != len(self.tokens)
            or any(not token or token.isspace() for token in self.tokens)
        ):
            raise ValueError(
                "CTC vocabulary requires unique blank, speaker-change, and text tokens"
            )

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def speaker_change_id(self) -> int:
        return self.tokens.index("<spk>")

    def encode(self, tokens: tuple[str, ...]) -> tuple[int, ...]:
        """Map known token strings to integer identifiers."""
        lookup = {token: index for index, token in enumerate(self.tokens)}
        try:
            return tuple(lookup[token] for token in tokens)
        except KeyError as error:
            raise ValueError(f"unknown CTC token: {error.args[0]}") from error

    def decode(self, token_ids: tuple[int, ...]) -> tuple[str, ...]:
        """Map valid integer identifiers back to token strings."""
        if any(token_id < 0 or token_id >= len(self.tokens) for token_id in token_ids):
            raise ValueError("CTC token identifier is outside the vocabulary")
        return tuple(self.tokens[token_id] for token_id in token_ids)


# -- extensions --
