"""Speaker-attributed decoding mechanics for synthetic CTC output."""

from __future__ import annotations

from dataclasses import dataclass

from choravox.ctc import CtcVocabulary


@dataclass(frozen=True, slots=True)
class AttributedToken:
    text: str
    speaker: str

    def __post_init__(self) -> None:
        if not self.text or not self.speaker:
            raise ValueError("attributed token text and speaker must be nonempty")


def serialize_sot(
    tokens: tuple[AttributedToken, ...], vocabulary: CtcVocabulary
) -> tuple[int, ...]:
    """Insert a speaker-change token whenever the attributed speaker changes."""
    if not tokens:
        return ()
    output: list[str] = []
    previous = tokens[0].speaker
    for token in tokens:
        if token.speaker != previous:
            output.append("<spk>")
        output.append(token.text)
        previous = token.speaker
    return vocabulary.encode(tuple(output))


def parse_sot(
    token_ids: tuple[int, ...], vocabulary: CtcVocabulary, speaker_order: tuple[str, ...]
) -> tuple[AttributedToken, ...]:
    """Assign serialized tokens to successive speakers in a declared order."""
    if not speaker_order or any(not speaker for speaker in speaker_order):
        raise ValueError("SOT parsing requires a nonempty speaker order")
    texts = vocabulary.decode(token_ids)
    speaker_index = 0
    output = []
    for text in texts:
        if text == "<blank>":
            raise ValueError("collapsed SOT tokens must not contain blank")
        if text == "<spk>":
            speaker_index += 1
            if speaker_index >= len(speaker_order):
                raise ValueError("SOT stream contains more speaker changes than declared speakers")
        else:
            output.append(AttributedToken(text, speaker_order[speaker_index]))
    return tuple(output)


# -- extensions --
