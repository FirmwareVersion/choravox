"""Measure synthetic CTC speaker-attribution mechanics across overlap ratios."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from choravox.asr import AttributedToken, decode_speaker_attributed, serialize_sot
from choravox.config import MeetingConfig
from choravox.ctc import CtcVocabulary
from choravox.hashing import canonical_json
from choravox.meeting import script_from_config
from choravox.metrics import cpwer
from choravox.report import metric_curve_svg


def _ctc_scores(token_ids: tuple[int, ...], vocabulary_size: int) -> np.ndarray:
    scores = np.full((2 * len(token_ids), vocabulary_size), -8.0)
    for index, token_id in enumerate(token_ids):
        scores[2 * index, token_id] = -0.01
        scores[2 * index + 1, 0] = -0.01
    return scores


def _condition(overlap_ratio: float, seed: int) -> dict[str, object]:
    script = script_from_config(
        MeetingConfig(speakers=2, duration_s=4.0, overlap_ratio=overlap_ratio, seed=seed)
    )
    words = {
        speaker: tuple(
            word for turn in script.turns if turn.speaker == speaker for word in turn.words
        )[:3]
        for speaker in script.speakers
    }
    if any(not tokens for tokens in words.values()):
        raise RuntimeError("synthetic script did not cover every speaker")
    reference_tokens = tuple(
        AttributedToken(word, speaker) for speaker in script.speakers for word in words[speaker]
    )
    vocabulary = CtcVocabulary(
        ("<blank>", *tuple(sorted({token.text for token in reference_tokens})), "<spk>")
    )
    deletion_count = min(
        len(reference_tokens) - len(script.speakers), round(overlap_ratio * len(reference_tokens))
    )
    retained = list(reference_tokens)
    for _ in range(deletion_count):
        retained.pop()
    encoded = serialize_sot(tuple(retained), vocabulary)
    decoded = decode_speaker_attributed(
        _ctc_scores(encoded, len(vocabulary.tokens)),
        vocabulary,
        script.speakers,
        frame_shift_s=0.01,
    )
    reference = {speaker: " ".join(words[speaker]) for speaker in script.speakers}
    hypothesis = {speaker: decoded.transcript.text(speaker) for speaker in script.speakers}
    score = cpwer(reference, hypothesis)
    return {
        "cpwer": score.rate,
        "deleted_tokens": deletion_count,
        "overlap_ratio": overlap_ratio,
        "reference_words": score.reference_words,
    }


def run(output: Path, *, seed: int = 23) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    conditions = tuple(_condition(ratio, seed) for ratio in (0.0, 0.15, 0.3))
    points = tuple((float(item["overlap_ratio"]), float(item["cpwer"])) for item in conditions)
    (output / "overlap-cpwer.svg").write_text(
        metric_curve_svg(
            points,
            title="Synthetic overlap stress versus cpWER",
            x_label="Planted overlap ratio",
            y_label="cpWER",
        ),
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "adapter": "deterministic synthetic CTC score matrix",
        "conditions": conditions,
        "seed": seed,
        "synthetic": True,
    }
    (output / "summary.json").write_bytes(canonical_json(summary) + b"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    summary = run(args.output, seed=args.seed)
    print(f"synthetic overlap conditions={len(summary['conditions'])}; output={args.output}")


if __name__ == "__main__":
    main()
