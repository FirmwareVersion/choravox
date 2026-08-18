"""Run oracle-embedding diarization on one deterministic synthetic meeting."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from choravox.audio_io import write_wav
from choravox.config import MeetingConfig
from choravox.diarization import DiarizationConfig, diarize_embeddings
from choravox.hashing import canonical_json
from choravox.meeting import MeetingFactory
from choravox.metrics import diarization_error
from choravox.report import der_decomposition_svg, speaker_timeline_svg
from choravox.types import Segment


def run(output: Path, *, seed: int = 17) -> dict[str, object]:
    """Write audio, measured DER, and SVG views for a planted meeting."""
    meeting = MeetingFactory(
        MeetingConfig(speakers=2, duration_s=2.0, overlap_ratio=0.12, seed=seed),
        sample_rate=8000,
    ).build()
    output.mkdir(parents=True, exist_ok=True)
    write_wav(output / "meeting.wav", meeting.samples, meeting.sample_rate)

    speakers = meeting.script.speakers
    lookup = {speaker: index for index, speaker in enumerate(speakers)}
    oracle_embeddings = np.eye(len(speakers))[
        [lookup[turn.speaker] for turn in meeting.script.turns]
    ]
    result = diarize_embeddings(
        oracle_embeddings,
        DiarizationConfig(speaker_count=len(speakers), vb_iterations=2),
    )
    reference = tuple(
        Segment(turn.start_s, turn.end_s, turn.speaker) for turn in meeting.script.turns
    )
    hypothesis = tuple(
        Segment(turn.start_s, turn.end_s, f"cluster-{label}")
        for turn, label in zip(meeting.script.turns, result.labels, strict=True)
    )
    score = diarization_error(
        reference,
        hypothesis,
        frame_count=round(meeting.script.duration_s / 0.01),
        frame_shift_s=0.01,
    )
    (output / "speaker-timeline.svg").write_text(
        speaker_timeline_svg(hypothesis, duration_s=meeting.script.duration_s), encoding="utf-8"
    )
    (output / "der-decomposition.svg").write_text(der_decomposition_svg(score), encoding="utf-8")
    summary: dict[str, object] = {
        "der": score.rate,
        "errors": {
            "confusion": score.confusion,
            "false_alarm": score.false_alarm,
            "miss": score.miss,
        },
        "oracle_embedding_source": "planted synthetic speaker identity",
        "recovered_speakers": result.speaker_count,
        "seed": seed,
        "synthetic": True,
    }
    (output / "summary.json").write_bytes(canonical_json(summary) + b"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    summary = run(args.output, seed=args.seed)
    print(f"synthetic DER={summary['der']:.6f}; output={args.output}")


if __name__ == "__main__":
    main()
