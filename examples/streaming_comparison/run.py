"""Compare online and offline synthetic meeting quality with explicit latency."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from choravox.config import MeetingConfig
from choravox.hashing import canonical_json
from choravox.meeting import script_from_config
from choravox.metrics import cpwer, diarization_error
from choravox.report import latency_quality_svg
from choravox.streaming import (
    LatencyQualityPoint,
    compare_online_offline,
    streaming_latency,
)
from choravox.types import Segment


def _quantize_online(
    reference: tuple[Segment, ...], block_s: float, duration_s: float
) -> tuple[Segment, ...]:
    output = []
    for segment in reference:
        start = round(math.ceil(segment.start / block_s) * block_s, 6)
        end = min(duration_s, round(math.ceil(segment.end / block_s) * block_s, 6))
        if end > start:
            output.append(Segment(start, end, segment.speaker))
    return tuple(output)


def run(output: Path, *, seed: int = 31) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    script = script_from_config(
        MeetingConfig(speakers=2, duration_s=4.0, overlap_ratio=0.12, seed=seed)
    )
    reference_segments = tuple(
        Segment(turn.start_s, turn.end_s, turn.speaker) for turn in script.turns
    )
    frame_count = round(script.duration_s / 0.01)
    offline_der = diarization_error(
        reference_segments,
        reference_segments,
        frame_count=frame_count,
        frame_shift_s=0.01,
    )
    reference_text = {
        speaker: " ".join(
            word for turn in script.turns if turn.speaker == speaker for word in turn.words
        )
        for speaker in script.speakers
    }
    offline_cpwer = cpwer(reference_text, reference_text)
    comparisons = []
    chart_points = []
    for block_s in (0.02, 0.05, 0.1):
        online_segments = _quantize_online(reference_segments, block_s, script.duration_s)
        online_der = diarization_error(
            reference_segments,
            online_segments,
            frame_count=frame_count,
            frame_shift_s=0.01,
        )
        online_text = dict(reference_text)
        revisions = int(block_s >= 0.1)
        if revisions:
            speaker = script.speakers[-1]
            online_text[speaker] = " ".join(online_text[speaker].split()[:-1])
        online_cpwer = cpwer(reference_text, online_text)
        latency = streaming_latency(
            block_size=round(block_s * 8000),
            sample_rate=8000,
            lookahead_frames=2,
        )
        comparison = compare_online_offline(
            offline_der,
            online_der,
            offline_cpwer,
            online_cpwer,
            latency,
            revisions=revisions,
        )
        comparisons.append(
            {
                "block_s": block_s,
                "cpwer_gap": comparison.cpwer_gap,
                "der_gap": comparison.der_gap,
                "latency_s": comparison.latency_s,
                "revisions": comparison.revisions,
            }
        )
        chart_points.append(
            LatencyQualityPoint(
                comparison.latency_s,
                comparison.online_der,
                comparison.online_cpwer,
                block_s,
            )
        )
    (output / "latency-quality.html").write_text(
        latency_quality_svg(tuple(chart_points)), encoding="utf-8"
    )
    summary: dict[str, object] = {
        "comparisons": comparisons,
        "offline_cpwer": offline_cpwer.rate,
        "offline_der": offline_der.rate,
        "seed": seed,
        "synthetic": True,
    }
    (output / "summary.json").write_bytes(canonical_json(summary) + b"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    summary = run(args.output, seed=args.seed)
    print(f"synthetic online points={len(summary['comparisons'])}; output={args.output}")


if __name__ == "__main__":
    main()
