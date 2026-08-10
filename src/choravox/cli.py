"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from choravox.hashing import canonical_json


def _command_synthesize(args: argparse.Namespace) -> None:
    from choravox.annotations import (
        rttm_to_text,
        script_to_rttm,
        transcript_to_jsonl,
        words_to_transcript,
    )
    from choravox.audio_io import write_wav
    from choravox.config import MeetingConfig
    from choravox.meeting import MeetingFactory

    config = MeetingConfig(
        speakers=args.speakers,
        duration_s=args.duration,
        overlap_ratio=args.overlap,
        seed=args.seed,
    )
    meeting = MeetingFactory(config).build()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_wav(output / "meeting.wav", meeting.samples, meeting.sample_rate)
    (output / "reference.rttm").write_text(
        rttm_to_text(script_to_rttm(meeting.script, "synthetic-meeting"))
    )
    (output / "transcript.jsonl").write_text(
        transcript_to_jsonl(words_to_transcript(meeting.words))
    )
    _write_json(
        output / "metadata.json",
        {
            "duration_s": meeting.script.duration_s,
            "sample_rate": meeting.sample_rate,
            "seed": meeting.seed,
            "speakers": meeting.script.speakers,
            "synthetic": True,
        },
    )


def _command_vad(args: argparse.Namespace) -> None:
    from choravox.audio_io import read_wav
    from choravox.vad import VadConfig, detect_speech

    audio = read_wav(args.input)
    result = detect_speech(audio.samples, VadConfig(sample_rate=audio.sample_rate))
    _write_json(
        args.output,
        {
            "posterior": result.posterior.tolist(),
            "speech": result.speech.astype(int).tolist(),
            "synthetic_or_local_input": True,
            "threshold_db": result.threshold_db,
        },
    )


def _command_embed(args: argparse.Namespace) -> None:
    import numpy as np

    from choravox.embeddings import embed_features, initialize_gmm

    features = np.load(args.input, allow_pickle=False)
    if features.ndim != 2 or features.shape[0] < 2:
        raise CliError("embedding input must contain at least two feature frames")
    model = initialize_gmm(features, min(2, features.shape[0]), seed=0)
    embedding = embed_features(features, model)
    _write_json(args.output, {"dimension": embedding.size, "embedding": embedding.tolist()})


class CliError(ValueError):
    """A user-facing command failure without an internal traceback."""


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(canonical_json(value) + b"\n")


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CliError(f"cannot read JSON input: {path}") from error


def build_parser() -> argparse.ArgumentParser:
    """Build the complete offline command-line parser."""
    parser = argparse.ArgumentParser(prog="choravox", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print the package version")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    synthesize = commands.add_parser("synthesize", help="render a planted synthetic meeting")
    synthesize.add_argument("output", help="output directory")
    synthesize.add_argument("--seed", type=int, default=0)
    synthesize.add_argument("--duration", type=float, default=2.0)
    synthesize.add_argument("--speakers", type=int, default=2)
    synthesize.add_argument("--overlap", type=float, default=0.1)
    synthesize.set_defaults(handler=_command_synthesize)

    for name, help_text in (
        ("vad", "run energy/HMM voice activity detection"),
        ("embed", "extract a statistics speaker embedding from a feature matrix"),
        ("diarize", "cluster a matrix of speaker embeddings"),
        ("overlap", "derive overlap labels from planted activity"),
        ("transcribe", "greedily decode a CTC log-probability matrix"),
        ("align", "force-align target tokens to CTC log probabilities"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("input", help="input WAV or NumPy .npy file")
        command.add_argument("output", help="output JSON file")
    commands.choices["vad"].set_defaults(handler=_command_vad)
    commands.choices["embed"].set_defaults(handler=_command_embed)
    commands.choices["diarize"].add_argument("--speakers", type=int)
    commands.choices["transcribe"].add_argument("--tokens", required=True)
    commands.choices["align"].add_argument("--tokens", required=True)
    commands.choices["align"].add_argument("--target", required=True)

    evaluate = commands.add_parser("evaluate", help="compute cpWER from transcript JSON")
    evaluate.add_argument("reference")
    evaluate.add_argument("hypothesis")
    evaluate.add_argument("output")

    compare = commands.add_parser("compare", help="compare paired lower-is-better scores")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("output")

    report_command = commands.add_parser("report", help="render Markdown and HTML from JSON spec")
    report_command.add_argument("spec")
    report_command.add_argument("output")

    demo = commands.add_parser("demo", help="run the deterministic offline mechanics demo")
    demo.add_argument("output")
    demo.add_argument("--seed", type=int, default=7)

    stream = commands.add_parser("stream-demo", help="compare synthetic online/offline mechanics")
    stream.add_argument("output")
    stream.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    from choravox import __version__

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
    elif handler := getattr(args, "handler", None):
        try:
            handler(args)
        except (CliError, OSError, ValueError) as error:
            print(f"choravox: error: {error}", file=sys.stderr)
            return 2
    else:
        parser.print_help()
    return 0
