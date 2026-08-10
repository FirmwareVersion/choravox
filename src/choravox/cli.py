"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from choravox.hashing import canonical_json


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
