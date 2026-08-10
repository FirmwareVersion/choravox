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
    """Build the initial command-line parser."""
    parser = argparse.ArgumentParser(prog="choravox", description=__doc__)
    parser.add_argument("--version", action="store_true", help="print the package version")
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
