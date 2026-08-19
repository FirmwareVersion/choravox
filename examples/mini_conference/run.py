"""Produce a complete deterministic mini-conference artifact bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from choravox.cli import main as cli_main
from choravox.hashing import canonical_json
from choravox.report import read_report_artifacts


def run(output: Path, *, seed: int = 41) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    meeting_directory = output / "meeting"
    report_directory = output / "report"
    synthesize_status = cli_main(
        [
            "synthesize",
            str(meeting_directory),
            "--seed",
            str(seed),
            "--duration",
            "2.0",
            "--overlap",
            "0.1",
        ]
    )
    demo_status = cli_main(["demo", str(report_directory), "--seed", str(seed)])
    if synthesize_status != 0 or demo_status != 0:
        raise RuntimeError("mini-conference command failed")
    report = read_report_artifacts(report_directory)
    summary: dict[str, object] = {
        "meeting_files": sorted(path.name for path in meeting_directory.iterdir()),
        "report_digest": report.digest,
        "report_files": sorted(path.name for path in report_directory.iterdir()),
        "seed": seed,
        "synthetic": True,
    }
    (output / "summary.json").write_bytes(canonical_json(summary) + b"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    summary = run(args.output, seed=args.seed)
    print(f"synthetic mini-conference digest={summary['report_digest']}; output={args.output}")


if __name__ == "__main__":
    main()
