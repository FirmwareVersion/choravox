from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def run_example(name: str, output: Path) -> tuple[dict, str]:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name / "run.py"), str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((output / "summary.json").read_text()), completed.stdout


def test_diarization_example_recovers_clean_oracle_speakers(tmp_path) -> None:
    summary, stdout = run_example("diarization_meeting", tmp_path / "run")
    assert summary["synthetic"] is True
    assert summary["recovered_speakers"] == 2
    assert summary["der"] == 0.0
    assert "synthetic DER=0.000000" in stdout
    assert (tmp_path / "run" / "speaker-timeline.svg").read_text().startswith("<svg")


def test_asr_overlap_example_measures_monotone_synthetic_stress(tmp_path) -> None:
    summary, stdout = run_example("asr_overlap_sweep", tmp_path / "run")
    rates = [condition["cpwer"] for condition in summary["conditions"]]
    assert rates[0] == 0.0
    assert rates == sorted(rates)
    assert "synthetic overlap conditions=3" in stdout
    assert (tmp_path / "run" / "overlap-cpwer.svg").read_text().startswith("<svg")
