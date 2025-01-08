from __future__ import annotations

import tomllib
from pathlib import Path

from choravox import __version__
from choravox.cli import main


def test_version_is_consistent(capsys: object) -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == __version__
    assert f"## [{__version__}]" in (root / "CHANGELOG.md").read_text(encoding="utf-8")


def test_cli_reports_version(capsys: object) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__  # type: ignore[attr-defined]
