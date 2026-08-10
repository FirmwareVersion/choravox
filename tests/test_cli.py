from __future__ import annotations

from choravox import cli


def test_cli_json_helpers_use_canonical_utf8(tmp_path) -> None:
    path = tmp_path / "nested" / "value.json"
    cli._write_json(path, {"z": 1, "a": "会议"})
    assert path.read_text() == '{"a":"会议","z":1}\n'
    assert cli._read_json(path) == {"a": "会议", "z": 1}


def test_cli_reports_invalid_json_without_traceback(tmp_path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{")
    try:
        cli._read_json(path)
    except cli.CliError as error:
        assert "cannot read JSON" in str(error)
    else:
        raise AssertionError("invalid JSON was accepted")
    assert capsys.readouterr().err == ""


def test_cli_parser_exposes_every_pipeline_command() -> None:
    parser = cli.build_parser()
    action = next(action for action in parser._actions if action.dest == "command")
    assert set(action.choices) == {
        "synthesize",
        "vad",
        "embed",
        "diarize",
        "overlap",
        "transcribe",
        "align",
        "evaluate",
        "compare",
        "report",
        "demo",
        "stream-demo",
    }


def test_cli_root_help_and_version_return_success(capsys) -> None:
    assert cli.main([]) == 0
    assert "COMMAND" in capsys.readouterr().out
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip()
