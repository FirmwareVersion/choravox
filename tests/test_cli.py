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
