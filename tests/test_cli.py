from __future__ import annotations

import json

import numpy as np
import pytest

from choravox import cli
from choravox.audio_io import read_wav, write_wav


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


def test_synthesize_command_writes_audio_and_planted_annotations(tmp_path) -> None:
    output = tmp_path / "meeting"
    assert cli.main(["synthesize", str(output), "--seed", "3", "--duration", "1.0"]) == 0
    assert read_wav(output / "meeting.wav").sample_rate == 16000
    assert (output / "reference.rttm").read_text().startswith("SPEAKER synthetic-meeting")
    assert (output / "transcript.jsonl").read_text().startswith("{")
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["seed"] == 3 and metadata["synthetic"] is True


def test_synthesize_command_is_byte_deterministic(tmp_path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    assert cli.main(["synthesize", str(left), "--seed", "5", "--duration", "1.0"]) == 0
    assert cli.main(["synthesize", str(right), "--seed", "5", "--duration", "1.0"]) == 0
    for name in ("meeting.wav", "reference.rttm", "transcript.jsonl", "metadata.json"):
        assert (left / name).read_bytes() == (right / name).read_bytes()


def test_vad_command_emits_frame_posteriors(tmp_path) -> None:
    input_path, output_path = tmp_path / "audio.wav", tmp_path / "vad.json"
    samples = np.concatenate((np.zeros(800), np.sin(np.linspace(0, 20, 1600)) * 0.2))
    write_wav(input_path, samples, 8000)
    assert cli.main(["vad", str(input_path), str(output_path)]) == 0
    payload = json.loads(output_path.read_text())
    assert len(payload["posterior"]) == len(payload["speech"])
    assert 0.0 <= min(payload["posterior"]) <= max(payload["posterior"]) <= 1.0


def test_embed_command_normalizes_statistics_vector(tmp_path) -> None:
    input_path, output_path = tmp_path / "features.npy", tmp_path / "embedding.json"
    np.save(input_path, np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]))
    assert cli.main(["embed", str(input_path), str(output_path)]) == 0
    payload = json.loads(output_path.read_text())
    assert payload["dimension"] == len(payload["embedding"])
    assert np.linalg.norm(payload["embedding"]) == pytest.approx(1.0)


def test_diarize_command_recovers_clean_speaker_count(tmp_path) -> None:
    input_path, output_path = tmp_path / "embeddings.npy", tmp_path / "diarization.json"
    np.save(input_path, np.array([[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [0.02, 0.98]]))
    assert cli.main(["diarize", str(input_path), str(output_path), "--speakers", "2"]) == 0
    payload = json.loads(output_path.read_text())
    assert payload["speaker_count"] == 2
    assert payload["labels"][0] == payload["labels"][1]


def test_overlap_command_reads_planted_activity_matrix(tmp_path) -> None:
    input_path, output_path = tmp_path / "activity.npy", tmp_path / "overlap.json"
    np.save(input_path, np.array([[1, 0], [1, 1], [0, 1]], dtype=bool))
    assert cli.main(["overlap", str(input_path), str(output_path)]) == 0
    assert json.loads(output_path.read_text())["overlap"] == [0, 1, 0]
