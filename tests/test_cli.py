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


def test_transcribe_command_collapses_ctc_path(tmp_path) -> None:
    input_path, output_path = tmp_path / "scores.npy", tmp_path / "transcript.json"
    np.save(input_path, np.log([[0.1, 0.8, 0.1, 1e-6], [0.8, 0.1, 0.1, 1e-6]]))
    assert cli.main(["transcribe", str(input_path), str(output_path), "--tokens", "sito,mava"]) == 0
    assert json.loads(output_path.read_text())["tokens"] == ["sito"]


def test_align_command_emits_token_frame_spans(tmp_path) -> None:
    input_path, output_path = tmp_path / "scores.npy", tmp_path / "alignment.json"
    np.save(input_path, np.log([[0.1, 0.9, 1e-6], [0.9, 0.1, 1e-6]]))
    assert (
        cli.main(
            ["align", str(input_path), str(output_path), "--tokens", "sito", "--target", "sito"]
        )
        == 0
    )
    token = json.loads(output_path.read_text())["tokens"][0]
    assert (token["start_frame"], token["end_frame"]) == (0, 1)


def test_evaluate_command_computes_permutation_invariant_cpwer(tmp_path) -> None:
    reference = tmp_path / "ref.json"
    hypothesis = tmp_path / "hyp.json"
    output = tmp_path / "out.json"
    cli._write_json(reference, {"a": "one two", "b": "three"})
    cli._write_json(hypothesis, {"x": "three", "y": "one two"})
    assert cli.main(["evaluate", str(reference), str(hypothesis), str(output)]) == 0
    assert json.loads(output.read_text())["rate"] == 0.0


def test_compare_command_reports_paired_improvements(tmp_path) -> None:
    baseline = tmp_path / "base.json"
    candidate = tmp_path / "new.json"
    output = tmp_path / "out.json"
    cli._write_json(baseline, {"a": 0.3, "b": 0.2})
    cli._write_json(candidate, {"a": 0.1, "b": 0.2})
    assert cli.main(["compare", str(baseline), str(candidate), str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["improvements"] == 1 and payload["ties"] == 1


def report_spec() -> dict:
    return {
        "cpwer": 0.0,
        "diarization": {
            "confusion": 0,
            "evaluated_frames": 10,
            "false_alarm": 0,
            "miss": 0,
            "reference_speaker_frames": 10,
        },
        "duration_s": 1.0,
        "latency_points": [{"block_s": 0.05, "cpwer": 0.0, "der": 0.0, "latency_s": 0.1}],
        "seed": 3,
        "segments": [{"end": 1.0, "speaker": "a", "start": 0.0}],
        "snr_points": [[0.0, 0.0]],
        "title": "合成报告",
        "words": [
            {
                "confidence": 1.0,
                "end": 0.2,
                "speaker": "a",
                "start": 0.0,
                "text": "sito",
            }
        ],
    }


def test_report_command_writes_self_contained_bundle(tmp_path) -> None:
    spec_path, output = tmp_path / "spec.json", tmp_path / "report"
    cli._write_json(spec_path, report_spec())
    assert cli.main(["report", str(spec_path), str(output)]) == 0
    assert (output / "report.md").read_text().startswith("# 合成报告")
    assert (output / "report.html").read_text().startswith("<!doctype html>")
    assert (output / "manifest.json").exists()
