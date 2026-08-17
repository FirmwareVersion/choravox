"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from choravox.ctc import CtcVocabulary
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


def _command_diarize(args: argparse.Namespace) -> None:
    import numpy as np

    from choravox.diarization import DiarizationConfig, diarize_embeddings

    embeddings = np.load(args.input, allow_pickle=False)
    result = diarize_embeddings(embeddings, DiarizationConfig(speaker_count=args.speakers))
    _write_json(
        args.output,
        {
            "labels": result.labels.tolist(),
            "posteriors": result.posteriors.tolist(),
            "speaker_count": result.speaker_count,
        },
    )


def _command_overlap(args: argparse.Namespace) -> None:
    import numpy as np

    from choravox.overlap import overlap_targets

    activity = np.load(args.input, allow_pickle=False)
    targets = overlap_targets(activity)
    _write_json(args.output, {"overlap": targets.astype(int).tolist()})


def _vocabulary(text: str) -> CtcVocabulary:
    tokens = tuple(token.strip() for token in text.split(",") if token.strip())
    return CtcVocabulary(("<blank>", *tokens, "<spk>"))


def _command_transcribe(args: argparse.Namespace) -> None:
    import numpy as np

    from choravox.ctc import greedy_decode

    vocabulary = _vocabulary(args.tokens)
    result = greedy_decode(np.load(args.input, allow_pickle=False))
    _write_json(
        args.output,
        {
            "score": result.score,
            "token_ids": result.token_ids,
            "tokens": vocabulary.decode(result.token_ids),
        },
    )


def _command_align(args: argparse.Namespace) -> None:
    import numpy as np

    from choravox.ctc import forced_align

    vocabulary = _vocabulary(args.tokens)
    target = tuple(token.strip() for token in args.target.split(",") if token.strip())
    result = forced_align(np.load(args.input, allow_pickle=False), vocabulary.encode(target))
    _write_json(
        args.output,
        {
            "score": result.score,
            "tokens": [
                {
                    "confidence": item.confidence,
                    "end_frame": item.end_frame,
                    "start_frame": item.start_frame,
                    "text": text,
                }
                for text, item in zip(target, result.tokens, strict=True)
            ],
        },
    )


def _command_evaluate(args: argparse.Namespace) -> None:
    from choravox.metrics import cpwer

    reference = _read_json(args.reference)
    hypothesis = _read_json(args.hypothesis)
    if not isinstance(reference, dict) or not isinstance(hypothesis, dict):
        raise CliError("evaluation inputs must be speaker-to-transcript objects")
    result = cpwer(reference, hypothesis)
    _write_json(
        args.output,
        {
            "errors": result.errors,
            "mapping": result.mapping,
            "rate": result.rate,
            "reference_words": result.reference_words,
            "synthetic": True,
        },
    )


def _command_compare(args: argparse.Namespace) -> None:
    from choravox.evaluation import compare_scores

    baseline = _read_json(args.baseline)
    candidate = _read_json(args.candidate)
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise CliError("comparison inputs must be recording-to-score objects")
    result = compare_scores(baseline, candidate)
    _write_json(
        args.output,
        {
            "improvements": result.improvements,
            "mean_difference": result.mean_difference,
            "regressions": result.regressions,
            "ties": result.ties,
        },
    )


def _report_from_spec(spec: dict[str, Any], output: str | Path) -> None:
    from choravox import __version__
    from choravox.metrics import DiarizationError
    from choravox.report import (
        CoverageFacts,
        RunMetadata,
        standard_report_charts,
        synthesize_report,
        write_report_artifacts,
    )
    from choravox.streaming import LatencyQualityPoint
    from choravox.types import Segment, Transcript, Word

    segments = tuple(Segment(**item) for item in spec["segments"])
    transcript = Transcript(tuple(Word(**item) for item in spec["words"]))
    diarization = DiarizationError(**spec["diarization"])
    latency = tuple(LatencyQualityPoint(**item) for item in spec["latency_points"])
    metadata = RunMetadata.from_config(
        package_version=__version__,
        seed=int(spec["seed"]),
        config=spec,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    )
    charts = standard_report_charts(
        segments=segments,
        duration_s=float(spec["duration_s"]),
        diarization=diarization,
        snr_points=tuple(tuple(item) for item in spec["snr_points"]),
        latency_points=latency,
    )
    artifacts = synthesize_report(
        title=str(spec["title"]),
        metadata=metadata,
        transcript=transcript,
        diarization=diarization,
        cpwer=float(spec["cpwer"]),
        facts=CoverageFacts(True, True, True, False),
        charts=charts,
    )
    write_report_artifacts(output, artifacts, overwrite=True)


def _command_report(args: argparse.Namespace) -> None:
    spec = _read_json(args.spec)
    if not isinstance(spec, dict):
        raise CliError("report specification must be a JSON object")
    _report_from_spec(spec, args.output)


def _demo_spec(seed: int, *, streaming: bool) -> dict[str, Any]:
    from choravox.annotations import words_to_transcript
    from choravox.config import MeetingConfig
    from choravox.meeting import MeetingFactory
    from choravox.metrics import cpwer, diarization_error
    from choravox.types import Segment
    from choravox.vad import VadConfig, detect_speech

    meeting = MeetingFactory(
        MeetingConfig(speakers=2, duration_s=2.0, overlap_ratio=0.1, seed=seed), sample_rate=8000
    ).build()
    vad = detect_speech(meeting.samples, VadConfig(sample_rate=meeting.sample_rate))
    segments = tuple(
        Segment(turn.start_s, turn.end_s, turn.speaker) for turn in meeting.script.turns
    )
    frame_count = round(meeting.script.duration_s / 0.01)
    der = diarization_error(segments, segments, frame_count=frame_count, frame_shift_s=0.01)
    transcript = words_to_transcript(meeting.words)
    speaker_text = {speaker: transcript.text(speaker) for speaker in transcript.speakers}
    cpwer_result = cpwer(speaker_text, speaker_text)
    return {
        "cpwer": cpwer_result.rate,
        "diarization": {
            "confusion": der.confusion,
            "evaluated_frames": der.evaluated_frames,
            "false_alarm": der.false_alarm,
            "miss": der.miss,
            "reference_speaker_frames": der.reference_speaker_frames,
        },
        "duration_s": meeting.script.duration_s,
        "latency_points": [
            {
                "block_s": 0.05,
                "cpwer": cpwer_result.rate,
                "der": der.rate,
                "latency_s": 0.06 if streaming else 0.0,
            }
        ],
        "mechanics": {
            "diarization_adapter": "planted/oracle segments",
            "speech_frames": int(vad.speech.sum()),
            "transcript_adapter": "planted synthetic words",
        },
        "seed": seed,
        "segments": [
            {"end": item.end, "speaker": item.speaker, "start": item.start} for item in segments
        ],
        "snr_points": [[0.0, der.rate]],
        "title": "choravox synthetic streaming demo" if streaming else "choravox synthetic demo",
        "words": [
            {
                "confidence": word.confidence,
                "end": word.end,
                "speaker": word.speaker,
                "start": word.start,
                "text": word.text,
            }
            for word in transcript.words
        ],
    }


def _command_demo(args: argparse.Namespace) -> None:
    spec = _demo_spec(args.seed, streaming=False)
    _report_from_spec(spec, args.output)
    point = spec["latency_points"][0]
    print(f"synthetic DER={point['der']:.6f} cpWER={spec['cpwer']:.6f}")


def _command_stream_demo(args: argparse.Namespace) -> None:
    spec = _demo_spec(args.seed, streaming=True)
    _report_from_spec(spec, args.output)
    point = spec["latency_points"][0]
    print(
        f"synthetic online DER={point['der']:.6f} "
        f"cpWER={spec['cpwer']:.6f} latency={point['latency_s']:.3f}s"
    )


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
    commands.choices["diarize"].set_defaults(handler=_command_diarize)
    commands.choices["overlap"].set_defaults(handler=_command_overlap)
    commands.choices["transcribe"].add_argument("--tokens", required=True)
    commands.choices["transcribe"].set_defaults(handler=_command_transcribe)
    commands.choices["align"].add_argument("--tokens", required=True)
    commands.choices["align"].add_argument("--target", required=True)
    commands.choices["align"].set_defaults(handler=_command_align)

    evaluate = commands.add_parser("evaluate", help="compute cpWER from transcript JSON")
    evaluate.add_argument("reference")
    evaluate.add_argument("hypothesis")
    evaluate.add_argument("output")
    evaluate.set_defaults(handler=_command_evaluate)

    compare = commands.add_parser("compare", help="compare paired lower-is-better scores")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("output")
    compare.set_defaults(handler=_command_compare)

    report_command = commands.add_parser("report", help="render Markdown and HTML from JSON spec")
    report_command.add_argument("spec")
    report_command.add_argument("output")
    report_command.set_defaults(handler=_command_report)

    demo = commands.add_parser("demo", help="run the deterministic offline mechanics demo")
    demo.add_argument("output")
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(handler=_command_demo)

    stream = commands.add_parser("stream-demo", help="compare synthetic online/offline mechanics")
    stream.add_argument("output")
    stream.add_argument("--seed", type=int, default=7)
    stream.set_defaults(handler=_command_stream_demo)
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
