"""Content-addressed bundles for reproducible synthetic meetings."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

from choravox.annotations import (
    PhoneAnnotation,
    rttm_to_text,
    script_to_rttm,
    stm_to_text,
    transcript_to_jsonl,
    words_to_phones,
    words_to_stm,
    words_to_transcript,
)
from choravox.audio_io import wav_bytes
from choravox.hashing import canonical_json, content_hash
from choravox.meeting import SyntheticMeeting

_DIGEST = re.compile(r"[0-9a-f]{64}")


def bytes_digest(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact artifact bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Manifest metadata for one relative-path byte artifact."""

    path: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        candidate = PurePosixPath(self.path)
        if not self.path or candidate.is_absolute() or ".." in candidate.parts or "\\" in self.path:
            raise ValueError("artifact path must be safe and relative")
        if not self.media_type or any(char.isspace() for char in self.media_type):
            raise ValueError("artifact media_type must be a nonempty token")
        if self.size_bytes < 0 or _DIGEST.fullmatch(self.sha256) is None:
            raise ValueError("artifact size or SHA-256 digest is invalid")


def artifact_record(path: str, media_type: str, data: bytes) -> ArtifactRecord:
    """Measure exact artifact bytes into a validated manifest record."""
    return ArtifactRecord(path, media_type, len(data), bytes_digest(data))


@dataclass(frozen=True, slots=True)
class MeetingManifest:
    """Version-one inventory and provenance for a synthetic meeting bundle."""

    recording_id: str
    seed: int
    config_hash: str
    conditions: dict[str, object]
    artifacts: tuple[ArtifactRecord, ...]

    def __post_init__(self) -> None:
        if not self.recording_id or any(char.isspace() for char in self.recording_id):
            raise ValueError("manifest recording_id must be a nonempty token")
        if self.seed < 0 or _DIGEST.fullmatch(self.config_hash) is None:
            raise ValueError("manifest seed or config_hash is invalid")
        paths = [record.path for record in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("manifest artifact paths must be unique")


def manifest_to_json(manifest: MeetingManifest) -> str:
    """Encode a deterministic manifest with an envelope integrity digest."""
    from choravox.hashing import canonical_json, content_hash

    payload = {
        "artifacts": [asdict(record) for record in manifest.artifacts],
        "conditions": manifest.conditions,
        "config_hash": manifest.config_hash,
        "recording_id": manifest.recording_id,
        "seed": manifest.seed,
    }
    document = {
        "digest": content_hash(payload),
        "payload": payload,
        "schema": "choravox.meeting-manifest",
        "version": 1,
    }
    return canonical_json(document).decode("utf-8") + "\n"


def manifest_from_json(text: str) -> MeetingManifest:
    """Decode and integrity-check a version-one meeting manifest."""
    from choravox.hashing import content_hash

    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("manifest must be valid JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("schema") != "choravox.meeting-manifest"
        or document.get("version") != 1
    ):
        raise ValueError("unsupported meeting manifest schema or version")
    payload = document.get("payload")
    if not isinstance(payload, dict) or document.get("digest") != content_hash(payload):
        raise ValueError("meeting manifest digest mismatch")
    try:
        records = tuple(ArtifactRecord(**record) for record in payload["artifacts"])
        return MeetingManifest(
            str(payload["recording_id"]),
            int(payload["seed"]),
            str(payload["config_hash"]),
            dict(payload["conditions"]),
            records,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("meeting manifest payload is malformed") from error


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    """In-memory named bytes paired with their content manifest."""

    files: tuple[tuple[str, bytes], ...]
    manifest: MeetingManifest


def verify_bundle(bundle: ArtifactBundle) -> None:
    """Reject missing, extra, reordered, resized, or modified artifact bytes."""
    names = [name for name, _ in bundle.files]
    expected = [record.path for record in bundle.manifest.artifacts]
    if names != expected:
        raise ValueError("bundle file order and names must match the manifest")
    for (name, data), record in zip(bundle.files, bundle.manifest.artifacts, strict=True):
        if len(data) != record.size_bytes or bytes_digest(data) != record.sha256:
            raise ValueError(f"bundle artifact digest mismatch: {name}")


def phones_to_jsonl(phones: tuple[PhoneAnnotation, ...]) -> str:
    """Encode planted phone timestamps with a versioned integrity header."""
    payload = [asdict(phone) for phone in phones]
    header = {
        "digest": content_hash(payload),
        "schema": "choravox.phones",
        "version": 1,
    }
    return (
        b"\n".join((canonical_json(header), *(canonical_json(item) for item in payload))).decode(
            "utf-8"
        )
        + "\n"
    )


def phones_from_jsonl(text: str) -> tuple[PhoneAnnotation, ...]:
    """Decode planted phone timestamps and reject payload corruption."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("phone JSONL requires a version header")
    try:
        header = json.loads(lines[0])
        payload = [json.loads(line) for line in lines[1:]]
    except json.JSONDecodeError as error:
        raise ValueError("phone JSONL contains invalid JSON") from error
    if (
        not isinstance(header, dict)
        or header.get("schema") != "choravox.phones"
        or header.get("version") != 1
    ):
        raise ValueError("unsupported phone schema or version")
    if header.get("digest") != content_hash(payload):
        raise ValueError("phone JSONL digest mismatch")
    try:
        return tuple(PhoneAnnotation(**item) for item in payload)
    except (TypeError, ValueError) as error:
        raise ValueError("phone JSONL payload is malformed") from error


def build_meeting_bundle(
    meeting: SyntheticMeeting,
    recording_id: str,
    *,
    conditions: dict[str, object] | None = None,
) -> ArtifactBundle:
    """Assemble audio and every planted annotation into verified in-memory bytes."""
    audio = wav_bytes(meeting.samples, meeting.sample_rate, encoding="float32")
    rttm = rttm_to_text(script_to_rttm(meeting.script, recording_id)).encode("utf-8")
    stm = stm_to_text(words_to_stm(meeting.words, recording_id)).encode("utf-8")
    transcript = transcript_to_jsonl(words_to_transcript(meeting.words)).encode("utf-8")
    phones = phones_to_jsonl(words_to_phones(meeting.words)).encode("utf-8")
    files = (
        ("meeting.wav", audio),
        ("turns.rttm", rttm),
        ("words.stm", stm),
        ("transcript.jsonl", transcript),
        ("phones.jsonl", phones),
    )
    media_types = (
        "audio/wav",
        "text/plain",
        "text/plain",
        "application/x-ndjson",
        "application/x-ndjson",
    )
    records = tuple(
        artifact_record(name, media_type, data)
        for (name, data), media_type in zip(files, media_types, strict=True)
    )
    script_payload = {
        "duration_s": meeting.script.duration_s,
        "turns": [asdict(turn) for turn in meeting.script.turns],
    }
    manifest = MeetingManifest(
        recording_id,
        meeting.seed,
        content_hash(script_payload),
        {} if conditions is None else conditions,
        records,
    )
    bundle = ArtifactBundle(files, manifest)
    verify_bundle(bundle)
    return bundle


# -- extensions --
