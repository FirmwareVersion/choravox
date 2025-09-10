from __future__ import annotations

import json

import pytest

from choravox import artifacts


def test_artifact_record_measures_exact_bytes() -> None:
    record = artifacts.artifact_record("audio/meeting.wav", "audio/wav", b"RIFFdata")
    assert record.path == "audio/meeting.wav"
    assert record.size_bytes == 8
    assert record.sha256 == artifacts.bytes_digest(b"RIFFdata")
    assert len(record.sha256) == 64


def test_bytes_digest_has_known_sha256_value() -> None:
    assert (
        artifacts.bytes_digest(b"choravox")
        == "66e3c58f57e8d45c021aaf65bbb91dc0b5fd3581d1ebc887a102dfe32d4f7e95"
    )


def test_artifact_record_rejects_absolute_and_parent_paths() -> None:
    for path in ("/tmp/audio.wav", "../audio.wav", "nested/../../audio.wav", r"nested\audio.wav"):
        with pytest.raises(ValueError, match="safe and relative"):
            artifacts.ArtifactRecord(path, "audio/wav", 1, "0" * 64)


def test_artifact_record_rejects_bad_media_size_and_digest() -> None:
    with pytest.raises(ValueError, match="media_type"):
        artifacts.ArtifactRecord("a.wav", "audio wav", 1, "0" * 64)
    with pytest.raises(ValueError, match="size"):
        artifacts.ArtifactRecord("a.wav", "audio/wav", -1, "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        artifacts.ArtifactRecord("a.wav", "audio/wav", 1, "ABC")


def test_artifact_digest_changes_with_one_byte() -> None:
    assert artifacts.bytes_digest(b"abc") != artifacts.bytes_digest(b"abd")


def _manifest() -> artifacts.MeetingManifest:
    records = (artifacts.artifact_record("meeting.wav", "audio/wav", b"audio"),)
    return artifacts.MeetingManifest("demo-01", 7, "a" * 64, {"snr_db": 10.0}, records)


def test_manifest_json_roundtrip_is_byte_stable() -> None:
    encoded = artifacts.manifest_to_json(_manifest())
    decoded = artifacts.manifest_from_json(encoded)
    assert decoded == _manifest()
    assert artifacts.manifest_to_json(decoded) == encoded
    assert json.loads(encoded)["schema"] == "choravox.meeting-manifest"


def test_manifest_preserves_condition_metadata_types() -> None:
    decoded = artifacts.manifest_from_json(artifacts.manifest_to_json(_manifest()))
    assert decoded.conditions == {"snr_db": 10.0}
    assert isinstance(decoded.conditions["snr_db"], float)


def test_manifest_rejects_duplicate_artifact_paths() -> None:
    record = artifacts.artifact_record("same.bin", "application/octet-stream", b"x")
    with pytest.raises(ValueError, match="unique"):
        artifacts.MeetingManifest("rec", 0, "b" * 64, {}, (record, record))


def test_manifest_decoder_detects_changed_payload() -> None:
    encoded = artifacts.manifest_to_json(_manifest())
    changed = encoded.replace('"seed":7', '"seed":8')
    with pytest.raises(ValueError, match="digest mismatch"):
        artifacts.manifest_from_json(changed)


def test_manifest_rejects_unknown_version_and_bad_config_hash() -> None:
    encoded = artifacts.manifest_to_json(_manifest()).replace('"version":1', '"version":2')
    with pytest.raises(ValueError, match="unsupported"):
        artifacts.manifest_from_json(encoded)
    with pytest.raises(ValueError, match="config_hash"):
        artifacts.MeetingManifest("rec", 0, "short", {}, ())


def test_bundle_verifier_accepts_exact_declared_bytes() -> None:
    files = (("a.txt", b"alpha"), ("b.bin", b"beta"))
    records = tuple(
        artifacts.artifact_record(name, "application/octet-stream", data) for name, data in files
    )
    manifest = artifacts.MeetingManifest("rec", 0, "c" * 64, {}, records)
    artifacts.verify_bundle(artifacts.ArtifactBundle(files, manifest))


def test_bundle_verifier_detects_one_byte_change() -> None:
    record = artifacts.artifact_record("a.txt", "text/plain", b"alpha")
    bundle = artifacts.ArtifactBundle(
        (("a.txt", b"alphb"),), artifacts.MeetingManifest("rec", 0, "c" * 64, {}, (record,))
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        artifacts.verify_bundle(bundle)


def test_bundle_verifier_rejects_missing_and_extra_files() -> None:
    record = artifacts.artifact_record("a.txt", "text/plain", b"alpha")
    manifest = artifacts.MeetingManifest("rec", 0, "d" * 64, {}, (record,))
    with pytest.raises(ValueError, match="order and names"):
        artifacts.verify_bundle(artifacts.ArtifactBundle((), manifest))
    with pytest.raises(ValueError, match="order and names"):
        artifacts.verify_bundle(
            artifacts.ArtifactBundle((("a.txt", b"alpha"), ("b.txt", b"beta")), manifest)
        )


def test_bundle_verifier_rejects_reordered_files() -> None:
    files = (("a.txt", b"a"), ("b.txt", b"b"))
    records = tuple(artifacts.artifact_record(name, "text/plain", data) for name, data in files)
    manifest = artifacts.MeetingManifest("rec", 0, "e" * 64, {}, records)
    with pytest.raises(ValueError, match="order"):
        artifacts.verify_bundle(artifacts.ArtifactBundle(tuple(reversed(files)), manifest))


def test_bundle_verifier_checks_size_before_digest() -> None:
    record = artifacts.artifact_record("a.txt", "text/plain", b"alpha")
    manifest = artifacts.MeetingManifest("rec", 0, "f" * 64, {}, (record,))
    with pytest.raises(ValueError, match="digest mismatch"):
        artifacts.verify_bundle(
            artifacts.ArtifactBundle((("a.txt", b"shorter-or-longer"),), manifest)
        )


# -- extensions --
