from __future__ import annotations

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


# -- extensions --
