from __future__ import annotations

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


# -- extensions --
