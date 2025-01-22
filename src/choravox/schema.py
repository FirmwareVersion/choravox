"""Versioned envelopes shared by serialized artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from choravox.hashing import content_hash


@dataclass(frozen=True, slots=True)
class Envelope:
    """A schema identifier, version, payload, and integrity digest."""

    schema: str
    version: int
    payload: Any
    digest: str


def make_envelope(schema: str, version: int, payload: Any) -> Envelope:
    """Build an integrity-protected schema envelope."""
    if not schema or version < 1:
        raise ValueError("schema must be nonempty and version positive")
    return Envelope(schema, version, payload, content_hash(payload))


def validate_envelope(envelope: Envelope, schema: str, versions: set[int]) -> Any:
    """Validate identity, supported version, and payload integrity."""
    if envelope.schema != schema:
        raise ValueError(f"expected schema {schema!r}, got {envelope.schema!r}")
    if envelope.version not in versions:
        raise ValueError(f"unsupported {schema} version {envelope.version}")
    if envelope.digest != content_hash(envelope.payload):
        raise ValueError("artifact digest mismatch")
    return envelope.payload


# -- extensions --
