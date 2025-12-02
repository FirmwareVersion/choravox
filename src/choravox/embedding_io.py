"""Versioned serialization for speaker embedding artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from choravox.hashing import canonical_json, content_hash

FloatArray = NDArray[np.float64]
SCHEMA = "choravox.embeddings"
VERSION = 1


@dataclass(frozen=True, slots=True)
class EmbeddingRecord:
    """One identified embedding from a synthetic utterance."""

    utterance_id: str
    speaker_id: str
    values: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if not self.utterance_id or not self.speaker_id:
            raise ValueError("embedding identities must be nonempty")
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("embedding values must be a nonempty finite vector")


def embeddings_to_json(records: tuple[EmbeddingRecord, ...]) -> str:
    """Encode same-dimensional embedding records with an integrity digest."""
    if not records:
        raise ValueError("at least one embedding record is required")
    dimension = records[0].values.size
    if any(record.values.size != dimension for record in records):
        raise ValueError("embedding records must share one dimension")
    payload = {
        "dimension": int(dimension),
        "records": [
            {
                "speaker_id": record.speaker_id,
                "utterance_id": record.utterance_id,
                "values": np.asarray(record.values, dtype=np.float64).tolist(),
            }
            for record in records
        ],
    }
    document = {
        "digest": content_hash(payload),
        "payload": payload,
        "schema": SCHEMA,
        "version": VERSION,
    }
    return canonical_json(document).decode("utf-8")


def embeddings_from_json(text: str) -> tuple[EmbeddingRecord, ...]:
    """Decode strict version-one embedding records and verify integrity."""
    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("embedding artifact must be valid JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "digest",
        "payload",
        "schema",
        "version",
    }:
        raise ValueError("embedding artifact envelope is malformed")
    if document["schema"] != SCHEMA or document["version"] != VERSION:
        raise ValueError("unsupported embedding artifact schema or version")
    payload = document["payload"]
    if not isinstance(payload, dict) or document["digest"] != content_hash(payload):
        raise ValueError("embedding artifact digest mismatch")
    try:
        dimension = int(payload["dimension"])
        raw_records = payload["records"]
        records = tuple(
            EmbeddingRecord(
                str(item["utterance_id"]),
                str(item["speaker_id"]),
                np.asarray(item["values"], dtype=np.float64),
            )
            for item in raw_records
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("embedding artifact payload is malformed") from error
    if set(payload) != {"dimension", "records"} or dimension < 1 or not records:
        raise ValueError("embedding artifact payload is malformed")
    if any(record.values.size != dimension for record in records):
        raise ValueError("embedding artifact dimension mismatch")
    return records
