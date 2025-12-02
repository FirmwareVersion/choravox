"""Versioned serialization for speaker embedding artifacts."""

from __future__ import annotations

from dataclasses import dataclass

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
