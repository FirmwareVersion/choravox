from __future__ import annotations

import json

import numpy as np
import pytest

from choravox import embedding_io


def test_embedding_encoding_is_canonical_and_contains_schema() -> None:
    records = (embedding_io.EmbeddingRecord("utt-1", "speaker-a", np.array([0.6, 0.8])),)
    first = embedding_io.embeddings_to_json(records)
    second = embedding_io.embeddings_to_json(records)
    assert first == second
    assert '"schema":"choravox.embeddings"' in first
    assert '"version":1' in first


def test_embedding_encoding_requires_records_with_shared_dimension() -> None:
    with pytest.raises(ValueError, match="at least one"):
        embedding_io.embeddings_to_json(())
    records = (
        embedding_io.EmbeddingRecord("one", "a", np.ones(2)),
        embedding_io.EmbeddingRecord("two", "b", np.ones(3)),
    )
    with pytest.raises(ValueError, match="share"):
        embedding_io.embeddings_to_json(records)


def test_embedding_record_validates_identity_and_vector() -> None:
    with pytest.raises(ValueError, match="identities"):
        embedding_io.EmbeddingRecord("", "speaker", np.ones(2))
    with pytest.raises(ValueError, match="finite vector"):
        embedding_io.EmbeddingRecord("utt", "speaker", np.array([np.nan]))


def test_embedding_artifact_roundtrip_preserves_records_exactly() -> None:
    records = (
        embedding_io.EmbeddingRecord("utt-1", "a", np.array([0.6, 0.8])),
        embedding_io.EmbeddingRecord("utt-2", "b", np.array([-1.0, 0.0])),
    )
    decoded = embedding_io.embeddings_from_json(embedding_io.embeddings_to_json(records))
    assert [(item.utterance_id, item.speaker_id) for item in decoded] == [
        ("utt-1", "a"),
        ("utt-2", "b"),
    ]
    np.testing.assert_array_equal(decoded[0].values, records[0].values)
    np.testing.assert_array_equal(decoded[1].values, records[1].values)


def test_embedding_decoder_rejects_tampering_and_wrong_version() -> None:
    record = (embedding_io.EmbeddingRecord("utt", "a", np.ones(2)),)
    document = json.loads(embedding_io.embeddings_to_json(record))
    document["payload"]["records"][0]["values"][0] = 2.0
    with pytest.raises(ValueError, match="digest"):
        embedding_io.embeddings_from_json(json.dumps(document))
    document = json.loads(embedding_io.embeddings_to_json(record))
    document["version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        embedding_io.embeddings_from_json(json.dumps(document))
