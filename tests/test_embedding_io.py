from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from choravox import embedding_io, embeddings


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


def test_embedding_artifact_has_stable_golden_digest() -> None:
    records = (
        embedding_io.EmbeddingRecord("alpha-01", "alpha", np.array([0.0, 1.0])),
        embedding_io.EmbeddingRecord("beta-01", "beta", np.array([-0.5, 0.5])),
    )
    encoded = embedding_io.embeddings_to_json(records).encode()
    assert (
        hashlib.sha256(encoded).hexdigest()
        == "4bcfe2b63750c55251e5af2f36ab7e9a2efcaf46fff3a2aeb335d1bb2962025b"
    )


def test_embedding_decoder_rejects_shape_spoofing_with_valid_digest() -> None:
    from choravox.hashing import content_hash

    record = (embedding_io.EmbeddingRecord("utt", "speaker", np.ones(2)),)
    document = json.loads(embedding_io.embeddings_to_json(record))
    document["payload"]["dimension"] = 3
    document["digest"] = content_hash(document["payload"])
    with pytest.raises(ValueError, match="dimension mismatch"):
        embedding_io.embeddings_from_json(json.dumps(document))


def test_speaker_model_bundle_roundtrip_preserves_parameters() -> None:
    bundle = embedding_io.SpeakerModels(
        embeddings.DiagonalGmm(np.ones(1), np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]])),
        embeddings.PldaModel(np.array([0.5, -0.5]), np.eye(2)),
    )
    decoded = embedding_io.models_from_json(embedding_io.models_to_json(bundle))
    np.testing.assert_array_equal(decoded.ubm.means, bundle.ubm.means)
    np.testing.assert_array_equal(decoded.ubm.variances, bundle.ubm.variances)
    np.testing.assert_array_equal(decoded.plda.projection, bundle.plda.projection)


def test_speaker_model_bundle_rejects_integrity_failure() -> None:
    bundle = embedding_io.SpeakerModels(
        embeddings.DiagonalGmm(np.ones(1), np.zeros((1, 1)), np.ones((1, 1))),
        embeddings.PldaModel(np.zeros(1), np.ones((1, 1))),
    )
    document = json.loads(embedding_io.models_to_json(bundle))
    document["payload"]["ubm"]["means"][0][0] = 2.0
    with pytest.raises(ValueError, match="digest"):
        embedding_io.models_from_json(json.dumps(document))
