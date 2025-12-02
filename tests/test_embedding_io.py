from __future__ import annotations

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
