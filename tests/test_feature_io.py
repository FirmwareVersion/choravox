from __future__ import annotations

import numpy as np

from choravox.feature_io import features_from_json, features_to_json


def test_feature_artifact_roundtrip_is_exact() -> None:
    matrix = np.array([[0.0, -1.25], [3.5, 9.0]])
    encoded = features_to_json(matrix, 16_000, 160)
    decoded = features_from_json(encoded)
    assert (decoded.sample_rate, decoded.hop_length) == (16_000, 160)
    np.testing.assert_array_equal(decoded.values, matrix)


def test_feature_artifact_encoding_is_canonical_and_repeatable() -> None:
    matrix = np.array([[1.0, 2.0]])
    first = features_to_json(matrix, 8000, 80)
    assert first == features_to_json(matrix, 8000, 80)
    assert " " not in first and first.startswith('{"digest":')
