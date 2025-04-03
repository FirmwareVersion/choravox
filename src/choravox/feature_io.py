"""Stable JSON serialization for extracted feature matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.hashing import canonical_json, content_hash

FloatArray = NDArray[np.float64]
SCHEMA = "choravox.features"
VERSION = 1


@dataclass(frozen=True, slots=True)
class FeatureArtifact:
    """Decoded feature values and their timebase."""

    sample_rate: int
    hop_length: int
    values: FloatArray


def features_to_json(values: ArrayLike, sample_rate: int, hop_length: int) -> str:
    """Encode finite two-dimensional features with an integrity digest."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a finite two-dimensional matrix")
    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError("feature timebase values must be positive")
    payload = {
        "hop_length": int(hop_length),
        "sample_rate": int(sample_rate),
        "shape": list(matrix.shape),
        "values": matrix.tolist(),
    }
    document = {
        "digest": content_hash(payload),
        "payload": payload,
        "schema": SCHEMA,
        "version": VERSION,
    }
    return canonical_json(document).decode("utf-8")


def features_from_json(text: str) -> FeatureArtifact:
    """Decode and validate a version-one feature artifact."""
    try:
        document: Any = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("feature artifact must be valid JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "digest",
        "payload",
        "schema",
        "version",
    }:
        raise ValueError("feature artifact envelope is malformed")
    if document["schema"] != SCHEMA or document["version"] != VERSION:
        raise ValueError("unsupported feature artifact schema or version")
    payload = document["payload"]
    if not isinstance(payload, dict) or document["digest"] != content_hash(payload):
        raise ValueError("feature artifact digest mismatch")
    try:
        matrix = np.asarray(payload["values"], dtype=np.float64)
        shape = tuple(payload["shape"])
        sample_rate = int(payload["sample_rate"])
        hop_length = int(payload["hop_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("feature artifact payload is malformed") from error
    if matrix.ndim != 2 or matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError("feature artifact matrix is malformed")
    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError("feature artifact timebase is invalid")
    return FeatureArtifact(sample_rate, hop_length, matrix)
