"""Deterministic frame-level overlap detection for synthetic meetings."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


def overlap_targets(activity: ArrayLike) -> BoolArray:
    """Mark frames containing at least two planted active speakers."""
    matrix = np.asarray(activity)
    if matrix.ndim != 2:
        raise ValueError("speaker activity must be a frame-by-speaker matrix")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("speaker activity must be binary")
    return np.sum(matrix.astype(np.int64), axis=1) >= 2


# -- extensions --
