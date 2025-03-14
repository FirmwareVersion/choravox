"""Deterministic framing and spectral feature primitives."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform

FloatArray = NDArray[np.float64]


def frame_signal(
    samples: ArrayLike,
    frame_length: int,
    hop_length: int,
    *,
    end: Literal["drop", "pad"] = "drop",
    pad_value: float = 0.0,
) -> FloatArray:
    """Slice a samples-first waveform into frames without copying channels together."""
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    if end not in {"drop", "pad"}:
        raise ValueError("end must be 'drop' or 'pad'")
    if not np.isfinite(pad_value):
        raise ValueError("pad_value must be finite")
    waveform = as_waveform(samples)
    sample_count = waveform.shape[0]
    if sample_count == 0 or (end == "drop" and sample_count < frame_length):
        return np.empty((0, frame_length, *waveform.shape[1:]), dtype=np.float64)
    if end == "drop":
        frame_count = 1 + (sample_count - frame_length) // hop_length
    else:
        frame_count = 1 + max(0, math.ceil((sample_count - frame_length) / hop_length))
    output = np.full((frame_count, frame_length, *waveform.shape[1:]), pad_value, dtype=np.float64)
    for index in range(frame_count):
        start = index * hop_length
        available = min(frame_length, sample_count - start)
        output[index, :available] = waveform[start : start + available]
    return output


# -- extensions --
