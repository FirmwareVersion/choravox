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


def overlap_add(frames: ArrayLike, hop_length: int, *, length: int | None = None) -> FloatArray:
    """Reconstruct samples by averaging every overlapping frame contribution."""
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[1] == 0:
        raise ValueError("frames must have shape (frames, samples[, channels])")
    if not np.all(np.isfinite(values)):
        raise ValueError("frames must be finite")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")
    if length is not None and length < 0:
        raise ValueError("length must be nonnegative")
    natural = 0 if values.shape[0] == 0 else (values.shape[0] - 1) * hop_length + values.shape[1]
    requested = natural if length is None else length
    output = np.zeros((max(natural, requested), *values.shape[2:]), dtype=np.float64)
    weight = np.zeros(max(natural, requested), dtype=np.float64)
    for index, frame in enumerate(values):
        start = index * hop_length
        output[start : start + values.shape[1]] += frame
        weight[start : start + values.shape[1]] += 1.0
    if output.ndim == 2:
        output = np.divide(output, weight[:, None], out=output, where=weight[:, None] > 0)
    else:
        output = np.divide(output, weight, out=output, where=weight > 0)
    return np.asarray(output[:requested], dtype=np.float64)


# -- extensions --
