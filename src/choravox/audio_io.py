"""Portable PCM and RIFF/WAV codecs for samples-first audio."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform

FloatArray = NDArray[np.float64]
Encoding = Literal["pcm16", "float32"]


def encode_pcm16(samples: ArrayLike) -> bytes:
    """Quantize normalized audio to little-endian signed sixteen-bit PCM."""
    waveform = np.clip(as_waveform(samples), -1.0, np.nextafter(1.0, 0.0))
    quantized = np.rint(waveform * 32768.0).clip(-32768, 32767).astype("<i2")
    return quantized.tobytes(order="C")


def decode_pcm16(data: bytes, *, channels: int = 1) -> FloatArray:
    """Decode interleaved little-endian PCM to normalized samples-first audio."""
    if channels <= 0:
        raise ValueError("channels must be positive")
    if len(data) % (2 * channels):
        raise ValueError("PCM byte count must contain complete channel frames")
    values = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
    return values if channels == 1 else values.reshape(-1, channels)


# -- extensions --
