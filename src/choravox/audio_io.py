"""Portable PCM and RIFF/WAV codecs for samples-first audio."""

from __future__ import annotations

import os
import struct
from pathlib import Path
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


def encode_float32(samples: ArrayLike) -> bytes:
    """Encode finite audio as interleaved little-endian IEEE float32 values."""
    return as_waveform(samples).astype("<f4").tobytes(order="C")


def decode_float32(data: bytes, *, channels: int = 1) -> FloatArray:
    """Decode interleaved little-endian float32 audio into float64 processing values."""
    if channels <= 0:
        raise ValueError("channels must be positive")
    if len(data) % (4 * channels):
        raise ValueError("float PCM byte count must contain complete channel frames")
    values = np.frombuffer(data, dtype="<f4").astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("float PCM samples must be finite")
    return values if channels == 1 else values.reshape(-1, channels)


def wav_bytes(samples: ArrayLike, sample_rate: int, *, encoding: Encoding = "pcm16") -> bytes:
    """Build a canonical RIFF/WAVE document from mono or samples-first channels."""
    waveform = as_waveform(samples)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    channels = 1 if waveform.ndim == 1 else waveform.shape[1]
    if encoding == "pcm16":
        format_tag, bits, payload = 1, 16, encode_pcm16(waveform)
    elif encoding == "float32":
        format_tag, bits, payload = 3, 32, encode_float32(waveform)
    else:
        raise ValueError("encoding must be 'pcm16' or 'float32'")
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack("<HHIIHH", format_tag, channels, sample_rate, byte_rate, block_align, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def write_wav(
    path: str | os.PathLike[str],
    samples: ArrayLike,
    sample_rate: int,
    *,
    encoding: Encoding = "pcm16",
) -> None:
    """Write a canonical wave file to disk."""
    Path(path).write_bytes(wav_bytes(samples, sample_rate, encoding=encoding))


# -- extensions --
