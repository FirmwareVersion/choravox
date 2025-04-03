from __future__ import annotations

import struct

import numpy as np

from choravox import audio_io


def test_pcm16_codec_has_little_endian_golden_bytes() -> None:
    encoded = audio_io.encode_pcm16([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert encoded == struct.pack("<5h", -32768, -16384, 0, 16384, 32767)


def test_pcm16_roundtrip_stays_within_one_quantization_step() -> None:
    signal = np.linspace(-1.0, 1.0, 101)
    decoded = audio_io.decode_pcm16(audio_io.encode_pcm16(signal))
    np.testing.assert_allclose(decoded, signal, atol=1 / 32768)


# -- extensions --
