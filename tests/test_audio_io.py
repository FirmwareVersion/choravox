from __future__ import annotations

import struct

import numpy as np
import pytest

from choravox import audio_io


def test_pcm16_codec_has_little_endian_golden_bytes() -> None:
    encoded = audio_io.encode_pcm16([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert encoded == struct.pack("<5h", -32768, -16384, 0, 16384, 32767)


def test_pcm16_roundtrip_stays_within_one_quantization_step() -> None:
    signal = np.linspace(-1.0, 1.0, 101)
    decoded = audio_io.decode_pcm16(audio_io.encode_pcm16(signal))
    np.testing.assert_allclose(decoded, signal, atol=1 / 32768)


def test_pcm16_interleaves_and_restores_samples_first_channels() -> None:
    stereo = np.array([[0.25, -0.25], [0.5, -0.5]])
    encoded = audio_io.encode_pcm16(stereo)
    assert encoded == struct.pack("<4h", 8192, -8192, 16384, -16384)
    np.testing.assert_array_equal(audio_io.decode_pcm16(encoded, channels=2), stereo)


def test_pcm16_saturates_out_of_range_samples() -> None:
    decoded = audio_io.decode_pcm16(audio_io.encode_pcm16([-2.0, 2.0]))
    np.testing.assert_array_equal(decoded, [-1.0, 32767 / 32768])


def test_pcm16_rejects_incomplete_channel_frames() -> None:
    with pytest.raises(ValueError, match="complete"):
        audio_io.decode_pcm16(b"\x00\x00", channels=2)
    with pytest.raises(ValueError, match="positive"):
        audio_io.decode_pcm16(b"", channels=0)


def test_float32_codec_preserves_representable_values() -> None:
    values = np.array([-1.5, -0.0, 0.25, 2.0], dtype=np.float32)
    encoded = audio_io.encode_float32(values)
    assert encoded == struct.pack("<4f", *values)
    np.testing.assert_array_equal(audio_io.decode_float32(encoded), values.astype(np.float64))


def test_float32_codec_keeps_multichannel_geometry() -> None:
    stereo = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    restored = audio_io.decode_float32(audio_io.encode_float32(stereo), channels=2)
    np.testing.assert_array_equal(restored, stereo.astype(np.float64))


def test_float32_decoder_rejects_nonfinite_payloads() -> None:
    with pytest.raises(ValueError, match="finite"):
        audio_io.decode_float32(struct.pack("<f", float("nan")))


def test_float32_decoder_rejects_partial_samples_and_frames() -> None:
    with pytest.raises(ValueError, match="complete"):
        audio_io.decode_float32(b"\x00\x00")
    with pytest.raises(ValueError, match="complete"):
        audio_io.decode_float32(struct.pack("<f", 1.0), channels=2)


def test_float_encoder_rejects_nonfinite_source() -> None:
    with pytest.raises(ValueError, match="finite"):
        audio_io.encode_float32([np.inf])


def test_wav_writer_emits_canonical_header(tmp_path: object) -> None:
    from pathlib import Path

    target = Path(str(tmp_path)) / "tone.wav"
    audio_io.write_wav(target, [0.0, 0.5], 8000)
    document = target.read_bytes()
    assert document[:12] == b"RIFF" + struct.pack("<I", 40) + b"WAVE"
    assert document[12:36] == b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
    assert document[36:44] == b"data" + struct.pack("<I", 4)


def test_wav_writer_marks_float_encoding_and_channels() -> None:
    document = audio_io.wav_bytes(np.zeros((3, 2)), 16_000, encoding="float32")
    format_tag, channels = struct.unpack("<HH", document[20:24])
    assert (format_tag, channels) == (3, 2)


@pytest.mark.parametrize("encoding", ["pcm16", "float32"])
def test_wave_codec_roundtrips_multichannel_audio(encoding: audio_io.Encoding) -> None:
    stereo = np.array([[0.0, 0.25], [-0.5, 0.75]])
    decoded = audio_io.parse_wav(audio_io.wav_bytes(stereo, 22_050, encoding=encoding))
    assert (decoded.sample_rate, decoded.encoding) == (22_050, encoding)
    tolerance = 1 / 32768 if encoding == "pcm16" else 1e-7
    np.testing.assert_allclose(decoded.samples, stereo, atol=tolerance)


def test_wave_file_reader_restores_written_audio(tmp_path: object) -> None:
    from pathlib import Path

    target = Path(str(tmp_path)) / "stereo.wav"
    audio_io.write_wav(target, np.zeros((8, 2)), 16_000)
    assert audio_io.read_wav(target).samples.shape == (8, 2)


# -- extensions --
