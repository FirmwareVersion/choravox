from __future__ import annotations

import numpy as np
import pytest

from choravox import vad


def test_frame_energy_reports_silence_floor_and_full_scale() -> None:
    silence = vad.frame_energy_db(np.zeros(800), 8000, frame_ms=20.0, hop_ms=10.0)
    full = vad.frame_energy_db(np.ones(800), 8000, frame_ms=20.0, hop_ms=10.0)
    np.testing.assert_array_equal(silence.dbfs, -120.0)
    np.testing.assert_allclose(full.dbfs[:-1], 0.0, atol=1e-9)


def test_frame_energy_exposes_sample_derived_timing() -> None:
    measured = vad.frame_energy_db(np.ones(800), 8000, frame_ms=20.0, hop_ms=10.0)
    assert measured.frame_duration_s == 0.02
    assert measured.starts_s[:3].tolist() == [0.0, 0.01, 0.02]
    assert measured.centers_s[:3].tolist() == [0.01, 0.02, 0.03]


def test_frame_energy_downmixes_multichannel_audio() -> None:
    stereo = np.column_stack((np.ones(400), np.ones(400)))
    mono = vad.frame_energy_db(np.ones(400), 8000)
    measured = vad.frame_energy_db(stereo, 8000)
    np.testing.assert_array_equal(measured.dbfs, mono.dbfs)


def test_frame_energy_empty_audio_has_empty_time_axis() -> None:
    measured = vad.frame_energy_db([], 16000)
    assert measured.dbfs.shape == (0,)
    assert measured.starts_s.shape == measured.centers_s.shape == (0,)


def test_frame_energy_rejects_invalid_timing_and_floor() -> None:
    with pytest.raises(ValueError, match="positive and ordered"):
        vad.frame_energy_db(np.ones(10), 8000, frame_ms=10.0, hop_ms=20.0)
    with pytest.raises(ValueError, match="floor_db"):
        vad.frame_energy_db(np.ones(10), 8000, floor_db=1.0)


# -- extensions --
