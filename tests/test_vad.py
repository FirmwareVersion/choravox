from __future__ import annotations

import numpy as np

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


# -- extensions --
