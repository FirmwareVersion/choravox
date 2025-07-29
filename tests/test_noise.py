from __future__ import annotations

import numpy as np
import pytest

from choravox import audio, noise


def test_white_noise_is_centered_at_requested_rms() -> None:
    samples = noise.white_noise(4096, seed=7, rms_level=0.2)
    assert np.mean(samples) == pytest.approx(0.0, abs=1e-9)
    assert audio.rms(samples) == pytest.approx(0.2, abs=1e-8)


def test_white_noise_rejects_empty_or_invalid_level() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        noise.white_noise(0, seed=1)
    with pytest.raises(ValueError, match="rms_level"):
        noise.white_noise(8, seed=1, rms_level=float("nan"))


# -- extensions --
