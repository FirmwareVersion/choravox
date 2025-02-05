from __future__ import annotations

import numpy as np
import pytest

from choravox import audio


def test_as_waveform_preserves_samples_first_geometry() -> None:
    mono = audio.as_waveform([0, 1, -1])
    stereo = audio.as_waveform([[0, 1], [1, 0]])
    assert mono.shape == (3,) and mono.dtype == np.float64
    assert stereo.shape == (2, 2)


def test_as_waveform_rejects_nonfinite_and_rank_three_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        audio.as_waveform([0.0, np.inf])
    with pytest.raises(ValueError, match="shape"):
        audio.as_waveform(np.zeros((2, 2, 2)))


# -- extensions --
