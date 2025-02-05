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


def test_to_mono_averages_or_weights_channels() -> None:
    stereo = np.array([[1.0, -1.0], [0.0, 1.0]])
    np.testing.assert_allclose(audio.to_mono(stereo), [0.0, 0.5])
    np.testing.assert_allclose(audio.to_mono(stereo, [1.0, 0.0]), [1.0, 0.0])


def test_to_mono_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="one finite value"):
        audio.to_mono(np.zeros((4, 2)), [1.0])
    with pytest.raises(ValueError, match="sum to zero"):
        audio.to_mono(np.zeros((4, 2)), [1.0, -1.0])


# -- extensions --
