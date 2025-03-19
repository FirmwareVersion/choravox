from __future__ import annotations

import numpy as np
import pytest

from choravox import features


def test_frame_signal_uses_declared_hop_and_drops_incomplete_tail() -> None:
    framed = features.frame_signal(np.arange(7), frame_length=4, hop_length=2)
    np.testing.assert_array_equal(framed, [[0, 1, 2, 3], [2, 3, 4, 5]])


def test_frame_signal_can_pad_the_last_partial_frame() -> None:
    framed = features.frame_signal([1.0, 2.0, 3.0], 2, 2, end="pad", pad_value=-1.0)
    np.testing.assert_array_equal(framed, [[1.0, 2.0], [3.0, -1.0]])


def test_frame_signal_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="positive"):
        features.frame_signal([1.0], 0, 1)
    with pytest.raises(ValueError, match="end"):
        features.frame_signal([1.0], 1, 1, end="mirror")  # type: ignore[arg-type]


def test_overlap_add_inverts_complete_overlapping_frames() -> None:
    signal = np.arange(8, dtype=np.float64)
    frames = features.frame_signal(signal, 4, 2)
    np.testing.assert_array_equal(features.overlap_add(frames, 2), signal)


def test_overlap_add_respects_requested_trim_and_padding() -> None:
    frames = features.frame_signal([1.0, 2.0, 3.0], 2, 1)
    np.testing.assert_array_equal(features.overlap_add(frames, 1, length=2), [1.0, 2.0])
    np.testing.assert_array_equal(features.overlap_add(frames, 1, length=5), [1, 2, 3, 0, 0])


def test_multichannel_frame_roundtrip_preserves_each_channel() -> None:
    stereo = np.column_stack((np.arange(6), -np.arange(6)))
    frames = features.frame_signal(stereo, 3, 1)
    assert frames.shape == (4, 3, 2)
    np.testing.assert_array_equal(features.overlap_add(frames, 1), stereo)


def test_empty_framing_retains_rank_and_channel_count() -> None:
    assert features.frame_signal([], 4, 2).shape == (0, 4)
    assert features.frame_signal(np.empty((0, 2)), 4, 2).shape == (0, 4, 2)


def test_overlap_add_rejects_malformed_or_nonfinite_frames() -> None:
    with pytest.raises(ValueError, match="shape"):
        features.overlap_add([1.0, 2.0], 1)
    with pytest.raises(ValueError, match="finite"):
        features.overlap_add([[0.0, np.nan]], 1)


def test_symmetric_hann_has_zero_endpoints() -> None:
    window = features.analysis_window(5, "hann", periodic=False)
    np.testing.assert_allclose(window, [0.0, 0.5, 1.0, 0.5, 0.0], atol=1e-15)


def test_periodic_window_does_not_duplicate_its_first_endpoint() -> None:
    window = features.analysis_window(8, "hann", periodic=True)
    assert window[0] == 0.0
    assert window[-1] > 0.0


def test_analysis_window_validates_kind_and_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        features.analysis_window(0)
    with pytest.raises(ValueError, match="unknown"):
        features.analysis_window(8, "triangle")  # type: ignore[arg-type]


def test_sample_sizes_reject_fractional_and_boolean_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        features.frame_signal([1.0], 1.5, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        features.analysis_window(True)  # type: ignore[arg-type]


def test_frame_spectrum_places_a_bin_centered_tone() -> None:
    tone = np.sin(2 * np.pi * np.arange(16) * 3 / 16)
    spectrum = features.frame_spectrum(tone[None, :], scale="magnitude")
    assert int(np.argmax(spectrum[0])) == 3
    assert spectrum.shape == (1, 9)


def test_frame_spectrum_power_is_squared_magnitude() -> None:
    frames = np.array([[1.0, -0.5, 0.25, 0.0]])
    magnitude = features.frame_spectrum(frames, n_fft=8, scale="magnitude")
    power = features.frame_spectrum(frames, n_fft=8, scale="power")
    np.testing.assert_allclose(power, magnitude**2, rtol=1e-14, atol=1e-14)


# -- extensions --
