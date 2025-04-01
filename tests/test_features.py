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


@pytest.mark.parametrize("n_fft", [15, 16])
def test_one_sided_power_obeys_parseval(n_fft: int) -> None:
    generator = np.random.default_rng(101)
    frame = generator.normal(size=n_fft)
    power = features.frame_spectrum(frame[None, :], scale="power")[0]
    edge = power[0] + (power[-1] if n_fft % 2 == 0 else 0.0)
    stop = -1 if n_fft % 2 == 0 else None
    spectral_energy = (edge + 2.0 * np.sum(power[1:stop])) / n_fft
    assert spectral_energy == pytest.approx(float(np.sum(frame**2)), rel=1e-12)


def test_spectrum_applies_window_per_channel() -> None:
    frames = np.ones((1, 4, 2))
    spectrum = features.frame_spectrum(frames, window=[1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(spectrum, np.ones((1, 3, 2)))


def test_spectrum_rejects_short_fft_and_bad_window() -> None:
    with pytest.raises(ValueError, match="at least"):
        features.frame_spectrum(np.ones((1, 8)), n_fft=4)
    with pytest.raises(ValueError, match="one finite"):
        features.frame_spectrum(np.ones((1, 8)), window=np.ones(7))


@pytest.mark.parametrize("scale", ["slaney", "htk"])
def test_mel_frequency_transforms_roundtrip(scale: features.MelScale) -> None:
    frequencies = np.array([0.0, 100.0, 999.0, 1000.0, 8000.0])
    restored = features.mel_to_hz(features.hz_to_mel(frequencies, scale), scale)
    np.testing.assert_allclose(restored, frequencies, rtol=1e-12, atol=1e-12)


def test_slaney_scale_is_continuous_at_one_kilohertz() -> None:
    assert features.hz_to_mel(1000.0).item() == pytest.approx(15.0)
    assert features.mel_to_hz(15.0).item() == pytest.approx(1000.0)


def test_mel_transforms_reject_negative_domain() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        features.hz_to_mel([-1.0])
    with pytest.raises(ValueError, match="nonnegative"):
        features.mel_to_hz([-1.0])


def test_mel_filterbank_has_ordered_normalized_triangles() -> None:
    bank = features.mel_filterbank(16_000, 512, 24)
    assert bank.shape == (24, 257)
    np.testing.assert_allclose(np.sum(bank, axis=1), np.ones(24), atol=1e-15)
    assert np.all(bank >= 0.0)
    assert np.all(np.diff(np.argmax(bank, axis=1)) > 0)


def test_mel_filterbank_honors_frequency_bounds() -> None:
    bank = features.mel_filterbank(16_000, 512, 8, min_hz=1000.0, max_hz=4000.0)
    frequencies = np.linspace(0.0, 8000.0, 257)
    assert np.all(bank[:, frequencies < 1000.0] == 0.0)
    assert np.all(bank[:, frequencies > 4000.0] == 0.0)


def test_filterbank_rejects_bounds_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        features.mel_filterbank(16_000, 512, 20, max_hz=9000.0)


def test_filterbank_rejects_unresolvable_band_count() -> None:
    with pytest.raises(ValueError, match="too small"):
        features.mel_filterbank(16_000, 8, 40)


def test_filterbank_is_byte_deterministic() -> None:
    first = features.mel_filterbank(16_000, 400, 20, min_hz=80.0, max_hz=7600.0)
    second = features.mel_filterbank(16_000, 400, 20, min_hz=80.0, max_hz=7600.0)
    assert first.tobytes() == second.tobytes()


def test_filterbank_without_normalization_retains_triangle_peaks() -> None:
    bank = features.mel_filterbank(8_000, 1024, 4, scale="htk", normalize=False)
    assert np.all(np.max(bank, axis=1) > 0.98)
    assert not np.allclose(np.sum(bank, axis=1), 1.0)


def test_filterbank_projection_has_hand_computed_values() -> None:
    spectra = np.array([[1.0, 2.0, 4.0], [2.0, 0.0, 1.0]])
    bank = np.array([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
    np.testing.assert_allclose(features.apply_filterbank(spectra, bank), [[1.0, 3.5], [2.0, 0.75]])


def test_filterbank_projection_preserves_channels() -> None:
    spectra = np.ones((3, 5, 2))
    projected = features.apply_filterbank(spectra, np.eye(5)[::2])
    assert projected.shape == (3, 3, 2)


def test_log_compression_floors_silence() -> None:
    compressed = features.log_compress([0.0, 1e-12, 1.0], floor=1e-6)
    np.testing.assert_allclose(compressed, [np.log(1e-6), np.log(1e-6), 0.0])


def test_log_compression_supports_base_ten() -> None:
    np.testing.assert_allclose(features.log_compress([0.01, 1.0], base="10"), [-2.0, 0.0])


def test_filterbank_projection_rejects_bin_mismatch_and_negative_power() -> None:
    with pytest.raises(ValueError, match="bins"):
        features.apply_filterbank(np.ones((2, 4)), np.ones((3, 5)))
    with pytest.raises(ValueError, match="nonnegative"):
        features.apply_filterbank(np.array([[1.0, -1.0]]), np.ones((1, 2)))


def test_log_compression_rejects_invalid_floor_and_base() -> None:
    with pytest.raises(ValueError, match="positive"):
        features.log_compress([1.0], floor=0.0)
    with pytest.raises(ValueError, match="base"):
        features.log_compress([1.0], base="two")  # type: ignore[arg-type]


def test_silent_log_mel_vector_is_finite() -> None:
    bank = features.mel_filterbank(16_000, 512, 20)
    energies = features.apply_filterbank(np.zeros((4, 257)), bank)
    assert np.all(np.isfinite(features.log_compress(energies)))


def test_cmvn_centers_and_scales_each_feature() -> None:
    matrix = np.array([[1.0, 10.0], [2.0, 12.0], [3.0, 14.0]])
    normalized = features.cmvn(matrix)
    np.testing.assert_allclose(np.mean(normalized, axis=0), [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(np.std(normalized, axis=0), [1.0, 1.0], atol=1e-15)


def test_cmvn_can_only_remove_the_mean() -> None:
    centered = features.cmvn([[1.0], [3.0]], normalize_variance=False)
    np.testing.assert_allclose(centered[:, 0], [-1.0, 1.0])


def test_cmvn_maps_constant_dimensions_to_exact_zero() -> None:
    normalized = features.cmvn([[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]])
    np.testing.assert_array_equal(normalized[:, 1], np.zeros(3))


def test_cmvn_constant_multichannel_plane_is_stable() -> None:
    matrix = np.ones((3, 2, 2))
    np.testing.assert_array_equal(features.cmvn(matrix), np.zeros_like(matrix))


def test_cmvn_estimates_statistics_from_masked_frames() -> None:
    matrix = np.array([[0.0], [2.0], [100.0]])
    normalized = features.cmvn(matrix, mask=[True, True, False])
    np.testing.assert_allclose(normalized[:, 0], [-1.0, 1.0, 99.0])


def test_cmvn_rejects_empty_or_mismatched_masks() -> None:
    with pytest.raises(ValueError, match="at least one"):
        features.cmvn(np.ones((2, 3)), mask=[False, False])
    with pytest.raises(ValueError, match="at least one"):
        features.cmvn(np.ones((2, 3)), mask=[True])


def test_cmvn_preserves_frame_and_feature_geometry() -> None:
    generator = np.random.default_rng(7)
    matrix = generator.normal(size=(9, 4, 2))
    assert features.cmvn(matrix).shape == matrix.shape


def test_delta_features_recover_linear_interior_slope() -> None:
    matrix = np.arange(7, dtype=np.float64)[:, None] * np.array([[1.0, -2.0]])
    deltas = features.delta_features(matrix, width=2)
    np.testing.assert_allclose(deltas[2:-2], np.tile([1.0, -2.0], (3, 1)))


def test_delta_features_are_zero_for_constant_sequences() -> None:
    np.testing.assert_array_equal(features.delta_features(np.ones((5, 3))), np.zeros((5, 3)))


# -- extensions --
