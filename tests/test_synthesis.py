from __future__ import annotations

import numpy as np
import pytest

from choravox import synthesis


def test_resonator_coefficients_encode_stable_poles() -> None:
    coefficients = synthesis.resonator_coefficients(1000.0, 100.0, 16_000)
    poles = np.roots([1.0, -coefficients.a1, -coefficients.a2])
    assert np.all(np.abs(poles) < 1.0)
    np.testing.assert_allclose(np.abs(poles[0]), np.abs(poles[1]), atol=1e-15)


def test_narrower_resonator_places_poles_nearer_unit_circle() -> None:
    narrow = synthesis.resonator_coefficients(1000.0, 50.0, 16_000)
    wide = synthesis.resonator_coefficients(1000.0, 300.0, 16_000)
    assert abs(narrow.a2) > abs(wide.a2)


def test_resonator_rejects_frequency_at_or_above_nyquist() -> None:
    with pytest.raises(ValueError, match="frequency"):
        synthesis.resonator_coefficients(8000.0, 100.0, 16_000)
    with pytest.raises(ValueError, match="frequency"):
        synthesis.resonator_coefficients(0.0, 100.0, 16_000)


def test_resonator_rejects_nonpositive_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        synthesis.resonator_coefficients(1000.0, 0.0, 16_000)


def test_resonator_coefficients_repeat_bit_exactly() -> None:
    first = synthesis.resonator_coefficients(730.0, 80.0, 16_000)
    second = synthesis.resonator_coefficients(730.0, 80.0, 16_000)
    assert first == second


def test_resonator_impulse_starts_at_feedforward_gain() -> None:
    coefficients = synthesis.resonator_coefficients(1000.0, 200.0, 8000)
    impulse = np.zeros(20)
    impulse[0] = 1.0
    response = synthesis.apply_resonator(impulse, coefficients)
    assert response[0] == coefficients.b0
    assert response[1] == pytest.approx(coefficients.a1 * coefficients.b0)


def test_resonator_preserves_samples_first_channels() -> None:
    stereo = np.column_stack((np.ones(10), -np.ones(10)))
    output = synthesis.apply_resonator(stereo, synthesis.resonator_coefficients(500.0, 100.0, 8000))
    assert output.shape == stereo.shape
    np.testing.assert_allclose(output[:, 0], -output[:, 1])


def test_resonator_amplifies_target_relative_to_distant_tone() -> None:
    time = np.arange(8000) / 8000
    target = np.sin(2 * np.pi * 1000 * time)
    distant = np.sin(2 * np.pi * 3000 * time)
    coefficients = synthesis.resonator_coefficients(1000.0, 100.0, 8000)
    target_output = synthesis.apply_resonator(target, coefficients)[1000:]
    distant_output = synthesis.apply_resonator(distant, coefficients)[1000:]
    assert np.sqrt(np.mean(target_output**2)) > np.sqrt(np.mean(distant_output**2)) * 10.0


def test_zero_input_has_zero_resonator_state() -> None:
    coefficients = synthesis.resonator_coefficients(1000.0, 100.0, 8000)
    np.testing.assert_array_equal(
        synthesis.apply_resonator(np.zeros(50), coefficients), np.zeros(50)
    )


def test_empty_resonator_input_keeps_channel_rank() -> None:
    coefficients = synthesis.resonator_coefficients(1000.0, 100.0, 8000)
    assert synthesis.apply_resonator(np.empty((0, 2)), coefficients).shape == (0, 2)


def test_formant_filter_keeps_excitation_length() -> None:
    from choravox.phonetics import get_phone, speaker_profile

    source = np.random.default_rng(1).normal(size=1000)
    shaped = synthesis.apply_formants(source, get_phone("aa"), speaker_profile("a", 0), 16_000)
    assert shaped.shape == source.shape


def test_fricative_shaping_emphasizes_sample_differences() -> None:
    from choravox.phonetics import get_phone, speaker_profile

    source = np.ones(20)
    shaped = synthesis.apply_formants(source, get_phone("s"), speaker_profile("a", 0), 16_000)
    assert shaped[0] == 1.0
    np.testing.assert_allclose(shaped[1:], 0.08)


def test_vowel_filters_produce_distinct_spectral_envelopes() -> None:
    from choravox.phonetics import get_phone, speaker_profile

    source = np.random.default_rng(77).normal(size=16_000)
    profile = speaker_profile("speaker", 4)
    aa = synthesis.apply_formants(source, get_phone("aa"), profile, 16_000)
    iy = synthesis.apply_formants(source, get_phone("iy"), profile, 16_000)
    aa_spectrum = np.abs(np.fft.rfft(aa[2000:]))
    iy_spectrum = np.abs(np.fft.rfft(iy[2000:]))
    assert np.corrcoef(aa_spectrum, iy_spectrum)[0, 1] < 0.95


def test_formant_scale_moves_first_resonance_upward() -> None:
    from choravox.phonetics import SpeakerProfile, get_phone

    source = np.zeros(8000)
    source[0] = 1.0
    low = SpeakerProfile("low", 120, 2, 0.85, 0.1, 0.01)
    high = SpeakerProfile("high", 120, 2, 1.15, 0.1, 0.01)
    low_peak = np.argmax(
        np.abs(np.fft.rfft(synthesis.apply_formants(source, get_phone("aa"), low, 8000)))
    )
    high_peak = np.argmax(
        np.abs(np.fft.rfft(synthesis.apply_formants(source, get_phone("aa"), high, 8000)))
    )
    assert high_peak > low_peak


def test_phone_envelope_has_exact_endpoints_and_plateau() -> None:
    envelope = synthesis.phone_envelope(10, 3, 3)
    assert envelope[0] == 0.0 and envelope[-1] < 1e-30
    np.testing.assert_array_equal(envelope[3:7], np.ones(4))


def test_phone_envelope_is_symmetric_for_equal_fades() -> None:
    envelope = synthesis.phone_envelope(20, 5, 5)
    np.testing.assert_allclose(envelope, envelope[::-1], atol=1e-15)


def test_phone_envelope_allows_no_fades() -> None:
    np.testing.assert_array_equal(synthesis.phone_envelope(4, 0, 0), np.ones(4))


def test_phone_envelope_can_partition_entire_phone() -> None:
    envelope = synthesis.phone_envelope(6, 3, 3)
    assert envelope.shape == (6,)
    assert np.all((envelope >= 0.0) & (envelope <= 1.0))


def test_phone_envelope_rejects_overlapping_fades() -> None:
    with pytest.raises(ValueError, match="fit"):
        synthesis.phone_envelope(5, 3, 3)
    with pytest.raises(ValueError, match="fit"):
        synthesis.phone_envelope(0, 0, 0)


# -- extensions --
