from __future__ import annotations

import numpy as np
import pytest

from choravox import phonetics


def test_phone_inventory_has_stable_symbols_and_classes() -> None:
    assert tuple(phonetics.PHONE_INVENTORY) == (
        "aa",
        "ae",
        "eh",
        "iy",
        "uw",
        "n",
        "f",
        "s",
        "sh",
        "t",
    )
    assert {phone.manner for phone in phonetics.PHONE_INVENTORY.values()} == {
        "vowel",
        "fricative",
        "stop",
        "nasal",
    }


def test_phone_lookup_returns_immutable_inventory_record() -> None:
    phone = phonetics.get_phone("iy")
    assert phone.voiced and phone.formants_hz == (270.0, 2290.0, 3010.0)
    with pytest.raises(ValueError, match="unknown"):
        phonetics.get_phone("zz")


def test_voiced_phones_have_ordered_matched_formants() -> None:
    for phone in phonetics.PHONE_INVENTORY.values():
        if phone.voiced:
            assert len(phone.formants_hz) == len(phone.bandwidths_hz) >= 3
            assert all(
                left < right
                for left, right in zip(phone.formants_hz, phone.formants_hz[1:], strict=False)
            )


def test_phone_rejects_missing_or_mismatched_acoustics() -> None:
    with pytest.raises(ValueError, match="require formant"):
        phonetics.Phone("bad", "vowel", True)
    with pytest.raises(ValueError, match="one bandwidth"):
        phonetics.Phone("bad", "vowel", True, (500.0,), ())


def test_inventory_mapping_cannot_be_mutated() -> None:
    with pytest.raises(TypeError):
        phonetics.PHONE_INVENTORY["x"] = phonetics.get_phone("s")  # type: ignore[index]


def test_speaker_profiles_repeat_by_name_and_seed() -> None:
    first = phonetics.speaker_profile("speaker-a", 17)
    second = phonetics.speaker_profile("speaker-a", 17)
    assert first == second


def test_speaker_profiles_separate_named_streams() -> None:
    first = phonetics.speaker_profile("speaker-a", 17)
    second = phonetics.speaker_profile("speaker-b", 17)
    assert first != second
    assert first.speaker_id == "speaker-a" and second.speaker_id == "speaker-b"


def test_generated_profiles_stay_inside_declared_voice_ranges() -> None:
    profiles = [phonetics.speaker_profile(f"speaker-{index}", 44) for index in range(50)]
    assert all(95.0 <= profile.base_pitch_hz <= 240.0 for profile in profiles)
    assert all(1.5 <= profile.pitch_range_semitones <= 6.0 for profile in profiles)
    assert all(0.88 <= profile.formant_scale <= 1.12 for profile in profiles)


def test_profile_validation_rejects_nonphysical_controls() -> None:
    with pytest.raises(ValueError, match="base_pitch"):
        phonetics.SpeakerProfile("x", 30.0, 2.0, 1.0, 0.1, 0.01)
    with pytest.raises(ValueError, match="speaker_id"):
        phonetics.speaker_profile("", 1)


def test_many_named_profiles_are_not_all_identical() -> None:
    pitches = {phonetics.speaker_profile(f"s{index}", 5).base_pitch_hz for index in range(20)}
    assert len(pitches) == 20


def test_pitch_contour_has_sample_accurate_length() -> None:
    profile = phonetics.speaker_profile("speaker-a", 9)
    contour = phonetics.pitch_contour(0.125, 16_000, profile, seed=3)
    assert contour.shape == (2000,)
    assert np.all(contour > 0.0)


def test_pitch_contour_repeats_exactly() -> None:
    profile = phonetics.speaker_profile("speaker-a", 9)
    first = phonetics.pitch_contour(0.2, 8000, profile, seed=11)
    second = phonetics.pitch_contour(0.2, 8000, profile, seed=11)
    np.testing.assert_array_equal(first, second)


def test_flat_profile_produces_constant_pitch() -> None:
    profile = phonetics.SpeakerProfile("flat", 120.0, 0.0, 1.0, 0.0, 0.0)
    contour = phonetics.pitch_contour(0.1, 8000, profile, seed=1)
    np.testing.assert_array_equal(contour, np.full(800, 120.0))


def test_pitch_range_changes_contour_span() -> None:
    narrow = phonetics.SpeakerProfile("voice", 140.0, 1.0, 1.0, 0.0, 0.0)
    wide = phonetics.SpeakerProfile("voice", 140.0, 6.0, 1.0, 0.0, 0.0)
    narrow_track = phonetics.pitch_contour(1.0, 1000, narrow, seed=2)
    wide_track = phonetics.pitch_contour(1.0, 1000, wide, seed=2)
    assert np.ptp(wide_track) > np.ptp(narrow_track) * 4.0


def test_pitch_contour_defines_empty_duration() -> None:
    profile = phonetics.speaker_profile("speaker-a", 0)
    assert phonetics.pitch_contour(0.0, 16_000, profile, seed=0).shape == (0,)
    with pytest.raises(ValueError, match="valid"):
        phonetics.pitch_contour(-1.0, 16_000, profile, seed=0)


def test_glottal_source_tracks_declared_fundamental() -> None:
    contour = np.full(800, 100.0)
    source = phonetics.glottal_source(contour, 8000)
    spectrum = np.abs(np.fft.rfft(source))
    frequencies = np.fft.rfftfreq(source.size, 1 / 8000)
    assert frequencies[int(np.argmax(spectrum[1:]) + 1)] == 100.0


def test_glottal_source_is_peak_normalized() -> None:
    contour = np.linspace(100.0, 140.0, 1000)
    source = phonetics.glottal_source(contour, 16_000)
    assert np.max(np.abs(source)) == 1.0


def test_glottal_source_respects_nyquist_harmonic_limit() -> None:
    contour = np.full(400, 1800.0)
    many = phonetics.glottal_source(contour, 8000, harmonics=20)
    two = phonetics.glottal_source(contour, 8000, harmonics=2)
    np.testing.assert_array_equal(many, two)


def test_glottal_source_rejects_nonpositive_pitch() -> None:
    with pytest.raises(ValueError, match="positive vector"):
        phonetics.glottal_source(np.array([100.0, 0.0]), 8000)


def test_empty_pitch_track_produces_empty_excitation() -> None:
    assert phonetics.glottal_source(np.empty(0), 8000).shape == (0,)


def test_unvoiced_source_has_calibrated_rms() -> None:
    source = phonetics.unvoiced_source(4000, seed=7)
    assert np.sqrt(np.mean(source**2)) == pytest.approx(0.1, abs=1e-9)


def test_unvoiced_source_repeats_by_namespace() -> None:
    first = phonetics.unvoiced_source(100, seed=7, namespace="s")
    second = phonetics.unvoiced_source(100, seed=7, namespace="s")
    np.testing.assert_array_equal(first, second)


def test_highpass_control_increases_upper_band_energy() -> None:
    flat = phonetics.unvoiced_source(8192, seed=12, highpass=0.0)
    shaped = phonetics.unvoiced_source(8192, seed=12, highpass=0.95)
    flat_spectrum = np.abs(np.fft.rfft(flat)) ** 2
    shaped_spectrum = np.abs(np.fft.rfft(shaped)) ** 2
    split = flat_spectrum.size // 2
    flat_ratio = np.sum(flat_spectrum[split:]) / np.sum(flat_spectrum[:split])
    shaped_ratio = np.sum(shaped_spectrum[split:]) / np.sum(shaped_spectrum[:split])
    assert shaped_ratio > flat_ratio * 3.0


def test_unvoiced_namespaces_are_independent() -> None:
    first = phonetics.unvoiced_source(64, seed=3, namespace="f")
    second = phonetics.unvoiced_source(64, seed=3, namespace="s")
    assert not np.array_equal(first, second)


def test_unvoiced_source_validates_length_and_shaping() -> None:
    with pytest.raises(ValueError, match="supported"):
        phonetics.unvoiced_source(-1, seed=0)
    with pytest.raises(ValueError, match="supported"):
        phonetics.unvoiced_source(10, seed=0, highpass=1.1)


# -- extensions --
