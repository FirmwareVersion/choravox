from __future__ import annotations

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


# -- extensions --
