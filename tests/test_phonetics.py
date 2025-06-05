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


# -- extensions --
