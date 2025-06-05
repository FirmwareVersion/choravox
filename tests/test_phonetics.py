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


# -- extensions --
