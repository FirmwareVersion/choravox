"""Small phoneme-like inventory for deterministic synthetic speech."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

Manner = Literal["vowel", "fricative", "stop", "nasal"]


@dataclass(frozen=True, slots=True)
class Phone:
    """Acoustic target for one synthetic phoneme-like unit."""

    symbol: str
    manner: Manner
    voiced: bool
    formants_hz: tuple[float, ...] = ()
    bandwidths_hz: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol or any(char.isspace() for char in self.symbol):
            raise ValueError("phone symbol must be a nonempty token")
        values = (*self.formants_hz, *self.bandwidths_hz)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("phone frequencies and bandwidths must be finite and positive")
        if len(self.formants_hz) != len(self.bandwidths_hz):
            raise ValueError("every formant requires one bandwidth")
        if self.formants_hz != tuple(sorted(self.formants_hz)):
            raise ValueError("formants must be strictly ordered")
        if self.voiced and not self.formants_hz:
            raise ValueError("voiced phones require formant targets")


_PHONES = (
    Phone("aa", "vowel", True, (730.0, 1090.0, 2440.0), (80.0, 90.0, 120.0)),
    Phone("ae", "vowel", True, (660.0, 1720.0, 2410.0), (80.0, 100.0, 120.0)),
    Phone("eh", "vowel", True, (530.0, 1840.0, 2480.0), (70.0, 95.0, 120.0)),
    Phone("iy", "vowel", True, (270.0, 2290.0, 3010.0), (60.0, 100.0, 130.0)),
    Phone("uw", "vowel", True, (300.0, 870.0, 2240.0), (60.0, 90.0, 120.0)),
    Phone("n", "nasal", True, (250.0, 1500.0, 2500.0), (100.0, 150.0, 180.0)),
    Phone("f", "fricative", False),
    Phone("s", "fricative", False),
    Phone("sh", "fricative", False),
    Phone("t", "stop", False),
)
PHONE_INVENTORY = MappingProxyType({phone.symbol: phone for phone in _PHONES})


def get_phone(symbol: str) -> Phone:
    """Resolve a phone symbol with a useful domain error."""
    try:
        return PHONE_INVENTORY[symbol]
    except KeyError as error:
        raise ValueError(f"unknown phone symbol: {symbol}") from error


# -- extensions --
