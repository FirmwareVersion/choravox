"""Condition slicing and aggregation for synthetic meeting evaluations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class IntervalBin:
    """Named left-closed numeric interval with an optional open upper tail."""

    label: str
    lower: float
    upper: float | None

    def __post_init__(self) -> None:
        if not self.label or not np.isfinite(self.lower):
            raise ValueError("bin label and lower bound must be valid")
        if self.upper is not None and (not np.isfinite(self.upper) or self.upper <= self.lower):
            raise ValueError("bin upper bound must exceed its lower bound")

    def contains(self, value: float) -> bool:
        """Return whether a finite value lies in this half-open interval."""
        return bool(
            np.isfinite(value)
            and value >= self.lower
            and (self.upper is None or value < self.upper)
        )


def locate_bin(value: float, bins: tuple[IntervalBin, ...]) -> int | None:
    """Locate one value while rejecting overlapping interval definitions."""
    matches = [index for index, interval in enumerate(bins) if interval.contains(value)]
    if len(matches) > 1:
        raise ValueError("condition bins must not overlap")
    return None if not matches else matches[0]


# -- extensions --
