"""Condition slicing and aggregation for synthetic meeting evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


ConditionName = Literal["snr_db", "overlap_ratio", "turn_duration_s"]


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """One measured synthetic recording with planted condition metadata."""

    recording_id: str
    value: float
    weight: float
    snr_db: float
    overlap_ratio: float
    turn_duration_s: float

    def __post_init__(self) -> None:
        numbers = (self.value, self.weight, self.snr_db, self.overlap_ratio, self.turn_duration_s)
        if not self.recording_id or not all(np.isfinite(item) for item in numbers):
            raise ValueError("metric observation must contain finite identified values")
        if self.weight <= 0.0 or not 0.0 <= self.overlap_ratio <= 1.0:
            raise ValueError("weight and overlap ratio are outside their domains")
        if self.turn_duration_s <= 0.0:
            raise ValueError("turn duration must be positive")


@dataclass(frozen=True, slots=True)
class SliceSummary:
    label: str
    count: int
    weight: float
    mean: float | None


def slice_observations(
    observations: tuple[MetricObservation, ...],
    condition: ConditionName,
    bins: tuple[IntervalBin, ...],
) -> tuple[SliceSummary, ...]:
    """Compute weighted metric means over one planted meeting condition."""
    buckets: list[list[MetricObservation]] = [[] for _ in bins]
    for observation in observations:
        index = locate_bin(getattr(observation, condition), bins)
        if index is not None:
            buckets[index].append(observation)
    summaries = []
    for interval, bucket in zip(bins, buckets, strict=True):
        weight = sum(item.weight for item in bucket)
        mean = None if not bucket else sum(item.value * item.weight for item in bucket) / weight
        summaries.append(
            SliceSummary(
                interval.label,
                len(bucket),
                round(weight, 6),
                None if mean is None else round(mean, 12),
            )
        )
    return tuple(summaries)


# -- extensions --
