"""Stable label handling for speaker-attributed artifacts."""

from __future__ import annotations

import re
from collections.abc import Iterable

_UNSAFE = re.compile(r"[^a-z0-9_-]+")


def normalize_label(value: str) -> str:
    """Convert a display label to a portable lowercase identifier."""
    label = _UNSAFE.sub("-", value.strip().lower()).strip("-")
    if not label:
        raise ValueError("label has no portable characters")
    return label


def unique_labels(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize labels and append deterministic numeric suffixes on collision."""
    result: list[str] = []
    counts: dict[str, int] = {}
    for value in values:
        base = normalize_label(value)
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}-{counts[base]}")
    return tuple(result)


# -- extensions --
