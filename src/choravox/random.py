"""Seed handling for reproducible synthetic experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def derive_seed(seed: int, namespace: str) -> int:
    """Derive a stable unsigned 64-bit seed for one named component."""
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not namespace:
        raise ValueError("namespace must be nonempty")
    payload = f"choravox\0{seed}\0{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def rng(seed: int, namespace: str = "root") -> np.random.Generator:
    """Build an independent NumPy generator from a named seed."""
    return np.random.default_rng(derive_seed(seed, namespace))


def deterministic_split(
    size: int, ratios: Iterable[float], seed: int
) -> tuple[NDArray[np.int64], ...]:
    """Partition shuffled indices without dropping examples."""
    values = tuple(ratios)
    if size < 0 or not values or any(value < 0 for value in values):
        raise ValueError("size and ratios must be nonnegative")
    total = sum(values)
    if total <= 0:
        raise ValueError("at least one ratio must be positive")
    permutation = rng(seed, "split").permutation(size).astype(np.int64)
    cumulative = 0
    boundaries: list[int] = []
    for value in values[:-1]:
        cumulative += round(size * value / total)
        boundaries.append(min(size, cumulative))
    return tuple(np.split(permutation, boundaries))


# -- extensions --
