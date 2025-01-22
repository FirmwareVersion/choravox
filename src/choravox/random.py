"""Seed handling for reproducible synthetic experiments."""

from __future__ import annotations

import hashlib

import numpy as np


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


# -- extensions --
