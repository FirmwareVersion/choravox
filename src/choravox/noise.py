"""Deterministic synthetic noise generation and calibrated mixing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import add_noise_at_snr, as_waveform, snr_db
from choravox.random import rng

FloatArray = NDArray[np.float64]


def _calibrate_rms(samples: FloatArray, rms_level: float) -> FloatArray:
    current = float(np.sqrt(np.mean(np.square(samples))))
    if current == 0.0:
        raise ValueError("noise realization must have nonzero RMS")
    return np.asarray(samples * (rms_level / current), dtype=np.float64)


def white_noise(sample_count: int, *, seed: int, rms_level: float = 1.0) -> FloatArray:
    """Generate zero-mean Gaussian noise at a requested RMS level."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(rms_level) or rms_level <= 0.0:
        raise ValueError("rms_level must be finite and positive")
    samples = rng(seed, "white-noise").standard_normal(sample_count)
    samples -= np.mean(samples)
    return np.round(_calibrate_rms(samples, rms_level), 9)


def pink_noise(sample_count: int, *, seed: int, rms_level: float = 1.0) -> FloatArray:
    """Generate periodic finite-length pink noise by shaping white-noise FFT bins."""
    if sample_count < 2:
        raise ValueError("pink noise requires at least two samples")
    if not math.isfinite(rms_level) or rms_level <= 0.0:
        raise ValueError("rms_level must be finite and positive")
    white = rng(seed, "pink-noise").standard_normal(sample_count)
    spectrum = np.fft.rfft(white)
    spectrum[0] = 0.0
    spectrum[1:] /= np.sqrt(np.arange(1, spectrum.size, dtype=np.float64))
    samples = np.fft.irfft(spectrum, n=sample_count)
    return np.round(_calibrate_rms(samples, rms_level), 9)


def mix_at_snr(signal: ArrayLike, disturbance: ArrayLike, target_db: float) -> FloatArray:
    """Add a disturbance and verify its measured SNR after deterministic scaling."""
    if not math.isfinite(target_db):
        raise ValueError("target_db must be finite")
    clean = as_waveform(signal)
    observed = add_noise_at_snr(clean, disturbance, target_db)
    measured = snr_db(clean, observed)
    if not math.isclose(measured, target_db, abs_tol=1e-5):
        raise RuntimeError("noise calibration failed")
    return observed


def snr_sweep(
    signal: ArrayLike, disturbance: ArrayLike, levels_db: tuple[float, ...]
) -> tuple[FloatArray, ...]:
    """Mix the same disturbance at a stable sequence of requested SNR levels."""
    if not levels_db:
        raise ValueError("levels_db must be nonempty")
    return tuple(mix_at_snr(signal, disturbance, level) for level in levels_db)


def babble_noise(
    sources: tuple[ArrayLike, ...], sample_count: int, *, seed: int, rms_level: float = 1.0
) -> FloatArray:
    """Mix wrapped, seeded excerpts of synthetic mono speakers into babble."""
    if len(sources) < 2:
        raise ValueError("babble requires at least two synthetic sources")
    if sample_count <= 0 or not math.isfinite(rms_level) or rms_level <= 0.0:
        raise ValueError("sample_count and rms_level must be positive")
    mixture = np.zeros(sample_count, dtype=np.float64)
    for index, source in enumerate(sources):
        waveform = as_waveform(source)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("babble sources must be nonempty mono waveforms")
        generator = rng(seed, f"babble-source:{index}")
        offset = int(generator.integers(0, waveform.size))
        indices = (np.arange(sample_count) + offset) % waveform.size
        mixture += waveform[indices] - np.mean(waveform)
    if not np.any(mixture):
        raise ValueError("babble sources must contain nonconstant audio")
    return np.round(_calibrate_rms(mixture, rms_level), 9)


@dataclass(frozen=True, slots=True)
class NoiseCondition:
    """Serializable controls for one synthetic noise realization."""

    kind: str
    seed: int
    rms_level: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"white", "pink", "babble"}:
            raise ValueError("noise kind must be white, pink, or babble")
        if self.seed < 0 or not math.isfinite(self.rms_level) or self.rms_level <= 0.0:
            raise ValueError("noise seed and rms_level must be valid")


def render_noise(
    condition: NoiseCondition,
    sample_count: int,
    *,
    babble_sources: tuple[ArrayLike, ...] = (),
) -> FloatArray:
    """Render one declared synthetic condition without accessing external data."""
    if condition.kind == "white":
        return white_noise(sample_count, seed=condition.seed, rms_level=condition.rms_level)
    if condition.kind == "pink":
        return pink_noise(sample_count, seed=condition.seed, rms_level=condition.rms_level)
    return babble_noise(
        babble_sources, sample_count, seed=condition.seed, rms_level=condition.rms_level
    )


# -- extensions --
