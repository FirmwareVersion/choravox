"""Deterministic framing and spectral feature primitives."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform

FloatArray = NDArray[np.float64]


def _positive_size(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def frame_signal(
    samples: ArrayLike,
    frame_length: int,
    hop_length: int,
    *,
    end: Literal["drop", "pad"] = "drop",
    pad_value: float = 0.0,
) -> FloatArray:
    """Slice a samples-first waveform into frames without copying channels together."""
    frame_length = _positive_size(frame_length, "frame_length")
    hop_length = _positive_size(hop_length, "hop_length")
    if end not in {"drop", "pad"}:
        raise ValueError("end must be 'drop' or 'pad'")
    if not np.isfinite(pad_value):
        raise ValueError("pad_value must be finite")
    waveform = as_waveform(samples)
    sample_count = waveform.shape[0]
    if sample_count == 0 or (end == "drop" and sample_count < frame_length):
        return np.empty((0, frame_length, *waveform.shape[1:]), dtype=np.float64)
    if end == "drop":
        frame_count = 1 + (sample_count - frame_length) // hop_length
    else:
        frame_count = 1 + max(0, math.ceil((sample_count - frame_length) / hop_length))
    output = np.full((frame_count, frame_length, *waveform.shape[1:]), pad_value, dtype=np.float64)
    for index in range(frame_count):
        start = index * hop_length
        available = min(frame_length, sample_count - start)
        output[index, :available] = waveform[start : start + available]
    return output


def overlap_add(frames: ArrayLike, hop_length: int, *, length: int | None = None) -> FloatArray:
    """Reconstruct samples by averaging every overlapping frame contribution."""
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[1] == 0:
        raise ValueError("frames must have shape (frames, samples[, channels])")
    if not np.all(np.isfinite(values)):
        raise ValueError("frames must be finite")
    hop_length = _positive_size(hop_length, "hop_length")
    if length is not None and length < 0:
        raise ValueError("length must be nonnegative")
    natural = 0 if values.shape[0] == 0 else (values.shape[0] - 1) * hop_length + values.shape[1]
    requested = natural if length is None else length
    output = np.zeros((max(natural, requested), *values.shape[2:]), dtype=np.float64)
    weight = np.zeros(max(natural, requested), dtype=np.float64)
    for index, frame in enumerate(values):
        start = index * hop_length
        output[start : start + values.shape[1]] += frame
        weight[start : start + values.shape[1]] += 1.0
    if output.ndim == 2:
        output = np.divide(output, weight[:, None], out=output, where=weight[:, None] > 0)
    else:
        output = np.divide(output, weight, out=output, where=weight > 0)
    return np.asarray(output[:requested], dtype=np.float64)


WindowKind = Literal["rectangular", "hann", "hamming", "blackman"]


def analysis_window(length: int, kind: WindowKind = "hann", *, periodic: bool = True) -> FloatArray:
    """Create a standard analysis window with explicit periodicity."""
    length = _positive_size(length, "window length")
    if kind == "rectangular" or length == 1:
        return np.ones(length, dtype=np.float64)
    size = length + 1 if periodic else length
    if kind == "hann":
        values = np.hanning(size)
    elif kind == "hamming":
        values = np.hamming(size)
    elif kind == "blackman":
        values = np.blackman(size)
    else:
        raise ValueError(f"unknown window kind: {kind}")
    return np.asarray(values[:length], dtype=np.float64)


SpectrumScale = Literal["magnitude", "power"]


def frame_spectrum(
    frames: ArrayLike,
    *,
    n_fft: int | None = None,
    window: ArrayLike | None = None,
    scale: SpectrumScale = "power",
) -> FloatArray:
    """Compute a one-sided real FFT spectrum along each frame's sample axis."""
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[1] == 0:
        raise ValueError("frames must have shape (frames, samples[, channels])")
    if not np.all(np.isfinite(values)):
        raise ValueError("frames must be finite")
    fft_size = values.shape[1] if n_fft is None else _positive_size(n_fft, "n_fft")
    if fft_size < values.shape[1]:
        raise ValueError("n_fft must be at least the frame length")
    if window is not None:
        weights = np.asarray(window, dtype=np.float64)
        if weights.shape != (values.shape[1],) or not np.all(np.isfinite(weights)):
            raise ValueError("window must contain one finite weight per frame sample")
        weights = weights[:, None] if values.ndim == 3 else weights
        values = values * weights
    transformed = np.fft.rfft(values, n=fft_size, axis=1)
    magnitude = np.abs(transformed)
    if scale == "magnitude":
        return np.asarray(magnitude, dtype=np.float64)
    if scale == "power":
        return np.asarray(np.square(magnitude), dtype=np.float64)
    raise ValueError("scale must be 'magnitude' or 'power'")


MelScale = Literal["slaney", "htk"]


def hz_to_mel(frequencies: ArrayLike, scale: MelScale = "slaney") -> FloatArray:
    """Map nonnegative frequencies to the selected mel convention."""
    values = np.asarray(frequencies, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("frequencies must be finite and nonnegative")
    if scale == "htk":
        return np.asarray(2595.0 * np.log10(1.0 + values / 700.0), dtype=np.float64)
    if scale != "slaney":
        raise ValueError("scale must be 'slaney' or 'htk'")
    linear = values / (200.0 / 3.0)
    logarithmic = 15.0 + np.log(np.maximum(values, 1000.0) / 1000.0) / (np.log(6.4) / 27.0)
    return np.asarray(np.where(values >= 1000.0, logarithmic, linear), dtype=np.float64)


def mel_to_hz(mels: ArrayLike, scale: MelScale = "slaney") -> FloatArray:
    """Invert the selected mel convention for nonnegative mel values."""
    values = np.asarray(mels, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("mels must be finite and nonnegative")
    if scale == "htk":
        return np.asarray(700.0 * (10.0 ** (values / 2595.0) - 1.0), dtype=np.float64)
    if scale != "slaney":
        raise ValueError("scale must be 'slaney' or 'htk'")
    linear = values * (200.0 / 3.0)
    logarithmic = 1000.0 * np.exp((values - 15.0) * (np.log(6.4) / 27.0))
    return np.asarray(np.where(values >= 15.0, logarithmic, linear), dtype=np.float64)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    *,
    min_hz: float = 0.0,
    max_hz: float | None = None,
    scale: MelScale = "slaney",
    normalize: bool = True,
) -> FloatArray:
    """Construct triangular mel filters over one-sided FFT bins."""
    sample_rate = _positive_size(sample_rate, "sample_rate")
    n_fft = _positive_size(n_fft, "n_fft")
    n_mels = _positive_size(n_mels, "n_mels")
    upper = sample_rate / 2.0 if max_hz is None else float(max_hz)
    if not np.isfinite(min_hz) or not np.isfinite(upper) or min_hz < 0 or upper <= min_hz:
        raise ValueError("mel frequency bounds must be finite and increasing")
    if upper > sample_rate / 2.0:
        raise ValueError("max_hz must not exceed the Nyquist frequency")
    mel_edges = np.linspace(hz_to_mel(min_hz, scale), hz_to_mel(upper, scale), n_mels + 2)
    edges = mel_to_hz(mel_edges, scale)
    bins = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    bank = np.zeros((n_mels, bins.size), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = edges[index : index + 3]
        bank[index] = np.maximum(
            0.0, np.minimum((bins - left) / (center - left), (right - bins) / (right - center))
        )
    if normalize:
        totals = np.sum(bank, axis=1)
        if np.any(totals == 0.0):
            raise ValueError("n_fft is too small to resolve every requested mel band")
        bank /= totals[:, None]
    return bank


def apply_filterbank(spectra: ArrayLike, filters: ArrayLike) -> FloatArray:
    """Project frame spectra onto filters while preserving optional channels."""
    values = np.asarray(spectra, dtype=np.float64)
    bank = np.asarray(filters, dtype=np.float64)
    if values.ndim not in (2, 3) or bank.ndim != 2 or values.shape[1] != bank.shape[1]:
        raise ValueError("spectra bins must match a two-dimensional filterbank")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(bank)):
        raise ValueError("spectra and filters must be finite")
    if np.any(values < 0.0) or np.any(bank < 0.0):
        raise ValueError("spectra and filters must be nonnegative")
    if values.ndim == 2:
        return np.asarray(np.einsum("fb,mb->fm", values, bank), dtype=np.float64)
    return np.asarray(np.einsum("fbc,mb->fmc", values, bank), dtype=np.float64)


LogBase = Literal["e", "10"]


def log_compress(energies: ArrayLike, *, floor: float = 1e-10, base: LogBase = "e") -> FloatArray:
    """Floor nonnegative energies before deterministic log compression."""
    values = np.asarray(energies, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("energies must be finite and nonnegative")
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and positive")
    bounded = np.maximum(values, floor)
    if base == "e":
        result = np.log(bounded)
    elif base == "10":
        result = np.log10(bounded)
    else:
        raise ValueError("base must be 'e' or '10'")
    return np.asarray(result, dtype=np.float64)


def cmvn(
    feature_matrix: ArrayLike,
    *,
    normalize_variance: bool = True,
    mask: ArrayLike | None = None,
    epsilon: float = 1e-8,
) -> FloatArray:
    """Apply utterance mean and optional variance normalization over selected frames."""
    values = np.asarray(feature_matrix, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("features must be a nonempty finite frame matrix")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    selected = np.ones(values.shape[0], dtype=np.bool_)
    if mask is not None:
        selected = np.asarray(mask, dtype=np.bool_)
        if selected.shape != (values.shape[0],) or not np.any(selected):
            raise ValueError("mask must select at least one feature frame")
    reference = values[selected]
    centered = values - np.mean(reference, axis=0)
    if not normalize_variance:
        return np.asarray(centered, dtype=np.float64)
    deviation = np.std(reference, axis=0)
    constant = deviation < epsilon
    normalized = centered / np.where(constant, 1.0, deviation)
    normalized[..., constant] = 0.0
    return np.asarray(normalized, dtype=np.float64)


def delta_features(feature_matrix: ArrayLike, *, width: int = 2) -> FloatArray:
    """Estimate regression deltas along time using replicated boundary frames."""
    values = np.asarray(feature_matrix, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("features must be a nonempty finite frame matrix")
    width = _positive_size(width, "delta width")
    padded = np.pad(values, ((width, width), *((0, 0),) * (values.ndim - 1)), mode="edge")
    output = np.zeros_like(values)
    for offset in range(1, width + 1):
        output += offset * (
            padded[width + offset : width + offset + values.shape[0]]
            - padded[width - offset : width - offset + values.shape[0]]
        )
    output /= 2.0 * sum(offset * offset for offset in range(1, width + 1))
    return output


# -- extensions --
