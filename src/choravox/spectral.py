"""Short-time spectral transforms and compact descriptors."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform
from choravox.features import WindowKind, analysis_window, frame_signal

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def stft(
    samples: ArrayLike,
    frame_length: int,
    hop_length: int,
    *,
    n_fft: int | None = None,
    window: WindowKind = "hann",
    center: bool = True,
) -> ComplexArray:
    """Compute a one-sided STFT with optional zero-centered boundary frames."""
    waveform = as_waveform(samples)
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    fft_size = frame_length if n_fft is None else n_fft
    if fft_size < frame_length:
        raise ValueError("n_fft must be at least frame_length")
    if center:
        width = frame_length // 2
        padding = ((width, width), *((0, 0),) * (waveform.ndim - 1))
        waveform = np.pad(waveform, padding)
    frames = frame_signal(waveform, frame_length, hop_length, end="pad")
    weights = analysis_window(frame_length, window)
    frames = frames * weights[None, :, None] if frames.ndim == 3 else frames * weights[None, :]
    return np.asarray(np.fft.rfft(frames, n=fft_size, axis=1), dtype=np.complex128)


def istft(
    spectra: ArrayLike,
    frame_length: int,
    hop_length: int,
    *,
    length: int | None = None,
    window: WindowKind = "hann",
    center: bool = True,
) -> FloatArray:
    """Invert an STFT by overlap-adding frames with squared-window normalization."""
    values = np.asarray(spectra, dtype=np.complex128)
    if values.ndim not in (2, 3) or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("spectra must have shape (frames, bins[, channels])")
    if not np.all(np.isfinite(values)):
        raise ValueError("spectra must be finite")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")
    inferred_fft = 2 * (values.shape[1] - 1)
    if frame_length > inferred_fft:
        raise ValueError("frame_length exceeds the inferred FFT size")
    frames = np.fft.irfft(values, n=inferred_fft, axis=1)[:, :frame_length]
    weights = analysis_window(frame_length, window)
    natural = (values.shape[0] - 1) * hop_length + frame_length
    output = np.zeros((natural, *values.shape[2:]), dtype=np.float64)
    normalization = np.zeros(natural, dtype=np.float64)
    for index, frame in enumerate(frames):
        start = index * hop_length
        if output.ndim == 2:
            output[start : start + frame_length] += frame * weights[:, None]
        else:
            output[start : start + frame_length] += frame * weights
        normalization[start : start + frame_length] += weights**2
    if output.ndim == 2:
        output = np.divide(
            output,
            normalization[:, None],
            out=np.zeros_like(output),
            where=normalization[:, None] > 1e-12,
        )
    else:
        output = np.divide(
            output, normalization, out=np.zeros_like(output), where=normalization > 1e-12
        )
    if center:
        width = frame_length // 2
        output = output[width:] if width else output
    if length is not None:
        if length < 0 or length > output.shape[0]:
            raise ValueError("length must fit the reconstructed signal")
        output = output[:length]
    return np.asarray(output, dtype=np.float64)


def periodogram(
    samples: ArrayLike,
    sample_rate: int,
    *,
    n_fft: int | None = None,
    window: WindowKind = "hann",
) -> tuple[FloatArray, FloatArray]:
    """Estimate a one-sided power spectral density for one mono observation."""
    waveform = as_waveform(samples)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("periodogram input must be nonempty mono audio")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    fft_size = waveform.size if n_fft is None else n_fft
    if fft_size < waveform.size:
        raise ValueError("n_fft must not truncate the observation")
    weights = analysis_window(waveform.size, window, periodic=False)
    transformed = np.fft.rfft(waveform * weights, n=fft_size)
    density = np.square(np.abs(transformed)) / (sample_rate * np.sum(weights**2))
    if fft_size % 2 == 0:
        density[1:-1] *= 2.0
    else:
        density[1:] *= 2.0
    frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    return np.asarray(frequencies, dtype=np.float64), np.asarray(density, dtype=np.float64)


def spectral_centroid(spectra: ArrayLike, frequencies: ArrayLike) -> FloatArray:
    """Return the energy-weighted center frequency of every frame."""
    values, axis = _validate_descriptor_inputs(spectra, frequencies)
    totals = np.sum(values, axis=1)
    weighted = np.sum(values * axis[None, :], axis=1)
    centroid: FloatArray = np.asarray(
        np.divide(weighted, totals, out=np.zeros_like(weighted), where=totals > 0.0),
        dtype=np.float64,
    )
    return centroid


def spectral_bandwidth(spectra: ArrayLike, frequencies: ArrayLike) -> FloatArray:
    """Return the weighted standard deviation around each spectral centroid."""
    values, axis = _validate_descriptor_inputs(spectra, frequencies)
    centers = spectral_centroid(values, axis)
    totals = np.sum(values, axis=1)
    variance = np.sum(values * np.square(axis[None, :] - centers[:, None]), axis=1)
    variance = np.divide(variance, totals, out=np.zeros_like(variance), where=totals > 0.0)
    bandwidth: FloatArray = np.asarray(np.sqrt(variance), dtype=np.float64)
    return bandwidth


def _validate_descriptor_inputs(
    spectra: ArrayLike, frequencies: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(spectra, dtype=np.float64)
    axis = np.asarray(frequencies, dtype=np.float64)
    if values.ndim != 2 or axis.shape != (values.shape[1],):
        raise ValueError("spectra and frequency axis have incompatible shapes")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(axis)):
        raise ValueError("spectral descriptor inputs must be finite")
    if np.any(values < 0.0) or np.any(axis < 0.0) or np.any(np.diff(axis) < 0.0):
        raise ValueError("spectra and frequencies must be nonnegative and ordered")
    return values, axis


def spectral_flux(spectra: ArrayLike, *, normalize: bool = True) -> FloatArray:
    """Measure positive frame-to-frame spectral changes with a zero first frame."""
    values = np.asarray(spectra, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("spectra must be a finite nonnegative frame matrix")
    working = values.copy()
    if normalize:
        totals = np.sum(working, axis=1, keepdims=True)
        working = np.divide(working, totals, out=np.zeros_like(working), where=totals > 0.0)
    differences = np.maximum(working[1:] - working[:-1], 0.0)
    output = np.zeros(values.shape[0], dtype=np.float64)
    output[1:] = np.sqrt(np.sum(np.square(differences), axis=1))
    return output


def dct_type2(feature_matrix: ArrayLike, n_coefficients: int | None = None) -> FloatArray:
    """Apply an orthonormal DCT-II across the last feature dimension."""
    values = np.asarray(feature_matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("features must be a finite nonempty matrix")
    count = values.shape[1] if n_coefficients is None else n_coefficients
    if not 1 <= count <= values.shape[1]:
        raise ValueError("n_coefficients must fit the feature dimension")
    positions = np.arange(values.shape[1], dtype=np.float64) + 0.5
    orders = np.arange(count, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * orders * positions / values.shape[1])
    basis[0] *= np.sqrt(1.0 / values.shape[1])
    basis[1:] *= np.sqrt(2.0 / values.shape[1])
    return np.asarray(values @ basis.T, dtype=np.float64)


def inverse_dct_type2(coefficients: ArrayLike) -> FloatArray:
    """Invert a full orthonormal DCT-II coefficient matrix."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be a finite nonempty matrix")
    size = values.shape[1]
    positions = np.arange(size, dtype=np.float64) + 0.5
    orders = np.arange(size, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * orders * positions / size)
    basis[0] *= np.sqrt(1.0 / size)
    basis[1:] *= np.sqrt(2.0 / size)
    return np.asarray(values @ basis, dtype=np.float64)


# -- extensions --
