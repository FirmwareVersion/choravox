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
        output = output[width:-width] if width else output
    if length is not None:
        if length < 0 or length > output.shape[0]:
            raise ValueError("length must fit the reconstructed signal")
        output = output[:length]
    return np.asarray(output, dtype=np.float64)


# -- extensions --
