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


# -- extensions --
