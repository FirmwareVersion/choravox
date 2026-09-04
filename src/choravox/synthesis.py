"""Deterministic source-filter synthesis for phoneme-like sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import as_waveform, normalize_rms
from choravox.phonetics import (
    Phone,
    SpeakerProfile,
    get_phone,
    glottal_source,
    pitch_contour,
    unvoiced_source,
)
from choravox.random import derive_seed

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Resonator:
    """Second-order all-pole resonator coefficients."""

    b0: float
    a1: float
    a2: float


def resonator_coefficients(frequency_hz: float, bandwidth_hz: float, sample_rate: int) -> Resonator:
    """Create stable coefficients for one digital formant resonance."""
    if sample_rate <= 0 or not 0.0 < frequency_hz < sample_rate / 2:
        raise ValueError("formant frequency must lie between zero and Nyquist")
    if not 0.0 < bandwidth_hz < sample_rate / 2:
        raise ValueError("formant bandwidth must lie between zero and Nyquist")
    radius = float(np.exp(-np.pi * bandwidth_hz / sample_rate))
    angle = 2.0 * np.pi * frequency_hz / sample_rate
    return Resonator(1.0 - radius, 2.0 * radius * float(np.cos(angle)), -(radius**2))


def apply_resonator(samples: ArrayLike, coefficients: Resonator) -> FloatArray:
    """Apply a causal resonator independently along every waveform channel."""
    waveform = as_waveform(samples)
    output = np.empty_like(waveform)
    if waveform.ndim == 1:
        previous_one_scalar = previous_two_scalar = 0.0
        for index, sample in enumerate(waveform):
            value = (
                coefficients.b0 * sample
                + coefficients.a1 * previous_one_scalar
                + coefficients.a2 * previous_two_scalar
            )
            output[index] = value
            previous_two_scalar, previous_one_scalar = previous_one_scalar, value
    else:
        previous_one_vector = np.zeros(waveform.shape[1], dtype=np.float64)
        previous_two_vector = np.zeros(waveform.shape[1], dtype=np.float64)
        for index, sample in enumerate(waveform):
            value = (
                coefficients.b0 * sample
                + coefficients.a1 * previous_one_vector
                + coefficients.a2 * previous_two_vector
            )
            output[index] = value
            previous_two_vector, previous_one_vector = previous_one_vector, value
    return output


def apply_formants(
    source: ArrayLike, phone: Phone, profile: SpeakerProfile, sample_rate: int
) -> FloatArray:
    """Apply scaled formants or a simple fricative emphasis to one excitation."""
    output = as_waveform(source)
    if output.ndim != 1:
        raise ValueError("formant synthesis expects mono excitation")
    if phone.voiced:
        for frequency, bandwidth in zip(phone.formants_hz, phone.bandwidths_hz, strict=True):
            scaled = min(frequency * profile.formant_scale, sample_rate * 0.48)
            output = apply_resonator(output, resonator_coefficients(scaled, bandwidth, sample_rate))
    elif phone.manner == "fricative" and output.size:
        emphasized = output.copy()
        emphasized[1:] = output[1:] - 0.92 * output[:-1]
        output = emphasized
    return np.asarray(output, dtype=np.float64)


def phone_envelope(sample_count: int, attack: int, release: int) -> FloatArray:
    """Create a raised-cosine phone envelope with exact zero endpoints."""
    if sample_count <= 0 or attack < 0 or release < 0 or attack + release > sample_count:
        raise ValueError("phone envelope lengths must be nonnegative and fit")
    envelope = np.ones(sample_count, dtype=np.float64)
    if attack:
        envelope[:attack] = np.sin(np.linspace(0.0, np.pi / 2.0, attack)) ** 2
    if release:
        envelope[-release:] = np.cos(np.linspace(0.0, np.pi / 2.0, release)) ** 2
    return envelope


def synthesize_phone(
    phone: str | Phone,
    duration_s: float,
    profile: SpeakerProfile,
    sample_rate: int,
    *,
    seed: int,
) -> FloatArray:
    """Render one phone from seeded excitation through its acoustic target."""
    target = get_phone(phone) if isinstance(phone, str) else phone
    sample_count = round(duration_s * sample_rate)
    if duration_s <= 0.0 or sample_rate <= 0 or sample_count <= 0:
        raise ValueError("phone duration and sample rate must be positive")
    if target.voiced:
        contour = pitch_contour(duration_s, sample_rate, profile, seed=seed)
        excitation = glottal_source(contour, sample_rate)
        if profile.breathiness > 0.0:
            breath = unvoiced_source(sample_count, seed=seed, namespace=f"breath:{target.symbol}")
            excitation = excitation + profile.breathiness * breath
    else:
        excitation = unvoiced_source(sample_count, seed=seed, namespace=f"phone:{target.symbol}")
    shaped = apply_formants(excitation, target, profile, sample_rate)
    fade = min(round(0.01 * sample_rate), sample_count // 4)
    shaped *= phone_envelope(sample_count, fade, fade)
    return np.round(normalize_rms(shaped, 0.1, peak_limit=0.95), 9)


@dataclass(frozen=True, slots=True)
class PhoneToken:
    """One requested phone and its duration."""

    symbol: str
    duration_s: float

    def __post_init__(self) -> None:
        get_phone(self.symbol)
        if self.duration_s <= 0.0:
            raise ValueError("phone token duration must be positive")


@dataclass(frozen=True, slots=True)
class PhoneSpan:
    """Planted time span for a rendered phone."""

    symbol: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class SynthesizedSequence:
    """Rendered waveform and planted phone boundaries."""

    samples: FloatArray
    phones: tuple[PhoneSpan, ...]


def synthesize_sequence(
    tokens: tuple[PhoneToken, ...],
    profile: SpeakerProfile,
    sample_rate: int,
    *,
    seed: int,
    gap_s: float = 0.0,
) -> SynthesizedSequence:
    """Concatenate independently seeded phones and plant exact sample-derived spans."""
    if not tokens:
        raise ValueError("at least one phone token is required")
    gap_samples = round(gap_s * sample_rate)
    if gap_s < 0.0 or gap_samples < 0:
        raise ValueError("gap_s must be nonnegative")
    pieces: list[FloatArray] = []
    spans: list[PhoneSpan] = []
    cursor = 0
    for index, token in enumerate(tokens):
        token_seed = derive_seed(seed, f"phone:{index}:{token.symbol}")
        rendered = synthesize_phone(
            token.symbol, token.duration_s, profile, sample_rate, seed=token_seed
        )
        start = cursor
        pieces.append(rendered)
        cursor += rendered.size
        spans.append(
            PhoneSpan(token.symbol, round(start / sample_rate, 9), round(cursor / sample_rate, 9))
        )
        if index + 1 < len(tokens) and gap_samples:
            pieces.append(np.zeros(gap_samples, dtype=np.float64))
            cursor += gap_samples
    return SynthesizedSequence(np.concatenate(pieces), tuple(spans))


# -- extensions --
