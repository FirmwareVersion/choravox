"""Deterministic frame-energy and statistical voice activity detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import to_mono
from choravox.features import frame_signal

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FrameEnergy:
    """Frame timing paired with RMS energy in dBFS."""

    starts_s: FloatArray
    centers_s: FloatArray
    dbfs: FloatArray
    frame_duration_s: float


def frame_energy_db(
    samples: ArrayLike,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    floor_db: float = -120.0,
) -> FrameEnergy:
    """Measure mono frame RMS on a sample-derived time axis."""
    if sample_rate <= 0 or frame_ms <= 0.0 or hop_ms <= 0.0 or hop_ms > frame_ms:
        raise ValueError("VAD sample rate and frame timing must be positive and ordered")
    if not math.isfinite(floor_db) or floor_db > 0.0:
        raise ValueError("floor_db must be finite and nonpositive")
    frame_length = round(sample_rate * frame_ms / 1000.0)
    hop_length = round(sample_rate * hop_ms / 1000.0)
    frames = frame_signal(to_mono(samples), frame_length, hop_length, end="pad")
    if frames.shape[0] == 0:
        values = np.empty(0, dtype=np.float64)
    else:
        power = np.mean(np.square(frames), axis=1)
        values = np.maximum(floor_db, 10.0 * np.log10(np.maximum(power, 10.0 ** (floor_db / 10.0))))
        values = np.round(values, 6)
    starts = np.arange(values.size, dtype=np.float64) * hop_length / sample_rate
    duration = frame_length / sample_rate
    return FrameEnergy(starts, starts + duration / 2.0, values, duration)


def adaptive_threshold_db(
    energy_db: ArrayLike, *, noise_quantile: float = 0.2, margin_db: float = 6.0
) -> float:
    """Estimate a speech threshold above a lower-tail noise floor."""
    values = np.asarray(energy_db, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("energy_db must be a nonempty finite vector")
    if not 0.0 <= noise_quantile <= 1.0 or not math.isfinite(margin_db) or margin_db < 0.0:
        raise ValueError("noise_quantile and margin_db are invalid")
    noise_floor = float(np.quantile(values, noise_quantile))
    return round(noise_floor + margin_db, 6)


def energy_posterior(
    energy_db: ArrayLike, threshold_db: float, *, temperature_db: float = 2.0
) -> FloatArray:
    """Map frame energy to calibrated logistic speech probabilities."""
    values = np.asarray(energy_db, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("energy_db must be a finite vector")
    if (
        not math.isfinite(threshold_db)
        or not math.isfinite(temperature_db)
        or temperature_db <= 0.0
    ):
        raise ValueError("posterior threshold and temperature must be finite and valid")
    logits = np.clip((values - threshold_db) / temperature_db, -60.0, 60.0)
    return np.round(1.0 / (1.0 + np.exp(-logits)), 9)


def hysteresis_decode(
    posterior: ArrayLike,
    *,
    onset: float = 0.6,
    offset: float = 0.4,
    initial_speech: bool = False,
) -> BoolArray:
    """Decode speech state with separate entry and exit thresholds."""
    values = np.asarray(posterior, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("posterior must be a finite probability vector")
    if not 0.0 <= offset <= onset <= 1.0:
        raise ValueError("hysteresis requires 0 <= offset <= onset <= 1")
    output = np.empty(values.size, dtype=np.bool_)
    active = initial_speech
    for index, value in enumerate(values):
        if active and value < offset:
            active = False
        elif not active and value >= onset:
            active = True
        output[index] = active
    return output


def apply_hangover(decisions: ArrayLike, frames: int) -> BoolArray:
    """Retain speech for a fixed number of frames after every positive decision."""
    values = np.asarray(decisions)
    if values.ndim != 1 or values.dtype.kind != "b":
        raise ValueError("decisions must be a boolean vector")
    if frames < 0:
        raise ValueError("hangover frames must be nonnegative")
    output = values.astype(np.bool_, copy=True)
    remaining = 0
    for index, value in enumerate(values):
        if value:
            remaining = frames
        elif remaining > 0:
            output[index] = True
            remaining -= 1
    return output


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Validated controls for deterministic frame-energy VAD."""

    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    noise_quantile: float = 0.2
    margin_db: float = 6.0
    temperature_db: float = 2.0
    onset: float = 0.6
    offset: float = 0.4
    hangover_frames: int = 3
    hmm_smoothing: bool = True

    def __post_init__(self) -> None:
        values = (
            self.frame_ms,
            self.hop_ms,
            self.noise_quantile,
            self.margin_db,
            self.temperature_db,
            self.onset,
            self.offset,
        )
        if self.sample_rate <= 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("VAD configuration values must be finite and sample_rate positive")
        if self.frame_ms <= 0.0 or not 0.0 < self.hop_ms <= self.frame_ms:
            raise ValueError("VAD frame and hop timing are invalid")
        if (
            not 0.0 <= self.noise_quantile <= 1.0
            or self.margin_db < 0.0
            or self.temperature_db <= 0.0
        ):
            raise ValueError("VAD noise and posterior calibration are invalid")
        if not 0.0 <= self.offset <= self.onset <= 1.0 or self.hangover_frames < 0:
            raise ValueError("VAD state thresholds and hangover are invalid")


@dataclass(frozen=True, slots=True)
class VadResult:
    """Energy, threshold, posterior, and final frame decisions."""

    frames: FrameEnergy
    threshold_db: float
    posterior: FloatArray
    speech: BoolArray


def _hmm_inputs(
    posterior: ArrayLike, silence_stay: float, speech_stay: float
) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(posterior, dtype=np.float64)
    if (
        values.ndim != 1
        or not np.all(np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("posterior must be a finite probability vector")
    if not 0.0 < silence_stay < 1.0 or not 0.0 < speech_stay < 1.0:
        raise ValueError("HMM stay probabilities must lie in (0, 1)")
    emissions = np.column_stack((1.0 - values, values))
    emissions = np.clip(emissions, 1e-12, 1.0)
    transitions = np.array(
        [[silence_stay, 1.0 - silence_stay], [1.0 - speech_stay, speech_stay]],
        dtype=np.float64,
    )
    return emissions, transitions


def hmm_speech_posterior(
    posterior: ArrayLike, *, silence_stay: float = 0.95, speech_stay: float = 0.95
) -> FloatArray:
    """Apply scaled forward-backward smoothing to frame speech evidence."""
    emissions, transitions = _hmm_inputs(posterior, silence_stay, speech_stay)
    if emissions.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    forward = np.empty_like(emissions)
    forward[0] = np.array([0.5, 0.5]) * emissions[0]
    forward[0] /= np.sum(forward[0])
    for index in range(1, emissions.shape[0]):
        forward[index] = (forward[index - 1] @ transitions) * emissions[index]
        forward[index] /= np.sum(forward[index])
    backward = np.ones_like(emissions)
    for index in range(emissions.shape[0] - 2, -1, -1):
        backward[index] = transitions @ (emissions[index + 1] * backward[index + 1])
        backward[index] /= np.sum(backward[index])
    smoothed = forward * backward
    smoothed /= np.sum(smoothed, axis=1, keepdims=True)
    return np.round(smoothed[:, 1], 9)


def hmm_viterbi(
    posterior: ArrayLike, *, silence_stay: float = 0.95, speech_stay: float = 0.95
) -> BoolArray:
    """Decode the maximum-probability two-state sequence by Viterbi."""
    emissions, transitions = _hmm_inputs(posterior, silence_stay, speech_stay)
    if emissions.shape[0] == 0:
        return np.empty(0, dtype=np.bool_)
    log_emissions = np.log(emissions)
    log_transitions = np.log(transitions)
    scores = np.empty_like(emissions)
    backpointers = np.zeros(emissions.shape, dtype=np.int64)
    scores[0] = math.log(0.5) + log_emissions[0]
    for index in range(1, emissions.shape[0]):
        candidates = scores[index - 1][:, None] + log_transitions
        backpointers[index] = np.argmax(candidates, axis=0)
        scores[index] = np.max(candidates, axis=0) + log_emissions[index]
    states = np.empty(emissions.shape[0], dtype=np.int64)
    states[-1] = int(np.argmax(scores[-1]))
    for index in range(states.size - 1, 0, -1):
        states[index - 1] = backpointers[index, states[index]]
    return states.astype(np.bool_)


# -- extensions --
