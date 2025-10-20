"""Deterministic frame-energy and statistical voice activity detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from choravox.audio import to_mono
from choravox.features import frame_signal
from choravox.types import Segment

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


def detect_speech(samples: ArrayLike, config: VadConfig | None = None) -> VadResult:
    """Run energy measurement, adaptive calibration, HMM smoothing, and state rules."""
    settings = VadConfig() if config is None else config
    frames = frame_energy_db(
        samples,
        settings.sample_rate,
        frame_ms=settings.frame_ms,
        hop_ms=settings.hop_ms,
    )
    if frames.dbfs.size == 0:
        empty_float = np.empty(0, dtype=np.float64)
        empty_bool = np.empty(0, dtype=np.bool_)
        return VadResult(frames, -120.0, empty_float, empty_bool)
    threshold = adaptive_threshold_db(
        frames.dbfs,
        noise_quantile=settings.noise_quantile,
        margin_db=settings.margin_db,
    )
    posterior = energy_posterior(frames.dbfs, threshold, temperature_db=settings.temperature_db)
    if settings.hmm_smoothing:
        posterior = hmm_speech_posterior(posterior)
    speech = hysteresis_decode(posterior, onset=settings.onset, offset=settings.offset)
    speech = apply_hangover(speech, settings.hangover_frames)
    return VadResult(frames, threshold, posterior, speech)


def decisions_to_segments(
    decisions: ArrayLike,
    hop_s: float,
    frame_duration_s: float,
    *,
    speaker: str = "speech",
) -> tuple[Segment, ...]:
    """Merge consecutive positive frame decisions into half-open segments."""
    values = np.asarray(decisions)
    if values.ndim != 1 or values.dtype.kind != "b":
        raise ValueError("decisions must be a boolean vector")
    if hop_s <= 0.0 or frame_duration_s <= 0.0 or not speaker:
        raise ValueError("segment timing and speaker must be valid")
    segments: list[Segment] = []
    start_index: int | None = None
    for index, active in enumerate(np.append(values, False)):
        if active and start_index is None:
            start_index = index
        elif not active and start_index is not None:
            segments.append(
                Segment(
                    round(start_index * hop_s, 9),
                    round((index - 1) * hop_s + frame_duration_s, 9),
                    speaker,
                )
            )
            start_index = None
    return tuple(segments)


def segments_to_decisions(
    segments: tuple[Segment, ...], frame_starts_s: ArrayLike, frame_duration_s: float
) -> BoolArray:
    """Mark frames fully contained by at least one same-clock speech segment."""
    starts = np.asarray(frame_starts_s, dtype=np.float64)
    if starts.ndim != 1 or not np.all(np.isfinite(starts)) or frame_duration_s <= 0.0:
        raise ValueError("frame starts and duration must be valid")
    ends = starts + frame_duration_s
    return np.asarray(
        [
            any(
                start >= segment.start - 1e-12 and end <= segment.end + 1e-12
                for segment in segments
            )
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=np.bool_,
    )


def statistical_features(
    samples: ArrayLike,
    sample_rate: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> FloatArray:
    """Extract frame log-energy and zero-crossing rate for statistical VAD."""
    energy = frame_energy_db(samples, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    frame_length = round(sample_rate * frame_ms / 1000.0)
    hop_length = round(sample_rate * hop_ms / 1000.0)
    frames = frame_signal(to_mono(samples), frame_length, hop_length, end="pad")
    if frames.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    signs = frames >= 0.0
    crossings = (
        np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
        if frame_length > 1
        else np.zeros(frames.shape[0])
    )
    return np.round(np.column_stack((energy.dbfs, crossings)), 6)


@dataclass(frozen=True, slots=True)
class GaussianVadModel:
    """Two-class diagonal Gaussian emissions estimated from planted frames."""

    silence_mean: FloatArray
    silence_variance: FloatArray
    speech_mean: FloatArray
    speech_variance: FloatArray
    speech_prior: float

    def __post_init__(self) -> None:
        arrays = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.silence_mean,
                self.silence_variance,
                self.speech_mean,
                self.speech_variance,
            )
        )
        if any(array.ndim != 1 or array.size == 0 for array in arrays):
            raise ValueError("Gaussian VAD parameters must be nonempty vectors")
        if any(
            array.shape != arrays[0].shape or not np.all(np.isfinite(array)) for array in arrays
        ):
            raise ValueError("Gaussian VAD parameter shapes and values must match")
        if np.any(arrays[1] <= 0.0) or np.any(arrays[3] <= 0.0):
            raise ValueError("Gaussian VAD variances must be positive")
        if not 0.0 < self.speech_prior < 1.0:
            raise ValueError("Gaussian VAD speech_prior must lie in (0, 1)")


def fit_gaussian_vad(
    features: ArrayLike,
    speech: ArrayLike,
    *,
    variance_floor: float = 1e-4,
) -> GaussianVadModel:
    """Estimate two diagonal emissions from planted speech labels."""
    values = np.asarray(features, dtype=np.float64)
    labels = np.asarray(speech)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("training features must be a nonempty finite matrix")
    if labels.shape != (values.shape[0],) or labels.dtype.kind != "b":
        raise ValueError("training speech labels must align as booleans")
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and positive")
    if not np.any(labels) or np.all(labels):
        raise ValueError("training labels must contain speech and silence")

    def statistics(selected: FloatArray) -> tuple[FloatArray, FloatArray]:
        mean = np.mean(selected, axis=0)
        variance = np.maximum(np.var(selected, axis=0), variance_floor)
        return np.round(mean, 6), np.round(variance, 6)

    silence_mean, silence_variance = statistics(values[~labels])
    speech_mean, speech_variance = statistics(values[labels])
    return GaussianVadModel(
        silence_mean,
        silence_variance,
        speech_mean,
        speech_variance,
        round(float(np.mean(labels)), 6),
    )


def gaussian_speech_posterior(features: ArrayLike, model: GaussianVadModel) -> FloatArray:
    """Evaluate normalized two-class diagonal Gaussian posterior probabilities."""
    values = np.asarray(features, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1:] != model.speech_mean.shape
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("inference features must be a finite matrix matching the model")

    def log_density(mean: FloatArray, variance: FloatArray) -> FloatArray:
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * variance) + np.square(values - mean) / variance, axis=1
        )

    silence = log_density(model.silence_mean, model.silence_variance) + math.log(
        1.0 - model.speech_prior
    )
    speech = log_density(model.speech_mean, model.speech_variance) + math.log(model.speech_prior)
    difference = np.clip(silence - speech, -60.0, 60.0)
    return np.round(1.0 / (1.0 + np.exp(difference)), 9)


def detect_statistical(
    samples: ArrayLike,
    model: GaussianVadModel,
    config: VadConfig | None = None,
) -> VadResult:
    """Run planted-trained Gaussian emissions through shared HMM state mechanics."""
    settings = VadConfig() if config is None else config
    frames = frame_energy_db(
        samples, settings.sample_rate, frame_ms=settings.frame_ms, hop_ms=settings.hop_ms
    )
    features = statistical_features(
        samples, settings.sample_rate, frame_ms=settings.frame_ms, hop_ms=settings.hop_ms
    )
    posterior = gaussian_speech_posterior(features, model)
    if settings.hmm_smoothing:
        posterior = hmm_speech_posterior(posterior)
    speech = hysteresis_decode(posterior, onset=settings.onset, offset=settings.offset)
    speech = apply_hangover(speech, settings.hangover_frames)
    threshold = float((model.silence_mean[0] + model.speech_mean[0]) / 2.0)
    return VadResult(frames, round(threshold, 6), posterior, speech)


def _runs(values: BoolArray) -> tuple[tuple[int, int, bool], ...]:
    if values.size == 0:
        return ()
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [values.size]))
    return tuple(
        (int(start), int(end), bool(values[start])) for start, end in zip(starts, ends, strict=True)
    )


def enforce_minimum_durations(
    decisions: ArrayLike,
    *,
    min_speech_frames: int = 1,
    min_silence_frames: int = 1,
) -> BoolArray:
    """Remove short speech islands, then fill short internal silence gaps."""
    values = np.asarray(decisions)
    if values.ndim != 1 or values.dtype.kind != "b":
        raise ValueError("decisions must be a boolean vector")
    if min_speech_frames < 1 or min_silence_frames < 1:
        raise ValueError("minimum state durations must be positive")
    output = values.astype(np.bool_, copy=True)
    for start, end, active in _runs(output):
        if active and end - start < min_speech_frames:
            output[start:end] = False
    for start, end, active in _runs(output):
        internal = start > 0 and end < output.size
        if not active and internal and end - start < min_silence_frames:
            output[start:end] = True
    return output


# -- extensions --
