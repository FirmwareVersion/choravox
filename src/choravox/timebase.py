"""Exact conventions for sample, frame, and wall-clock indices."""

from __future__ import annotations


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    """Convert a nonnegative sample boundary to quantized seconds."""
    if samples < 0 or sample_rate <= 0:
        raise ValueError("samples must be nonnegative and sample_rate positive")
    return round(samples / sample_rate, 9)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    """Map seconds to the nearest sample boundary using half-up rounding."""
    if seconds < 0 or sample_rate <= 0:
        raise ValueError("seconds must be nonnegative and sample_rate positive")
    return int(seconds * sample_rate + 0.5)


def frame_spans(
    length: int, frame_length: int, hop_length: int, *, pad_end: bool = False
) -> tuple[tuple[int, int], ...]:
    """Return half-open frame spans, optionally including a padded tail."""
    if length < 0 or frame_length <= 0 or hop_length <= 0:
        raise ValueError("invalid frame geometry")
    if length == 0:
        return ()
    starts = range(0, length if pad_end else max(0, length - frame_length + 1), hop_length)
    spans = tuple((start, min(start + frame_length, length)) for start in starts)
    if not pad_end:
        return tuple(span for span in spans if span[1] - span[0] == frame_length)
    return spans


# -- extensions --
