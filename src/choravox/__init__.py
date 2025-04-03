"""Deterministic mechanics for synthetic multi-speaker meetings."""

from choravox._version import __version__
from choravox.config import AudioConfig, FeatureConfig, MeetingConfig
from choravox.types import Segment, Transcript, Word

__all__ = [
    "AudioConfig",
    "FeatureConfig",
    "MeetingConfig",
    "Segment",
    "Transcript",
    "Word",
    "__version__",
]
