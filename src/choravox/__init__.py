"""Deterministic mechanics for synthetic multi-speaker meetings."""

from choravox._version import __version__
from choravox.config import AudioConfig, MeetingConfig
from choravox.types import Segment, Transcript, Word

__all__ = [
    "AudioConfig",
    "MeetingConfig",
    "Segment",
    "Transcript",
    "Word",
    "__version__",
]
