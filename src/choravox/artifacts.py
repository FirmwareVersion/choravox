"""Content-addressed bundles for reproducible synthetic meetings."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

_DIGEST = re.compile(r"[0-9a-f]{64}")


def bytes_digest(data: bytes) -> str:
    """Return a lowercase SHA-256 digest for exact artifact bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Manifest metadata for one relative-path byte artifact."""

    path: str
    media_type: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        candidate = PurePosixPath(self.path)
        if not self.path or candidate.is_absolute() or ".." in candidate.parts or "\\" in self.path:
            raise ValueError("artifact path must be safe and relative")
        if not self.media_type or any(char.isspace() for char in self.media_type):
            raise ValueError("artifact media_type must be a nonempty token")
        if self.size_bytes < 0 or _DIGEST.fullmatch(self.sha256) is None:
            raise ValueError("artifact size or SHA-256 digest is invalid")


def artifact_record(path: str, media_type: str, data: bytes) -> ArtifactRecord:
    """Measure exact artifact bytes into a validated manifest record."""
    return ArtifactRecord(path, media_type, len(data), bytes_digest(data))


# -- extensions --
