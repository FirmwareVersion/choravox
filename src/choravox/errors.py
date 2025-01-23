"""Public exception hierarchy with actionable context."""

from __future__ import annotations


class ChoraVoxError(Exception):
    """Base class for expected toolkit failures."""


class ValidationError(ChoraVoxError, ValueError):
    """An input or serialized artifact violates a declared contract."""


class OptionalDependencyError(ChoraVoxError, ImportError):
    """A requested feature needs an explicitly optional dependency."""


def require(condition: bool, message: str) -> None:
    """Raise a public validation error when a contract is false."""
    if not condition:
        raise ValidationError(message)


def missing_extra(extra: str, feature: str) -> OptionalDependencyError:
    """Create a consistent installation hint for an optional feature."""
    return OptionalDependencyError(f"{feature} requires the optional '{extra}' extra")
