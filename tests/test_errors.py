from __future__ import annotations

import pytest

from choravox import errors


def test_require_uses_public_validation_error() -> None:
    errors.require(True, "unused")
    with pytest.raises(errors.ChoraVoxError, match="bad interval"):
        errors.require(False, "bad interval")


def test_optional_dependency_error_names_extra_and_feature() -> None:
    error = errors.missing_extra("torch", "CTC acoustic model")
    assert isinstance(error, ImportError)
    assert "torch" in str(error) and "CTC acoustic model" in str(error)
