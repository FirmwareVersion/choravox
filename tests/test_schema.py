from __future__ import annotations

from dataclasses import replace

import pytest

from choravox import schema


def test_schema_envelope_validates_untouched_payload() -> None:
    envelope = schema.make_envelope("choravox.words", 1, [{"text": "hello"}])
    assert schema.validate_envelope(envelope, "choravox.words", {1}) == [{"text": "hello"}]


def test_schema_envelope_detects_identity_and_content_drift() -> None:
    envelope = schema.make_envelope("choravox.words", 1, ["hello"])
    with pytest.raises(ValueError, match="expected schema"):
        schema.validate_envelope(envelope, "choravox.turns", {1})
    with pytest.raises(ValueError, match="digest mismatch"):
        schema.validate_envelope(replace(envelope, payload=["changed"]), "choravox.words", {1})


# -- extensions --
