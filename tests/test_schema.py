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


def test_envelope_digest_has_stable_golden_value() -> None:
    envelope = schema.make_envelope("choravox.example", 1, {"seed": 7, "labels": ["a", "b"]})
    assert envelope.digest == "1e6667528640a9a1ec6ed01d21d75246c570a82d6ffd22aa5073fca736854c97"


# -- extensions --
