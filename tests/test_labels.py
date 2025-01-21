from __future__ import annotations

import pytest

from choravox import labels


def test_normalize_label_collapses_unsafe_runs() -> None:
    assert labels.normalize_label(" Speaker A / Near ") == "speaker-a-near"
    assert labels.normalize_label("SPK_01") == "spk_01"


def test_unique_labels_resolves_normalization_collisions() -> None:
    assert labels.unique_labels(["Speaker A", "speaker-a", "Speaker B"]) == (
        "speaker-a",
        "speaker-a-2",
        "speaker-b",
    )
    with pytest.raises(ValueError, match="portable"):
        labels.normalize_label("中文")


# -- extensions --
