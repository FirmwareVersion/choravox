from __future__ import annotations

import pytest

from choravox import microphone


def test_microphone_array_preserves_channel_order_and_centroid() -> None:
    array = microphone.MicrophoneArray(((-0.1, 0.0, 1.0), (0.1, 0.0, 1.0)))
    assert array.channel_count == 2
    assert array.centroid_m == (0.0, 0.0, 1.0)
    assert array.positions_m[0][0] < array.positions_m[1][0]


def test_microphone_array_rejects_empty_duplicate_or_nonfinite_positions() -> None:
    with pytest.raises(ValueError, match="at least one"):
        microphone.MicrophoneArray(())
    with pytest.raises(ValueError, match="distinct"):
        microphone.MicrophoneArray(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="finite"):
        microphone.MicrophoneArray(((float("nan"), 0.0, 0.0),))


# -- extensions --
