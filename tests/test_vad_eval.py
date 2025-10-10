from __future__ import annotations

import numpy as np
import pytest

from choravox import vad_eval


def test_binary_vad_counts_match_hand_computed_errors() -> None:
    reference = np.array([False, True, True, False, True, False])
    hypothesis = np.array([False, False, True, True, True, False])
    counts = vad_eval.binary_counts(reference, hypothesis)
    assert counts == vad_eval.VadCounts(1, 1, 3, 3)
    assert counts.miss_rate == pytest.approx(1 / 3, abs=1e-6)
    assert counts.false_alarm_rate == pytest.approx(1 / 3, abs=1e-6)
    assert counts.error_rate == pytest.approx(1 / 3, abs=1e-6)


def test_perfect_binary_vad_has_exact_zero_rates() -> None:
    reference = np.array([False, True, True, False])
    counts = vad_eval.binary_counts(reference, reference)
    assert counts.missed == counts.false_alarm == 0
    assert counts.miss_rate == counts.false_alarm_rate == counts.error_rate == 0.0


def test_vad_rates_handle_all_speech_and_all_silence() -> None:
    speech = vad_eval.binary_counts(np.ones(4, dtype=np.bool_), np.zeros(4, dtype=np.bool_))
    silence = vad_eval.binary_counts(np.zeros(4, dtype=np.bool_), np.ones(4, dtype=np.bool_))
    assert (speech.miss_rate, speech.false_alarm_rate) == (1.0, 0.0)
    assert (silence.miss_rate, silence.false_alarm_rate) == (0.0, 1.0)


def test_vad_counts_reject_impossible_error_totals() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        vad_eval.VadCounts(2, 0, 1, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        vad_eval.VadCounts(-1, 0, 1, 1)


def test_binary_counts_rejects_shape_and_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        vad_eval.binary_counts(np.array([True]), np.array([True, False]))
    with pytest.raises(ValueError, match="boolean vector"):
        vad_eval.binary_counts([0, 1], np.array([False, True]))


# -- extensions --
