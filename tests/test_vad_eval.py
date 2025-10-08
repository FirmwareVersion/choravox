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


# -- extensions --
