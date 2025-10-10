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


def test_planted_frame_activity_uses_frame_centers() -> None:
    from choravox.dialogue import DialogueScript, DialogueTurn

    script = DialogueScript(1.0, (DialogueTurn(0.2, 0.6, "spk", ("mava",)),))
    starts = np.array([0.0, 0.15, 0.2, 0.55, 0.6])
    activity = vad_eval.planted_frame_activity(script, starts, 0.1)
    np.testing.assert_array_equal(activity, [False, True, True, False, False])


def test_overlapping_turns_still_produce_binary_speech_truth() -> None:
    from choravox.dialogue import DialogueScript, DialogueTurn

    script = DialogueScript(
        1.0,
        (DialogueTurn(0.0, 0.7, "a", ("mava",)), DialogueTurn(0.3, 0.8, "b", ("sito",))),
    )
    activity = vad_eval.planted_frame_activity(script, [0.25, 0.45, 0.75], 0.1)
    np.testing.assert_array_equal(activity, [True, True, False])


def test_planted_activity_empty_script_is_all_silence() -> None:
    from choravox.dialogue import DialogueScript

    activity = vad_eval.planted_frame_activity(DialogueScript(1.0, ()), [0.0, 0.1], 0.025)
    np.testing.assert_array_equal(activity, [False, False])


def test_planted_activity_rejects_negative_or_matrix_starts() -> None:
    from choravox.dialogue import DialogueScript

    with pytest.raises(ValueError, match="nonnegative vector"):
        vad_eval.planted_frame_activity(DialogueScript(1.0, ()), [-0.1], 0.02)
    with pytest.raises(ValueError, match="nonnegative vector"):
        vad_eval.planted_frame_activity(DialogueScript(1.0, ()), [[0.0]], 0.02)


def test_planted_activity_rejects_nonpositive_duration() -> None:
    from choravox.dialogue import DialogueScript

    with pytest.raises(ValueError, match="positive"):
        vad_eval.planted_frame_activity(DialogueScript(1.0, ()), [0.0], 0.0)


# -- extensions --
