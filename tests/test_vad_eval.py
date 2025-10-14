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


def test_evaluate_result_counts_known_decision_errors() -> None:
    from choravox.dialogue import DialogueScript, DialogueTurn
    from choravox.vad import FrameEnergy, VadResult

    starts = np.array([0.0, 0.1, 0.2, 0.3])
    frames = FrameEnergy(starts, starts + 0.05, np.zeros(4), 0.1)
    result = VadResult(frames, -20.0, np.full(4, 0.5), np.array([False, False, True, True]))
    script = DialogueScript(0.4, (DialogueTurn(0.1, 0.3, "spk", ("mava",)),))
    assert vad_eval.evaluate_result(result, script) == vad_eval.VadCounts(1, 1, 2, 2)


def test_evaluate_result_rejects_misaligned_decisions() -> None:
    from choravox.dialogue import DialogueScript
    from choravox.vad import FrameEnergy, VadResult

    frames = FrameEnergy(np.array([0.0]), np.array([0.05]), np.array([0.0]), 0.1)
    result = VadResult(frames, 0.0, np.array([]), np.array([], dtype=np.bool_))
    with pytest.raises(ValueError, match="align"):
        vad_eval.evaluate_result(result, DialogueScript(0.1, ()))


def test_clean_synthetic_meeting_vad_beats_all_silence_baseline() -> None:
    from choravox.config import MeetingConfig
    from choravox.meeting import MeetingFactory
    from choravox.vad import VadConfig, detect_speech

    meeting = MeetingFactory(
        MeetingConfig(duration_s=2.0, overlap_ratio=0.1, silence_ratio=0.3, seed=7), 8000
    ).build()
    result = detect_speech(meeting.samples, VadConfig(sample_rate=8000, margin_db=6.0))
    counts = vad_eval.evaluate_result(result, meeting.script)
    reference = vad_eval.planted_frame_activity(
        meeting.script, result.frames.starts_s, result.frames.frame_duration_s
    )
    silent = vad_eval.binary_counts(reference, np.zeros(reference.size, dtype=np.bool_))
    assert counts.error_rate < silent.error_rate


def test_planted_evaluation_is_deterministic() -> None:
    from choravox.config import MeetingConfig
    from choravox.meeting import MeetingFactory
    from choravox.vad import VadConfig, detect_speech

    meeting = MeetingFactory(MeetingConfig(duration_s=1.0, seed=2), 4000).build()
    result = detect_speech(meeting.samples, VadConfig(sample_rate=4000))
    assert vad_eval.evaluate_result(result, meeting.script) == vad_eval.evaluate_result(
        result, meeting.script
    )


def test_det_curve_matches_hand_computed_thresholds() -> None:
    posterior = [0.1, 0.4, 0.6, 0.9]
    reference = np.array([False, True, False, True])
    points = vad_eval.det_curve(posterior, reference, thresholds=[0.0, 0.5, 1.0])
    assert points[0] == vad_eval.DetPoint(0.0, 0.0, 1.0)
    assert points[1] == vad_eval.DetPoint(0.5, 0.5, 0.5)
    assert points[2] == vad_eval.DetPoint(1.0, 1.0, 0.0)


def test_default_det_curve_has_inclusive_probability_grid() -> None:
    points = vad_eval.det_curve([0.2, 0.8], np.array([False, True]))
    assert len(points) == 101
    assert points[0].threshold == 0.0
    assert points[-1].threshold == 1.0


def test_det_miss_rises_and_false_alarm_falls_with_threshold() -> None:
    posterior = np.linspace(0.0, 1.0, 20)
    reference = posterior >= 0.55
    points = vad_eval.det_curve(posterior, reference)
    misses = np.array([point.miss_rate for point in points])
    false_alarms = np.array([point.false_alarm_rate for point in points])
    assert np.all(np.diff(misses) >= 0.0)
    assert np.all(np.diff(false_alarms) <= 0.0)


def test_det_curve_rejects_misaligned_or_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="aligned"):
        vad_eval.det_curve([0.5], np.array([True, False]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        vad_eval.det_curve([1.1], np.array([True]))
    with pytest.raises(ValueError, match="thresholds"):
        vad_eval.det_curve([0.5], np.array([True]), thresholds=[-0.1])


def test_det_curve_empty_input_preserves_requested_thresholds() -> None:
    points = vad_eval.det_curve([], np.array([], dtype=np.bool_), thresholds=[0.25, 0.75])
    assert points == (vad_eval.DetPoint(0.25, 0.0, 0.0), vad_eval.DetPoint(0.75, 0.0, 0.0))


def test_snr_slices_assign_frames_to_half_open_bins() -> None:
    reference = np.array([True, True, False, False])
    hypothesis = np.array([False, True, True, False])
    slices = vad_eval.slice_by_snr(
        reference, hypothesis, [-10.0, -0.1, 0.0, 9.9], (-10.0, 0.0, 10.0)
    )
    assert slices[0].counts == vad_eval.VadCounts(1, 0, 2, 0)
    assert slices[1].counts == vad_eval.VadCounts(0, 1, 0, 2)
    assert (slices[0].lower_db, slices[0].upper_db) == (-10.0, 0.0)


def test_empty_snr_bin_has_zero_rates() -> None:
    slices = vad_eval.slice_by_snr(np.array([True]), np.array([True]), [5.0], (-10.0, 0.0, 10.0))
    assert slices[0].counts.error_rate == 0.0
    assert slices[0].counts.reference_speech == 0


def test_snr_slices_reject_unsorted_edges() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        vad_eval.slice_by_snr(np.array([True]), np.array([True]), [0.0], (0.0, 0.0, 10.0))


def test_snr_slices_reject_misaligned_frame_metadata() -> None:
    with pytest.raises(ValueError, match="aligned"):
        vad_eval.slice_by_snr(np.array([True]), np.array([True]), [0.0, 1.0], (-10.0, 10.0))


def test_snr_slice_counts_sum_when_edges_cover_all_frames() -> None:
    reference = np.array([True, False, True, False, True])
    hypothesis = np.array([False, True, True, False, False])
    levels = np.array([-15.0, -5.0, 0.0, 5.0, 15.0])
    slices = vad_eval.slice_by_snr(reference, hypothesis, levels, (-20.0, -10.0, 10.0, 20.0))
    overall = vad_eval.binary_counts(reference, hypothesis)
    assert sum(item.counts.missed for item in slices) == overall.missed
    assert sum(item.counts.false_alarm for item in slices) == overall.false_alarm
    assert sum(item.counts.reference_speech for item in slices) == overall.reference_speech


# -- extensions --
