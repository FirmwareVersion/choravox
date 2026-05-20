from __future__ import annotations

import numpy as np
import pytest

from choravox import asr
from choravox.ctc import CtcCandidate, CtcVocabulary


def vocabulary() -> CtcVocabulary:
    return CtcVocabulary(("<blank>", "sito", "mava", "noki", "<spk>"))


def test_sot_roundtrip_preserves_speaker_turn_sequence() -> None:
    tokens = (
        asr.AttributedToken("sito", "a"),
        asr.AttributedToken("mava", "a"),
        asr.AttributedToken("noki", "b"),
    )
    encoded = asr.serialize_sot(tokens, vocabulary())
    assert encoded == (1, 2, 4, 3)
    assert asr.parse_sot(encoded, vocabulary(), ("a", "b")) == tokens


def test_empty_sot_serialization_is_explicit() -> None:
    assert asr.serialize_sot((), vocabulary()) == ()


def test_sot_parser_rejects_blank_and_excess_speaker_changes() -> None:
    with pytest.raises(ValueError, match="blank"):
        asr.parse_sot((0,), vocabulary(), ("a",))
    with pytest.raises(ValueError, match="more speaker"):
        asr.parse_sot((1, 4, 2), vocabulary(), ("a",))


def test_sot_roundtrip_handles_rapid_alternating_speakers() -> None:
    tokens = tuple(
        asr.AttributedToken(text, speaker)
        for text, speaker in zip(("sito", "mava", "noki"), ("a", "b", "c"), strict=True)
    )
    encoded = asr.serialize_sot(tokens, vocabulary())
    assert encoded == (1, 4, 2, 4, 3)
    assert asr.parse_sot(encoded, vocabulary(), ("a", "b", "c")) == tokens


def test_sot_encoding_does_not_insert_change_within_a_turn() -> None:
    tokens = tuple(asr.AttributedToken(text, "a") for text in ("sito", "mava", "noki"))
    assert asr.serialize_sot(tokens, vocabulary()) == (1, 2, 3)


def test_sot_parse_requires_a_valid_speaker_roster() -> None:
    with pytest.raises(ValueError, match="speaker order"):
        asr.parse_sot((1,), vocabulary(), ())
    with pytest.raises(ValueError, match="speaker order"):
        asr.parse_sot((1,), vocabulary(), ("",))


def test_attributed_tokens_validate_both_labels() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        asr.AttributedToken("", "a")
    with pytest.raises(ValueError, match="nonempty"):
        asr.AttributedToken("sito", "")


def test_nbest_attribution_combines_acoustic_and_embedding_scores() -> None:
    candidates = (CtcCandidate((1,), -0.2), CtcCandidate((2,), -0.4))
    ranked = asr.attribute_nbest(candidates, {"a": 0.1, "b": 0.8}, embedding_weight=0.5)
    assert ranked[0].speaker == "b" and ranked[0].token_ids == (1,)
    assert ranked[0].combined_score == 0.2
    assert len(ranked) == 4


def test_zero_embedding_weight_preserves_acoustic_ranking() -> None:
    candidates = (CtcCandidate((1,), -0.1), CtcCandidate((2,), -0.3))
    ranked = asr.attribute_nbest(candidates, {"a": -1.0, "b": 1.0}, embedding_weight=0.0)
    assert [item.token_ids for item in ranked[:2]] == [(1,), (1,)]


def test_nbest_attribution_breaks_equal_scores_deterministically() -> None:
    ranked = asr.attribute_nbest(
        (CtcCandidate((2,), 0.0), CtcCandidate((1,), 0.0)), {"b": 0.0, "a": 0.0}
    )
    assert [(item.speaker, item.token_ids) for item in ranked] == [
        ("a", (1,)),
        ("a", (2,)),
        ("b", (1,)),
        ("b", (2,)),
    ]


def test_embedding_weight_can_change_top_speaker_without_transcript() -> None:
    candidates = (CtcCandidate((1, 2), -0.1),)
    acoustic = asr.attribute_nbest(candidates, {"a": 0.2, "b": 0.8}, embedding_weight=0.0)
    rescored = asr.attribute_nbest(candidates, {"a": 0.2, "b": 0.8}, embedding_weight=2.0)
    assert acoustic[0].speaker == "a"
    assert rescored[0].speaker == "b"
    assert rescored[0].token_ids == acoustic[0].token_ids


def test_nbest_attribution_validates_inputs_and_scores() -> None:
    with pytest.raises(ValueError, match="requires"):
        asr.attribute_nbest((), {"a": 1.0})
    with pytest.raises(ValueError, match="weight"):
        asr.attribute_nbest((CtcCandidate((1,), 0.0),), {"a": 1.0}, embedding_weight=-1.0)
    with pytest.raises(ValueError, match="roster"):
        asr.attribute_nbest((CtcCandidate((1,), 0.0),), {"a": float("nan")})


def test_candidate_cross_product_is_complete() -> None:
    candidates = tuple(CtcCandidate((index,), -float(index)) for index in range(1, 4))
    ranked = asr.attribute_nbest(candidates, {"a": 0.1, "b": 0.2})
    assert len(ranked) == len(candidates) * 2
    assert {(item.speaker, item.token_ids) for item in ranked} == {
        (speaker, candidate.token_ids) for speaker in ("a", "b") for candidate in candidates
    }


def test_enrollment_selects_matching_synthetic_speaker() -> None:
    enrollment = np.array([1.0, 0.0])
    probes = np.array([[0.0, 1.0], [0.9, 0.1]])
    scores = asr.enrollment_similarities(enrollment, probes, ("a", "b"))
    assert scores["b"] > scores["a"]
    assert asr.select_target_speaker(enrollment, probes, ("a", "b"))[0] == "b"


def test_target_speaker_tie_breaks_by_label() -> None:
    enrollment = np.array([1.0, 0.0])
    probes = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert asr.select_target_speaker(enrollment, probes, ("b", "a"))[0] == "a"


def test_enrollment_similarity_is_invariant_to_positive_scale() -> None:
    enrollment = np.array([2.0, 1.0])
    probes = np.array([[1.0, 0.0], [2.0, 1.0]])
    first = asr.enrollment_similarities(enrollment, probes, ("a", "b"))
    second = asr.enrollment_similarities(enrollment * 4.0, probes * 3.0, ("a", "b"))
    assert first == second


def test_permuting_probes_with_labels_preserves_named_scores() -> None:
    enrollment = np.array([1.0, 1.0])
    probes = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    first = asr.enrollment_similarities(enrollment, probes, ("a", "b", "c"))
    second = asr.enrollment_similarities(enrollment, probes[[2, 0, 1]], ("c", "a", "b"))
    assert first == second


def test_enrollment_matching_validates_probe_geometry() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        asr.enrollment_similarities([1.0, 0.0], [[1.0, 0.0]], ("a", "b"))
    with pytest.raises(ValueError, match="incompatible"):
        asr.enrollment_similarities([1.0], [[1.0], [1.0]], ("a", "a"))


def test_zero_enrollment_vector_is_rejected_by_cosine_scoring() -> None:
    with pytest.raises(ValueError, match="positive"):
        asr.enrollment_similarities([0.0, 0.0], [[1.0, 0.0]], ("a",))


def test_target_speaker_mask_selects_enrollment_like_frames() -> None:
    frames = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]])
    mask = asr.target_speaker_mask(frames, [1.0, 0.0], threshold=0.7)
    assert mask.tolist() == [True, True, False, False]


def test_target_speaker_threshold_endpoints_are_defined() -> None:
    frames = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert asr.target_speaker_mask(frames, [1.0, 0.0], threshold=-1.0).tolist() == [True, True]
    assert asr.target_speaker_mask(frames, [1.0, 0.0], threshold=1.0).tolist() == [True, False]


def test_target_mask_is_invariant_to_common_embedding_scale() -> None:
    frames = np.array([[1.0, 0.0], [0.0, 1.0]])
    first = asr.target_speaker_mask(frames, [1.0, 0.0])
    second = asr.target_speaker_mask(frames * 5.0, [3.0, 0.0])
    np.testing.assert_array_equal(first, second)


def test_target_mask_preserves_empty_frame_axis() -> None:
    result = asr.target_speaker_mask(np.empty((0, 2)), [1.0, 0.0])
    assert result.shape == (0,) and result.dtype == np.bool_


def test_target_mask_validates_dimensions_and_threshold() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        asr.target_speaker_mask(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="threshold"):
        asr.target_speaker_mask(np.ones((1, 2)), np.ones(2), threshold=1.1)


def test_target_mask_rejects_zero_probe_frames() -> None:
    with pytest.raises(ValueError, match="positive"):
        asr.target_speaker_mask(np.array([[0.0, 0.0]]), [1.0, 0.0])


def test_attributed_words_preserve_speaker_changes_and_confidence() -> None:
    tokens = (asr.AttributedToken("sito", "a"), asr.AttributedToken("mava", "b"))
    transcript = asr.attributed_words(tokens, ((0, 2, 0.9), (3, 5, 0.8)), frame_shift_s=0.1)
    assert [(word.text, word.speaker, word.start, word.end) for word in transcript.words] == [
        ("sito", "a", 0.0, 0.2),
        ("mava", "b", 0.3, 0.5),
    ]
    assert [word.confidence for word in transcript.words] == [0.9, 0.8]


def test_attributed_words_validate_alignment_shape_and_spans() -> None:
    token = asr.AttributedToken("sito", "a")
    with pytest.raises(ValueError, match="must match"):
        asr.attributed_words((token,), (), frame_shift_s=0.1)
    with pytest.raises(ValueError, match="span"):
        asr.attributed_words((token,), ((1, 1, 0.5),), frame_shift_s=0.1)


# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
