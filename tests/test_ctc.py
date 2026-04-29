from __future__ import annotations

import numpy as np
import pytest

from choravox import ctc


def vocabulary() -> ctc.CtcVocabulary:
    return ctc.CtcVocabulary(("<blank>", "sito", "mava", "<spk>"))


def test_ctc_vocabulary_roundtrips_words_and_speaker_changes() -> None:
    vocab = vocabulary()
    tokens = ("sito", "<spk>", "mava")
    assert vocab.decode(vocab.encode(tokens)) == tokens
    assert vocab.blank_id == 0 and vocab.speaker_change_id == 3


def test_ctc_vocabulary_rejects_unknown_tokens_and_identifiers() -> None:
    vocab = vocabulary()
    with pytest.raises(ValueError, match="unknown"):
        vocab.encode(("outside",))
    with pytest.raises(ValueError, match="outside"):
        vocab.decode((4,))


def test_ctc_vocabulary_requires_blank_at_zero_and_speaker_change() -> None:
    invalid = (
        ("word", "<blank>", "<spk>"),
        ("<blank>", "word", "other"),
        ("<blank>", "<spk>", "<spk>"),
        ("<blank>", "<spk>"),
    )
    for tokens in invalid:
        with pytest.raises(ValueError, match="vocabulary"):
            ctc.CtcVocabulary(tokens)


def test_vocabulary_encoding_is_stable_under_repeated_use() -> None:
    vocab = vocabulary()
    first = vocab.encode(("mava", "sito", "mava"))
    second = vocab.encode(("mava", "sito", "mava"))
    assert first == second == (2, 1, 2)


def test_vocabulary_decodes_empty_sequence_without_special_cases() -> None:
    assert vocabulary().decode(()) == ()


def test_ctc_collapse_keeps_repeats_separated_by_blank() -> None:
    assert ctc.collapse_ctc_path((0, 1, 1, 0, 1, 2, 2, 0)) == (1, 1, 2)


def test_greedy_decode_returns_collapsed_path_and_score() -> None:
    scores = np.log(np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]))
    result = ctc.greedy_decode(scores)
    assert result.path == (0, 1, 1, 0)
    assert result.token_ids == (1,)
    assert result.score == pytest.approx(np.log(0.8**3 * 0.7))


def test_ctc_collapse_matches_manual_oracle_for_short_paths() -> None:
    from itertools import product

    def oracle(path: tuple[int, ...]) -> tuple[int, ...]:
        deduplicated = tuple(
            value for index, value in enumerate(path) if index == 0 or value != path[index - 1]
        )
        return tuple(value for value in deduplicated if value != 0)

    for length in range(5):
        for path in product(range(3), repeat=length):
            assert ctc.collapse_ctc_path(path) == oracle(path)


def test_blank_only_greedy_path_decodes_to_empty_transcript() -> None:
    result = ctc.greedy_decode(np.array([[0.0, -2.0], [0.0, -1.0]]))
    assert result.token_ids == () and result.path == (0, 0)


def test_greedy_ties_choose_lowest_token_identifier() -> None:
    result = ctc.greedy_decode(np.zeros((2, 3)))
    assert result.path == (0, 0)


def test_greedy_decode_validates_score_geometry_and_blank() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        ctc.greedy_decode(np.empty((0, 2)))
    with pytest.raises(ValueError, match="invalid"):
        ctc.greedy_decode(np.ones((2, 2)), blank_id=2)


def test_nbest_decode_recovers_alternative_token_sequences() -> None:
    scores = np.log(np.array([[0.6, 0.3, 0.1], [0.4, 0.35, 0.25]]))
    candidates = ctc.nbest_decode(scores, beam_size=3)
    assert candidates[0].token_ids == ()
    assert {item.token_ids for item in candidates} == {(), (1,), (2,)}
    assert [item.score for item in candidates] == sorted(
        (item.score for item in candidates), reverse=True
    )


def test_nbest_decode_uses_lexical_ties() -> None:
    candidates = ctc.nbest_decode(np.zeros((1, 3)), beam_size=3)
    assert [item.token_ids for item in candidates] == [(), (1,), (2,)]


def test_wide_beam_matches_brute_force_max_path_per_transcript() -> None:
    from itertools import product

    scores = np.log(np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.4, 0.2, 0.4]]))
    expected: dict[tuple[int, ...], float] = {}
    for path in product(range(3), repeat=3):
        transcript = ctc.collapse_ctc_path(path)
        score = sum(float(scores[frame, token]) for frame, token in enumerate(path))
        expected[transcript] = max(expected.get(transcript, -np.inf), score)
    observed = ctc.nbest_decode(scores, beam_size=len(expected))
    assert [item.token_ids for item in observed] == [
        item[0] for item in sorted(expected.items(), key=lambda item: (-item[1], item[0]))
    ]
    np.testing.assert_allclose(
        [item.score for item in observed], sorted(expected.values(), reverse=True)
    )


def test_nbest_first_candidate_matches_greedy_when_beam_is_wide() -> None:
    scores = np.log(np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1]]))
    assert ctc.nbest_decode(scores, beam_size=8)[0].token_ids == ctc.greedy_decode(scores).token_ids


def test_nbest_decode_validates_beam_controls() -> None:
    with pytest.raises(ValueError, match="controls"):
        ctc.nbest_decode(np.ones((1, 2)), beam_size=0)
    with pytest.raises(ValueError, match="controls"):
        ctc.nbest_decode(np.ones((1, 2)), blank_id=4)


# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
# -- extensions --
