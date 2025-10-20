from __future__ import annotations

import numpy as np

from choravox import embeddings


def test_sufficient_statistics_match_hand_computed_assignment() -> None:
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    responsibilities = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    stats = embeddings.accumulate_statistics(features, responsibilities)
    np.testing.assert_array_equal(stats.counts, [1.5, 1.5])
    np.testing.assert_array_equal(stats.first_order, [[2.5, 4.0], [6.5, 8.0]])
    np.testing.assert_array_equal(stats.second_order, [[5.5, 12.0], [29.5, 44.0]])


def test_empty_statistics_preserve_component_and_feature_shape() -> None:
    stats = embeddings.accumulate_statistics(np.empty((0, 3)), np.empty((0, 2)))
    assert stats.counts.shape == (2,)
    assert stats.first_order.shape == stats.second_order.shape == (2, 3)


# -- extensions --
