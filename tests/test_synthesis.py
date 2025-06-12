from __future__ import annotations

import numpy as np
import pytest

from choravox import synthesis


def test_resonator_coefficients_encode_stable_poles() -> None:
    coefficients = synthesis.resonator_coefficients(1000.0, 100.0, 16_000)
    poles = np.roots([1.0, -coefficients.a1, -coefficients.a2])
    assert np.all(np.abs(poles) < 1.0)
    np.testing.assert_allclose(np.abs(poles[0]), np.abs(poles[1]), atol=1e-15)


def test_narrower_resonator_places_poles_nearer_unit_circle() -> None:
    narrow = synthesis.resonator_coefficients(1000.0, 50.0, 16_000)
    wide = synthesis.resonator_coefficients(1000.0, 300.0, 16_000)
    assert abs(narrow.a2) > abs(wide.a2)


def test_resonator_rejects_frequency_at_or_above_nyquist() -> None:
    with pytest.raises(ValueError, match="frequency"):
        synthesis.resonator_coefficients(8000.0, 100.0, 16_000)
    with pytest.raises(ValueError, match="frequency"):
        synthesis.resonator_coefficients(0.0, 100.0, 16_000)


def test_resonator_rejects_nonpositive_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        synthesis.resonator_coefficients(1000.0, 0.0, 16_000)


def test_resonator_coefficients_repeat_bit_exactly() -> None:
    first = synthesis.resonator_coefficients(730.0, 80.0, 16_000)
    second = synthesis.resonator_coefficients(730.0, 80.0, 16_000)
    assert first == second


def test_resonator_impulse_starts_at_feedforward_gain() -> None:
    coefficients = synthesis.resonator_coefficients(1000.0, 200.0, 8000)
    impulse = np.zeros(20)
    impulse[0] = 1.0
    response = synthesis.apply_resonator(impulse, coefficients)
    assert response[0] == coefficients.b0
    assert response[1] == pytest.approx(coefficients.a1 * coefficients.b0)


def test_resonator_preserves_samples_first_channels() -> None:
    stereo = np.column_stack((np.ones(10), -np.ones(10)))
    output = synthesis.apply_resonator(stereo, synthesis.resonator_coefficients(500.0, 100.0, 8000))
    assert output.shape == stereo.shape
    np.testing.assert_allclose(output[:, 0], -output[:, 1])


# -- extensions --
