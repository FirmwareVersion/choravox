from __future__ import annotations

import numpy as np

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


# -- extensions --
