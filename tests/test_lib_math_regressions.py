"""
Regression tests for lib/math bugs found by ruff (syntax / undefined names).

- causal_inference.py had `x**2.sum()`, which is a SyntaxError, so the whole
  module failed to import.
- volatility_models.py called an unimported `CubicSpline`, so the term
  structure and surface interpolation raised NameError.
"""

from __future__ import annotations

import numpy as np

from lib.math.causal_inference import synthetic_control, two_stage_least_squares
from lib.math.volatility_models import vol_term_structure


def test_two_stage_least_squares_recovers_effect():
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    u = rng.normal(size=n)  # confounder
    d = 0.8 * z + u + rng.normal(scale=0.5, size=n)
    y = 2.0 * d + 1.5 * u + rng.normal(scale=0.5, size=n)

    out = two_stage_least_squares(y, d, z)

    assert abs(out["beta_iv"] - 2.0) < 0.15
    assert out["first_stage_f"] > 10
    assert out["weak_instrument"] is False


def test_synthetic_control_returns_finite_pre_rmse():
    rng = np.random.default_rng(1)
    t, n = 60, 5
    y = np.cumsum(rng.normal(size=(t, n)), axis=0)
    y[40:, 0] += 5.0

    out = synthetic_control(y, treated_unit=0, treatment_start=40)

    assert np.isfinite(out["pre_rmse"])
    assert out["avg_treatment_effect"] > 0


def test_vol_term_structure_interpolates():
    mats = np.array([0.1, 0.25, 0.5, 1.0])
    vols = np.array([0.30, 0.28, 0.26, 0.25])

    at_nodes = vol_term_structure(vols.copy(), mats)
    np.testing.assert_allclose(at_nodes, vols, rtol=1e-6)

    mid = vol_term_structure(vols.copy(), mats, np.array([0.75]))
    assert 0.24 < mid[0] < 0.27
