"""
tests/test_portfolio_construction.py -- Tests for portfolio construction modules.

Modules under test:
  - execution/portfolio_construction/mean_variance.py
  - execution/portfolio_construction/risk_parity.py

Covers:
  - MeanVarianceOptimizer: max Sharpe, min variance, efficient frontier, Black-Litterman
  - PortfolioConstraints: bounds, sector, turnover
  - RiskParityOptimizer: ERC, risk budgeting, vol parity
  - CovarianceEstimator: sample, Ledoit-Wolf, EWMA, denoised
  - Numerical properties: weights sum to 1, non-negative, within bounds
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from execution.portfolio_construction.mean_variance import (
    MeanVarianceOptimizer,
    PortfolioConstraints,
    _ensure_psd,
)
from execution.portfolio_construction.risk_parity import (
    RiskParityOptimizer,
    CovarianceEstimator,
    RiskContribution,
    _risk_contributions_raw,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cov(n: int = 4, rng_seed: int = 42) -> np.ndarray:
    """Generate a random PSD covariance matrix of size n x n."""
    rng = np.random.default_rng(rng_seed)
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n
    cov += np.eye(n) * 0.01  -- ensure positive definite
    return cov


def make_mu(n: int = 4, rng_seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.uniform(0.05, 0.20, size=n)


def make_returns_df(n_assets: int = 4, n_obs: int = 200, rng_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    data = rng.standard_normal((n_obs, n_assets)) * 0.01
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)


# ---------------------------------------------------------------------------
# _ensure_psd helper
# ---------------------------------------------------------------------------


def test_ensure_psd_makes_positive_definite():
    """A near-singular matrix becomes PSD after projection."""
    A = np.array([[1.0, 0.9999], [0.9999, 1.0]])
    psd = _ensure_psd(A)
    eigvals = np.linalg.eigvalsh(psd)
    assert np.all(eigvals >= 0.0)


def test_ensure_psd_identity_unchanged():
    I = np.eye(4)
    result = _ensure_psd(I)
    np.testing.assert_allclose(result, I, atol=1e-8)


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer -- min_variance
# ---------------------------------------------------------------------------


def test_min_variance_weights_sum_to_one():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    w = opt.min_variance(cov, bounds=(0.05, 0.50))
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)


def test_min_variance_weights_within_bounds():
    opt = MeanVarianceOptimizer()
    cov = make_cov(5)
    bounds = (0.05, 0.40)
    w = opt.min_variance(cov, bounds=bounds)
    assert np.all(w >= bounds[0] - 1e-6)
    assert np.all(w <= bounds[1] + 1e-6)


def test_min_variance_minimizes_portfolio_vol():
    """Min variance portfolio should have lower vol than equal weight."""
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    w_mv = opt.min_variance(cov, bounds=(0.0, 1.0))
    w_eq = np.full(4, 0.25)
    vol_mv = opt.portfolio_vol(w_mv, cov)
    vol_eq = opt.portfolio_vol(w_eq, cov)
    assert vol_mv <= vol_eq + 1e-6


def test_min_variance_long_only_no_short():
    opt = MeanVarianceOptimizer()
    cov = make_cov(6)
    w = opt.min_variance(cov, bounds=(0.0, 0.50))
    assert np.all(w >= -1e-8)


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer -- max_sharpe
# ---------------------------------------------------------------------------


def test_max_sharpe_weights_sum_to_one():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = make_mu(4)
    w = opt.max_sharpe(mu, cov, rf=0.02, bounds=(0.0, 0.40))
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)


def test_max_sharpe_higher_sharpe_than_equal_weight():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = make_mu(4)
    w_ms = opt.max_sharpe(mu, cov, rf=0.02, bounds=(0.0, 0.50))
    w_eq = np.full(4, 0.25)
    sharpe_ms = opt.portfolio_sharpe(w_ms, mu, cov, rf=0.02)
    sharpe_eq = opt.portfolio_sharpe(w_eq, mu, cov, rf=0.02)
    assert sharpe_ms >= sharpe_eq - 1e-4


def test_max_sharpe_all_negative_mu_falls_back_to_min_var():
    """When all returns <= rf, falls back to min variance with warning."""
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = np.array([-0.05, -0.10, -0.02, -0.08])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        weights = opt.max_sharpe(mu, cov, rf=0.0, bounds=(0.0, 0.50))
        assert any("min-variance" in str(warning.message).lower() for warning in w)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)


def test_max_sharpe_with_sector_constraints():
    """Max Sharpe with sector constraint should respect per-sector cap."""
    opt = MeanVarianceOptimizer()
    n = 6
    cov = make_cov(n)
    mu = np.array([0.12, 0.15, 0.08, 0.20, 0.10, 0.13])
    constraints = PortfolioConstraints(
        max_weight=0.30,
        max_sector_weight={"tech": 0.40},
        asset_sector_map={0: "tech", 1: "tech", 2: "finance", 3: "finance",
                          4: "health", 5: "health"},
    )
    w = opt.max_sharpe(mu, cov, rf=0.02, constraints=constraints)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)
    tech_weight = w[0] + w[1]
    assert tech_weight <= 0.40 + 1e-5


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer -- efficient_frontier
# ---------------------------------------------------------------------------


def test_efficient_frontier_returns_list():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = make_mu(4)
    frontier = opt.efficient_frontier(mu, cov, n_points=10, bounds=(0.0, 0.50))
    assert len(frontier) >= 1


def test_efficient_frontier_returns_sorted_ascending():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = make_mu(4)
    frontier = opt.efficient_frontier(mu, cov, n_points=20, bounds=(0.0, 0.40))
    rets = [f[0] for f in frontier]
    assert rets == sorted(rets)


def test_efficient_frontier_weights_sum_to_one():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = make_mu(4)
    frontier = opt.efficient_frontier(mu, cov, n_points=5, bounds=(0.0, 0.50))
    for ret, vol, w in frontier:
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer -- max_return_for_vol
# ---------------------------------------------------------------------------


def test_max_return_for_vol_respects_vol_target():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = make_mu(4)
    target_vol = 0.15
    w = opt.max_return_for_vol(mu, cov, target_vol=target_vol, bounds=(0.0, 0.50))
    actual_vol = opt.portfolio_vol(w, cov)
    assert actual_vol <= target_vol + 1e-4


def test_max_return_for_vol_weights_sum_to_one():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu = make_mu(4)
    w = opt.max_return_for_vol(mu, cov, target_vol=0.20)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer -- Black-Litterman
# ---------------------------------------------------------------------------


def test_black_litterman_no_views_returns_prior():
    opt = MeanVarianceOptimizer()
    cov = make_cov(4)
    mu_prior = make_mu(4)
    mu_bl, cov_bl = opt.black_litterman(mu_prior, cov, views=[], view_confidences=np.array([]))
    np.testing.assert_allclose(mu_bl, mu_prior, atol=1e-10)
    np.testing.assert_allclose(cov_bl, cov, atol=1e-8)


def test_black_litterman_view_shifts_posterior_toward_view():
    """A positive view on asset 0 should increase its posterior return."""
    opt = MeanVarianceOptimizer()
    n = 4
    cov = make_cov(n)
    mu_prior = np.array([0.08, 0.10, 0.12, 0.07])
    view_vec = np.zeros(n)
    view_vec[0] = 1.0  -- view on asset 0
    views = [(view_vec, 0.25)]  -- predict 25% return for asset 0
    confidences = np.array([0.70])
    mu_bl, _ = opt.black_litterman(mu_prior, cov, views, confidences)
    assert mu_bl[0] > mu_prior[0]  -- view pulled posterior up


# ---------------------------------------------------------------------------
# PortfolioConstraints
# ---------------------------------------------------------------------------


def test_portfolio_constraints_build_bounds_long_only():
    pc = PortfolioConstraints(max_weight=0.30, min_weight=0.0, long_only=True)
    bounds = pc.build_bounds(5)
    assert len(bounds) == 5
    for lo, hi in bounds:
        assert lo == 0.0
        assert hi == 0.30


def test_portfolio_constraints_turnover_constraint_applied():
    """When max_turnover is set, the scipy constraint should include turnover."""
    current = np.array([0.25, 0.25, 0.25, 0.25])
    pc = PortfolioConstraints(max_turnover=0.10, current_weights=current)
    constrs = pc.build_scipy_constraints(4)
    -- At least sum-to-1 + turnover
    assert len(constrs) >= 2


def test_portfolio_constraints_sector_constraint_included():
    pc = PortfolioConstraints(
        max_sector_weight={"tech": 0.40},
        asset_sector_map={0: "tech", 1: "tech", 2: "other"},
    )
    constrs = pc.build_scipy_constraints(3)
    -- sum-to-1 + sector
    assert len(constrs) >= 2


# ---------------------------------------------------------------------------
# RiskParityOptimizer -- equal risk contribution
# ---------------------------------------------------------------------------


def test_erc_weights_sum_to_one():
    opt = RiskParityOptimizer()
    cov = make_cov(4)
    w = opt.equal_risk_contribution(cov, bounds=(0.01, 0.50))
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)


def test_erc_risk_contributions_approximately_equal():
    """ERC property: all RC_i should be approximately equal (sum to 1/n each)."""
    opt = RiskParityOptimizer()
    cov = make_cov(4)
    w = opt.equal_risk_contribution(cov, bounds=(0.01, 0.50))
    rc = _risk_contributions_raw(w, cov)
    -- Each RC should be close to 0.25 (= 1/4)
    target = 1.0 / 4
    for r in rc:
        assert abs(r - target) < 0.10  -- within 10% of target


def test_erc_two_asset_equal_vol():
    """Two assets with equal vol and zero correlation -> equal weights."""
    opt = RiskParityOptimizer()
    cov = np.array([[0.04, 0.0], [0.0, 0.04]])
    w = opt.equal_risk_contribution(cov, bounds=(0.01, 0.99))
    np.testing.assert_allclose(w[0], w[1], atol=0.05)


def test_erc_weights_within_bounds():
    opt = RiskParityOptimizer()
    cov = make_cov(5)
    bounds = (0.05, 0.40)
    w = opt.equal_risk_contribution(cov, bounds=bounds)
    assert np.all(w >= bounds[0] - 1e-6)
    assert np.all(w <= bounds[1] + 1e-6)


# ---------------------------------------------------------------------------
# RiskParityOptimizer -- risk budgeting
# ---------------------------------------------------------------------------


def test_risk_budgeted_weights_sum_to_one():
    opt = RiskParityOptimizer()
    cov = make_cov(4)
    budgets = np.array([0.40, 0.30, 0.20, 0.10])
    w = opt.risk_budgeted(cov, budgets, bounds=(0.01, 0.50))
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)


def test_risk_budgeted_unequal_budgets_skew_weights():
    """Asset with 50% budget should have higher weight than asset with 10%."""
    opt = RiskParityOptimizer()
    -- Diagonal cov so interpretation is clean
    cov = np.diag([0.02, 0.02, 0.02, 0.02])
    budgets = np.array([0.50, 0.20, 0.20, 0.10])
    w = opt.risk_budgeted(cov, budgets, bounds=(0.01, 0.90))
    assert w[0] > w[3]


def test_risk_budgeted_wrong_budget_length_raises():
    opt = RiskParityOptimizer()
    cov = make_cov(4)
    with pytest.raises(ValueError, match="budgets length"):
        opt.risk_budgeted(cov, np.array([0.5, 0.5]))  -- only 2 budgets for 4 assets


# ---------------------------------------------------------------------------
# RiskParityOptimizer -- vol parity
# ---------------------------------------------------------------------------


def test_vol_parity_inverse_vol_weights():
    opt = RiskParityOptimizer()
    vols = np.array([0.10, 0.20, 0.40])
    w = opt.vol_parity(vols)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-10)
    -- w proportional to 1/vol: 1/0.1 : 1/0.2 : 1/0.4 = 10 : 5 : 2.5 -> 17.5 total
    expected = np.array([10, 5, 2.5]) / 17.5
    np.testing.assert_allclose(w, expected, rtol=1e-6)


def test_vol_parity_zero_vol_raises():
    opt = RiskParityOptimizer()
    with pytest.raises(ValueError):
        opt.vol_parity(np.array([0.10, 0.0, 0.20]))


# ---------------------------------------------------------------------------
# RiskContribution decomposition
# ---------------------------------------------------------------------------


def test_risk_contributions_sum_to_one():
    opt = RiskParityOptimizer()
    cov = make_cov(5)
    w = np.full(5, 0.20)
    rcs = opt.compute_risk_contributions(w, cov)
    total = sum(r.percent_risk_contribution for r in rcs)
    np.testing.assert_allclose(total, 1.0, atol=1e-6)


def test_risk_contributions_labels_match():
    opt = RiskParityOptimizer()
    cov = make_cov(3)
    w = np.array([0.3, 0.4, 0.3])
    names = ["A", "B", "C"]
    rcs = opt.compute_risk_contributions(w, cov, asset_names=names)
    assert [r.asset for r in rcs] == names


def test_effective_n_equals_n_for_equal_weight():
    opt = RiskParityOptimizer()
    n = 5
    w = np.full(n, 1.0 / n)
    eff_n = opt.get_effective_n(w)
    np.testing.assert_allclose(eff_n, float(n), rtol=1e-5)


def test_effective_n_equals_one_for_concentrated():
    opt = RiskParityOptimizer()
    w = np.array([1.0, 0.0, 0.0, 0.0])
    eff_n = opt.get_effective_n(w)
    np.testing.assert_allclose(eff_n, 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# CovarianceEstimator
# ---------------------------------------------------------------------------


def test_sample_cov_shape():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=4, n_obs=150)
    cov = est.sample_cov(df, min_periods=60)
    assert cov.shape == (4, 4)


def test_sample_cov_is_symmetric():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=4, n_obs=150)
    cov = est.sample_cov(df, min_periods=60)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_sample_cov_insufficient_data_raises():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=4, n_obs=30)
    with pytest.raises(ValueError, match="at least"):
        est.sample_cov(df, min_periods=60)


def test_ledoit_wolf_shape_and_symmetry():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=5, n_obs=200)
    cov = est.ledoit_wolf(df)
    assert cov.shape == (5, 5)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)


def test_ledoit_wolf_psd():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=5, n_obs=100)
    cov = est.ledoit_wolf(df)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-9)


def test_ewma_cov_shape():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=4, n_obs=100)
    cov = est.ewma_cov(df, lambda_=0.94)
    assert cov.shape == (4, 4)


def test_ewma_cov_is_psd():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=4, n_obs=100)
    cov = est.ewma_cov(df, lambda_=0.94)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-9)


def test_denoised_cov_shape():
    est = CovarianceEstimator()
    df = make_returns_df(n_assets=6, n_obs=300)
    cov = est.denoised_cov(df, clip_pct=0.90)
    assert cov.shape == (6, 6)


# ---------------------------------------------------------------------------
# Portfolio statistics helpers
# ---------------------------------------------------------------------------


def test_portfolio_sharpe_positive_for_positive_return():
    opt = MeanVarianceOptimizer()
    mu = np.array([0.15, 0.12])
    cov = np.array([[0.04, 0.01], [0.01, 0.03]])
    w = np.array([0.5, 0.5])
    sharpe = opt.portfolio_sharpe(w, mu, cov, rf=0.02)
    assert sharpe > 0.0


def test_portfolio_vol_matches_manual_calc():
    opt = MeanVarianceOptimizer()
    w = np.array([1.0, 0.0])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    vol = opt.portfolio_vol(w, cov)
    np.testing.assert_allclose(vol, 0.20, rtol=1e-9)


def test_portfolio_return_is_weighted_sum():
    opt = MeanVarianceOptimizer()
    mu = np.array([0.10, 0.20])
    w = np.array([0.6, 0.4])
    ret = opt.portfolio_return(w, mu)
    np.testing.assert_allclose(ret, 0.6 * 0.10 + 0.4 * 0.20, rtol=1e-10)


# ---------------------------------------------------------------------------
# PortfolioConstructor integration test (if available)
# ---------------------------------------------------------------------------


def test_portfolio_constructor_signal_to_weights():
    """
    Smoke test: simulate a signal -> weights pipeline using optimizer directly.
    This replaces a full PortfolioConstructor integration that may not exist yet.
    """
    rng = np.random.default_rng(99)
    n = 6
    cov = make_cov(n)
    mu = rng.uniform(0.05, 0.20, n)

    opt = MeanVarianceOptimizer()
    pc = PortfolioConstraints(
        max_weight=0.25,
        min_weight=0.01,
        long_only=True,
    )
    w = opt.max_sharpe(mu, cov, rf=0.02, constraints=pc)

    -- Basic sanity checks
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)
    assert np.all(w >= 0.0 - 1e-6)
    assert np.all(w <= 0.25 + 1e-5)
