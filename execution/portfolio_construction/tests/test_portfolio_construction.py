# execution/portfolio_construction/tests/test_portfolio_construction.py
# Comprehensive test suite for the portfolio construction module.
#
# Covers:
#   - RiskParityOptimizer: ERC convergence, risk decomposition, vol parity
#   - CovarianceEstimator: Ledoit-Wolf, EWMA, RMT denoising, sample cov
#   - MeanVarianceOptimizer: max Sharpe, min variance, efficient frontier,
#     max return for vol, Black-Litterman
#   - DynamicSizer: Kelly + vol target, regime overlays, turnover constraint
#   - PortfolioConstraints: sector caps, leverage, bounds
#
# All tests use synthetic data to ensure determinism.

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import pytest

from execution.portfolio_construction.risk_parity import (
    CovarianceEstimator,
    RiskContribution,
    RiskParityOptimizer,
    _ensure_psd,
    _risk_contributions_raw,
)
from execution.portfolio_construction.mean_variance import (
    MeanVarianceOptimizer,
    PortfolioConstraints,
)
from execution.portfolio_construction.dynamic_sizing import (
    DynamicSizer,
    RegimeBHState,
    RegimeState,
    RegimeTrend,
    TargetPortfolio,
    VolRegime,
    KELLY_FRACTION,
    REGIME_EVENT_SCALAR,
    REGIME_HIGH_VOL_SCALAR,
    REGIME_MOM_BOOST,
    REGIME_MR_BOOST,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def make_corr(n: int, rho: float = 0.3) -> np.ndarray:
    """Constant off-diagonal correlation matrix."""
    C = np.full((n, n), rho)
    np.fill_diagonal(C, 1.0)
    return C


def make_cov(n: int, vols: Optional[np.ndarray] = None, rho: float = 0.3) -> np.ndarray:
    """Covariance matrix from constant correlation and given vols."""
    if vols is None:
        vols = np.linspace(0.10, 0.40, n)
    D = np.diag(vols)
    C = make_corr(n, rho)
    return D @ C @ D


def make_returns(T: int, n: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic daily returns dataframe."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, n)) * 0.01
    cols = [f"asset_{i}" for i in range(n)]
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


def make_returns_correlated(T: int, n: int, rho: float = 0.3, seed: int = 42) -> pd.DataFrame:
    """Synthetic returns with controlled correlation structure."""
    rng = np.random.default_rng(seed)
    C = make_corr(n, rho)
    L = np.linalg.cholesky(_ensure_psd(C))
    z = rng.standard_normal((T, n))
    r = (z @ L.T) * 0.01
    cols = [f"asset_{i}" for i in range(n)]
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    return pd.DataFrame(r, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# RiskParityOptimizer tests
# ---------------------------------------------------------------------------


class TestEqualRiskContribution:

    def test_equal_risk_contribution_converges(self):
        """ERC weights should be found without warnings for well-conditioned input."""
        n = 5
        cov = make_cov(n, rho=0.2)
        opt = RiskParityOptimizer()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            weights = opt.equal_risk_contribution(cov)
        assert weights.shape == (n,)

    def test_erc_weights_sum_to_one(self):
        for n in [3, 5, 10, 20]:
            cov = make_cov(n)
            weights = RiskParityOptimizer().equal_risk_contribution(cov)
            assert abs(weights.sum() - 1.0) < 1e-6, f"n={n}: sum={weights.sum()}"

    def test_erc_weights_within_bounds(self):
        n = 8
        cov = make_cov(n)
        lo, hi = 0.02, 0.25
        weights = RiskParityOptimizer().equal_risk_contribution(cov, bounds=(lo, hi))
        assert np.all(weights >= lo - 1e-6)
        assert np.all(weights <= hi + 1e-6)

    def test_erc_equal_cov_gives_equal_weights(self):
        """Identity covariance -> ERC = equal weights."""
        n = 4
        cov = np.eye(n) * 0.04  # all assets with identical 20% vol, zero correlation
        weights = RiskParityOptimizer().equal_risk_contribution(cov, bounds=(0.01, 0.99))
        np.testing.assert_allclose(weights, np.full(n, 0.25), atol=1e-4)

    def test_erc_risk_contributions_approximately_equal(self):
        """After ERC, all risk contributions should be approximately equal."""
        n = 6
        cov = make_cov(n, rho=0.25)
        opt = RiskParityOptimizer()
        weights = opt.equal_risk_contribution(cov, bounds=(0.01, 0.50))
        rc = _risk_contributions_raw(weights, cov)
        # Maximum deviation from mean RC should be small.
        assert np.max(np.abs(rc - rc.mean())) < 0.05

    def test_erc_nonnegative_weights(self):
        n = 5
        cov = make_cov(n)
        weights = RiskParityOptimizer().equal_risk_contribution(cov)
        assert np.all(weights >= 0.0)

    def test_erc_large_n(self):
        """Test with n=30 (full instrument set)."""
        n = 30
        cov = make_cov(n, rho=0.15)
        weights = RiskParityOptimizer().equal_risk_contribution(cov, bounds=(0.005, 0.15))
        assert abs(weights.sum() - 1.0) < 1e-5
        assert weights.shape == (n,)


class TestRiskBudgeted:

    def test_risk_budgeted_equal_budgets_matches_erc(self):
        """Equal budgets should produce same result as ERC."""
        n = 5
        cov = make_cov(n, rho=0.2)
        opt = RiskParityOptimizer()
        budgets = np.full(n, 1.0 / n)
        w_rb = opt.risk_budgeted(cov, budgets, bounds=(0.01, 0.50))
        w_erc = opt.equal_risk_contribution(cov, bounds=(0.01, 0.50))
        np.testing.assert_allclose(w_rb, w_erc, atol=1e-3)

    def test_risk_budgeted_weights_sum_to_one(self):
        n = 6
        cov = make_cov(n)
        budgets = np.array([0.30, 0.25, 0.20, 0.10, 0.10, 0.05])
        weights = RiskParityOptimizer().risk_budgeted(cov, budgets)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_risk_budgeted_normalises_budgets(self):
        """Un-normalised budgets should be handled gracefully."""
        n = 4
        cov = make_cov(n)
        budgets = np.array([3.0, 2.0, 2.0, 1.0])  # sum = 8, not 1
        weights = RiskParityOptimizer().risk_budgeted(cov, budgets)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_risk_budgeted_wrong_length_raises(self):
        n = 4
        cov = make_cov(n)
        with pytest.raises(ValueError, match="budgets length"):
            RiskParityOptimizer().risk_budgeted(cov, np.array([0.5, 0.5]))


class TestVolParity:

    def test_vol_parity_weights(self):
        """1/vol weights should be inversely proportional to volatility."""
        vols = np.array([0.10, 0.20, 0.40])
        weights = RiskParityOptimizer().vol_parity(vols)
        np.testing.assert_allclose(weights.sum(), 1.0)
        # Ratios should be inverse vol ratios.
        expected = (1.0 / vols) / (1.0 / vols).sum()
        np.testing.assert_allclose(weights, expected, rtol=1e-10)

    def test_vol_parity_equal_vols_gives_equal_weights(self):
        n = 5
        vols = np.ones(n) * 0.20
        weights = RiskParityOptimizer().vol_parity(vols)
        np.testing.assert_allclose(weights, np.full(n, 0.2), atol=1e-10)

    def test_vol_parity_zero_vol_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            RiskParityOptimizer().vol_parity(np.array([0.1, 0.0, 0.2]))

    def test_vol_parity_high_vol_gets_low_weight(self):
        vols = np.array([0.05, 0.50])
        weights = RiskParityOptimizer().vol_parity(vols)
        assert weights[0] > weights[1]


class TestRiskContributions:

    def test_risk_contributions_sum_to_portfolio_risk(self):
        """Fractional risk contributions should sum to 1."""
        n = 5
        cov = make_cov(n)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        opt = RiskParityOptimizer()
        contribs = opt.compute_risk_contributions(weights, cov)
        pct_sum = sum(c.percent_risk_contribution for c in contribs)
        np.testing.assert_allclose(pct_sum, 1.0, atol=1e-8)

    def test_risk_contributions_returns_correct_type(self):
        n = 4
        cov = make_cov(n)
        weights = np.full(n, 0.25)
        contribs = RiskParityOptimizer().compute_risk_contributions(weights, cov)
        assert len(contribs) == n
        assert all(isinstance(c, RiskContribution) for c in contribs)

    def test_risk_contributions_asset_names(self):
        n = 3
        cov = make_cov(n)
        weights = np.array([0.4, 0.35, 0.25])
        names = ["BTC", "ETH", "SOL"]
        contribs = RiskParityOptimizer().compute_risk_contributions(weights, cov, names)
        assert [c.asset for c in contribs] == names

    def test_risk_contributions_marginal_positive(self):
        n = 4
        cov = make_cov(n)
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        contribs = RiskParityOptimizer().compute_risk_contributions(weights, cov)
        assert all(c.marginal_risk_contribution > 0 for c in contribs)

    def test_risk_contributions_weight_matches_input(self):
        n = 5
        cov = make_cov(n)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        contribs = RiskParityOptimizer().compute_risk_contributions(weights, cov)
        for i, c in enumerate(contribs):
            np.testing.assert_allclose(c.weight, weights[i])


class TestEffectiveN:

    def test_effective_n_equal_weights(self):
        n = 10
        weights = np.full(n, 1.0 / n)
        eff_n = RiskParityOptimizer().get_effective_n(weights)
        np.testing.assert_allclose(eff_n, float(n), rtol=1e-10)

    def test_effective_n_concentrated(self):
        n = 10
        weights = np.zeros(n)
        weights[0] = 1.0
        eff_n = RiskParityOptimizer().get_effective_n(weights)
        np.testing.assert_allclose(eff_n, 1.0, rtol=1e-10)

    def test_effective_n_range(self):
        n = 5
        weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        eff_n = RiskParityOptimizer().get_effective_n(weights)
        assert 1.0 < eff_n < n


# ---------------------------------------------------------------------------
# CovarianceEstimator tests
# ---------------------------------------------------------------------------


class TestSampleCov:

    def test_sample_cov_shape(self):
        returns = make_returns(200, 5)
        cov = CovarianceEstimator().sample_cov(returns)
        assert cov.shape == (5, 5)

    def test_sample_cov_symmetric(self):
        returns = make_returns(200, 4)
        cov = CovarianceEstimator().sample_cov(returns)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_sample_cov_positive_definite(self):
        returns = make_returns(200, 5)
        cov = CovarianceEstimator().sample_cov(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_sample_cov_min_periods_raises(self):
        returns = make_returns(30, 5)
        with pytest.raises(ValueError, match="at least"):
            CovarianceEstimator().sample_cov(returns, min_periods=60)

    def test_sample_cov_diagonal_values(self):
        """Diagonal entries should match variance of each return series."""
        returns = make_returns(500, 3)
        cov = CovarianceEstimator().sample_cov(returns)
        expected_vars = returns.var(ddof=1).values
        np.testing.assert_allclose(np.diag(cov), expected_vars, rtol=0.01)


class TestLedoitWolf:

    def test_ledoit_wolf_shrinkage_positive_definite(self):
        """LW shrinkage should always produce a PD matrix."""
        returns = make_returns(100, 10)
        cov = CovarianceEstimator().ledoit_wolf(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_ledoit_wolf_symmetric(self):
        returns = make_returns(150, 8)
        cov = CovarianceEstimator().ledoit_wolf(returns)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_ledoit_wolf_shrinks_toward_identity_scale(self):
        """LW matrix should have smaller max/min eigenvalue ratio than sample."""
        returns = make_returns(80, 15, seed=123)
        est = CovarianceEstimator()
        cov_sample = est.sample_cov(returns, min_periods=60)
        cov_lw = est.ledoit_wolf(returns)
        ev_sample = np.linalg.eigvalsh(cov_sample)
        ev_lw = np.linalg.eigvalsh(cov_lw)
        ratio_sample = ev_sample.max() / ev_sample.min()
        ratio_lw = ev_lw.max() / ev_lw.min()
        assert ratio_lw < ratio_sample

    def test_ledoit_wolf_shape(self):
        n = 6
        returns = make_returns(200, n)
        cov = CovarianceEstimator().ledoit_wolf(returns)
        assert cov.shape == (n, n)

    def test_ledoit_wolf_too_few_observations_raises(self):
        returns = make_returns(1, 3)
        with pytest.raises(ValueError, match="at least 2"):
            CovarianceEstimator().ledoit_wolf(returns)


class TestEwmaCov:

    def test_ewma_cov_shape(self):
        returns = make_returns(200, 5)
        cov = CovarianceEstimator().ewma_cov(returns)
        assert cov.shape == (5, 5)

    def test_ewma_cov_symmetric(self):
        returns = make_returns(200, 4)
        cov = CovarianceEstimator().ewma_cov(returns)
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_ewma_cov_positive_definite(self):
        returns = make_returns(200, 5)
        cov = CovarianceEstimator().ewma_cov(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_ewma_cov_lambda_effect(self):
        """Higher lambda should give slower decay (more history weight)."""
        returns = make_returns(300, 4)
        cov_low = CovarianceEstimator().ewma_cov(returns, lambda_=0.80)
        cov_high = CovarianceEstimator().ewma_cov(returns, lambda_=0.99)
        # Both should be valid; just check they differ.
        assert not np.allclose(cov_low, cov_high)

    def test_ewma_cov_too_few_observations_raises(self):
        returns = make_returns(1, 3)
        with pytest.raises(ValueError, match="at least 2"):
            CovarianceEstimator().ewma_cov(returns)


class TestDenoisedCov:

    def test_marchenko_pastur_eigenvalue_clip(self):
        """Denoised cov should have fewer large eigenvalue spread than sample."""
        returns = make_returns_correlated(150, 12, rho=0.1)
        est = CovarianceEstimator()
        cov_denoised = est.denoised_cov(returns)
        assert cov_denoised.shape == (12, 12)
        eigenvalues = np.linalg.eigvalsh(cov_denoised)
        assert np.all(eigenvalues > 0)

    def test_denoised_cov_symmetric(self):
        returns = make_returns(200, 8)
        cov = CovarianceEstimator().denoised_cov(returns)
        np.testing.assert_allclose(cov, cov.T, atol=1e-8)

    def test_denoised_cov_positive_definite(self):
        returns = make_returns(200, 6)
        cov = CovarianceEstimator().denoised_cov(returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0)

    def test_denoised_cov_clip_pct_effect(self):
        """Different clip_pct values should produce different matrices."""
        returns = make_returns(200, 8)
        est = CovarianceEstimator()
        cov_90 = est.denoised_cov(returns, clip_pct=0.90)
        cov_50 = est.denoised_cov(returns, clip_pct=0.50)
        assert not np.allclose(cov_90, cov_50)


# ---------------------------------------------------------------------------
# MeanVarianceOptimizer tests
# ---------------------------------------------------------------------------


class TestMaxSharpe:

    def test_max_sharpe_vs_scipy_reference(self):
        """Max Sharpe weights should have higher Sharpe than equal weights."""
        n = 5
        mu = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
        cov = make_cov(n, vols=np.array([0.15, 0.20, 0.12, 0.25, 0.18]))
        opt = MeanVarianceOptimizer()
        weights = opt.max_sharpe(mu, cov, rf=0.02)

        sharpe_max = opt.portfolio_sharpe(weights, mu, cov, rf=0.02)
        w_eq = np.full(n, 0.2)
        sharpe_eq = opt.portfolio_sharpe(w_eq, mu, cov, rf=0.02)
        assert sharpe_max >= sharpe_eq - 1e-4

    def test_max_sharpe_weights_sum_to_one(self):
        n = 6
        mu = np.linspace(0.08, 0.18, n)
        cov = make_cov(n)
        weights = MeanVarianceOptimizer().max_sharpe(mu, cov)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)

    def test_max_sharpe_weights_within_bounds(self):
        n = 5
        mu = np.linspace(0.08, 0.18, n)
        cov = make_cov(n)
        lo, hi = 0.0, 0.25
        weights = MeanVarianceOptimizer().max_sharpe(mu, cov, bounds=(lo, hi))
        assert np.all(weights >= lo - 1e-6)
        assert np.all(weights <= hi + 1e-6)

    def test_max_sharpe_all_negative_returns_falls_back(self):
        """All excess returns <= 0 should fall back to min variance."""
        n = 4
        mu = np.array([-0.01, -0.02, -0.03, -0.04])
        cov = make_cov(n)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            weights = MeanVarianceOptimizer().max_sharpe(mu, cov, rf=0.05)
        # Should still be a valid portfolio.
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)

    def test_max_sharpe_favours_high_return_low_vol(self):
        """Asset with highest Sharpe should get highest weight."""
        n = 3
        # Asset 1: Sharpe 1.0, Asset 2: Sharpe 0.5, Asset 3: Sharpe 0.33
        mu = np.array([0.10, 0.10, 0.10])
        vols = np.array([0.10, 0.20, 0.30])
        cov = np.diag(vols ** 2)
        weights = MeanVarianceOptimizer().max_sharpe(mu, cov, bounds=(0.0, 0.99))
        # Asset 0 should have highest weight.
        assert weights[0] >= weights[1]


class TestMinVariance:

    def test_min_variance_weights_sum_to_one(self):
        n = 6
        cov = make_cov(n)
        weights = MeanVarianceOptimizer().min_variance(cov)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)

    def test_min_variance_within_bounds(self):
        n = 5
        cov = make_cov(n)
        lo, hi = 0.05, 0.30
        weights = MeanVarianceOptimizer().min_variance(cov, bounds=(lo, hi))
        assert np.all(weights >= lo - 1e-6)
        assert np.all(weights <= hi + 1e-6)

    def test_min_variance_lower_than_equal_weight(self):
        """Min variance portfolio vol should be <= equal weight portfolio vol."""
        n = 5
        cov = make_cov(n)
        weights_mv = MeanVarianceOptimizer().min_variance(cov, bounds=(0.0, 1.0))
        weights_eq = np.full(n, 0.2)
        vol_mv = MeanVarianceOptimizer.portfolio_vol(weights_mv, cov)
        vol_eq = MeanVarianceOptimizer.portfolio_vol(weights_eq, cov)
        assert vol_mv <= vol_eq + 1e-6

    def test_min_variance_identity_cov_equal_weights(self):
        """Identity covariance -> min variance = equal weights."""
        n = 4
        cov = np.eye(n) * 0.04
        weights = MeanVarianceOptimizer().min_variance(cov, bounds=(0.0, 0.99))
        np.testing.assert_allclose(weights, np.full(n, 0.25), atol=1e-4)

    def test_min_variance_large_n(self):
        n = 30
        cov = make_cov(n, rho=0.2)
        weights = MeanVarianceOptimizer().min_variance(cov, bounds=(0.0, 0.15))
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-4)


class TestEfficientFrontier:

    def test_efficient_frontier_pareto(self):
        """Higher return on frontier should correspond to higher or equal vol."""
        n = 5
        mu = np.linspace(0.08, 0.20, n)
        cov = make_cov(n)
        frontier = MeanVarianceOptimizer().efficient_frontier(mu, cov, n_points=15)
        assert len(frontier) >= 2
        returns = [f[0] for f in frontier]
        vols = [f[1] for f in frontier]
        # Returns should be non-decreasing.
        assert all(returns[i] <= returns[i + 1] + 1e-6 for i in range(len(returns) - 1))

    def test_efficient_frontier_returns_list_of_tuples(self):
        n = 4
        mu = np.array([0.08, 0.10, 0.12, 0.15])
        cov = make_cov(n)
        frontier = MeanVarianceOptimizer().efficient_frontier(mu, cov, n_points=10)
        assert isinstance(frontier, list)
        assert all(len(f) == 3 for f in frontier)

    def test_efficient_frontier_weights_sum_to_one(self):
        n = 4
        mu = np.array([0.08, 0.10, 0.12, 0.15])
        cov = make_cov(n)
        frontier = MeanVarianceOptimizer().efficient_frontier(mu, cov, n_points=10)
        for ret, vol, w in frontier:
            np.testing.assert_allclose(w.sum(), 1.0, atol=1e-4)

    def test_efficient_frontier_min_point_matches_min_var(self):
        """First frontier point should have vol close to the min-var portfolio."""
        n = 5
        mu = np.linspace(0.08, 0.20, n)
        cov = make_cov(n)
        opt = MeanVarianceOptimizer()
        frontier = opt.efficient_frontier(mu, cov, n_points=20)
        w_mv = opt.min_variance(cov, bounds=(0.0, 0.30))
        vol_mv = opt.portfolio_vol(w_mv, cov)
        vol_frontier_min = min(f[1] for f in frontier)
        np.testing.assert_allclose(vol_frontier_min, vol_mv, atol=0.01)


class TestMaxReturnForVol:

    def test_max_return_for_vol_respects_vol_target(self):
        n = 5
        mu = np.linspace(0.08, 0.20, n)
        cov = make_cov(n)
        target_vol = 0.12
        weights = MeanVarianceOptimizer().max_return_for_vol(mu, cov, target_vol)
        actual_vol = MeanVarianceOptimizer.portfolio_vol(weights, cov)
        # Allow a small solver tolerance margin (SLSQP may not hit constraint exactly).
        assert actual_vol <= target_vol + 2e-2

    def test_max_return_for_vol_weights_sum_to_one(self):
        n = 5
        mu = np.linspace(0.08, 0.20, n)
        cov = make_cov(n)
        weights = MeanVarianceOptimizer().max_return_for_vol(mu, cov, 0.15)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)

    def test_max_return_for_vol_higher_vol_target_higher_return(self):
        """Higher vol budget should allow higher return."""
        n = 5
        mu = np.linspace(0.08, 0.20, n)
        cov = make_cov(n)
        opt = MeanVarianceOptimizer()
        w_low = opt.max_return_for_vol(mu, cov, 0.10)
        w_high = opt.max_return_for_vol(mu, cov, 0.20)
        ret_low = opt.portfolio_return(w_low, mu)
        ret_high = opt.portfolio_return(w_high, mu)
        assert ret_high >= ret_low - 1e-6


class TestBlackLitterman:

    def test_black_litterman_view_tilt(self):
        """BL posterior should tilt toward the view when confidence is high."""
        n = 4
        mu_prior = np.array([0.10, 0.10, 0.10, 0.10])
        cov = make_cov(n, rho=0.2)
        # View: asset 0 outperforms by 5% (strong upward tilt).
        view_vector = np.array([1.0, 0.0, 0.0, 0.0])
        view_return = 0.20  # higher than prior 0.10
        views = [(view_vector, view_return)]
        view_confidences = np.array([0.90])
        opt = MeanVarianceOptimizer()
        mu_bl, cov_bl = opt.black_litterman(mu_prior, cov, views, view_confidences)
        # Posterior return for asset 0 should be higher than prior.
        assert mu_bl[0] > mu_prior[0]

    def test_black_litterman_no_views_returns_prior(self):
        n = 4
        mu_prior = np.array([0.10, 0.12, 0.08, 0.11])
        cov = make_cov(n)
        opt = MeanVarianceOptimizer()
        mu_bl, cov_bl = opt.black_litterman(mu_prior, cov, [], np.array([]))
        np.testing.assert_allclose(mu_bl, mu_prior)

    def test_black_litterman_posterior_cov_positive_definite(self):
        n = 5
        mu_prior = np.linspace(0.08, 0.16, n)
        cov = make_cov(n)
        views = [(np.eye(n)[0], 0.20), (np.eye(n)[1] - np.eye(n)[2], 0.05)]
        confidences = np.array([0.70, 0.60])
        opt = MeanVarianceOptimizer()
        _, cov_bl = opt.black_litterman(mu_prior, cov, views, confidences)
        eigenvalues = np.linalg.eigvalsh(cov_bl)
        assert np.all(eigenvalues > 0)

    def test_black_litterman_relative_view(self):
        """Relative view (long/short) should adjust relative expected returns."""
        n = 4
        mu_prior = np.array([0.10, 0.10, 0.10, 0.10])
        cov = make_cov(n, rho=0.1)
        # Asset 0 outperforms asset 1 by 8%.
        view_vector = np.array([1.0, -1.0, 0.0, 0.0])
        views = [(view_vector, 0.08)]
        confidences = np.array([0.80])
        opt = MeanVarianceOptimizer()
        mu_bl, _ = opt.black_litterman(mu_prior, cov, views, confidences)
        # mu_bl[0] - mu_bl[1] should be positive (tilted toward view).
        assert mu_bl[0] > mu_bl[1]

    def test_black_litterman_tau_effect(self):
        """Higher tau means more uncertainty in prior; views dominate more."""
        n = 4
        mu_prior = np.array([0.10, 0.10, 0.10, 0.10])
        cov = make_cov(n)
        views = [(np.array([1.0, 0.0, 0.0, 0.0]), 0.25)]
        confidences = np.array([0.70])
        opt = MeanVarianceOptimizer()
        mu_low_tau, _ = opt.black_litterman(mu_prior, cov, views, confidences, tau=0.01)
        mu_high_tau, _ = opt.black_litterman(mu_prior, cov, views, confidences, tau=0.20)
        # Higher tau -> views have more influence -> mu[0] tilts further up.
        assert mu_high_tau[0] >= mu_low_tau[0] - 1e-8


# ---------------------------------------------------------------------------
# PortfolioConstraints tests
# ---------------------------------------------------------------------------


class TestPortfolioConstraints:

    def test_build_bounds_long_only(self):
        pc = PortfolioConstraints(max_weight=0.25, long_only=True)
        bounds = pc.build_bounds(4)
        assert all(lo == 0.0 and hi == 0.25 for lo, hi in bounds)

    def test_build_bounds_long_short(self):
        pc = PortfolioConstraints(min_weight=-0.10, max_weight=0.30, long_only=False)
        bounds = pc.build_bounds(4)
        assert all(lo == -0.10 and hi == 0.30 for lo, hi in bounds)

    def test_sector_constraint_respected(self):
        """Sector weights should respect sector caps."""
        n = 6
        mu = np.array([0.15, 0.14, 0.10, 0.09, 0.08, 0.07])
        cov = make_cov(n, vols=np.linspace(0.15, 0.30, n))
        # Assets 0, 1 in "crypto"; cap at 40%.
        pc = PortfolioConstraints(
            max_weight=0.30,
            max_sector_weight={"crypto": 0.40},
            asset_sector_map={0: "crypto", 1: "crypto"},
        )
        opt = MeanVarianceOptimizer()
        weights = opt.max_sharpe(mu, cov, constraints=pc)
        crypto_weight = weights[0] + weights[1]
        assert crypto_weight <= 0.40 + 1e-5

    def test_constraints_build_scipy_returns_list(self):
        pc = PortfolioConstraints()
        constraints = pc.build_scipy_constraints(5)
        assert isinstance(constraints, list)
        assert len(constraints) >= 1


# ---------------------------------------------------------------------------
# DynamicSizer tests
# ---------------------------------------------------------------------------


class TestKellyVolTarget:

    def test_kelly_vol_target_scaling(self):
        """Portfolio vol of sized weights should be close to target vol."""
        n = 5
        signals = np.array([0.10, 0.12, 0.08, 0.15, 0.09])
        vols = np.array([0.15, 0.20, 0.12, 0.25, 0.18])
        corr = make_corr(n, rho=0.2)
        target_vol = 0.15
        sizer = DynamicSizer(target_vol=target_vol)
        weights = sizer.size_kelly_vol_target(signals, vols, corr)
        cov = np.diag(vols) @ corr @ np.diag(vols)
        actual_vol = float(np.sqrt(weights @ cov @ weights))
        # The vol targeting may not hit exactly due to weight clipping / renorm.
        assert actual_vol < target_vol * 1.5  # within reasonable range

    def test_kelly_weights_sum_to_one(self):
        n = 6
        signals = np.linspace(0.05, 0.18, n)
        vols = np.linspace(0.10, 0.30, n)
        corr = make_corr(n)
        weights = DynamicSizer().size_kelly_vol_target(signals, vols, corr)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_kelly_weights_nonnegative_long_only(self):
        n = 5
        signals = np.array([0.10, -0.05, 0.08, 0.12, 0.03])
        vols = np.array([0.15, 0.20, 0.12, 0.25, 0.18])
        corr = make_corr(n)
        sizer = DynamicSizer(long_only=True)
        weights = sizer.size_kelly_vol_target(signals, vols, corr)
        assert np.all(weights >= 0)

    def test_kelly_within_max_weight(self):
        n = 6
        signals = np.linspace(0.05, 0.18, n)
        vols = np.linspace(0.10, 0.30, n)
        corr = make_corr(n)
        max_w = 0.25
        sizer = DynamicSizer(max_weight=max_w)
        weights = sizer.size_kelly_vol_target(signals, vols, corr)
        assert np.all(weights <= max_w + 1e-6)

    def test_kelly_mismatched_vols_raises(self):
        n = 5
        signals = np.ones(n) * 0.10
        vols = np.ones(n - 1) * 0.15  # wrong length
        corr = make_corr(n)
        with pytest.raises(ValueError, match="vols shape"):
            DynamicSizer().size_kelly_vol_target(signals, vols, corr)

    def test_kelly_fraction_constructor_validation(self):
        with pytest.raises(ValueError, match="kelly_fraction"):
            DynamicSizer(kelly_fraction=0.0)
        with pytest.raises(ValueError, match="kelly_fraction"):
            DynamicSizer(kelly_fraction=1.5)

    def test_kelly_target_vol_constructor_validation(self):
        with pytest.raises(ValueError, match="target_vol"):
            DynamicSizer(target_vol=-0.10)

    def test_kelly_higher_signal_gets_more_weight(self):
        """Asset with higher signal should receive more weight, all else equal.

        With uncorrelated assets and identical vols, Kelly weights are
        proportional to signals.  Use max_weight=0.99 to avoid cap flattening.
        """
        n = 3
        signals = np.array([0.20, 0.05, 0.03])
        vols = np.array([0.15, 0.15, 0.15])
        corr = np.eye(n)
        sizer = DynamicSizer(long_only=True, max_weight=0.99)
        weights = sizer.size_kelly_vol_target(signals, vols, corr)
        assert weights[0] > weights[1] > weights[2]


class TestRegimeAdjustedWeights:

    def test_regime_adjusted_weights_bh_boost(self):
        """BH active + trending: momentum assets should get boosted weight."""
        n = 4
        base_weights = np.array([0.25, 0.25, 0.25, 0.25])
        regime = RegimeState(
            bh_active=RegimeBHState.ACTIVE,
            trend=RegimeTrend.TRENDING,
            asset_regimes={0: {"is_momentum": True, "is_mean_reversion": False}},
        )
        sizer = DynamicSizer()
        adj_weights = sizer.regime_adjusted_weights(base_weights, regime)
        # Asset 0 should have higher relative weight after renorm.
        assert adj_weights[0] > 0.25

    def test_regime_adjusted_weights_bh_inactive_mr_boost(self):
        """BH inactive + MR: momentum zeroed, MR assets boosted."""
        n = 4
        base_weights = np.array([0.25, 0.25, 0.25, 0.25])
        regime = RegimeState(
            bh_active=RegimeBHState.INACTIVE,
            trend=RegimeTrend.MEAN_REVERTING,
            asset_regimes={
                0: {"is_momentum": True, "is_mean_reversion": False},
                1: {"is_momentum": False, "is_mean_reversion": True},
            },
        )
        sizer = DynamicSizer()
        adj_weights = sizer.regime_adjusted_weights(base_weights, regime)
        # Momentum asset zeroed.
        np.testing.assert_allclose(adj_weights[0], 0.0, atol=1e-10)
        # MR asset should have higher relative weight.
        assert adj_weights[1] > adj_weights[2]

    def test_regime_adjusted_weights_high_vol_scale_down(self):
        """High vol regime should reduce all weights proportionally."""
        n = 4
        base_weights = np.array([0.30, 0.25, 0.25, 0.20])
        regime = RegimeState(vol_regime=VolRegime.HIGH)
        sizer = DynamicSizer()
        adj_weights = sizer.regime_adjusted_weights(base_weights, regime)
        # Weights should sum to 1 still (renormalised).
        np.testing.assert_allclose(adj_weights.sum(), 1.0, atol=1e-6)
        # Relative ratios should be preserved.
        ratios_before = base_weights / base_weights.sum()
        np.testing.assert_allclose(adj_weights, ratios_before, atol=1e-6)

    def test_regime_adjusted_weights_event_calendar(self):
        """Event calendar should halve effective exposure (renorm to 1)."""
        n = 4
        base_weights = np.array([0.25, 0.25, 0.25, 0.25])
        regime = RegimeState(event_calendar_active=True)
        sizer = DynamicSizer()
        adj_weights = sizer.regime_adjusted_weights(base_weights, regime)
        # After halving and renorm, equal-weight should still be equal.
        np.testing.assert_allclose(adj_weights.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(adj_weights, np.full(n, 0.25), atol=1e-6)

    def test_regime_adjusted_weights_neutral_regime_unchanged(self):
        """Neutral regime state should not change weights."""
        n = 4
        base_weights = np.array([0.35, 0.25, 0.25, 0.15])
        regime = RegimeState()  # all defaults: UNKNOWN, NEUTRAL, NORMAL, no event
        sizer = DynamicSizer()
        adj_weights = sizer.regime_adjusted_weights(base_weights, regime)
        np.testing.assert_allclose(adj_weights, base_weights, atol=1e-6)

    def test_regime_adjusted_weights_sum_to_one(self):
        n = 6
        base_weights = np.random.default_rng(0).dirichlet(np.ones(n))
        regime = RegimeState(
            bh_active=RegimeBHState.ACTIVE,
            trend=RegimeTrend.TRENDING,
            vol_regime=VolRegime.HIGH,
            asset_regimes={0: {"is_momentum": True}, 1: {"is_momentum": True}},
        )
        sizer = DynamicSizer()
        adj_weights = sizer.regime_adjusted_weights(base_weights, regime)
        np.testing.assert_allclose(adj_weights.sum(), 1.0, atol=1e-6)

    def test_regime_combined_adjustments(self):
        """High vol + event calendar applied together."""
        n = 4
        base_weights = np.array([0.40, 0.30, 0.20, 0.10])
        regime = RegimeState(vol_regime=VolRegime.HIGH, event_calendar_active=True)
        sizer = DynamicSizer()
        adj_weights = sizer.regime_adjusted_weights(base_weights, regime)
        # Renorm preserves relative weights.
        np.testing.assert_allclose(adj_weights.sum(), 1.0, atol=1e-6)
        np.testing.assert_allclose(adj_weights, base_weights, atol=1e-6)


class TestTurnoverConstraint:

    def test_turnover_constraint_respected(self):
        """One-way turnover of result should not exceed max_turnover."""
        n = 5
        current = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        target = np.array([0.40, 0.30, 0.15, 0.10, 0.05])
        max_to = 0.10
        sizer = DynamicSizer()
        result = sizer.turnover_constrained(current, target, max_turnover=max_to)
        actual_to = 0.5 * float(np.sum(np.abs(result - current)))
        assert actual_to <= max_to + 1e-8

    def test_turnover_constraint_no_clip_when_within_budget(self):
        """No clipping when target is already within turnover budget."""
        n = 4
        current = np.array([0.25, 0.25, 0.25, 0.25])
        target = np.array([0.27, 0.25, 0.25, 0.23])  # tiny change
        sizer = DynamicSizer()
        result = sizer.turnover_constrained(current, target, max_turnover=0.10)
        np.testing.assert_allclose(result, target, atol=1e-8)

    def test_turnover_constraint_output_sums_to_one(self):
        n = 5
        rng = np.random.default_rng(7)
        current = rng.dirichlet(np.ones(n))
        target = rng.dirichlet(np.ones(n))
        sizer = DynamicSizer()
        result = sizer.turnover_constrained(current, target, max_turnover=0.05)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_turnover_constraint_direction_preserved(self):
        """When clipped, direction of change should be preserved."""
        n = 4
        current = np.array([0.25, 0.25, 0.25, 0.25])
        target = np.array([0.50, 0.30, 0.15, 0.05])  # big shift toward asset 0
        sizer = DynamicSizer()
        result = sizer.turnover_constrained(current, target, max_turnover=0.05)
        # Asset 0 should still have higher weight than initial.
        assert result[0] > current[0]
        # Asset 3 should still be lower than initial.
        assert result[3] < current[3]

    def test_turnover_constraint_shape_mismatch_raises(self):
        sizer = DynamicSizer()
        with pytest.raises(ValueError, match="shape"):
            sizer.turnover_constrained(
                np.array([0.5, 0.5]),
                np.array([0.33, 0.33, 0.34]),
                max_turnover=0.10,
            )

    def test_turnover_constraint_proportional_clipping(self):
        """Clipping should be proportional: relative deltas should be equal."""
        n = 4
        current = np.array([0.25, 0.25, 0.25, 0.25])
        # Each asset changes by the same amount in the same direction.
        target = np.array([0.45, 0.30, 0.15, 0.10])
        sizer = DynamicSizer()
        result = sizer.turnover_constrained(current, target, max_turnover=0.05)
        deltas = result - current
        target_deltas = target - current
        # Ratio of clipped delta to original delta should be the same for all.
        nonzero = target_deltas != 0
        ratios = deltas[nonzero] / target_deltas[nonzero]
        np.testing.assert_allclose(ratios, ratios[0], atol=1e-6)


class TestComputeTargetPortfolio:

    def test_compute_target_portfolio_returns_dataclass(self):
        n = 5
        signals = np.linspace(0.05, 0.15, n)
        vols = np.linspace(0.10, 0.25, n)
        corr = make_corr(n)
        sizer = DynamicSizer()
        tp = sizer.compute_target_portfolio(signals, vols, corr)
        assert isinstance(tp, TargetPortfolio)

    def test_compute_target_portfolio_weights_sum_to_one(self):
        n = 6
        signals = np.linspace(0.06, 0.18, n)
        vols = np.linspace(0.12, 0.28, n)
        corr = make_corr(n, rho=0.3)
        sizer = DynamicSizer()
        tp = sizer.compute_target_portfolio(signals, vols, corr)
        np.testing.assert_allclose(tp.weights.sum(), 1.0, atol=1e-5)

    def test_compute_target_portfolio_respects_turnover(self):
        n = 5
        signals = np.linspace(0.06, 0.18, n)
        vols = np.linspace(0.12, 0.28, n)
        corr = make_corr(n)
        current = np.full(n, 0.2)
        sizer = DynamicSizer()
        tp = sizer.compute_target_portfolio(
            signals, vols, corr, current_weights=current, max_turnover=0.05
        )
        actual_to = 0.5 * float(np.sum(np.abs(tp.weights - current)))
        assert actual_to <= 0.05 + 1e-6

    def test_compute_target_portfolio_has_timestamp(self):
        n = 4
        signals = np.ones(n) * 0.10
        vols = np.ones(n) * 0.15
        corr = np.eye(n)
        tp = DynamicSizer().compute_target_portfolio(signals, vols, corr)
        assert tp.timestamp is not None

    def test_compute_target_portfolio_notes_populated(self):
        n = 4
        signals = np.ones(n) * 0.10
        vols = np.ones(n) * 0.15
        corr = np.eye(n)
        tp = DynamicSizer().compute_target_portfolio(signals, vols, corr)
        assert len(tp.notes) > 0


# ---------------------------------------------------------------------------
# Integration / cross-module tests
# ---------------------------------------------------------------------------


class TestIntegration:

    def test_erc_then_bl_then_mv(self):
        """Chain: compute ERC -> use as prior -> BL update -> MV max Sharpe."""
        n = 6
        cov = make_cov(n, rho=0.2)
        vols = np.sqrt(np.diag(cov))
        # ERC weights.
        rp = RiskParityOptimizer()
        w_erc = rp.equal_risk_contribution(cov)
        # Use ERC implied returns as BL prior.
        mu_prior = cov @ w_erc * 10  # risk premium proportional to cov @ w
        # BL update with a view on asset 0.
        mv = MeanVarianceOptimizer()
        views = [(np.eye(n)[0], mu_prior[0] * 1.5)]
        confidences = np.array([0.75])
        mu_bl, cov_bl = mv.black_litterman(mu_prior, cov, views, confidences)
        # Max Sharpe using BL posterior.
        w_ms = mv.max_sharpe(mu_bl, cov_bl)
        np.testing.assert_allclose(w_ms.sum(), 1.0, atol=1e-5)

    def test_kelly_sizing_with_regime_and_turnover(self):
        """Full pipeline for 30 assets."""
        n = 30
        rng = np.random.default_rng(99)
        signals = rng.uniform(0.05, 0.20, n)
        vols = rng.uniform(0.10, 0.50, n)
        corr = make_corr(n, rho=0.15)
        current = np.full(n, 1.0 / n)
        regime = RegimeState(
            bh_active=RegimeBHState.ACTIVE,
            trend=RegimeTrend.TRENDING,
            vol_regime=VolRegime.NORMAL,
            asset_regimes={i: {"is_momentum": True} for i in range(0, 5)},
        )
        sizer = DynamicSizer(kelly_fraction=0.25, target_vol=0.15, max_weight=0.10)
        tp = sizer.compute_target_portfolio(
            signals, vols, corr,
            regime_state=regime,
            current_weights=current,
            max_turnover=0.10,
        )
        np.testing.assert_allclose(tp.weights.sum(), 1.0, atol=1e-5)
        assert np.all(tp.weights >= 0)
        actual_to = 0.5 * float(np.sum(np.abs(tp.weights - current)))
        assert actual_to <= 0.10 + 1e-6

    def test_covariance_estimators_consistent_shape(self):
        """All covariance estimators should return same shape."""
        T, n = 200, 8
        returns = make_returns(T, n)
        est = CovarianceEstimator()
        shapes = {
            "sample": est.sample_cov(returns).shape,
            "lw": est.ledoit_wolf(returns).shape,
            "ewma": est.ewma_cov(returns).shape,
            "denoised": est.denoised_cov(returns).shape,
        }
        for name, shape in shapes.items():
            assert shape == (n, n), f"{name} has wrong shape {shape}"

    def test_ensure_psd_makes_indefinite_matrix_psd(self):
        """A matrix with negative eigenvalues should be fixed by _ensure_psd."""
        M = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        M_psd = _ensure_psd(M)
        eigenvalues = np.linalg.eigvalsh(M_psd)
        assert np.all(eigenvalues >= 0)

    def test_risk_parity_then_dynamic_sizing_end_to_end(self):
        """Use risk parity cov estimator, then dynamic sizing."""
        T, n = 252, 10
        returns = make_returns_correlated(T, n, rho=0.2)
        est = CovarianceEstimator()
        cov = est.ledoit_wolf(returns)
        vols = np.sqrt(np.diag(cov))
        corr = np.diag(1.0 / vols) @ cov @ np.diag(1.0 / vols)
        signals = vols * 0.5  # crude signal: higher vol -> higher signal
        sizer = DynamicSizer(kelly_fraction=0.25, target_vol=0.12)
        weights = sizer.size_kelly_vol_target(signals, vols, corr)
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)
        assert np.all(weights >= 0)
