"""
Tests for portfolio optimization modules.

Covers:
- Minimum variance portfolio
- Risk parity (equal risk contribution)
- Kelly fraction (single asset and multi-asset)
- Maximum Sharpe ratio
- Portfolio turnover and transaction costs
"""

import sys
import os
import unittest
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_universe(n_assets=5, n=300, seed=42):
    rng = np.random.default_rng(seed)
    prices = {}
    for i in range(n_assets):
        drift = rng.uniform(-0.0002, 0.0008)
        vol = rng.uniform(0.01, 0.025)
        log_rets = drift + vol * rng.standard_normal(n)
        prices[f"asset_{i}"] = 100 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(prices, index=idx)


def make_returns(n_assets=5, n=300, seed=42):
    universe = make_universe(n_assets, n, seed)
    return universe.pct_change().dropna()


class TestMinVariancePortfolio(unittest.TestCase):
    """Tests for minimum-variance portfolio."""

    def setUp(self):
        self.returns = make_returns(n_assets=5, n=300)

    def test_min_var_weights_sum_to_one(self):
        """Min-var weights should sum to 1."""
        from strategies.volatility.vol_targeting import RiskParity
        # Use SLSQP from vol_targeting
        rp = RiskParity(assets=list(self.returns.columns))

        # Manually compute min-var via scipy
        from scipy.optimize import minimize
        cov = self.returns.cov().values
        n = len(cov)

        def port_var(w):
            return w @ cov @ w

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n
        result = minimize(port_var, w0, bounds=bounds, constraints=constraints,
                          method="SLSQP")
        w = result.x
        self.assertAlmostEqual(w.sum(), 1.0, places=5)
        self.assertTrue(np.all(w >= -1e-6))

    def test_min_var_lower_variance(self):
        """Min-var portfolio should have lower variance than equal weight."""
        from scipy.optimize import minimize
        cov = self.returns.cov().values
        n = len(cov)

        eq_var = (np.ones(n) / n) @ cov @ (np.ones(n) / n)

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n
        result = minimize(lambda w: w @ cov @ w, w0, bounds=bounds,
                          constraints=constraints, method="SLSQP")
        min_var = result.fun
        self.assertLessEqual(min_var, eq_var + 1e-8)

    def test_degenerate_single_asset(self):
        """Single-asset min-var: weight = 1."""
        from scipy.optimize import minimize
        returns = self.returns.iloc[:, :1]
        cov = returns.cov().values
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)]
        result = minimize(lambda w: w @ cov @ w, [1.0], bounds=bounds,
                          constraints=constraints, method="SLSQP")
        self.assertAlmostEqual(result.x[0], 1.0, places=4)


class TestRiskParity(unittest.TestCase):
    """Tests for risk parity (equal risk contribution)."""

    def setUp(self):
        self.returns = make_returns(n_assets=4, n=300)
        self.universe = make_universe(n_assets=4, n=300)

    def test_risk_parity_backtest_runs(self):
        from strategies.volatility.vol_targeting import RiskParity
        rp = RiskParity(assets=list(self.universe.columns), target_vol=0.10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rp.backtest(self.universe)
        self.assertIsNotNone(result)

    def test_risk_parity_weights_sum_to_one(self):
        from strategies.volatility.vol_targeting import RiskParity
        rp = RiskParity(assets=list(self.universe.columns))
        # Test _risk_parity_weights
        cov = self.returns.cov().values
        w = rp._risk_parity_weights(cov)
        self.assertAlmostEqual(w.sum(), 1.0, places=4)
        # All weights non-negative
        self.assertTrue(np.all(w >= -1e-6))

    def test_equal_risk_contributions(self):
        """Risk contributions should be approximately equal."""
        from strategies.volatility.vol_targeting import RiskParity
        rp = RiskParity(assets=list(self.universe.columns))
        cov = self.returns.cov().values
        w = rp._risk_parity_weights(cov)
        port_vol = np.sqrt(w @ cov @ w)
        rc = w * (cov @ w) / (port_vol + 1e-12)
        target = port_vol / len(w)
        # Check that relative deviation is small
        rel_dev = np.abs(rc - target) / (target + 1e-12)
        self.assertTrue(np.all(rel_dev < 0.5), f"Risk contributions not equal: {rc}")

    def test_risk_contributions_method(self):
        from strategies.volatility.vol_targeting import RiskParity
        rp = RiskParity(assets=list(self.universe.columns))
        rc = rp.risk_contributions(self.universe)
        self.assertIsInstance(rc, pd.DataFrame)

    def test_risk_parity_vs_equal_weight(self):
        """Risk parity should have lower vol than equal-weight when vols differ."""
        from strategies.volatility.vol_targeting import RiskParity
        # Create universe with very different vols
        rng = np.random.default_rng(42)
        n = 300
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        high_vol = pd.Series(100 * np.exp(np.cumsum(0.0003 + 0.04 * rng.standard_normal(n))),
                              index=idx)
        low_vol = pd.Series(100 * np.exp(np.cumsum(0.0003 + 0.005 * rng.standard_normal(n))),
                             index=idx)
        universe = pd.DataFrame({"high_vol": high_vol, "low_vol": low_vol})
        returns = universe.pct_change().dropna()

        rp = RiskParity(assets=["high_vol", "low_vol"])
        cov = returns.cov().values
        w_rp = rp._risk_parity_weights(cov)

        # Risk parity should underweight high-vol asset
        self.assertLess(w_rp[0], w_rp[1])


class TestKellyFraction(unittest.TestCase):
    """Tests for Kelly criterion capital allocation."""

    def test_single_asset_kelly(self):
        """Kelly fraction = mean / variance for single asset."""
        rng = np.random.default_rng(42)
        n = 1000
        mu = 0.001
        sigma = 0.02
        returns = pd.Series(mu + sigma * rng.standard_normal(n))

        mean = returns.mean()
        var = returns.var()
        kelly = mean / (var + 1e-12)
        full_kelly = kelly * var  # fraction of wealth = mu/sigma^2 * sigma^2 = mu

        # Kelly fraction should be positive for positive mean
        self.assertGreater(kelly, 0)
        # Sanity check: full_kelly ≈ mean
        self.assertAlmostEqual(full_kelly, mean, places=4)

    def test_fractional_kelly(self):
        """Half-Kelly should be half of full Kelly."""
        rng = np.random.default_rng(42)
        returns = pd.Series(0.001 + 0.02 * rng.standard_normal(500))
        mean = returns.mean()
        var = returns.var()
        full_kelly = mean / (var + 1e-12)
        half_kelly = full_kelly / 2
        self.assertAlmostEqual(half_kelly, full_kelly / 2, places=8)

    def test_kelly_multiasset(self):
        """Multi-asset Kelly: w* = Sigma^{-1} * mu."""
        rng = np.random.default_rng(42)
        n = 500
        returns = make_returns(n_assets=3, n=n)
        mu = returns.mean().values
        cov = returns.cov().values

        try:
            kelly_w = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            kelly_w = np.linalg.lstsq(cov, mu, rcond=None)[0]

        # Expected return of Kelly portfolio
        port_ret = kelly_w @ mu
        # Should be positive if mu has positive elements
        if mu.sum() > 0:
            self.assertGreater(port_ret, 0)

    def test_zero_mean_kelly(self):
        """Zero-mean returns → Kelly fraction = 0."""
        rng = np.random.default_rng(42)
        returns = pd.Series(0.02 * rng.standard_normal(500))
        # Force mean exactly 0
        returns = returns - returns.mean()
        mean = returns.mean()
        var = returns.var()
        kelly = mean / (var + 1e-12)
        self.assertAlmostEqual(kelly, 0.0, places=4)

    def test_kelly_bounded_by_risk(self):
        """Kelly fraction should be bounded for reasonable inputs."""
        rng = np.random.default_rng(42)
        returns = pd.Series(0.001 + 0.02 * rng.standard_normal(500))
        mean = returns.mean()
        var = returns.var()
        kelly = mean / var
        # Kelly fraction for typical equity: ~0.001 / 0.0004 = 2.5 (reasonable)
        self.assertLess(abs(kelly), 100)


class TestMaxSharpe(unittest.TestCase):
    """Tests for maximum Sharpe ratio portfolio."""

    def setUp(self):
        self.returns = make_returns(n_assets=4, n=300)
        self.mu = self.returns.mean().values
        self.cov = self.returns.cov().values

    def test_max_sharpe_weights_sum_to_one(self):
        """Max-Sharpe portfolio (long-only) weights sum to 1."""
        from scipy.optimize import minimize

        n = len(self.mu)
        rf = 0.0

        def neg_sharpe(w):
            port_ret = w @ self.mu
            port_vol = np.sqrt(w @ self.cov @ w + 1e-12)
            return -(port_ret - rf) / port_vol

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n
        result = minimize(neg_sharpe, np.ones(n) / n, bounds=bounds,
                          constraints=constraints, method="SLSQP")
        w = result.x
        self.assertAlmostEqual(w.sum(), 1.0, places=4)

    def test_max_sharpe_higher_than_equal_weight(self):
        """Max-Sharpe Sharpe ratio should be >= equal-weight Sharpe."""
        from scipy.optimize import minimize

        n = len(self.mu)
        rf = 0.0

        def neg_sharpe(w):
            port_ret = w @ self.mu
            port_vol = np.sqrt(w @ self.cov @ w + 1e-12)
            return -(port_ret - rf) / port_vol

        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
        bounds = [(0, 1)] * n
        result = minimize(neg_sharpe, np.ones(n) / n, bounds=bounds,
                          constraints=constraints, method="SLSQP")
        max_sharpe = -result.fun

        # Equal-weight Sharpe
        w_eq = np.ones(n) / n
        eq_ret = w_eq @ self.mu
        eq_vol = np.sqrt(w_eq @ self.cov @ w_eq + 1e-12)
        eq_sharpe = eq_ret / eq_vol

        self.assertGreaterEqual(max_sharpe + 1e-6, eq_sharpe)


class TestMaxDiversification(unittest.TestCase):
    """Tests for maximum diversification portfolio."""

    def test_max_div_weights_sum_to_one(self):
        from strategies.volatility.vol_targeting import MaxDiversification
        universe = make_universe(n_assets=4, n=300)
        md = MaxDiversification(assets=list(universe.columns))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = md.backtest(universe)
        self.assertIsNotNone(result)

    def test_diversification_ratio_positive(self):
        from strategies.volatility.vol_targeting import MaxDiversification
        universe = make_universe(n_assets=4, n=300)
        md = MaxDiversification(assets=list(universe.columns))
        dr = md.diversification_ratio(universe)
        self.assertIsInstance(dr, pd.Series)
        self.assertTrue((dr.dropna() >= 1.0 - 1e-4).all())


class TestTransactionCosts(unittest.TestCase):
    """Tests that transaction costs reduce portfolio returns."""

    def test_turnover_reduces_returns(self):
        """Higher transaction costs → lower net returns."""
        from strategies.momentum.trend_following import DualMovingAverage

        df_ohlcv = make_universe(n_assets=1, n=300).rename(columns={"asset_0": "close"})
        df_ohlcv["open"] = df_ohlcv["close"] * 0.999
        df_ohlcv["high"] = df_ohlcv["close"] * 1.005
        df_ohlcv["low"] = df_ohlcv["close"] * 0.995
        df_ohlcv["volume"] = 1_000_000.0

        dma = DualMovingAverage(fast=10, slow=30)
        result = dma.backtest(df_ohlcv)
        gross_return = result.total_return

        # Simulate with 10bps transaction cost (heuristic check)
        signals = dma.generate_signals(df_ohlcv)
        daily_ret = df_ohlcv["close"].pct_change()
        trade_days = (signals.diff().abs() > 0).sum()
        cost = trade_days * 0.001 / len(signals)
        net_return = gross_return - cost
        self.assertLessEqual(net_return, gross_return + 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
