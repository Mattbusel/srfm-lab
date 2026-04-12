"""
test_portfolio.py — Tests for portfolio_tt.py and signal_extraction.py

Run with::

    pytest aeternus/tensor_net/tests/test_portfolio.py -v
"""

from __future__ import annotations

import math
import sys
import os

import numpy as np
import pytest

# Add tensor_net package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="module")
def n_assets():
    return 20


@pytest.fixture(scope="module")
def n_days():
    return 300


@pytest.fixture(scope="module")
def synthetic_returns(rng, n_assets, n_days):
    """Simulate daily log-returns: (T, N)."""
    cov = np.eye(n_assets) * 0.01 + 0.005 * np.ones((n_assets, n_assets))
    returns = rng.multivariate_normal(np.zeros(n_assets), cov, size=n_days)
    return returns.astype(np.float32)


@pytest.fixture(scope="module")
def synthetic_covariance(synthetic_returns):
    return np.cov(synthetic_returns.T).astype(np.float32)


@pytest.fixture(scope="module")
def equal_weights(n_assets):
    return np.ones(n_assets, dtype=np.float32) / n_assets


# ---------------------------------------------------------------------------
# TTCovarianceEstimator tests
# ---------------------------------------------------------------------------


class TestTTCovarianceEstimator:
    def test_import(self):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        assert TTCovarianceEstimator is not None

    def test_update_and_covariance(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        config = CovEstimatorConfig(n_assets=n_assets, min_observations=10)
        est = TTCovarianceEstimator(config)
        for r in synthetic_returns[:50]:
            est.update(r)
        cov = est.covariance()
        assert cov.shape == (n_assets, n_assets)
        # Diagonal should be positive
        assert np.all(np.diag(cov) > 0)

    def test_batch_update(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        config = CovEstimatorConfig(n_assets=n_assets, min_observations=5)
        est = TTCovarianceEstimator(config)
        est.update(synthetic_returns[:100])
        cov = est.covariance()
        assert cov.shape == (n_assets, n_assets)
        assert np.isfinite(cov).all()

    def test_correlation(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        config = CovEstimatorConfig(n_assets=n_assets, min_observations=5)
        est = TTCovarianceEstimator(config)
        est.update(synthetic_returns[:200])
        corr = est.correlation()
        assert corr.shape == (n_assets, n_assets)
        # Diagonal should be ~1
        np.testing.assert_allclose(np.diag(corr), np.ones(n_assets), atol=0.01)

    def test_mean_returns(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        config = CovEstimatorConfig(n_assets=n_assets, min_observations=5)
        est = TTCovarianceEstimator(config)
        est.update(synthetic_returns[:100])
        mu = est.mean_returns()
        assert mu.shape == (n_assets,)
        assert np.isfinite(mu).all()

    def test_annualised_vols(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        config = CovEstimatorConfig(n_assets=n_assets, min_observations=5)
        est = TTCovarianceEstimator(config)
        est.update(synthetic_returns[:100])
        vols = est.annualised_volatilities()
        assert vols.shape == (n_assets,)
        assert np.all(vols > 0)

    def test_reset(self, n_assets):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        config = CovEstimatorConfig(n_assets=n_assets, min_observations=5)
        est = TTCovarianceEstimator(config)
        est.update(np.ones((50, n_assets), dtype=np.float32) * 0.01)
        est.reset()
        assert est._n == 0.0
        assert np.all(est._S == 0.0)

    def test_few_observations_warning(self, n_assets):
        from tensor_net.portfolio_tt import TTCovarianceEstimator, CovEstimatorConfig
        config = CovEstimatorConfig(n_assets=n_assets, min_observations=100)
        est = TTCovarianceEstimator(config)
        est.update(np.random.randn(5, n_assets).astype(np.float32))
        with pytest.warns(UserWarning):
            cov = est.covariance()
        # Should return identity
        assert cov.shape == (n_assets, n_assets)


# ---------------------------------------------------------------------------
# TTMeanVarianceOptimiser tests
# ---------------------------------------------------------------------------


class TestTTMeanVarianceOptimiser:
    def test_import(self):
        from tensor_net.portfolio_tt import TTMeanVarianceOptimiser, MVOptConfig
        assert TTMeanVarianceOptimiser is not None

    def test_basic_optimise(self, synthetic_covariance, n_assets):
        from tensor_net.portfolio_tt import TTMeanVarianceOptimiser, MVOptConfig
        config = MVOptConfig(n_iters=100, lr=5e-3)
        opt = TTMeanVarianceOptimiser(config, n_assets=n_assets)
        mu = np.random.randn(n_assets).astype(np.float32) * 0.001
        w = opt.optimise(mu, synthetic_covariance)
        assert w.shape == (n_assets,)
        assert np.isfinite(w).all()
        np.testing.assert_allclose(w.sum(), 1.0, atol=0.05)

    def test_long_only_constraint(self, synthetic_covariance, n_assets):
        from tensor_net.portfolio_tt import TTMeanVarianceOptimiser, MVOptConfig
        config = MVOptConfig(allow_short=False, n_iters=50)
        opt = TTMeanVarianceOptimiser(config, n_assets=n_assets)
        mu = np.random.randn(n_assets).astype(np.float32) * 0.001
        w = opt.optimise(mu, synthetic_covariance)
        assert np.all(w >= -1e-5), "Long-only violated"

    def test_max_weight_constraint(self, synthetic_covariance, n_assets):
        from tensor_net.portfolio_tt import TTMeanVarianceOptimiser, MVOptConfig
        max_w = 0.15
        config = MVOptConfig(max_weight=max_w, n_iters=100)
        opt = TTMeanVarianceOptimiser(config, n_assets=n_assets)
        mu = np.ones(n_assets, dtype=np.float32) * 0.001
        w = opt.optimise(mu, synthetic_covariance)
        assert np.all(w <= max_w + 1e-4)

    def test_efficient_frontier(self, synthetic_covariance, n_assets):
        from tensor_net.portfolio_tt import TTMeanVarianceOptimiser, MVOptConfig
        config = MVOptConfig(n_iters=30)
        opt = TTMeanVarianceOptimiser(config, n_assets=n_assets)
        mu = np.random.randn(n_assets).astype(np.float32) * 0.001
        rets, vols, weights = opt.efficient_frontier(mu, synthetic_covariance, n_points=5)
        assert rets.shape == (5,)
        assert vols.shape == (5,)
        assert weights.shape == (5, n_assets)


# ---------------------------------------------------------------------------
# TTRiskParityPortfolio tests
# ---------------------------------------------------------------------------


class TestTTRiskParityPortfolio:
    def test_import(self):
        from tensor_net.portfolio_tt import TTRiskParityPortfolio
        assert TTRiskParityPortfolio is not None

    def test_equal_risk_parity(self, synthetic_covariance, n_assets):
        from tensor_net.portfolio_tt import TTRiskParityPortfolio, RiskParityConfig
        config = RiskParityConfig(n_iters=500)
        rp = TTRiskParityPortfolio(config, n_assets=n_assets)
        w = rp.optimise(synthetic_covariance)
        assert w.shape == (n_assets,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=0.05)

    def test_risk_contributions(self, synthetic_covariance, n_assets, equal_weights):
        from tensor_net.portfolio_tt import TTRiskParityPortfolio, RiskParityConfig
        config = RiskParityConfig(n_iters=100)
        rp = TTRiskParityPortfolio(config, n_assets=n_assets)
        rc = rp.risk_contributions(equal_weights, synthetic_covariance)
        assert rc.shape == (n_assets,)
        np.testing.assert_allclose(rc.sum(), 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# TTFactorModel tests
# ---------------------------------------------------------------------------


class TestTTFactorModel:
    def test_import(self):
        from tensor_net.portfolio_tt import TTFactorModel
        assert TTFactorModel is not None

    def test_fit_and_predict(self, synthetic_returns, n_assets, rng):
        from tensor_net.portfolio_tt import TTFactorModel, FactorModelConfig
        T = synthetic_returns.shape[0]
        K = 3
        factors = rng.normal(size=(T, K)).astype(np.float32)
        config = FactorModelConfig(n_assets=n_assets, n_factors=K)
        model = TTFactorModel(config)
        result = model.fit(synthetic_returns, factors)
        assert "r2" in result
        preds = model.predict(factors[:10])
        assert preds.shape == (10, n_assets)

    def test_covariance(self, synthetic_returns, n_assets, rng):
        from tensor_net.portfolio_tt import TTFactorModel, FactorModelConfig
        T = synthetic_returns.shape[0]
        K = 3
        factors = rng.normal(size=(T, K)).astype(np.float32)
        config = FactorModelConfig(n_assets=n_assets, n_factors=K)
        model = TTFactorModel(config)
        model.fit(synthetic_returns, factors)
        cov = model.covariance()
        assert cov.shape == (n_assets, n_assets)
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > -1e-4), "Factor model cov not PSD"

    def test_not_fitted_warning(self, n_assets):
        from tensor_net.portfolio_tt import TTFactorModel, FactorModelConfig
        config = FactorModelConfig(n_assets=n_assets)
        model = TTFactorModel(config)
        with pytest.warns(UserWarning):
            cov = model.covariance()


# ---------------------------------------------------------------------------
# TTBlackLitterman tests
# ---------------------------------------------------------------------------


class TestTTBlackLitterman:
    def test_import(self):
        from tensor_net.portfolio_tt import TTBlackLitterman
        assert TTBlackLitterman is not None

    def test_equilibrium_returns(self, synthetic_covariance, equal_weights):
        from tensor_net.portfolio_tt import TTBlackLitterman, BLConfig
        bl = TTBlackLitterman(n_assets=len(equal_weights))
        pi = bl.equilibrium_returns(equal_weights, synthetic_covariance)
        assert pi.shape == (len(equal_weights),)

    def test_posterior(self, synthetic_covariance, equal_weights, n_assets):
        from tensor_net.portfolio_tt import TTBlackLitterman, BLConfig
        bl = TTBlackLitterman(n_assets=n_assets)
        pi = bl.equilibrium_returns(equal_weights, synthetic_covariance)
        # 2 views
        P = np.zeros((2, n_assets), dtype=np.float32)
        P[0, 0] = 1.0; P[0, 1] = -1.0   # asset 0 outperforms asset 1
        P[1, 2] = 1.0                    # asset 2 has positive return
        q = np.array([0.001, 0.002], dtype=np.float32)
        mu_bl, sigma_bl = bl.posterior(pi, synthetic_covariance, P, q)
        assert mu_bl.shape == (n_assets,)
        assert sigma_bl.shape == (n_assets, n_assets)


# ---------------------------------------------------------------------------
# TransactionCostModel tests
# ---------------------------------------------------------------------------


class TestTransactionCostModel:
    def test_import(self):
        from tensor_net.portfolio_tt import TransactionCostModel
        assert TransactionCostModel is not None

    def test_zero_turnover(self, equal_weights, n_assets):
        from tensor_net.portfolio_tt import TransactionCostModel, TCostConfig
        tcm = TransactionCostModel(n_assets=n_assets)
        cost = tcm.cost(equal_weights, equal_weights)
        assert cost == pytest.approx(0.0, abs=1e-8)

    def test_full_turnover(self, n_assets):
        from tensor_net.portfolio_tt import TransactionCostModel, TCostConfig
        tcm = TransactionCostModel(n_assets=n_assets)
        w_old = np.zeros(n_assets, dtype=np.float32)
        w_old[0] = 1.0
        w_new = np.zeros(n_assets, dtype=np.float32)
        w_new[-1] = 1.0
        cost = tcm.cost(w_old, w_new)
        assert cost > 0.0

    def test_net_return(self, equal_weights, n_assets):
        from tensor_net.portfolio_tt import TransactionCostModel
        tcm = TransactionCostModel(n_assets=n_assets)
        gross = 0.01
        w_new = equal_weights.copy()
        w_new[0] += 0.05
        w_new /= w_new.sum()
        net = tcm.net_return(gross, equal_weights, w_new)
        assert net < gross


# ---------------------------------------------------------------------------
# PositionSizer tests
# ---------------------------------------------------------------------------


class TestPositionSizer:
    def test_fixed_fraction(self, equal_weights, n_assets):
        from tensor_net.portfolio_tt import PositionSizer, SizerConfig
        config = SizerConfig(method="fixed_fraction", fixed_fraction=0.5)
        sizer = PositionSizer(config)
        scaled = sizer.scale(equal_weights)
        np.testing.assert_allclose(scaled, equal_weights * 0.5, rtol=1e-5)

    def test_vol_target(self, equal_weights, synthetic_covariance, n_assets):
        from tensor_net.portfolio_tt import PositionSizer, SizerConfig
        config = SizerConfig(method="vol_target", vol_target=0.10)
        sizer = PositionSizer(config)
        scaled = sizer.scale(equal_weights, cov=synthetic_covariance)
        assert scaled.shape == (n_assets,)
        port_vol = float(np.sqrt(252 * scaled @ synthetic_covariance @ scaled))
        assert abs(port_vol - 0.10) < 0.5   # rough check


# ---------------------------------------------------------------------------
# CorrelationShockModel tests
# ---------------------------------------------------------------------------


class TestCorrelationShockModel:
    def test_uniform_shock(self, synthetic_covariance):
        from tensor_net.portfolio_tt import CorrelationShockModel, CorrelationShockConfig
        config = CorrelationShockConfig(shock_magnitude=0.2)
        model = CorrelationShockModel(config)
        cov_s = model.apply_uniform_shock(synthetic_covariance)
        assert cov_s.shape == synthetic_covariance.shape

    def test_random_scenarios(self, synthetic_covariance, equal_weights):
        from tensor_net.portfolio_tt import CorrelationShockModel, CorrelationShockConfig
        config = CorrelationShockConfig(n_scenarios=20)
        model = CorrelationShockModel(config)
        losses = model.random_scenarios(synthetic_covariance, equal_weights)
        assert losses.shape == (20,)
        assert np.isfinite(losses).all()

    def test_var_cvar(self, synthetic_covariance, equal_weights):
        from tensor_net.portfolio_tt import CorrelationShockModel, CorrelationShockConfig
        config = CorrelationShockConfig(n_scenarios=200)
        model = CorrelationShockModel(config)
        losses = model.random_scenarios(synthetic_covariance, equal_weights)
        var = model.var(losses)
        cvar = model.cvar(losses)
        assert var >= 0 or var < 0   # just finite check
        assert np.isfinite(var) and np.isfinite(cvar)


# ---------------------------------------------------------------------------
# PortfolioBacktester tests
# ---------------------------------------------------------------------------


class TestPortfolioBacktester:
    def test_import(self):
        from tensor_net.portfolio_tt import PortfolioBacktester
        assert PortfolioBacktester is not None

    def test_equal_weight_backtest(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import PortfolioBacktester, BacktestConfig, equal_weight
        config = BacktestConfig(rebalance_frequency=21, lookback=60, verbose=False)
        bt = PortfolioBacktester(config, weight_fn=equal_weight)
        result = bt.run(synthetic_returns)
        assert result.portfolio_value.shape[0] > 0
        assert np.isfinite(result.sharpe)
        assert np.isfinite(result.max_drawdown)

    def test_summary(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import PortfolioBacktester, BacktestConfig, equal_weight
        config = BacktestConfig(rebalance_frequency=21, lookback=60)
        bt = PortfolioBacktester(config, weight_fn=equal_weight)
        result = bt.run(synthetic_returns)
        summary = bt.summary(result)
        assert "sharpe" in summary
        assert "max_drawdown" in summary
        assert "annualised_return" in summary

    def test_inverse_vol_backtest(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import PortfolioBacktester, BacktestConfig, inverse_vol_weight
        config = BacktestConfig(rebalance_frequency=21, lookback=63)
        bt = PortfolioBacktester(config, weight_fn=inverse_vol_weight)
        result = bt.run(synthetic_returns)
        assert len(result.returns) > 0


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestPortfolioUtils:
    def test_compute_sharpe(self, synthetic_returns, n_assets, equal_weights):
        from tensor_net.portfolio_tt import compute_sharpe
        port_returns = synthetic_returns @ equal_weights
        sr = compute_sharpe(port_returns)
        assert np.isfinite(sr)

    def test_compute_max_drawdown(self, synthetic_returns, n_assets, equal_weights):
        from tensor_net.portfolio_tt import compute_max_drawdown
        pv = np.cumprod(1 + synthetic_returns @ equal_weights)
        mdd = compute_max_drawdown(pv)
        assert mdd <= 0.0

    def test_herfindahl(self, equal_weights, n_assets):
        from tensor_net.portfolio_tt import herfindahl_index, effective_n
        hhi = herfindahl_index(equal_weights)
        en = effective_n(equal_weights)
        np.testing.assert_allclose(hhi, 1.0 / n_assets, rtol=1e-4)
        np.testing.assert_allclose(en, float(n_assets), rtol=0.01)

    def test_diversification_ratio(self, equal_weights, synthetic_covariance):
        from tensor_net.portfolio_tt import diversification_ratio
        dr = diversification_ratio(equal_weights, synthetic_covariance)
        assert dr >= 1.0

    def test_momentum_weight(self, synthetic_returns, n_assets):
        from tensor_net.portfolio_tt import momentum_weight
        w = momentum_weight(synthetic_returns, top_k=5)
        assert w.shape == (n_assets,)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-5)
        assert (w > 0).sum() == 5

    def test_turnover_series(self, n_assets):
        from tensor_net.portfolio_tt import turnover_series
        weights = np.random.dirichlet(np.ones(n_assets), size=50)
        to = turnover_series(weights)
        assert to.shape == (49,)
        assert np.all(to >= 0)


# ---------------------------------------------------------------------------
# Signal extraction tests
# ---------------------------------------------------------------------------


class TestTTSignalExtractor:
    def test_import(self):
        from tensor_net.signal_extraction import TTSignalExtractor
        assert TTSignalExtractor is not None

    def test_fit_transform(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import TTSignalExtractor, SignalExtractorConfig
        config = SignalExtractorConfig(n_components=3)
        ext = TTSignalExtractor(config)
        signals = ext.fit_transform(synthetic_returns)
        T = synthetic_returns.shape[0]
        assert signals.shape == (T, 3)
        assert np.isfinite(signals).all()

    def test_explained_variance(self, synthetic_returns):
        from tensor_net.signal_extraction import TTSignalExtractor, SignalExtractorConfig
        config = SignalExtractorConfig(n_components=5)
        ext = TTSignalExtractor(config)
        ext.fit(synthetic_returns)
        evr = ext.explained_variance_ratio()
        assert evr.shape == (5,)
        assert np.all(evr >= 0)
        assert evr.sum() <= 1.01

    def test_transform_new_data(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import TTSignalExtractor, SignalExtractorConfig
        config = SignalExtractorConfig(n_components=2)
        ext = TTSignalExtractor(config)
        ext.fit(synthetic_returns[:200])
        signals = ext.transform(synthetic_returns[200:])
        assert signals.shape == (100, 2)


class TestMomentumSignalFactory:
    def test_cross_sectional(self, synthetic_returns):
        from tensor_net.signal_extraction import MomentumSignalFactory, MomentumConfig
        config = MomentumConfig(lookbacks=[21, 63])
        factory = MomentumSignalFactory(config)
        signals = factory.cross_sectional(synthetic_returns)
        assert 21 in signals or 63 in signals

    def test_blend(self, synthetic_returns):
        from tensor_net.signal_extraction import MomentumSignalFactory, MomentumConfig
        T, N = synthetic_returns.shape
        config = MomentumConfig(lookbacks=[21, 63])
        factory = MomentumSignalFactory(config)
        cs = factory.cross_sectional(synthetic_returns)
        if len(cs) >= 2:
            blended = factory.blend(cs)
            assert blended.shape == (T, N)


class TestMeanReversionSignal:
    def test_compute(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import MeanReversionSignal, MeanReversionConfig
        log_prices = np.cumsum(synthetic_returns, axis=0)
        config = MeanReversionConfig(lookback=10)
        sig = MeanReversionSignal(config)
        z = sig.compute(log_prices)
        assert z.shape == synthetic_returns.shape
        assert np.isfinite(z).all()


class TestVolatilitySignalFactory:
    def test_realised_vol(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import VolatilitySignalFactory, VolSignalConfig
        config = VolSignalConfig(rv_window=21)
        factory = VolatilitySignalFactory(config)
        rv = factory.realised_vol(synthetic_returns)
        assert rv.shape == synthetic_returns.shape
        assert np.all(rv[21:] >= 0)

    def test_garch_vol(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import VolatilitySignalFactory
        factory = VolatilitySignalFactory()
        gv = factory.garch_vol(synthetic_returns)
        assert gv.shape == synthetic_returns.shape
        assert np.all(gv >= 0)

    def test_vol_percentile(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import VolatilitySignalFactory
        factory = VolatilitySignalFactory()
        vp = factory.vol_percentile(synthetic_returns)
        assert vp.shape == synthetic_returns.shape
        assert np.all(vp[21:] >= 0)
        assert np.all(vp[21:] <= 1)


class TestCorrelationBreakSignal:
    def test_signal(self, synthetic_returns):
        from tensor_net.signal_extraction import CorrelationBreakSignal, CorrBreakConfig
        config = CorrBreakConfig(short_window=10, long_window=30)
        sig = CorrelationBreakSignal(config)
        T = synthetic_returns.shape[0]
        s = sig.correlation_break_series(synthetic_returns)
        assert s.shape == (T,)
        assert np.isfinite(s).all()
        assert np.all(s >= 0)

    def test_regime_indicator(self, synthetic_returns):
        from tensor_net.signal_extraction import CorrelationBreakSignal, CorrBreakConfig
        config = CorrBreakConfig(short_window=10, long_window=30, threshold=0.5)
        sig = CorrelationBreakSignal(config)
        s = sig.correlation_break_series(synthetic_returns)
        ind = sig.regime_indicator(s)
        assert set(np.unique(ind)).issubset({0.0, 1.0})


class TestSignalEvaluator:
    def test_ic_series(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import SignalEvaluator
        T, N = synthetic_returns.shape
        signal = np.random.randn(T, N).astype(np.float32)
        fwd = np.roll(synthetic_returns, -1, axis=0)
        try:
            evaluator = SignalEvaluator(signal, fwd)
            ic = evaluator.ic_series()
            assert ic.shape == (T,)
        except ImportError:
            pytest.skip("scipy not available")

    def test_summary(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import SignalEvaluator
        T, N = synthetic_returns.shape
        signal = np.random.randn(T, N).astype(np.float32)
        fwd = np.roll(synthetic_returns, -1, axis=0)
        try:
            evaluator = SignalEvaluator(signal, fwd)
            s = evaluator.summary()
            assert "mean_ic" in s
            assert "icir" in s
        except ImportError:
            pytest.skip("scipy not available")


class TestAlphaDecayAnalyser:
    def test_half_life(self):
        from tensor_net.signal_extraction import AlphaDecayAnalyser
        analyser = AlphaDecayAnalyser()
        ic = np.array([0.05 * (0.9 ** t) for t in range(100)], dtype=np.float32)
        hl = analyser.half_life(ic)
        assert np.isfinite(hl)

    def test_autocorrelation(self):
        from tensor_net.signal_extraction import AlphaDecayAnalyser
        analyser = AlphaDecayAnalyser()
        ic = np.random.randn(200).astype(np.float32)
        acf = analyser.autocorrelation(ic)
        assert acf[0] == pytest.approx(1.0, abs=0.01)
        assert acf.shape[0] == analyser.config.max_lag + 1


class TestSignalNeutraliser:
    def test_market_neutral(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import SignalNeutraliser, NeutralisationConfig
        config = NeutralisationConfig(neutralise_market=True, neutralise_sector=False)
        neutraliser = SignalNeutraliser(config)
        T, N = synthetic_returns.shape
        signal = np.random.randn(T, N).astype(np.float32)
        neutral = neutraliser.neutralise(signal)
        assert neutral.shape == (T, N)
        # Row means should be ~0
        row_means = np.mean(neutral, axis=1)
        np.testing.assert_allclose(row_means, np.zeros(T), atol=1e-5)


class TestSignalCombiner:
    def test_equal_combine(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import SignalCombiner, CombinerConfig
        config = CombinerConfig(method="equal")
        combiner = SignalCombiner(config)
        T, N = synthetic_returns.shape
        K = 3
        signals = np.random.randn(T, N, K).astype(np.float32)
        fwd = np.roll(synthetic_returns, -1, axis=0)
        combiner.fit(signals, fwd)
        composite = combiner.transform(signals)
        assert composite.shape == (T, N)
        assert np.isfinite(composite).all()

    def test_ic_weighted_combine(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import SignalCombiner, CombinerConfig
        config = CombinerConfig(method="ic_weighted")
        combiner = SignalCombiner(config)
        T, N = synthetic_returns.shape
        K = 2
        signals = np.random.randn(T, N, K).astype(np.float32)
        fwd = np.roll(synthetic_returns, -1, axis=0)
        combiner.fit(signals, fwd)
        w = combiner.weights
        assert w is not None
        assert w.shape == (K,)


# ---------------------------------------------------------------------------
# InformationCoefficient tests
# ---------------------------------------------------------------------------


class TestInformationCoefficient:
    def test_rolling_ic(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import InformationCoefficient
        ic_calc = InformationCoefficient(window=30)
        T, N = synthetic_returns.shape
        signal = np.random.randn(T, N).astype(np.float32)
        fwd = np.roll(synthetic_returns, -1, axis=0)
        rolling = ic_calc.rolling_ic(signal, fwd)
        assert rolling.shape == (T,)
        assert np.isfinite(rolling).all()

    def test_icir(self, synthetic_returns, n_assets):
        from tensor_net.signal_extraction import InformationCoefficient
        ic_calc = InformationCoefficient(window=30)
        T, N = synthetic_returns.shape
        signal = np.random.randn(T, N).astype(np.float32)
        fwd = np.roll(synthetic_returns, -1, axis=0)
        icir = ic_calc.icir(signal, fwd)
        assert np.isfinite(icir)
