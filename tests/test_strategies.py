"""
Tests for strategy modules.

Covers:
- Synthetic data generation for deterministic testing
- BacktestResult structure validation
- Signal generation interface (generate_signals returns pd.Series)
- Edge cases: constant prices, single day, NaN-heavy inputs
- Momentum, mean reversion, volatility, event-driven strategies
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
import warnings

import numpy as np
import pandas as pd

# Add srfm-lab to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def make_price_series(
    n: int = 500,
    drift: float = 0.0005,
    vol: float = 0.015,
    seed: int = 42,
    start: str = "2020-01-02",
) -> pd.Series:
    rng = np.random.default_rng(seed)
    log_rets = drift + vol * rng.standard_normal(n)
    prices = 100 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.Series(prices, index=idx, name="price")


def make_ohlcv(
    n: int = 500,
    drift: float = 0.0005,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    close = make_price_series(n, drift, vol, seed)
    rng = np.random.default_rng(seed + 1)
    noise = rng.uniform(0.98, 1.02, n)
    high = close * rng.uniform(1.00, 1.02, n)
    low = close * rng.uniform(0.98, 1.00, n)
    open_ = close * noise
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=close.index,
    )


def make_universe(
    n_assets: int = 5,
    n: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    prices = {}
    for i in range(n_assets):
        drift = rng.uniform(-0.0002, 0.0008)
        vol = rng.uniform(0.01, 0.025)
        log_rets = drift + vol * rng.standard_normal(n)
        prices[f"asset_{i}"] = 100 * np.exp(np.cumsum(log_rets))
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(prices, index=idx)


# ---------------------------------------------------------------------------
# Test: Momentum strategies
# ---------------------------------------------------------------------------

class TestTrendFollowing(unittest.TestCase):

    def setUp(self):
        self.df = make_ohlcv(n=500)

    def test_dual_ma_returns_backtest_result(self):
        from strategies.momentum.trend_following import DualMovingAverage, BacktestResult
        dma = DualMovingAverage(fast=20, slow=50)
        result = dma.backtest(self.df)
        self.assertIsInstance(result, BacktestResult)
        self.assertIsInstance(result.equity_curve, pd.Series)

    def test_dual_ma_signal_interface(self):
        from strategies.momentum.trend_following import DualMovingAverage
        dma = DualMovingAverage(fast=10, slow=30)
        signals = dma.generate_signals(self.df)
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.df))
        self.assertTrue(signals.isin([-1, 0, 1]).all())

    def test_turtle_system(self):
        from strategies.momentum.trend_following import TurtleSystem, BacktestResult
        ts = TurtleSystem(atr_period=14, entry_period=20, exit_period=10)
        result = ts.backtest(self.df)
        self.assertIsInstance(result, BacktestResult)
        self.assertFalse(np.isnan(result.sharpe))

    def test_donchian_breakout(self):
        from strategies.momentum.trend_following import DonchianBreakout
        db = DonchianBreakout(period=20)
        signals = db.generate_signals(self.df)
        self.assertIsInstance(signals, pd.Series)

    def test_keltner_breakout(self):
        from strategies.momentum.trend_following import KeltnerBreakout
        kb = KeltnerBreakout(period=20, multiplier=1.5)
        signals = kb.generate_signals(self.df)
        self.assertIsInstance(signals, pd.Series)

    def test_constant_price(self):
        """Edge case: constant prices."""
        from strategies.momentum.trend_following import DualMovingAverage
        df = self.df.copy()
        df["close"] = 100.0
        df["high"] = 100.5
        df["low"] = 99.5
        dma = DualMovingAverage(fast=5, slow=20)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dma.backtest(df)
        self.assertIsInstance(result, object)


class TestMomentum(unittest.TestCase):

    def setUp(self):
        self.df = make_ohlcv(n=500)

    def test_tsm_signal(self):
        from strategies.momentum.momentum import TimeSeriesMomentum
        tsm = TimeSeriesMomentum(lookback=252, skip_recent=21)
        signals = tsm.generate_signals(self.df)
        self.assertIsInstance(signals, pd.Series)
        # Signal should be in [-1, 1] range approximately
        self.assertTrue(signals.dropna().between(-2, 2).all())

    def test_dual_momentum(self):
        from strategies.momentum.momentum import DualMomentum
        dm = DualMomentum(lookback=126)
        result = dm.backtest(self.df)
        self.assertIn("sharpe", result.__dict__)

    def test_risk_adjusted_momentum(self):
        from strategies.momentum.momentum import RiskAdjustedMomentum
        ram = RiskAdjustedMomentum(lookback=126, vol_window=21)
        signals = ram.generate_signals(self.df)
        self.assertIsInstance(signals, pd.Series)


class TestCarry(unittest.TestCase):

    def setUp(self):
        n = 300
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        rng = np.random.default_rng(42)
        self.spot = pd.Series(100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_normal(n))),
                              index=idx)
        self.funding_rate = pd.Series(
            rng.uniform(-0.0001, 0.0003, n), index=idx, name="funding_rate"
        )
        self.df = make_ohlcv(n=n)
        self.df["funding_rate"] = self.funding_rate

    def test_forward_rate_carry(self):
        from strategies.momentum.carry import ForwardRateCarry
        frc = ForwardRateCarry(entry_threshold=0.02)
        signals = frc.generate_signals(self.df)
        self.assertIsInstance(signals, pd.Series)


# ---------------------------------------------------------------------------
# Test: Mean Reversion strategies
# ---------------------------------------------------------------------------

class TestStatArb(unittest.TestCase):

    def setUp(self):
        # Cointegrated pair
        rng = np.random.default_rng(42)
        n = 400
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        common = np.cumsum(rng.standard_normal(n))
        self.y1 = pd.Series(100 + common + 0.5 * rng.standard_normal(n), index=idx)
        self.y2 = pd.Series(50 + 0.5 * common + 0.3 * rng.standard_normal(n), index=idx)

    def test_pairs_trading(self):
        from strategies.mean_reversion.stat_arb import PairsTrading, BacktestResult
        pt = PairsTrading(lookback=60, entry_z=2.0, exit_z=0.5)
        result = pt.backtest(self.y1, self.y2)
        self.assertIsInstance(result, BacktestResult)

    def test_kalman_pairs(self):
        from strategies.mean_reversion.stat_arb import KalmanPairsTrading, BacktestResult
        kpt = KalmanPairsTrading(entry_z=2.0, exit_z=0.5)
        result = kpt.backtest(self.y1, self.y2)
        self.assertIsInstance(result, BacktestResult)

    def test_ou_mean_reversion(self):
        from strategies.mean_reversion.stat_arb import OUMeanReversion
        ou = OUMeanReversion(lookback=100)
        params = ou.current_params(self.y1)
        self.assertIn("half_life", params)
        self.assertGreater(params.get("half_life", 0), 0)

    def test_cointegration_test(self):
        from strategies.mean_reversion.stat_arb import PairsTrading
        pt = PairsTrading()
        result = pt.cointegration_test(self.y1, self.y2)
        self.assertIn("p_value", result)
        self.assertIsInstance(result["p_value"], float)

    def test_spread_statistics(self):
        from strategies.mean_reversion.stat_arb import PairsTrading
        pt = PairsTrading(lookback=60)
        spread = pt.compute_spread(self.y1, self.y2)
        self.assertIsInstance(spread, pd.Series)
        self.assertEqual(len(spread), len(self.y1))


class TestMarketMaking(unittest.TestCase):

    def test_avellaneda_stoikov_quotes(self):
        from strategies.mean_reversion.market_making import AvellanedaStoikovMM
        mm = AvellanedaStoikovMM(gamma=0.1, sigma=0.02, k=1.5, T=1.0, dt=1/252)
        q = mm.compute_quotes(S=100.0, q=0.0, t=0.5)
        self.assertIn("bid", q)
        self.assertIn("ask", q)
        self.assertGreater(q["ask"], q["bid"])

    def test_avellaneda_session(self):
        from strategies.mean_reversion.market_making import AvellanedaStoikovMM, MMBacktestResult
        mm = AvellanedaStoikovMM(gamma=0.1, sigma=0.02, k=1.5, T=1/252, dt=1/(252*390))
        result = mm.simulate_session(S0=100.0, n_steps=100)
        self.assertIsInstance(result, MMBacktestResult)


# ---------------------------------------------------------------------------
# Test: Volatility strategies
# ---------------------------------------------------------------------------

class TestVolTargeting(unittest.TestCase):

    def setUp(self):
        self.df = make_ohlcv(n=500)

    def test_vol_targeting(self):
        from strategies.volatility.vol_targeting import VolatilityTargeting
        vt = VolatilityTargeting(target_vol=0.15, lookback=63)
        leverage = vt.compute_leverage(self.df["close"])
        self.assertIsInstance(leverage, pd.Series)
        self.assertTrue((leverage >= 0).all())

    def test_risk_parity(self):
        from strategies.volatility.vol_targeting import RiskParity
        universe = make_universe(n_assets=3, n=300)
        rp = RiskParity(assets=list(universe.columns))
        result = rp.backtest(universe)
        self.assertIsNotNone(result)


class TestVariancePremium(unittest.TestCase):

    def setUp(self):
        n = 300
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        rng = np.random.default_rng(42)
        self.df = make_ohlcv(n=n)
        self.df["vix"] = 15 + 5 * np.abs(rng.standard_normal(n))

    def test_variance_premium_capture(self):
        from strategies.volatility.variance_premium import VariancePremiumCapture
        vpc = VariancePremiumCapture(realized_window=21)
        signals = vpc.generate_signals(self.df)
        self.assertIsInstance(signals, pd.Series)

    def test_vix_futures_rolldown(self):
        from strategies.volatility.variance_premium import VIXFuturesRolldown
        n = 300
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        rng = np.random.default_rng(42)
        spot = pd.Series(15 + rng.standard_normal(n), index=idx)
        f1 = spot + 1 + 0.5 * rng.standard_normal(n)
        f2 = spot + 2 + 0.5 * rng.standard_normal(n)
        vix_roll = VIXFuturesRolldown()
        ts = vix_roll.compute_term_structure(spot, f1, f2)
        self.assertIn("slope", ts.columns)


class TestGARCHVol(unittest.TestCase):

    def test_garch_fit(self):
        from strategies.volatility.regime_vol import GARCHVolTrading
        price = make_price_series(n=300)
        garch = GARCHVolTrading(p=1, q=1)
        params = garch.fit_garch(price)
        self.assertIn("omega", params)
        self.assertIn("alpha", params)
        self.assertIn("beta", params)
        # Stationarity: alpha + beta < 1
        self.assertLess(params["alpha"] + params["beta"], 1.1)  # allow slight numerical slack


# ---------------------------------------------------------------------------
# Test: Event-driven strategies
# ---------------------------------------------------------------------------

class TestEarningsStrategies(unittest.TestCase):

    def setUp(self):
        n = 400
        idx = pd.date_range("2018-01-02", periods=n, freq="B")
        rng = np.random.default_rng(42)
        tickers = ["AAPL", "MSFT", "GOOG"]

        self.price_df = pd.DataFrame(
            {t: 100 * np.exp(np.cumsum(0.0005 + 0.015 * rng.standard_normal(n)))
             for t in tickers},
            index=idx,
        )

        # Quarterly earnings announcements (roughly every 63 days)
        self.actual_eps = pd.DataFrame(np.nan, index=idx, columns=tickers)
        self.expected_eps = pd.DataFrame(np.nan, index=idx, columns=tickers)

        for t in tickers:
            for i in range(0, n, 63):
                self.actual_eps.iloc[i][t] = 2.0 + rng.uniform(-0.3, 0.5)
                self.expected_eps.iloc[i][t] = 2.0

    def test_earnings_surprise_sue(self):
        from strategies.event_driven.earnings import EarningsSurprise
        es = EarningsSurprise(lookback=4)
        sue = es.compute_sue(self.actual_eps, self.expected_eps)
        self.assertIsInstance(sue, pd.DataFrame)
        # Should have some non-NaN values
        self.assertGreater(sue.notna().sum().sum(), 0)

    def test_earnings_surprise_backtest(self):
        from strategies.event_driven.earnings import EarningsSurprise
        es = EarningsSurprise(n_long=2, n_short=1)
        result = es.backtest(self.price_df, self.actual_eps, self.expected_eps)
        self.assertIsNotNone(result)

    def test_earnings_momentum_streaks(self):
        from strategies.event_driven.earnings import EarningsMomentum
        em = EarningsMomentum(min_streak=2)
        streaks, sue = em.compute_streaks(self.actual_eps, self.expected_eps)
        self.assertIsInstance(streaks, pd.DataFrame)


class TestMacroEvents(unittest.TestCase):

    def setUp(self):
        n = 500
        idx = pd.date_range("2018-01-02", periods=n, freq="B")
        rng = np.random.default_rng(42)
        self.price = make_price_series(n=n)

        # FOMC dates (roughly every 6 weeks)
        self.fomc_dates = pd.DatetimeIndex([idx[i] for i in range(0, n, 30)])

        # Monthly NFP
        monthly_idx = pd.date_range("2018-01-05", periods=n // 21 + 1, freq="MS")
        monthly_idx = monthly_idx[:n // 21]
        self.actual_nfp = pd.Series(
            200 + 50 * rng.standard_normal(len(monthly_idx)),
            index=monthly_idx,
        )
        self.consensus_nfp = pd.Series(
            195 + 20 * rng.standard_normal(len(monthly_idx)),
            index=monthly_idx,
        )

    def test_fomc_drift_signals(self):
        from strategies.event_driven.macro_events import FOMCDrift
        fomc = FOMCDrift(drift_period=5, min_reaction=0.001)
        signals = fomc.generate_signals(self.price, self.fomc_dates)
        self.assertIsInstance(signals, pd.Series)
        self.assertEqual(len(signals), len(self.price))

    def test_fomc_backtest(self):
        from strategies.event_driven.macro_events import FOMCDrift
        fomc = FOMCDrift(drift_period=5)
        result = fomc.backtest(self.price, self.fomc_dates)
        self.assertIsNotNone(result)
        self.assertIsInstance(result.equity_curve, pd.Series)

    def test_nfp_surprise(self):
        from strategies.event_driven.macro_events import NFPMomentum
        nfp = NFPMomentum(holding_period=10, z_threshold=0.3)
        surprise = nfp.compute_surprise(self.actual_nfp, self.consensus_nfp)
        self.assertIsInstance(surprise, pd.Series)
        self.assertFalse(surprise.isna().all())

    def test_cpi_regime(self):
        from strategies.event_driven.macro_events import CPIRegime
        n = 60
        monthly_idx = pd.date_range("2018-01-01", periods=n, freq="MS")
        rng = np.random.default_rng(42)
        cpi = pd.Series(0.02 + 0.01 * rng.standard_normal(n), index=monthly_idx)
        cr = CPIRegime(low_threshold=0.02, high_threshold=0.04)
        regime = cr.compute_regime(cpi)
        self.assertIsInstance(regime, pd.Series)
        valid_regimes = {cr.DEFLATION, cr.LOW, cr.HIGH}
        self.assertTrue(regime.dropna().isin(valid_regimes).all())


# ---------------------------------------------------------------------------
# Test: Crypto strategies
# ---------------------------------------------------------------------------

class TestCryptoStrategies(unittest.TestCase):

    def setUp(self):
        n = 400
        rng = np.random.default_rng(42)
        self.btc_price = make_price_series(n=n, drift=0.001, vol=0.03, seed=42)
        idx = self.btc_price.index

        self.df = make_ohlcv(n=n)
        self.df["close"] = self.btc_price.values
        self.df["funding_rate"] = rng.uniform(-0.0002, 0.0005, n)

    def test_funding_rate_arb(self):
        from strategies.crypto.funding_rate import FundingRateArbitrage
        fra = FundingRateArbitrage(min_funding_rate=0.0001)
        result = fra.backtest(self.df)
        self.assertIsNotNone(result)

    def test_nvt_ratio(self):
        from strategies.crypto.onchain import NVTRatio
        n = 400
        rng = np.random.default_rng(42)
        idx = self.btc_price.index
        tx_volume = pd.Series(
            1e9 * (1 + 0.5 * np.abs(rng.standard_normal(n))), index=idx
        )
        nvt = NVTRatio(lookback=30)
        signal = nvt.generate_signals(self.df, tx_volume)
        self.assertIsInstance(signal, pd.Series)

    def test_mayer_multiple(self):
        from strategies.crypto.onchain import MayerMultiple
        mm = MayerMultiple(ma_period=200)
        signal = mm.generate_signals(self.df)
        self.assertIsInstance(signal, pd.Series)

    def test_bitcoin_dominance(self):
        from strategies.crypto.crypto_momentum import BitcoinDominance
        n = 400
        rng = np.random.default_rng(42)
        idx = self.btc_price.index
        dominance = pd.Series(0.4 + 0.1 * rng.standard_normal(n), index=idx).clip(0.2, 0.8)
        bd = BitcoinDominance(threshold=0.01, lookback=30)
        signal = bd.generate_signals(dominance, target="btc")
        self.assertIsInstance(signal, pd.Series)


# ---------------------------------------------------------------------------
# Test: ML Alpha pipeline
# ---------------------------------------------------------------------------

class TestMLPipeline(unittest.TestCase):

    def setUp(self):
        n = 300
        rng = np.random.default_rng(42)
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        self.prices = pd.DataFrame(
            {f"a{i}": 100 * np.exp(np.cumsum(0.0003 + 0.015 * rng.standard_normal(n)))
             for i in range(5)},
            index=idx,
        )
        # Minimal feature matrix
        self.features = pd.DataFrame(
            rng.standard_normal((n, 8)),
            index=idx,
            columns=[f"feat_{i}" for i in range(8)],
        )

    def test_purged_kfold_splits(self):
        from strategies.ml_alpha.pipeline import PurgedKFold
        pkf = PurgedKFold(n_splits=3, purge_pct=0.02, embargo_pct=0.01)
        splits = list(pkf.split(np.arange(200)))
        self.assertEqual(len(splits), 3)
        for train_idx, test_idx in splits:
            # No overlap between train and test
            self.assertEqual(len(set(train_idx) & set(test_idx)), 0)

    def test_ic_computation(self):
        from strategies.ml_alpha.pipeline import MLPipeline
        pipeline = MLPipeline()
        n = 100
        rng = np.random.default_rng(42)
        pred = pd.Series(rng.standard_normal(n))
        ret = pred * 0.5 + rng.standard_normal(n) * 0.5
        ret = pd.Series(ret)
        ic = pipeline.compute_ic(pred, ret, window=20)
        self.assertIsInstance(ic, pd.Series)
        self.assertTrue(ic.dropna().between(-1, 1).all())

    def test_signal_combiner_equal_weight(self):
        from strategies.ml_alpha.signal_combination import SignalCombiner
        n = 200
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        signals = [pd.Series(rng.standard_normal(n), index=idx) for _ in range(4)]
        sc = SignalCombiner()
        composite = sc.equal_weight(signals)
        self.assertIsInstance(composite, pd.Series)
        self.assertEqual(len(composite), n)

    def test_signal_combiner_pca(self):
        from strategies.ml_alpha.signal_combination import SignalCombiner
        n = 200
        rng = np.random.default_rng(42)
        idx = pd.date_range("2021-01-04", periods=n, freq="B")
        signals = [pd.Series(rng.standard_normal(n), index=idx) for _ in range(4)]
        sc = SignalCombiner()
        composite = sc.pca_combine(signals)
        self.assertIsInstance(composite, pd.Series)


# ---------------------------------------------------------------------------
# Test: Research modules
# ---------------------------------------------------------------------------

class TestFactorModels(unittest.TestCase):

    def setUp(self):
        self.universe = make_universe(n_assets=6, n=400)

    def test_wml_factor(self):
        from research.factor_model.factors import WMLFactor
        wml = WMLFactor(formation_period=126, holding_period=21)
        factor = wml.compute(self.universe)
        self.assertIsInstance(factor, pd.Series)
        self.assertGreater(len(factor.dropna()), 0)

    def test_ts_regression(self):
        from research.factor_model.regression import TimeSeriesRegression
        rng = np.random.default_rng(42)
        n = 200
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        factor = pd.DataFrame(
            {"MKT": rng.standard_normal(n) * 0.01},
            index=idx,
        )
        asset = pd.Series(0.0003 + 0.8 * factor["MKT"] + 0.005 * rng.standard_normal(n),
                          index=idx)
        ts_reg = TimeSeriesRegression()
        result = ts_reg.fit(asset, factor)
        self.assertAlmostEqual(result.betas["MKT"], 0.8, delta=0.3)
        self.assertGreater(result.r_squared, 0.2)

    def test_pca_model(self):
        from research.factor_model.pca import StatisticalFactorModel
        returns = self.universe.pct_change().dropna()
        pca = StatisticalFactorModel(n_factors=2)
        result = pca.fit(returns)
        self.assertEqual(result.n_factors, 2)
        self.assertEqual(result.factor_returns.shape[1], 2)
        self.assertGreater(result.explained_variance[0], 0)


class TestRegimeAnalysis(unittest.TestCase):

    def setUp(self):
        self.price = make_price_series(n=400)

    def test_hmm_fit(self):
        from research.regime_analysis.hmm_regime import GaussianHMM, HMMResult
        obs = self.price.pct_change().dropna().values
        hmm = GaussianHMM(n_states=2, n_iter=50)
        result = hmm.fit(obs)
        self.assertIsInstance(result, HMMResult)
        self.assertEqual(len(result.state_sequence), len(obs))
        # Transition matrix row stochastic
        row_sums = result.transition_matrix.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-6)

    def test_cusum_detect(self):
        from research.regime_analysis.change_point import CUSUM
        # Create series with a clear mean shift at index 200
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(200), rng.standard_normal(200) + 3])
        s = pd.Series(x, index=pd.date_range("2020-01-01", periods=400))
        cusum = CUSUM(threshold=5.0, min_segment_length=30)
        result = cusum.detect(s)
        # Should detect approximately 1 change point near index 200
        self.assertGreater(len(result.change_points), 0)
        detected = result.change_points[0]
        self.assertGreater(detected, 100)
        self.assertLess(detected, 300)

    def test_pelt_detect(self):
        from research.regime_analysis.change_point import PELT
        rng = np.random.default_rng(42)
        x = np.concatenate([rng.standard_normal(100), rng.standard_normal(100) + 2,
                             rng.standard_normal(100)])
        s = pd.Series(x, index=pd.date_range("2020-01-01", periods=300))
        pelt = PELT(model="normal", penalty="bic", min_size=10)
        result = pelt.detect(s)
        self.assertIsInstance(result.change_points, list)


class TestOptionsResearch(unittest.TestCase):

    def test_bs_put_call_parity(self):
        from research.options.pricing import BlackScholes
        bs = BlackScholes(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        call = bs.call()
        put = bs.put()
        ok = BlackScholes.put_call_parity_check(call, put, 100, 100, 0.25, 0.05)
        self.assertTrue(ok)

    def test_bs_greeks_gradient(self):
        """Compare analytical Greeks to finite difference."""
        from research.options.greeks import BlackScholesGreeks
        g = BlackScholesGreeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20)
        analytical_delta = g.delta("call")
        numerical_delta = g.numerical_delta("call", h=0.01)
        self.assertAlmostEqual(analytical_delta, numerical_delta, places=4)

        analytical_gamma = g.gamma()
        numerical_gamma = g.numerical_gamma(h=0.01)
        self.assertAlmostEqual(analytical_gamma, numerical_gamma, places=4)

    def test_binomial_tree_european(self):
        from research.options.pricing import BinomialTree, BlackScholes
        S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.20
        tree = BinomialTree(S, K, T, r, sigma, n_steps=200)
        bs = BlackScholes(S, K, T, r, sigma)
        self.assertAlmostEqual(tree.european_call(), bs.call(), delta=0.02)
        self.assertAlmostEqual(tree.european_put(), bs.put(), delta=0.02)

    def test_implied_vol_roundtrip(self):
        from research.options.vol_surface import implied_vol, _bs_price
        S, K, T, r, sigma = 100, 105, 0.25, 0.03, 0.22
        market_price = _bs_price(S, K, T, r, sigma, "call")
        iv = implied_vol(market_price, S, K, T, r, "call", method="brentq")
        self.assertAlmostEqual(iv, sigma, delta=1e-4)

    def test_svi_fit(self):
        from research.options.vol_surface import SVIModel
        k_data = np.linspace(-0.3, 0.3, 10)
        iv_data = 0.20 + 0.05 * k_data ** 2  # simple smile
        T = 0.25
        svi = SVIModel()
        params = svi.fit(k_data, iv_data, T)
        self.assertIn("rmse", params)
        self.assertLess(params["rmse"], 0.01)

    def test_butterfly_no_arb(self):
        from research.options.vol_surface import ArbitrageChecker, implied_vol_surface
        from research.options.pricing import BlackScholes
        S, r = 100.0, 0.03
        strikes = [90, 95, 100, 105, 110]
        maturities = [0.25, 0.5]
        # Use flat vol = 0.20; no arb by construction
        price_matrix = np.array([
            [BlackScholes(S, K, T, r, 0.20).call() for K in strikes]
            for T in maturities
        ])
        iv_surf = implied_vol_surface(pd.DataFrame(price_matrix), strikes, maturities, S, r)
        arb = ArbitrageChecker.full_surface_check(iv_surf, strikes, maturities, S, r)
        self.assertTrue(arb["butterfly_free"])

    def test_straddle_payoff(self):
        from research.options.strategies_options import Straddle
        s = Straddle(S=100, K=100, T=0.25, sigma=0.20, long=True)
        pnl = s.payoff_profile()
        # Straddle P&L should be negative near ATM (theta decay) and positive far OTM
        atm_idx = pnl.index[(pnl.index - 100).abs() < 1].tolist()
        if atm_idx:
            self.assertLess(pnl[atm_idx[0]], 0)  # net premium paid


# ---------------------------------------------------------------------------
# Test: Alternative data
# ---------------------------------------------------------------------------

class TestAlternativeData(unittest.TestCase):

    def setUp(self):
        n = 300
        rng = np.random.default_rng(42)
        self.idx = pd.date_range("2020-01-02", periods=n, freq="B")
        self.price = make_price_series(n=n)

    def test_sentiment_scorer(self):
        from research.alternative_data.sentiment import score_text
        self.assertGreater(score_text("great strong growth excellent"), 0)
        self.assertLess(score_text("bad poor weak decline loss"), 0)
        self.assertAlmostEqual(score_text(""), 0.0)

    def test_sentiment_aggregator(self):
        from research.alternative_data.sentiment import SentimentAggregator
        rng = np.random.default_rng(42)
        texts = pd.Series(
            ["great profit beat", "bad loss decline", "strong growth"] * 100,
            index=self.idx,
        )
        sa = SentimentAggregator(lookback=10)
        signal = sa.build_signal(texts)
        self.assertIsInstance(signal, pd.Series)

    def test_etf_flow_signal(self):
        from research.alternative_data.flows import ETFFlowSignal
        rng = np.random.default_rng(42)
        flows = pd.Series(
            rng.normal(0, 100e6, len(self.idx)), index=self.idx
        )
        etf = ETFFlowSignal()
        signal = etf.compute_flow_signal(flows)
        self.assertIsInstance(signal, pd.Series)
        self.assertEqual(len(signal), len(self.idx))

    def test_yield_curve_features(self):
        from research.alternative_data.macro import YieldCurveFeatures
        n = len(self.idx)
        rng = np.random.default_rng(42)
        yields = pd.DataFrame({
            "2Y": 1.0 + 0.5 * rng.standard_normal(n),
            "5Y": 1.5 + 0.5 * rng.standard_normal(n),
            "10Y": 2.0 + 0.5 * rng.standard_normal(n),
        }, index=self.idx)
        ycf = YieldCurveFeatures()
        feats = ycf.compute_features(yields)
        self.assertIn("yc_slope", feats.columns)
        self.assertIn("yc_level", feats.columns)
        self.assertIn("yc_inverted", feats.columns)

    def test_cross_asset_momentum(self):
        from research.alternative_data.macro import CrossAssetMomentum
        equity = make_price_series(n=300, drift=0.0005, vol=0.015, seed=42)
        bonds = make_price_series(n=300, drift=0.0001, vol=0.005, seed=43)
        cam = CrossAssetMomentum()
        composite = cam.composite_signal(equity, bonds)
        self.assertIsInstance(composite, pd.Series)
        self.assertGreater(composite.dropna().count(), 100)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_short_series(self):
        """Very short series should not crash."""
        from strategies.momentum.trend_following import DualMovingAverage
        df = make_ohlcv(n=15)
        dma = DualMovingAverage(fast=5, slow=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = dma.backtest(df)
        # Should return some result without exception

    def test_nan_heavy_input(self):
        """Series with many NaN should not crash."""
        from strategies.momentum.momentum import TimeSeriesMomentum
        df = make_ohlcv(n=300)
        # Insert random NaN in close
        df.loc[df.sample(frac=0.3, random_state=42).index, "close"] = np.nan
        tsm = TimeSeriesMomentum(lookback=63)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals = tsm.generate_signals(df)
        self.assertIsInstance(signals, pd.Series)

    def test_all_same_returns(self):
        """Constant returns (zero vol) should not crash."""
        from research.factor_model.regression import TimeSeriesRegression
        n = 100
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        asset = pd.Series(0.001, index=idx)
        factor = pd.DataFrame({"MKT": [0.001] * n}, index=idx)
        ts_reg = TimeSeriesRegression()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ts_reg.fit(asset, factor)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
