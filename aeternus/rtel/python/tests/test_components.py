"""
AETERNUS RTEL — Component tests for signal_engine, portfolio_optimizer,
data_pipeline, and backtest_runner.
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from rtel.signal_engine import (
    MomentumSignal, MeanReversionSignal, LOBImbalanceSignal,
    EMACrossoverSignal, RSISignal, CrossSectionalNormalizer,
    ICWeightedEnsemble, SignalDecayEstimator, SignalEngine,
)
from rtel.portfolio_optimizer import (
    EWMACovariance, MVOptimizer, ERCOptimizer, BlackLitterman,
    TransactionCostModel, PortfolioRebalancer, KellyPositionSizer,
    PortfolioOptimizationEngine, LedoitWolfShrinkage,
)
from rtel.data_pipeline import (
    RawTick, DataValidator, BarAggregator, OnlineNormalizer,
    AnomalyDetector, DataPipeline, SyntheticDataSource,
    LOBFeatureStage, NormalizationStage,
)
from rtel.backtest_runner import (
    BacktestConfig, BacktestRunner, BacktestStats, MultiAssetGBM,
    Portfolio, run_grid_search,
)


# ============================================================================
# Signal engine tests
# ============================================================================

class TestMomentumSignal:
    def test_no_signal_insufficient_data(self):
        sig = MomentumSignal(lookback=10)
        for i in range(5):
            sig.update(0, 100.0 + i)
        assert sig.compute(0) is None

    def test_positive_signal_for_uptrend(self):
        sig = MomentumSignal(lookback=5, skip_period=1)
        for i in range(25):
            sig.update(0, 100.0 + i * 0.5)  # steady uptrend
        val = sig.compute(0)
        assert val is not None
        assert val > 0.0, f"uptrend should give positive signal, got {val}"

    def test_negative_signal_for_downtrend(self):
        sig = MomentumSignal(lookback=5, skip_period=1)
        for i in range(25):
            sig.update(0, 100.0 - i * 0.5)
        val = sig.compute(0)
        assert val is not None
        assert val < 0.0

    def test_signal_in_range(self):
        sig = MomentumSignal(lookback=10)
        for i in range(50):
            sig.update(0, 100.0 + np.random.randn())
        val = sig.compute(0)
        if val is not None:
            assert -1.0 <= val <= 1.0


class TestMeanReversionSignal:
    def test_no_signal_insufficient_data(self):
        sig = MeanReversionSignal(lookback=20)
        for i in range(10):
            sig.update(0, 100.0)
        assert sig.compute(0) is None

    def test_oversold_gives_positive_signal(self):
        sig = MeanReversionSignal(lookback=20, z_cap=2.0)
        for _ in range(20):
            sig.update(0, 100.0)
        # Push price way down
        sig.update(0, 90.0)  # extreme low
        val = sig.compute(0)
        if val is not None:
            assert val > 0.0, "oversold should give buy signal"

    def test_overbought_gives_negative_signal(self):
        sig = MeanReversionSignal(lookback=20, z_cap=2.0)
        for _ in range(20):
            sig.update(0, 100.0)
        sig.update(0, 110.0)  # extreme high
        val = sig.compute(0)
        if val is not None:
            assert val < 0.0, "overbought should give sell signal"


class TestEMACrossover:
    def test_bull_cross(self):
        sig = EMACrossoverSignal(fast=3, slow=10)
        # Feed steady prices first
        for _ in range(30):
            sig.update(0, 100.0)
        # Now trending up sharply
        for i in range(20):
            sig.update(0, 100.0 + i * 2.0)
        val = sig.compute(0)
        assert val is not None
        # Fast EMA should be above slow after uptrend
        assert val > 0.0

    def test_bear_cross(self):
        sig = EMACrossoverSignal(fast=3, slow=10)
        for _ in range(30):
            sig.update(0, 100.0)
        for i in range(20):
            sig.update(0, 100.0 - i * 2.0)
        val = sig.compute(0)
        assert val is not None
        assert val < 0.0


class TestRSISignal:
    def test_overbought_signal(self):
        sig = RSISignal(period=14)
        # Consistent gains → RSI high
        for _ in range(14):
            sig.update(0, 100.0)
        for i in range(20):
            sig.update(0, 100.0 + i * 1.0)
        val = sig.compute(0)
        if val is not None:
            assert val <= 0.0, "overbought RSI should give sell signal"

    def test_insufficient_data(self):
        sig = RSISignal(period=14)
        for _ in range(5):
            sig.update(0, 100.0)
        assert sig.compute(0) is None


class TestCrossSectionalNormalizer:
    def test_normalize_mean_zero_std_one(self):
        signals = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        normalized = CrossSectionalNormalizer.normalize(signals)
        vals = list(normalized.values())
        assert abs(np.mean(vals)) < 1e-9
        assert abs(np.std(vals) - 1.0) < 0.1

    def test_rank_normalize_range(self):
        signals = {i: float(i) for i in range(10)}
        ranked  = CrossSectionalNormalizer.rank_normalize(signals)
        vals    = list(ranked.values())
        assert min(vals) == pytest.approx(-1.0)
        assert max(vals) == pytest.approx(1.0)

    def test_winsorize(self):
        signals = {0: 100.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: -100.0}
        winsorized = CrossSectionalNormalizer.winsorize(signals, z_cap=2.0)
        vals = list(winsorized.values())
        assert all(abs(v) <= 2.0 + 1e-9 for v in vals)


class TestICEnsemble:
    def test_equal_weight_fallback(self):
        ens = ICWeightedEnsemble(min_icir=0.0)
        ens.register_signal("a")
        ens.register_signal("b")
        w = ens.compute_weights()
        assert abs(w.get("a", 0) - 0.5) < 0.01
        assert abs(w.get("b", 0) - 0.5) < 0.01

    def test_combine_returns_clipped(self):
        ens = ICWeightedEnsemble()
        ens.register_signal("a")
        ens.update_signal("a", {0: 5.0, 1: -5.0})
        result = ens.combine([0, 1])
        for v in result.values():
            assert -1.0 <= v <= 1.0


class TestSignalEngine:
    def test_runs_without_error(self):
        engine = SignalEngine(n_assets=5, lookback_short=5, lookback_long=15)
        for step in range(50):
            prices = {i: 100.0 * (1 + 0.001 * step * (1 + 0.1*i)) for i in range(5)}
            engine.update_prices(prices)
            engine.update_forward_returns(prices)
        signals = engine.get_combined_signal()
        assert len(signals) == 5
        for v in signals.values():
            assert -1.1 <= v <= 1.1  # slight tolerance for clipping

    def test_normalized_signal_range(self):
        engine = SignalEngine(n_assets=3, lookback_short=5, lookback_long=15)
        for step in range(30):
            prices = {i: 100.0 + step * (0.1 + 0.05*i) for i in range(3)}
            engine.update_prices(prices)
            engine.update_forward_returns(prices)
        normalized = engine.get_normalized_signal()
        vals = list(normalized.values())
        if vals:
            assert min(vals) >= -1.0 - 1e-9
            assert max(vals) <= 1.0 + 1e-9

    def test_diagnostics(self):
        engine = SignalEngine(n_assets=2)
        for step in range(30):
            prices = {0: 100.0 + step, 1: 200.0 + step * 0.5}
            engine.update_prices(prices)
            engine.update_forward_returns(prices)
        diag = engine.diagnostics()
        assert "icirs" in diag
        assert "weights" in diag
        assert diag["n_assets"] == 2


# ============================================================================
# Portfolio optimizer tests
# ============================================================================

class TestEWMACovariance:
    def test_updates_without_error(self):
        cov_est = EWMACovariance(5)
        for _ in range(20):
            r = np.random.randn(5) * 0.01
            cov_est.update(r)
        cov = cov_est.covariance
        assert cov.shape == (5, 5)
        # Should be positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        assert all(e >= -1e-10 for e in eigvals)

    def test_correlation_diagonal_one(self):
        cov_est = EWMACovariance(4)
        for _ in range(30):
            cov_est.update(np.random.randn(4) * 0.01)
        corr = cov_est.correlation
        for i in range(4):
            assert abs(corr[i, i] - 1.0) < 1e-9


class TestMVOptimizer:
    def test_min_variance_sums_to_one(self):
        n    = 5
        cov  = np.eye(n) * 0.01
        mvo  = MVOptimizer(n)
        w    = mvo.min_variance(cov)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_max_sharpe_sums_to_one(self):
        n   = 5
        cov = np.eye(n) * 0.01
        mu  = np.array([0.05, 0.07, 0.04, 0.08, 0.06])
        mvo = MVOptimizer(n)
        w   = mvo.max_sharpe(mu, cov)
        assert abs(w.sum() - 1.0) < 1e-9
        assert all(wi >= -1e-9 for wi in w)

    def test_efficient_frontier(self):
        n   = 3
        cov = np.eye(n) * 0.01
        mu  = np.array([0.05, 0.07, 0.09])
        mvo = MVOptimizer(n)
        pts = mvo.efficient_frontier(mu, cov, n_points=5)
        assert len(pts) == 5
        for ret, vol, w in pts:
            assert vol >= 0
            assert abs(w.sum() - 1.0) < 1e-9


class TestERCOptimizer:
    def test_equal_risk_contributions(self):
        n   = 4
        cov = np.eye(n) * 0.01
        erc = ERCOptimizer(n)
        w   = erc.optimize(cov)
        assert abs(w.sum() - 1.0) < 1e-9
        # Equal vol → equal weights
        assert all(abs(wi - 0.25) < 0.01 for wi in w)

    def test_unequal_vols(self):
        n = 3
        cov = np.diag([0.01, 0.04, 0.09])  # vols = 0.1, 0.2, 0.3
        erc = ERCOptimizer(n)
        w   = erc.optimize(cov)
        # Higher vol → lower weight
        assert w[0] > w[1] > w[2]


class TestTransactionCostModel:
    def test_cost_increases_with_size(self):
        tc   = TransactionCostModel(spread_bps=5.0, impact_coeff=0.1)
        cost1 = tc.cost_for_trade(10_000, 1e6)
        cost2 = tc.cost_for_trade(100_000, 1e6)
        assert cost2 > cost1

    def test_zero_trade_zero_cost(self):
        tc = TransactionCostModel()
        assert tc.cost_for_trade(0.0, 1e6) == 0.0


class TestKellyPositionSizer:
    def test_positive_edge(self):
        kelly = KellyPositionSizer(kelly_fraction=0.5)
        size  = kelly.size_from_moments(mu=0.01, sigma=0.1)
        assert size > 0.0

    def test_negative_edge_no_position(self):
        kelly = KellyPositionSizer(kelly_fraction=0.5)
        size  = kelly.size_from_moments(mu=-0.01, sigma=0.1)
        # Negative Kelly → clipped to -max_position
        assert size <= 0.0

    def test_max_position_respected(self):
        kelly = KellyPositionSizer(kelly_fraction=1.0, max_position=0.1)
        size  = kelly.size_from_moments(mu=1.0, sigma=0.01)  # huge Kelly
        assert abs(size) <= 0.1 + 1e-9


class TestPortfolioOptimizationEngine:
    def test_runs_erc(self):
        engine = PortfolioOptimizationEngine(5, method="erc")
        for _ in range(20):
            r = np.random.randn(5) * 0.01
            engine.update_returns(r)
        signals = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
        prices  = np.ones(5) * 100.0
        w = engine.compute_target_weights(signals, prices)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_runs_mvo(self):
        engine = PortfolioOptimizationEngine(4, method="mvo")
        for _ in range(30):
            engine.update_returns(np.random.randn(4) * 0.01)
        signals = np.array([0.5, -0.3, 0.2, -0.1])
        prices  = np.ones(4) * 50.0
        w = engine.compute_target_weights(signals, prices)
        assert abs(w.sum() - 1.0) < 1e-9


# ============================================================================
# Data pipeline tests
# ============================================================================

class TestDataValidator:
    def test_valid_tick_accepted(self):
        v    = DataValidator()
        tick = RawTick(0, 0.0, 100.0, 100.1, 1000.0, 1000.0, 100.05, 500.0)
        ok, reason = v.validate(tick)
        assert ok
        assert reason == ""

    def test_inverted_spread_rejected(self):
        v    = DataValidator()
        tick = RawTick(0, 0.0, 100.1, 100.0, 1000.0, 1000.0, 100.05, 500.0)  # ask < bid
        ok, _ = v.validate(tick)
        assert not ok

    def test_large_spread_rejected(self):
        v    = DataValidator(max_spread_bps=10.0)
        tick = RawTick(0, 0.0, 95.0, 110.0, 1000.0, 1000.0, 100.0, 500.0)
        ok, _ = v.validate(tick)
        assert not ok

    def test_price_jump_rejected(self):
        v = DataValidator(max_price_jump_pct=2.0)
        tick1 = RawTick(0, 0.0, 100.0, 100.1, 1000.0, 1000.0, 100.05, 500.0)
        tick2 = RawTick(0, 1.0, 150.0, 150.1, 1000.0, 1000.0, 150.05, 500.0)
        v.validate(tick1)
        ok, reason = v.validate(tick2)
        assert not ok


class TestBarAggregator:
    def test_bar_completes_on_duration(self):
        agg   = BarAggregator(bar_duration_s=1.0)
        tick1 = RawTick(0, 0.0, 99.9, 100.1, 1000.0, 1000.0, 100.0, 100.0)
        tick2 = RawTick(0, 0.5, 100.4, 100.6, 1000.0, 1000.0, 100.5, 200.0)
        tick3 = RawTick(0, 1.5, 101.9, 102.1, 1000.0, 1000.0, 102.0, 300.0)
        assert agg.update(tick1) is None
        assert agg.update(tick2) is None
        bar = agg.update(tick3)
        assert bar is not None
        assert bar.asset_id == 0

    def test_ohlcv_correct(self):
        agg = BarAggregator(bar_duration_s=1.0)
        ticks = [
            RawTick(0, 0.0, 99.9, 100.1, 1000.0, 1000.0, 100.0, 100.0),
            RawTick(0, 0.3, 100.4, 100.6, 1000.0, 1000.0, 100.5, 100.0),
            RawTick(0, 0.7, 99.4, 99.6, 1000.0, 1000.0, 99.5, 100.0),
        ]
        for t in ticks:
            agg.update(t)
        next_tick = RawTick(0, 1.5, 101.0, 101.2, 1000.0, 1000.0, 101.1, 50.0)
        bar = agg.update(next_tick)
        assert bar is not None
        assert bar.high >= bar.low
        assert bar.n_ticks == 3


class TestOnlineNormalizer:
    def test_zero_mean_unit_std(self):
        norm = OnlineNormalizer(3)
        data = [np.array([1.0, 2.0, 3.0]),
                np.array([2.0, 3.0, 4.0]),
                np.array([3.0, 4.0, 5.0])]
        for d in data:
            norm.update(d)
        z = norm.transform(np.array([2.0, 3.0, 4.0]))
        assert np.all(np.abs(z) < 5.0)

    def test_clipping(self):
        norm = OnlineNormalizer(2, clip_z=2.0)
        for _ in range(20):
            norm.update(np.array([0.0, 0.0]))
        z = norm.transform(np.array([1000.0, -1000.0]))
        assert np.all(np.abs(z) <= 2.0 + 1e-9)


class TestDataPipeline:
    def test_processes_ticks(self):
        pipeline = DataPipeline(n_assets=3)
        source   = SyntheticDataSource(n_assets=3)
        for _ in range(30):
            ticks = source.next_ticks()
            pipeline.process_batch(ticks)
        stats = pipeline.stats()
        assert stats["n_ticks"] > 0

    def test_handler_called(self):
        pipeline = DataPipeline(n_assets=2)
        received = []
        pipeline.add_handler(lambda f: received.append(f))

        source = SyntheticDataSource(n_assets=2)
        for _ in range(10):
            ticks = source.next_ticks()
            pipeline.process_batch(ticks)

        assert len(received) > 0

    def test_feature_sequence_shape(self):
        pipeline = DataPipeline(n_assets=2, n_lob_features=8)
        source   = SyntheticDataSource(n_assets=2)
        for _ in range(20):
            pipeline.process_batch(source.next_ticks())

        seq = pipeline.get_feature_sequence(0, seq_len=8)
        if seq is not None:
            assert seq.shape == (8, 8)


class TestSyntheticDataSource:
    def test_generates_valid_ticks(self):
        source = SyntheticDataSource(n_assets=5)
        for _ in range(10):
            ticks = source.next_ticks()
            assert len(ticks) == 5
            for t in ticks:
                assert t.bid > 0
                assert t.ask >= t.bid
                assert t.bid_size >= 0

    def test_prices_evolve(self):
        source = SyntheticDataSource(n_assets=1, sigma=0.02)
        prices = []
        for _ in range(50):
            ticks = source.next_ticks()
            prices.append(ticks[0].mid())
        assert len(set(prices)) > 10, "prices should vary"


# ============================================================================
# Backtest runner tests
# ============================================================================

class TestPortfolioClass:
    def test_buy_and_sell(self):
        port = Portfolio(100_000.0, 3)
        port.apply_trade(0, 100.0, 50.0, 0.0)
        assert abs(port.positions[0] - 100.0) < 1e-9
        assert abs(port.cash - (100_000.0 - 5000.0)) < 1e-9

    def test_equity_calculation(self):
        port   = Portfolio(100_000.0, 2)
        port.apply_trade(0, 100.0, 50.0, 0.0)
        prices = np.array([55.0, 0.0])
        equity = port.equity(prices)
        assert abs(equity - (100_000.0 - 5000.0 + 5500.0)) < 1e-6

    def test_pnl_on_close(self):
        port = Portfolio(100_000.0, 1)
        port.apply_trade(0, 100.0, 100.0, 0.0)
        port.apply_trade(0, -100.0, 110.0, 0.0)
        assert abs(port.realized_pnl - 1000.0) < 1e-6


class TestMultiAssetGBM:
    def test_prices_positive(self):
        gbm = MultiAssetGBM(n_assets=5, mu=0.0, sigma=0.01, seed=42)
        for _ in range(50):
            prices = gbm.step()
            assert all(p > 0 for p in prices)

    def test_prices_vary(self):
        gbm = MultiAssetGBM(n_assets=1, mu=0.0, sigma=0.02, seed=123)
        prices = [gbm.step()[0] for _ in range(30)]
        assert len(set(prices)) > 10

    def test_makes_valid_ticks(self):
        gbm   = MultiAssetGBM(n_assets=3, mu=0.0, sigma=0.01, seed=0)
        ticks = gbm.make_ticks()
        assert len(ticks) == 3
        for t in ticks:
            assert t.bid > 0 and t.ask >= t.bid


class TestBacktestRunner:
    def test_runs_to_completion(self):
        cfg    = BacktestConfig(n_assets=3, n_steps=100)
        runner = BacktestRunner(cfg)
        stats  = runner.run()
        assert stats.n_steps >= 2
        assert len(stats.equity_curve) >= 2

    def test_equity_curve_positive(self):
        cfg    = BacktestConfig(n_assets=3, n_steps=100, initial_capital=50_000.0)
        runner = BacktestRunner(cfg)
        stats  = runner.run()
        assert all(e > 0 for e in stats.equity_curve)

    def test_sharpe_finite(self):
        cfg    = BacktestConfig(n_assets=5, n_steps=200)
        runner = BacktestRunner(cfg)
        stats  = runner.run()
        assert math.isfinite(stats.sharpe)

    def test_erc_method(self):
        cfg = BacktestConfig(n_assets=4, n_steps=100, opt_method="erc")
        runner = BacktestRunner(cfg)
        stats  = runner.run()
        assert stats.n_trades >= 0

    def test_mvo_method(self):
        cfg = BacktestConfig(n_assets=4, n_steps=100, opt_method="mvo")
        runner = BacktestRunner(cfg)
        stats  = runner.run()
        assert stats.n_trades >= 0


class TestBacktestStats:
    def test_compute_from_equity(self):
        equity = [100.0, 102.0, 101.5, 105.0, 103.0, 108.0]
        stats  = BacktestStats.from_equity_curve(
            equity, n_trades=10, commissions=50.0,
            trade_pnls=[100.0, -50.0, 200.0, -30.0],
        )
        assert stats.total_return > 0
        assert stats.max_drawdown >= 0
        assert 0.0 <= stats.win_rate <= 1.0

    def test_declining_equity_negative_return(self):
        equity = [100.0, 99.0, 98.0, 97.0]
        stats  = BacktestStats.from_equity_curve(equity, 0, 0.0, [])
        assert stats.total_return < 0


class TestGridSearch:
    def test_returns_sorted_by_sharpe(self):
        cfg = BacktestConfig(n_assets=3, n_steps=50)
        results = run_grid_search(cfg, methods=["erc", "min_var"],
                                  lookbacks=[10], verbose=False)
        sharpes = [r.sharpe for r in results]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_all_results_have_stats(self):
        cfg = BacktestConfig(n_assets=2, n_steps=50)
        results = run_grid_search(cfg, methods=["erc"],
                                  lookbacks=[10, 15], verbose=False)
        assert len(results) == 2
        for r in results:
            assert hasattr(r.stats, "sharpe")
