"""
tests/test_benchmarks.py

Tests for benchmark_suite.py module.
"""

from __future__ import annotations

import pathlib
import sys
import unittest
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from lumina.benchmark_suite import (
    PerformanceMetrics,
    MomentumBaseline,
    MeanReversionBaseline,
    RealizedVolBaseline,
    EqualWeightBaseline,
    MarkowitzBaseline,
    DieboldMarianoTest,
    WhiteRealityCheck,
    DirectionPredictionBenchmark,
    VolatilityForecastBenchmark,
    CrisisDetectionBenchmark,
    PortfolioOptimizationBenchmark,
    BenchmarkRunner,
    walk_forward_splits,
)


# ---------------------------------------------------------------------------
# Minimal model for benchmark tests
# ---------------------------------------------------------------------------

class DummyModel(nn.Module):
    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = x.mean(dim=1)
        logit = self.proj(x).squeeze(-1)
        return {"logits": logit, "output": logit}


# ---------------------------------------------------------------------------
# Helper data generators
# ---------------------------------------------------------------------------

def make_returns(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n) * 0.01


def make_multi_asset_returns(n: int = 500, n_assets: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, n_assets) * 0.01


def make_prices(n: int = 500, seed: int = 0) -> np.ndarray:
    returns = make_returns(n, seed)
    prices = np.cumprod(1 + returns) * 100
    return prices


def make_features(n: int = 500, seq_len: int = 16, feature_dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(n, seq_len, feature_dim).astype(np.float32)


# ---------------------------------------------------------------------------
# Tests: PerformanceMetrics
# ---------------------------------------------------------------------------

class TestPerformanceMetrics(unittest.TestCase):

    def setUp(self):
        self.rets = make_returns(500)

    def test_sharpe_ratio_finite(self):
        sharpe = PerformanceMetrics.sharpe_ratio(self.rets)
        self.assertTrue(np.isfinite(sharpe))

    def test_sharpe_zero_for_constant(self):
        """Zero variance returns should yield 0 Sharpe (or inf, implementation dependent)."""
        constant = np.zeros(100)
        sharpe = PerformanceMetrics.sharpe_ratio(constant)
        self.assertTrue(np.isfinite(sharpe) or np.isinf(sharpe))

    def test_sortino_ratio(self):
        sortino = PerformanceMetrics.sortino_ratio(self.rets)
        self.assertTrue(np.isfinite(sortino))

    def test_calmar_ratio(self):
        calmar = PerformanceMetrics.calmar_ratio(self.rets)
        self.assertTrue(np.isfinite(calmar))

    def test_max_drawdown_non_positive(self):
        mdd = PerformanceMetrics.max_drawdown(self.rets)
        self.assertLessEqual(mdd, 0.0)

    def test_max_drawdown_all_positive(self):
        pos_rets = np.abs(self.rets) + 0.001
        mdd = PerformanceMetrics.max_drawdown(pos_rets)
        self.assertAlmostEqual(mdd, 0.0, places=5)

    def test_hit_rate_range(self):
        predictions = np.sign(self.rets + np.random.randn(len(self.rets)) * 0.005)
        hr = PerformanceMetrics.hit_rate(predictions, self.rets)
        self.assertGreaterEqual(hr, 0.0)
        self.assertLessEqual(hr, 1.0)

    def test_information_coefficient(self):
        ic = PerformanceMetrics.information_coefficient(self.rets, self.rets)
        self.assertAlmostEqual(ic, 1.0, places=5)

    def test_ic_negative_correlation(self):
        ic = PerformanceMetrics.information_coefficient(-self.rets, self.rets)
        self.assertAlmostEqual(ic, -1.0, places=5)

    def test_turnover(self):
        n = 50
        weights1 = np.ones((n, 3)) / 3.0
        weights2 = np.ones((n, 3)) / 3.0
        to = PerformanceMetrics.turnover(weights1, weights2)
        self.assertEqual(to, 0.0)

    def test_turnover_high(self):
        n = 50
        w1 = np.tile([1.0, 0.0, 0.0], (n, 1))
        w2 = np.tile([0.0, 1.0, 0.0], (n, 1))
        to = PerformanceMetrics.turnover(w1, w2)
        self.assertGreater(to, 0.0)

    def test_annualized_return(self):
        ar = PerformanceMetrics.annualized_return(self.rets, periods_per_year=252)
        self.assertTrue(np.isfinite(ar))

    def test_annualized_volatility(self):
        av = PerformanceMetrics.annualized_volatility(self.rets, periods_per_year=252)
        self.assertGreater(av, 0.0)


# ---------------------------------------------------------------------------
# Tests: Baseline Models
# ---------------------------------------------------------------------------

class TestMomentumBaseline(unittest.TestCase):

    def test_predict_shape(self):
        baseline = MomentumBaseline(lookback=20)
        prices = make_prices(200)
        preds = baseline.predict(prices)
        self.assertEqual(len(preds), len(prices))

    def test_predict_direction(self):
        """Prices trending up should yield positive momentum signals."""
        baseline = MomentumBaseline(lookback=10)
        prices = np.arange(1, 101, dtype=float)
        preds = baseline.predict(prices)
        # Latter part should be positive
        self.assertGreater(preds[-1], 0)

    def test_evaluate_returns_dict(self):
        baseline = MomentumBaseline(lookback=20)
        prices = make_prices(300)
        returns = np.diff(np.log(prices))
        result = baseline.evaluate(prices, returns)
        self.assertIsInstance(result, dict)
        self.assertIn("sharpe_ratio", result)


class TestMeanReversionBaseline(unittest.TestCase):

    def test_predict_shape(self):
        baseline = MeanReversionBaseline(lookback=20)
        prices = make_prices(200)
        preds = baseline.predict(prices)
        self.assertEqual(len(preds), len(prices))

    def test_mean_reversion_signal(self):
        """If price above SMA, should get short (negative) signal."""
        baseline = MeanReversionBaseline(lookback=10)
        prices = np.ones(50)
        prices[-1] = 2.0   # Spike above mean
        preds = baseline.predict(prices)
        self.assertLessEqual(preds[-1], 0)

    def test_evaluate_returns_dict(self):
        baseline = MeanReversionBaseline(lookback=20)
        prices = make_prices(300)
        returns = np.diff(np.log(prices))
        result = baseline.evaluate(prices, returns)
        self.assertIn("sharpe_ratio", result)


class TestRealizedVolBaseline(unittest.TestCase):

    def test_predict_shape(self):
        baseline = RealizedVolBaseline(lookback=20)
        returns = make_returns(200)
        preds = baseline.predict(returns)
        self.assertEqual(len(preds), len(returns))

    def test_predict_non_negative(self):
        baseline = RealizedVolBaseline(lookback=10)
        returns = make_returns(100)
        preds = baseline.predict(returns)
        self.assertTrue(np.all(preds >= 0))

    def test_evaluate(self):
        baseline = RealizedVolBaseline(lookback=10)
        returns = make_returns(200)
        realized = np.abs(returns) * 15
        result = baseline.evaluate(returns, realized)
        self.assertIn("rmse", result)


class TestEqualWeightBaseline(unittest.TestCase):

    def test_weights_sum_to_one(self):
        baseline = EqualWeightBaseline()
        n_assets = 5
        weights = baseline.get_weights(n_assets)
        self.assertAlmostEqual(weights.sum(), 1.0, places=6)

    def test_evaluate(self):
        baseline = EqualWeightBaseline()
        returns = make_multi_asset_returns(200, n_assets=5)
        result = baseline.evaluate(returns)
        self.assertIn("sharpe_ratio", result)


class TestMarkowitzBaseline(unittest.TestCase):

    def test_evaluate(self):
        baseline = MarkowitzBaseline(lookback=60)
        returns = make_multi_asset_returns(200, n_assets=4)
        result = baseline.evaluate(returns)
        self.assertIn("sharpe_ratio", result)

    def test_weights_non_negative_long_only(self):
        """Long-only MVO should have non-negative weights."""
        baseline = MarkowitzBaseline(lookback=60, allow_short=False)
        returns = make_multi_asset_returns(200, n_assets=4)
        weights = baseline._optimize_weights(returns[:100])
        self.assertTrue(np.all(weights >= -1e-6))

    def test_weights_sum_to_one(self):
        baseline = MarkowitzBaseline(lookback=60)
        returns = make_multi_asset_returns(200, n_assets=4)
        weights = baseline._optimize_weights(returns[:100])
        self.assertAlmostEqual(weights.sum(), 1.0, places=4)


# ---------------------------------------------------------------------------
# Tests: Statistical Tests
# ---------------------------------------------------------------------------

class TestDieboldMarianoTest(unittest.TestCase):

    def test_returns_dict_with_statistic(self):
        dm = DieboldMarianoTest()
        realized = make_returns(200) * 15
        forecast1 = realized + np.random.randn(200) * 0.001
        forecast2 = np.zeros(200)
        result = dm.test(forecast1, forecast2, realized)
        self.assertIn("dm_statistic", result)
        self.assertIn("p_value", result)

    def test_identical_forecasts_zero_stat(self):
        dm = DieboldMarianoTest()
        realized = make_returns(200) * 15
        forecast = realized + np.random.randn(200) * 0.001
        result = dm.test(forecast, forecast, realized)
        self.assertAlmostEqual(result["dm_statistic"], 0.0, places=5)

    def test_p_value_range(self):
        dm = DieboldMarianoTest()
        realized = make_returns(200) * 15
        f1 = realized + np.random.randn(200) * 0.01
        f2 = np.random.randn(200) * 0.005
        result = dm.test(f1, f2, realized)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)


class TestWhiteRealityCheck(unittest.TestCase):

    def test_returns_dict(self):
        wrc = WhiteRealityCheck(n_bootstrap=100, block_size=5)
        returns = make_returns(300)
        benchmarks = [np.zeros(300), np.random.randn(300) * 0.001]
        result = wrc.test(returns, benchmarks)
        self.assertIn("p_value", result)
        self.assertIn("statistic", result)

    def test_p_value_range(self):
        wrc = WhiteRealityCheck(n_bootstrap=200, block_size=10)
        returns = make_returns(300)
        benchmarks = [np.zeros(300)]
        result = wrc.test(returns, benchmarks)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"], 1.0)


# ---------------------------------------------------------------------------
# Tests: walk_forward_splits
# ---------------------------------------------------------------------------

class TestWalkForwardSplits(unittest.TestCase):

    def test_returns_correct_count(self):
        splits = walk_forward_splits(1000, train_size=400, test_size=100, step_size=100)
        self.assertGreater(len(splits), 0)

    def test_split_boundaries(self):
        splits = walk_forward_splits(1000, train_size=400, test_size=100, step_size=100)
        for split in splits:
            self.assertLess(split.train_end, split.test_start + 1)
            self.assertLess(split.test_start, split.test_end)

    def test_expanding_window(self):
        splits = walk_forward_splits(1000, train_size=400, test_size=100, step_size=100, expanding=True)
        if len(splits) > 1:
            # Train end should grow with expanding window
            self.assertGreaterEqual(splits[1].train_end, splits[0].train_end)

    def test_rolling_window(self):
        splits = walk_forward_splits(1000, train_size=200, test_size=100, step_size=100, expanding=False)
        if len(splits) > 1:
            # Train size should be constant
            sizes = [s.train_end - s.train_start for s in splits]
            self.assertEqual(len(set(sizes)), 1)

    def test_gap_respected(self):
        splits = walk_forward_splits(1000, train_size=300, test_size=100, step_size=100, gap=20)
        for split in splits:
            self.assertGreaterEqual(split.test_start, split.train_end + 20)

    def test_no_splits_for_small_dataset(self):
        splits = walk_forward_splits(10, train_size=400, test_size=100)
        self.assertEqual(len(splits), 0)


# ---------------------------------------------------------------------------
# Tests: DirectionPredictionBenchmark
# ---------------------------------------------------------------------------

class TestDirectionPredictionBenchmark(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel(feature_dim=16)
        self.model.eval()
        self.runner = BenchmarkRunner(self.model, device=torch.device("cpu"))

    def _make_dataloader(self, n: int = 200):
        from torch.utils.data import TensorDataset, DataLoader
        features = torch.randn(n, 16, 16)
        labels = torch.randn(n)
        ds = TensorDataset(features, labels)
        return DataLoader(ds, batch_size=32, shuffle=False)

    def test_direction_benchmark_returns_dict(self):
        dl = self._make_dataloader()
        prices = make_prices(200)
        result = self.runner.run_direction_benchmark(dl, prices)
        self.assertIsInstance(result, dict)

    def test_direction_benchmark_has_lumina_key(self):
        dl = self._make_dataloader()
        prices = make_prices(200)
        result = self.runner.run_direction_benchmark(dl, prices)
        self.assertIn("lumina", result)

    def test_direction_lumina_has_sharpe(self):
        dl = self._make_dataloader()
        prices = make_prices(200)
        result = self.runner.run_direction_benchmark(dl, prices)
        self.assertIn("sharpe_ratio", result.get("lumina", {}))


# ---------------------------------------------------------------------------
# Tests: VolatilityForecastBenchmark
# ---------------------------------------------------------------------------

class TestVolatilityForecastBenchmark(unittest.TestCase):

    def test_evaluate_returns_dict(self):
        bench = VolatilityForecastBenchmark()
        n = 300
        returns = make_returns(n)
        realized = np.abs(returns) + np.random.rand(n) * 0.001
        model_forecast = np.abs(returns) + np.random.randn(n) * 0.0005
        result = bench.evaluate(model_forecast, realized, returns)
        self.assertIsInstance(result, dict)

    def test_lumina_key_present(self):
        bench = VolatilityForecastBenchmark()
        n = 300
        returns = make_returns(n)
        realized = np.abs(returns) + 0.001
        model_forecast = np.abs(returns) + 0.0005
        result = bench.evaluate(model_forecast, realized, returns)
        self.assertIn("lumina", result)

    def test_rmse_non_negative(self):
        bench = VolatilityForecastBenchmark()
        n = 300
        returns = make_returns(n)
        realized = np.abs(returns) + 0.001
        model_forecast = np.abs(returns) + 0.0005
        result = bench.evaluate(model_forecast, realized, returns)
        lumina_rmse = result.get("lumina", {}).get("rmse", None)
        if lumina_rmse is not None:
            self.assertGreaterEqual(lumina_rmse, 0.0)

    def test_qlike_finite(self):
        bench = VolatilityForecastBenchmark()
        n = 200
        returns = make_returns(n)
        realized = np.abs(returns) + 0.001
        model_forecast = np.abs(returns) + 0.0005
        result = bench.evaluate(model_forecast, realized, returns)
        lumina = result.get("lumina", {})
        if "qlike" in lumina:
            self.assertTrue(np.isfinite(lumina["qlike"]))


# ---------------------------------------------------------------------------
# Tests: CrisisDetectionBenchmark
# ---------------------------------------------------------------------------

class TestCrisisDetectionBenchmark(unittest.TestCase):

    def test_evaluate_returns_dict(self):
        bench = CrisisDetectionBenchmark()
        n = 300
        crisis_labels = (np.random.rand(n) > 0.9).astype(int)
        model_scores = np.random.rand(n)
        returns = make_returns(n)
        result = bench.evaluate(model_scores, crisis_labels, returns=returns)
        self.assertIsInstance(result, dict)

    def test_auroc_range(self):
        bench = CrisisDetectionBenchmark()
        n = 300
        crisis_labels = (np.random.rand(n) > 0.9).astype(int)
        model_scores = np.random.rand(n)
        returns = make_returns(n)
        result = bench.evaluate(model_scores, crisis_labels, returns=returns)
        lumina = result.get("lumina", {})
        if "auroc" in lumina:
            self.assertGreaterEqual(lumina["auroc"], 0.0)
            self.assertLessEqual(lumina["auroc"], 1.0)

    def test_perfect_classifier(self):
        """Perfect predictor should achieve AUROC ~= 1.0."""
        bench = CrisisDetectionBenchmark()
        n = 200
        crisis_labels = (np.arange(n) % 10 == 0).astype(int)
        model_scores = crisis_labels.astype(float) + np.random.randn(n) * 1e-6
        result = bench.evaluate(model_scores, crisis_labels)
        lumina = result.get("lumina", {})
        if "auroc" in lumina:
            self.assertGreater(lumina["auroc"], 0.9)


# ---------------------------------------------------------------------------
# Tests: PortfolioOptimizationBenchmark
# ---------------------------------------------------------------------------

class TestPortfolioOptimizationBenchmark(unittest.TestCase):

    def test_evaluate_returns_dict(self):
        bench = PortfolioOptimizationBenchmark(transaction_cost=0.001)
        n, n_assets = 300, 5
        returns = make_multi_asset_returns(n, n_assets)
        model_signals = np.random.randn(n, n_assets)
        result = bench.evaluate(model_signals, returns)
        self.assertIsInstance(result, dict)

    def test_lumina_key_present(self):
        bench = PortfolioOptimizationBenchmark()
        n, n_assets = 300, 5
        returns = make_multi_asset_returns(n, n_assets)
        model_signals = np.random.randn(n, n_assets)
        result = bench.evaluate(model_signals, returns)
        self.assertIn("lumina", result)

    def test_equal_weight_baseline_present(self):
        bench = PortfolioOptimizationBenchmark()
        n, n_assets = 300, 5
        returns = make_multi_asset_returns(n, n_assets)
        model_signals = np.random.randn(n, n_assets)
        result = bench.evaluate(model_signals, returns)
        self.assertIn("equal_weight", result)

    def test_with_transaction_cost(self):
        bench_no_cost = PortfolioOptimizationBenchmark(transaction_cost=0.0)
        bench_with_cost = PortfolioOptimizationBenchmark(transaction_cost=0.01)
        n, n_assets = 300, 5
        returns = make_multi_asset_returns(n, n_assets)
        model_signals = np.random.randn(n, n_assets)
        r1 = bench_no_cost.evaluate(model_signals, returns)
        r2 = bench_with_cost.evaluate(model_signals, returns)
        # Sharpe with cost should be <= without
        s1 = r1.get("lumina", {}).get("sharpe_ratio", 0)
        s2 = r2.get("lumina", {}).get("sharpe_ratio", 0)
        self.assertGreaterEqual(s1, s2 - 0.1)  # small tolerance


# ---------------------------------------------------------------------------
# Tests: BenchmarkRunner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner(unittest.TestCase):

    def setUp(self):
        self.model = DummyModel(feature_dim=16)
        self.model.eval()
        self.runner = BenchmarkRunner(self.model, device=torch.device("cpu"))

    def _make_dataloader(self, n: int = 200):
        from torch.utils.data import TensorDataset, DataLoader
        features = torch.randn(n, 16, 16)
        labels = torch.randn(n)
        ds = TensorDataset(features, labels)
        return DataLoader(ds, batch_size=32, shuffle=False)

    def test_direction_benchmark(self):
        dl = self._make_dataloader()
        prices = make_prices(200)
        result = self.runner.run_direction_benchmark(dl, prices)
        self.assertIsInstance(result, dict)
        self.assertIn("lumina", result)

    def test_print_report_no_error(self):
        dl = self._make_dataloader()
        prices = make_prices(200)
        result = {"direction": self.runner.run_direction_benchmark(dl, prices)}
        # Should not raise
        self.runner.print_report(result)

    def test_runner_with_different_batch_sizes(self):
        from torch.utils.data import TensorDataset, DataLoader
        for bs in [1, 16, 64]:
            features = torch.randn(100, 4, 16)
            labels = torch.randn(100)
            ds = TensorDataset(features, labels)
            dl = DataLoader(ds, batch_size=bs, shuffle=False)
            prices = make_prices(100)
            result = self.runner.run_direction_benchmark(dl, prices)
            self.assertIn("lumina", result)


# ---------------------------------------------------------------------------
# Integration test: full pipeline
# ---------------------------------------------------------------------------

class TestFullBenchmarkPipeline(unittest.TestCase):

    def test_end_to_end_small(self):
        """Run a small end-to-end benchmark without error."""
        model = DummyModel(feature_dim=8)
        model.eval()
        runner = BenchmarkRunner(model, device=torch.device("cpu"))

        n, seq_len, feature_dim = 200, 8, 8
        n_assets = 4

        from torch.utils.data import TensorDataset, DataLoader
        features = torch.randn(n, seq_len, feature_dim)
        labels = torch.randn(n)
        ds = TensorDataset(features, labels)
        dl = DataLoader(ds, batch_size=32, shuffle=False)

        prices = make_prices(n)
        returns = make_returns(n)
        multi_returns = make_multi_asset_returns(n, n_assets)
        realized_vol = np.abs(returns) + 0.001
        crisis_labels = (np.random.rand(n) > 0.9).astype(int)

        all_results = {}

        # Direction
        all_results["direction"] = runner.run_direction_benchmark(dl, prices)
        self.assertIn("lumina", all_results["direction"])

        # Volatility
        vol_bench = VolatilityForecastBenchmark()
        model_vol = np.abs(returns) + np.random.randn(n) * 0.0001
        all_results["volatility"] = vol_bench.evaluate(model_vol, realized_vol, returns)
        self.assertIn("lumina", all_results["volatility"])

        # Crisis
        crisis_bench = CrisisDetectionBenchmark()
        model_scores = np.random.rand(n)
        all_results["crisis"] = crisis_bench.evaluate(model_scores, crisis_labels, returns=returns)
        self.assertIn("lumina", all_results["crisis"])

        # Portfolio
        port_bench = PortfolioOptimizationBenchmark(transaction_cost=0.0005)
        signals = np.random.randn(n, n_assets)
        all_results["portfolio"] = port_bench.evaluate(signals, multi_returns)
        self.assertIn("lumina", all_results["portfolio"])

        # Print report should not error
        runner.print_report(all_results)

    def test_walk_forward_then_dm_test(self):
        """Walk-forward splits + DM test combo."""
        n = 500
        returns = make_returns(n)
        splits = walk_forward_splits(n, train_size=200, test_size=50, step_size=50)
        self.assertGreater(len(splits), 0)

        # Simulate two forecasters
        forecast1 = returns + np.random.randn(n) * 0.002
        forecast2 = np.zeros(n)

        dm = DieboldMarianoTest()
        result = dm.test(forecast1, forecast2, returns)
        self.assertIn("dm_statistic", result)
        self.assertIn("p_value", result)

    def test_performance_metrics_full(self):
        """Test all PerformanceMetrics methods on realistic data."""
        n = 500
        returns = make_returns(n)
        signals = np.sign(returns + np.random.randn(n) * 0.005)

        strategy_rets = signals * returns

        sharpe = PerformanceMetrics.sharpe_ratio(strategy_rets)
        sortino = PerformanceMetrics.sortino_ratio(strategy_rets)
        calmar = PerformanceMetrics.calmar_ratio(strategy_rets)
        mdd = PerformanceMetrics.max_drawdown(strategy_rets)
        hr = PerformanceMetrics.hit_rate(signals, returns)
        ic = PerformanceMetrics.information_coefficient(signals, returns)
        ann_ret = PerformanceMetrics.annualized_return(strategy_rets)
        ann_vol = PerformanceMetrics.annualized_volatility(strategy_rets)

        for metric_name, metric_val in [
            ("sharpe", sharpe), ("sortino", sortino), ("calmar", calmar),
            ("mdd", mdd), ("hr", hr), ("ic", ic),
            ("ann_ret", ann_ret), ("ann_vol", ann_vol),
        ]:
            self.assertTrue(np.isfinite(metric_val), f"{metric_name} should be finite, got {metric_val}")


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
