"""Comprehensive tests for evaluation.py extended components."""
import pytest
import numpy as np
import torch

class TestComputeBacktestResult:
    def test_positive_returns(self):
        from evaluation import compute_backtest_result
        returns = np.full(252, 0.001)  # 0.1% daily
        result = compute_backtest_result(returns)
        assert result.annualized_return > 0
        assert result.sharpe_ratio > 0

    def test_negative_returns(self):
        from evaluation import compute_backtest_result
        returns = np.full(252, -0.001)
        result = compute_backtest_result(returns)
        assert result.annualized_return < 0

    def test_max_drawdown_negative(self):
        from evaluation import compute_backtest_result
        returns = np.array([-0.01] * 20 + [0.01] * 20)
        result = compute_backtest_result(returns)
        assert result.max_drawdown <= 0

    def test_win_rate_all_wins(self):
        from evaluation import compute_backtest_result
        returns = np.abs(np.random.randn(100)) * 0.01
        result = compute_backtest_result(returns)
        assert result.win_rate == 1.0

    def test_win_rate_all_losses(self):
        from evaluation import compute_backtest_result
        returns = -np.abs(np.random.randn(100)) * 0.01
        result = compute_backtest_result(returns)
        assert result.win_rate == 0.0

    def test_with_benchmark(self):
        from evaluation import compute_backtest_result
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01
        benchmark = np.random.randn(252) * 0.008
        result = compute_backtest_result(returns, benchmark)
        assert result.information_ratio is not None
        assert result.beta is not None

    def test_to_dict(self):
        from evaluation import compute_backtest_result
        returns = np.random.randn(100) * 0.01
        result = compute_backtest_result(returns)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert 'sharpe_ratio' in d

    def test_empty_returns(self):
        from evaluation import compute_backtest_result
        result = compute_backtest_result(np.array([]))
        assert result.total_return == 0

    def test_single_period(self):
        from evaluation import compute_backtest_result
        result = compute_backtest_result(np.array([0.05]))
        assert result.total_return == pytest.approx(0.05, abs=1e-6)

class TestStrategyEvaluationSuite:
    def setup_method(self):
        from evaluation import StrategyEvaluationSuite
        self.suite = StrategyEvaluationSuite()
        np.random.seed(42)

    def test_add_and_rank(self):
        r1 = np.random.randn(252) * 0.01 + 0.0005
        r2 = np.random.randn(252) * 0.015 + 0.0003
        self.suite.add_strategy('strategy_a', r1)
        self.suite.add_strategy('strategy_b', r2)
        ranked = self.suite.rank_strategies('sharpe_ratio')
        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]

    def test_pairwise_comparison(self):
        r1 = np.random.randn(252) * 0.01 + 0.001
        r2 = np.random.randn(252) * 0.01 - 0.001
        self.suite.add_strategy('good', r1)
        self.suite.add_strategy('bad', r2)
        comparison = self.suite.pairwise_comparison('good', 'bad')
        assert 'sharpe_diff' in comparison
        assert comparison['better_sharpe'] == 'good'

    def test_full_report(self):
        r1 = np.random.randn(100) * 0.01
        self.suite.add_strategy('alpha', r1)
        report = self.suite.full_report()
        assert 'alpha' in report
        assert 'sharpe_ratio' in report['alpha']

class TestMLModelEvaluator:
    def setup_method(self):
        from evaluation import MLModelEvaluator
        self.evaluator = MLModelEvaluator()
        np.random.seed(42)

    def test_regression_metrics(self):
        y_true = np.random.randn(100) * 0.01
        y_pred = y_true + np.random.randn(100) * 0.003
        metrics = self.evaluator.compute_regression_metrics(y_true, y_pred)
        assert 'ic' in metrics
        assert 'r2' in metrics
        assert 'mse' in metrics

    def test_high_ic_perfect_prediction(self):
        y_true = np.random.randn(100)
        y_pred = y_true * 1.1  # linear transform -> IC=1
        metrics = self.evaluator.compute_regression_metrics(y_true, y_pred)
        assert metrics['ic'] > 0.99

    def test_classification_metrics(self):
        np.random.seed(0)
        y_true = np.random.randn(200) * 0.01
        y_pred_proba = np.random.uniform(0, 1, 200)
        metrics = self.evaluator.compute_classification_metrics(y_true, y_pred_proba)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_classification_perfect(self):
        y_true = np.array([0.01, -0.01, 0.02, -0.02, 0.005])
        y_pred_proba = np.array([0.9, 0.1, 0.95, 0.05, 0.8])
        metrics = self.evaluator.compute_classification_metrics(y_true, y_pred_proba)
        assert metrics['accuracy'] == 1.0

    def test_ic_series(self):
        forecasts = [np.random.randn(50) for _ in range(12)]
        realizations = [np.random.randn(50) for _ in range(12)]
        result = self.evaluator.information_coefficient_series(forecasts, realizations)
        assert 'ic_mean' in result
        assert 'icir' in result

class TestPortfolioConstructionEvaluator:
    def setup_method(self):
        from evaluation import PortfolioConstructionEvaluator
        self.eval = PortfolioConstructionEvaluator(n_assets=10)
        np.random.seed(0)

    def test_evaluate_equal_weights(self):
        returns = np.random.randn(252, 10) * 0.01
        weights = np.ones(10) / 10
        metrics = self.eval.evaluate_weights(weights, returns)
        assert 'sharpe_ratio' in metrics
        assert abs(metrics['max_weight'] - 0.1) < 1e-6

    def test_herfindahl_concentrated(self):
        returns = np.random.randn(100, 10) * 0.01
        weights = np.zeros(10)
        weights[0] = 1.0
        metrics = self.eval.evaluate_weights(weights, returns)
        assert metrics['weight_herfindahl'] == pytest.approx(1.0)
        assert metrics['effective_n_assets'] == pytest.approx(1.0, abs=1e-5)

    def test_mean_variance_utility(self):
        weights = np.array([0.3, 0.3, 0.4])
        expected_returns = np.array([0.1, 0.08, 0.12])
        cov = np.eye(3) * 0.04
        utility = self.eval.mean_variance_efficiency(weights, expected_returns, cov)
        assert isinstance(utility, float)

    def test_turnover_cost(self):
        w_before = np.array([0.25, 0.25, 0.25, 0.25])
        w_after = np.array([0.30, 0.20, 0.30, 0.20])
        cost = self.eval.turnover_cost(w_before, w_after)
        assert cost > 0
        assert cost < 1.0

@pytest.mark.parametrize('n,mu,sigma', [
    (100, 0.001, 0.01), (252, 0.0005, 0.015), (504, 0.0, 0.02),
    (1000, 0.0003, 0.008), (63, 0.002, 0.025), (21, 0.0, 0.03),
])
def test_backtest_result_no_nan(n, mu, sigma):
    from evaluation import compute_backtest_result
    np.random.seed(42)
    returns = np.random.randn(n) * sigma + mu
    result = compute_backtest_result(returns)
    assert not np.isnan(result.sharpe_ratio)
    assert not np.isnan(result.annualized_return)
    assert result.max_drawdown <= 0
