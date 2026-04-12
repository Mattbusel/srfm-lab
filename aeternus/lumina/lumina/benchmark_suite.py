"""
lumina/benchmark_suite.py

Comprehensive benchmarking suite for Lumina financial foundation model.

Covers:
  - Direction prediction benchmark (vs momentum, mean reversion baselines)
  - Volatility forecasting benchmark (vs GARCH, realized vol)
  - Crisis detection benchmark (vs VIX-based, CUSUM)
  - Portfolio optimization benchmark (vs Markowitz, 1/N)
  - Sharpe / Sortino / Calmar ratio computation
  - Walk-forward (out-of-sample) validation
  - Statistical significance tests (Diebold-Mariano, White's reality check)
  - Maximum Drawdown and recovery metrics
  - Information Coefficient (IC) and ICIR
  - Turnover-adjusted performance
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    from scipy import stats as scipy_stats
    from scipy.optimize import minimize
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Return / performance metrics
# ---------------------------------------------------------------------------

class PerformanceMetrics:
    """
    Computes standard financial performance metrics from a return series.
    """

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> float:
        """Sharpe ratio = (mean_ret - rf) / std_ret * sqrt(periods_per_year)."""
        excess = returns - risk_free_rate / periods_per_year
        std = np.std(excess, ddof=1)
        if std == 0:
            return 0.0
        sr = np.mean(excess) / std
        if annualize:
            sr *= math.sqrt(periods_per_year)
        return float(sr)

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> float:
        """Sortino ratio: uses downside deviation instead of total std."""
        excess = returns - risk_free_rate / periods_per_year
        downside = excess[excess < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = math.sqrt(np.mean(downside ** 2))
        if downside_std == 0:
            return 0.0
        sr = np.mean(excess) / downside_std
        if annualize:
            sr *= math.sqrt(periods_per_year)
        return float(sr)

    @staticmethod
    def calmar_ratio(
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        ann_ret = PerformanceMetrics.annualized_return(returns, periods_per_year)
        mdd = PerformanceMetrics.max_drawdown(returns)
        if mdd == 0:
            return float("inf")
        return ann_ret / abs(mdd)

    @staticmethod
    def annualized_return(
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """Compound annualized growth rate."""
        total = np.prod(1 + returns)
        n = len(returns)
        if n == 0:
            return 0.0
        return float(total ** (periods_per_year / n) - 1)

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Maximum drawdown (negative number or zero)."""
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / (peak + 1e-10)
        return float(dd.min())

    @staticmethod
    def max_drawdown_duration(returns: np.ndarray) -> int:
        """Length (in periods) of the longest drawdown."""
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        in_drawdown = cum < peak
        max_duration = 0
        current = 0
        for v in in_drawdown:
            if v:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration

    @staticmethod
    def information_coefficient(
        predictions: np.ndarray,
        realized: np.ndarray,
    ) -> float:
        """
        Information Coefficient (IC) = rank correlation between
        predicted and realized returns.
        """
        if len(predictions) < 2:
            return float("nan")
        if _SCIPY_AVAILABLE:
            ic, _ = scipy_stats.spearmanr(predictions, realized)
        else:
            # Fallback: Pearson correlation
            ic = float(np.corrcoef(predictions, realized)[0, 1])
        return float(ic)

    @staticmethod
    def icir(
        prediction_series: List[np.ndarray],
        realized_series: List[np.ndarray],
    ) -> float:
        """IC Information Ratio = mean(IC) / std(IC)."""
        ics = [
            PerformanceMetrics.information_coefficient(p, r)
            for p, r in zip(prediction_series, realized_series)
        ]
        ics = [ic for ic in ics if not math.isnan(ic)]
        if not ics:
            return 0.0
        std = np.std(ics, ddof=1)
        if std == 0:
            return 0.0
        return float(np.mean(ics) / std)

    @staticmethod
    def hit_rate(predictions: np.ndarray, realized: np.ndarray) -> float:
        """Fraction of correct direction predictions."""
        pred_sign = np.sign(predictions)
        real_sign = np.sign(realized)
        return float(np.mean(pred_sign == real_sign))

    @staticmethod
    def turnover(weights: np.ndarray) -> float:
        """
        Portfolio turnover = mean absolute change in weights per period.
        weights: (T, N) array of portfolio weights.
        """
        if weights.ndim == 1 or weights.shape[0] < 2:
            return 0.0
        diffs = np.abs(np.diff(weights, axis=0))
        return float(diffs.sum(axis=1).mean())

    @staticmethod
    def full_report(
        returns: np.ndarray,
        risk_free: float = 0.0,
        periods_per_year: int = 252,
    ) -> Dict[str, float]:
        """Full performance report."""
        return {
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(returns, risk_free, True, periods_per_year),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(returns, risk_free, True, periods_per_year),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(returns, periods_per_year),
            "annualized_return": PerformanceMetrics.annualized_return(returns, periods_per_year),
            "annualized_vol": float(np.std(returns, ddof=1) * math.sqrt(periods_per_year)),
            "max_drawdown": PerformanceMetrics.max_drawdown(returns),
            "max_drawdown_duration": PerformanceMetrics.max_drawdown_duration(returns),
            "skewness": float(pd.Series(returns).skew()),
            "kurtosis": float(pd.Series(returns).kurt()),
            "hit_rate": float(np.mean(returns > 0)),
            "total_return": float(np.prod(1 + returns) - 1),
        }


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class MomentumBaseline:
    """N-period momentum signal: predict sign of past N-period return."""

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Returns predicted direction (+1/-1) for each period."""
        T = len(prices)
        predictions = np.zeros(T)
        for t in range(self.lookback, T):
            past_return = (prices[t - 1] - prices[t - self.lookback - 1]) / (prices[t - self.lookback - 1] + 1e-10)
            predictions[t] = np.sign(past_return)
        return predictions


class MeanReversionBaseline:
    """Mean reversion: predict against recent drift (z-score based)."""

    def __init__(self, lookback: int = 20, z_threshold: float = 1.0):
        self.lookback = lookback
        self.z_threshold = z_threshold

    def predict(self, prices: np.ndarray) -> np.ndarray:
        T = len(prices)
        predictions = np.zeros(T)
        for t in range(self.lookback, T):
            window = prices[t - self.lookback: t]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma == 0:
                continue
            z = (prices[t - 1] - mu) / sigma
            if z > self.z_threshold:
                predictions[t] = -1.0   # Overbought -> short
            elif z < -self.z_threshold:
                predictions[t] = 1.0    # Oversold -> long
        return predictions


class GARCHBaseline:
    """GARCH(1,1) volatility forecast baseline."""

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q

    def fit_predict(self, returns: np.ndarray) -> np.ndarray:
        """Fit GARCH and return 1-step-ahead vol forecasts."""
        if not _ARCH_AVAILABLE:
            # Fallback: EWMA
            return self._ewma_vol(returns)
        try:
            model = arch_model(returns * 100, vol="Garch", p=self.p, q=self.q, rescale=False)
            result = model.fit(disp="off")
            forecasts = result.forecast(horizon=1)
            vol = np.sqrt(forecasts.variance.values[-len(returns):, 0]) / 100
            return vol
        except Exception:
            return self._ewma_vol(returns)

    @staticmethod
    def _ewma_vol(returns: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
        T = len(returns)
        var = np.zeros(T)
        var[0] = returns[0] ** 2
        for t in range(1, T):
            var[t] = lambda_ * var[t - 1] + (1 - lambda_) * returns[t - 1] ** 2
        return np.sqrt(var)


class RealizedVolBaseline:
    """Rolling realized volatility baseline."""

    def __init__(self, window: int = 20):
        self.window = window

    def predict(self, returns: np.ndarray) -> np.ndarray:
        T = len(returns)
        rvol = np.full(T, np.nan)
        for t in range(self.window, T):
            rvol[t] = np.std(returns[t - self.window: t], ddof=1) * math.sqrt(252)
        return rvol


class EqualWeightBaseline:
    """1/N equal-weight portfolio."""

    def predict_weights(self, n_assets: int, T: int) -> np.ndarray:
        weights = np.ones((T, n_assets)) / n_assets
        return weights


class MarkowitzBaseline:
    """Mean-variance optimal portfolio (Markowitz, 1952)."""

    def __init__(self, lookback: int = 60, target_vol: float = 0.10):
        self.lookback = lookback
        self.target_vol = target_vol

    def predict_weights(
        self,
        returns: np.ndarray,
        t: int,
    ) -> np.ndarray:
        """
        Compute optimal weights at time t using past lookback returns.
        returns: (T, N) array.
        """
        if t < self.lookback or not _SCIPY_AVAILABLE:
            n = returns.shape[1]
            return np.ones(n) / n

        hist = returns[t - self.lookback: t]
        mu = np.mean(hist, axis=0)
        cov = np.cov(hist, rowvar=False) + 1e-6 * np.eye(hist.shape[1])
        n = mu.shape[0]

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = w @ mu
            port_vol = math.sqrt(w @ cov @ w)
            return -port_ret / (port_vol + 1e-10)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        try:
            result = minimize(neg_sharpe, w0, bounds=bounds, constraints=constraints, method="SLSQP")
            weights = result.x
        except Exception:
            weights = np.ones(n) / n

        return weights / (weights.sum() + 1e-10)


class VIXBaseline:
    """VIX-based crisis indicator: flag if VIX > threshold."""

    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def predict_crisis(self, vix: np.ndarray) -> np.ndarray:
        return (vix > self.threshold).astype(int)


class CUSUMBaseline:
    """CUSUM change point detector for crisis detection."""

    def __init__(self, k: float = 0.5, h: float = 5.0):
        self.k = k
        self.h = h

    def predict_crisis(self, returns: np.ndarray) -> np.ndarray:
        T = len(returns)
        crisis = np.zeros(T, dtype=int)
        s_plus = 0.0
        s_minus = 0.0
        for t in range(1, T):
            s_plus = max(0, s_plus + returns[t] - self.k)
            s_minus = max(0, s_minus - returns[t] - self.k)
            if s_plus > self.h or s_minus > self.h:
                crisis[t] = 1
                s_plus = 0.0
                s_minus = 0.0
        return crisis


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------

class DieboldMarianoTest:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: Models have equal predictive accuracy.
    HA: Model 1 is better than Model 2.

    Returns: test statistic, p-value.
    """

    @staticmethod
    def test(
        errors_1: np.ndarray,
        errors_2: np.ndarray,
        h: int = 1,
        power: int = 2,
    ) -> Tuple[float, float]:
        """
        DM test.

        Args:
            errors_1: Forecast errors from model 1.
            errors_2: Forecast errors from model 2.
            h: Forecast horizon.
            power: 1 for MAE-based, 2 for MSE-based.

        Returns:
            (DM statistic, p-value) — negative = model 1 better.
        """
        d = np.abs(errors_1) ** power - np.abs(errors_2) ** power
        T = len(d)
        d_mean = np.mean(d)

        # Newey-West variance estimate
        var_d = DieboldMarianoTest._nw_variance(d, h)
        if var_d <= 0:
            return float("nan"), float("nan")

        dm_stat = d_mean / math.sqrt(var_d / T)

        if _SCIPY_AVAILABLE:
            p_value = float(2 * scipy_stats.t.sf(abs(dm_stat), df=T - 1))
        else:
            # Approximate with normal
            p_value = float(2 * (1 - min(0.9999, abs(dm_stat) / 4)))

        return float(dm_stat), p_value

    @staticmethod
    def _nw_variance(d: np.ndarray, h: int) -> float:
        """Newey-West heteroskedasticity and autocorrelation consistent variance."""
        T = len(d)
        d_centered = d - d.mean()
        gamma_0 = np.var(d_centered, ddof=0)
        nw_var = gamma_0
        for j in range(1, h):
            gamma_j = np.dot(d_centered[j:], d_centered[:-j]) / T
            nw_var += 2 * (1 - j / h) * gamma_j
        return max(0.0, nw_var)


class WhiteRealityCheck:
    """
    White's Reality Check for data snooping (White, 2000).
    Tests whether the best model in a set beats the benchmark after
    accounting for the selection bias from testing multiple models.
    """

    def __init__(self, n_bootstrap: int = 1000, block_size: int = 5):
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size

    def test(
        self,
        benchmark_returns: np.ndarray,
        model_returns: List[np.ndarray],
    ) -> Tuple[float, float]:
        """
        White's reality check.

        Args:
            benchmark_returns: Returns of the benchmark strategy.
            model_returns: List of returns from each candidate model.

        Returns:
            (Reality check p-value, best model performance).
        """
        T = len(benchmark_returns)
        # Performance differences vs benchmark
        perf_diffs = np.array([m - benchmark_returns for m in model_returns])
        mean_diffs = perf_diffs.mean(axis=1)
        V_n = max(mean_diffs.max(), 0)

        # Block bootstrap
        boot_max_vals = []
        n_blocks = T // self.block_size

        for _ in range(self.n_bootstrap):
            # Sample blocks with replacement
            block_starts = np.random.randint(0, T - self.block_size, n_blocks)
            boot_indices = np.concatenate([
                np.arange(s, s + self.block_size) for s in block_starts
            ])[:T]
            boot_diffs = perf_diffs[:, boot_indices] - mean_diffs[:, np.newaxis]
            boot_means = boot_diffs.mean(axis=1)
            boot_max_vals.append(boot_means.max())

        boot_max_vals = np.array(boot_max_vals)
        p_value = float((boot_max_vals >= V_n).mean())
        return p_value, float(V_n)


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplit:
    """One fold of a walk-forward validation."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    fold: int


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
    gap: int = 0,
    expanding: bool = True,
) -> List[WalkForwardSplit]:
    """
    Generate walk-forward (out-of-sample) validation splits.

    Args:
        n_samples: Total number of samples.
        train_size: Size of training window (for expanding: minimum size).
        test_size: Size of test window.
        step_size: How far to advance each fold (defaults to test_size).
        gap: Gap between train and test to avoid lookahead bias.
        expanding: If True, training window grows (expanding window).
                   If False, rolling window of fixed size.

    Returns:
        List of WalkForwardSplit objects.
    """
    step_size = step_size or test_size
    splits = []
    fold = 0
    test_start = train_size + gap

    while test_start + test_size <= n_samples:
        if expanding:
            train_start = 0
            train_end = test_start - gap
        else:
            train_start = max(0, test_start - gap - train_size)
            train_end = test_start - gap

        splits.append(WalkForwardSplit(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=min(test_start + test_size, n_samples),
            fold=fold,
        ))
        test_start += step_size
        fold += 1

    return splits


# ---------------------------------------------------------------------------
# Direction prediction benchmark
# ---------------------------------------------------------------------------

class DirectionPredictionBenchmark:
    """
    Benchmark for direction prediction (up/down).

    Compares:
      - Lumina model
      - Momentum baseline
      - Mean reversion baseline
      - Random walk (50/50 baseline)
    """

    def __init__(
        self,
        lookbacks: List[int] = None,
        transaction_cost: float = 0.001,
    ):
        self.lookbacks = lookbacks or [5, 10, 20, 60]
        self.transaction_cost = transaction_cost

    def evaluate(
        self,
        model_predictions: np.ndarray,
        realized_returns: np.ndarray,
        prices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Full direction prediction benchmark.

        Args:
            model_predictions: Model's predicted direction (+1/-1) or scores.
            realized_returns: Actual returns.
            prices: Price series.

        Returns:
            Dict with benchmark comparison.
        """
        T = len(realized_returns)
        results = {}

        # Lumina model
        model_strategy = np.sign(model_predictions) * realized_returns
        model_strategy -= self.transaction_cost * np.abs(np.diff(np.sign(model_predictions), prepend=0))
        results["lumina"] = {
            **PerformanceMetrics.full_report(model_strategy),
            "hit_rate": PerformanceMetrics.hit_rate(model_predictions, realized_returns),
            "ic": PerformanceMetrics.information_coefficient(model_predictions, realized_returns),
        }

        # Momentum baselines
        for lb in self.lookbacks:
            mom = MomentumBaseline(lb)
            mom_preds = mom.predict(prices)
            mom_rets = np.sign(mom_preds) * realized_returns
            mom_rets -= self.transaction_cost * np.abs(np.diff(np.sign(mom_preds), prepend=0))
            results[f"momentum_{lb}"] = {
                **PerformanceMetrics.full_report(mom_rets),
                "hit_rate": PerformanceMetrics.hit_rate(mom_preds, realized_returns),
            }

        # Mean reversion baseline
        mr = MeanReversionBaseline()
        mr_preds = mr.predict(prices)
        mr_rets = np.sign(mr_preds) * realized_returns
        mr_rets -= self.transaction_cost * np.abs(np.diff(np.sign(mr_preds), prepend=0))
        results["mean_reversion"] = {
            **PerformanceMetrics.full_report(mr_rets),
            "hit_rate": PerformanceMetrics.hit_rate(mr_preds, realized_returns),
        }

        # Buy & hold
        results["buy_hold"] = PerformanceMetrics.full_report(realized_returns)

        # Diebold-Mariano vs best baseline
        best_baseline_rets = max(
            [v["sharpe_ratio"] for k, v in results.items() if k != "lumina"],
            default=0.0,
        )
        dm_stat, dm_pval = DieboldMarianoTest.test(
            realized_returns - model_strategy,
            realized_returns - realized_returns,  # vs buy-hold
        )
        results["dm_test_vs_buyhold"] = {"statistic": dm_stat, "p_value": dm_pval}

        return results


# ---------------------------------------------------------------------------
# Volatility forecasting benchmark
# ---------------------------------------------------------------------------

class VolatilityForecastBenchmark:
    """
    Benchmark for 1-step-ahead realized volatility forecasting.

    Compares:
      - Lumina model
      - GARCH(1,1)
      - EWMA (RiskMetrics)
      - Rolling realized vol (historical)
    """

    def __init__(self, window: int = 20):
        self.window = window

    def evaluate(
        self,
        model_vol_forecasts: np.ndarray,
        realized_vol: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate volatility forecasting."""
        T = len(realized_vol)
        valid = ~np.isnan(realized_vol) & ~np.isnan(model_vol_forecasts)

        mv = model_vol_forecasts[valid]
        rv = realized_vol[valid]
        rets = returns[valid]

        def rmse(pred: np.ndarray, target: np.ndarray) -> float:
            return float(np.sqrt(np.mean((pred - target) ** 2)))

        def mae(pred: np.ndarray, target: np.ndarray) -> float:
            return float(np.mean(np.abs(pred - target)))

        def qlike(pred: np.ndarray, target: np.ndarray) -> float:
            """QLIKE loss (better for vol forecasting)."""
            eps = 1e-10
            return float(np.mean(target / (pred + eps) - np.log(target / (pred + eps)) - 1))

        results = {}

        # Lumina model
        results["lumina"] = {
            "rmse": rmse(mv, rv),
            "mae": mae(mv, rv),
            "qlike": qlike(mv, rv),
            "corr": float(np.corrcoef(mv, rv)[0, 1]) if len(mv) > 1 else 0.0,
        }

        # GARCH baseline
        garch = GARCHBaseline()
        garch_vol = garch.fit_predict(rets)
        garch_vol = garch_vol[valid]
        results["garch"] = {
            "rmse": rmse(garch_vol, rv),
            "mae": mae(garch_vol, rv),
            "qlike": qlike(garch_vol, rv),
            "corr": float(np.corrcoef(garch_vol, rv)[0, 1]) if len(garch_vol) > 1 else 0.0,
        }

        # EWMA baseline
        ewma_vol = GARCHBaseline._ewma_vol(rets)
        ewma_vol = ewma_vol[valid]
        results["ewma"] = {
            "rmse": rmse(ewma_vol, rv),
            "mae": mae(ewma_vol, rv),
            "qlike": qlike(ewma_vol, rv),
        }

        # Rolling realized vol baseline
        hist_vol = RealizedVolBaseline(self.window).predict(rets)
        hist_vol = hist_vol[valid]
        results["hist_vol"] = {
            "rmse": rmse(hist_vol, rv),
            "mae": mae(hist_vol, rv),
            "qlike": qlike(hist_vol, rv),
        }

        # DM test: Lumina vs GARCH
        dm_stat, dm_pval = DieboldMarianoTest.test(mv - rv, garch_vol - rv, power=2)
        results["dm_vs_garch"] = {"statistic": dm_stat, "p_value": dm_pval}

        # Rank models by RMSE
        models_rmse = {k: v["rmse"] for k, v in results.items() if "rmse" in v}
        results["ranking"] = dict(sorted(models_rmse.items(), key=lambda x: x[1]))

        return results


# ---------------------------------------------------------------------------
# Crisis detection benchmark
# ---------------------------------------------------------------------------

class CrisisDetectionBenchmark:
    """
    Benchmark for detecting market crises.

    Compares:
      - Lumina crisis probability scores
      - VIX threshold rule
      - CUSUM detector
    """

    def __init__(self, vix_threshold: float = 30.0):
        self.vix_threshold = vix_threshold

    def evaluate(
        self,
        model_scores: np.ndarray,     # Higher = more likely crisis
        true_labels: np.ndarray,       # 1 = crisis, 0 = normal
        vix: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate crisis detection performance.
        """
        from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
        from sklearn.metrics import average_precision_score

        def threshold_classify(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
            return (scores >= threshold).astype(int)

        results = {}

        # Lumina model
        try:
            auc = roc_auc_score(true_labels, model_scores)
            ap = average_precision_score(true_labels, model_scores)
        except Exception:
            auc, ap = 0.0, 0.0

        model_preds = threshold_classify(model_scores)
        results["lumina"] = {
            "auc_roc": float(auc),
            "avg_precision": float(ap),
            "f1": float(f1_score(true_labels, model_preds, zero_division=0)),
            "precision": float(np.sum((model_preds == 1) & (true_labels == 1)) / (np.sum(model_preds == 1) + 1e-10)),
            "recall": float(np.sum((model_preds == 1) & (true_labels == 1)) / (np.sum(true_labels == 1) + 1e-10)),
        }

        # VIX baseline
        if vix is not None:
            vix_bl = VIXBaseline(self.vix_threshold)
            vix_preds = vix_bl.predict_crisis(vix)
            try:
                vix_auc = roc_auc_score(true_labels, vix)
            except Exception:
                vix_auc = 0.0
            results["vix"] = {
                "auc_roc": float(vix_auc),
                "f1": float(f1_score(true_labels, vix_preds, zero_division=0)),
                "precision": float(np.sum((vix_preds == 1) & (true_labels == 1)) / (np.sum(vix_preds == 1) + 1e-10)),
                "recall": float(np.sum((vix_preds == 1) & (true_labels == 1)) / (np.sum(true_labels == 1) + 1e-10)),
            }

        # CUSUM baseline
        if returns is not None:
            cusum = CUSUMBaseline()
            cusum_preds = cusum.predict_crisis(returns)
            results["cusum"] = {
                "f1": float(f1_score(true_labels, cusum_preds, zero_division=0)),
                "precision": float(np.sum((cusum_preds == 1) & (true_labels == 1)) / (np.sum(cusum_preds == 1) + 1e-10)),
                "recall": float(np.sum((cusum_preds == 1) & (true_labels == 1)) / (np.sum(true_labels == 1) + 1e-10)),
            }

        return results


# ---------------------------------------------------------------------------
# Portfolio optimization benchmark
# ---------------------------------------------------------------------------

class PortfolioOptimizationBenchmark:
    """
    Benchmark for portfolio construction using model signals.

    Compares:
      - Lumina signal-based portfolio
      - Equal weight (1/N)
      - Markowitz MVO
      - Momentum portfolio
    """

    def __init__(
        self,
        rebalance_freq: int = 20,        # Rebalance every N periods
        transaction_cost: float = 0.001,
        n_longs: Optional[int] = None,   # Long only top N
    ):
        self.rebalance_freq = rebalance_freq
        self.transaction_cost = transaction_cost
        self.n_longs = n_longs

    def signals_to_weights(
        self,
        signals: np.ndarray,        # (N,) signal for each asset
        method: str = "rank",
    ) -> np.ndarray:
        """Convert signals to portfolio weights."""
        n = len(signals)
        if method == "rank":
            ranks = pd.Series(signals).rank(pct=True).values
            weights = ranks / (ranks.sum() + 1e-10)
        elif method == "long_only_top":
            n_long = self.n_longs or max(1, n // 3)
            weights = np.zeros(n)
            top_idx = np.argsort(signals)[-n_long:]
            weights[top_idx] = 1.0 / n_long
        elif method == "long_short":
            weights = (signals - signals.mean()) / (signals.std() + 1e-10)
            weights /= np.abs(weights).sum() + 1e-10
        else:
            weights = np.ones(n) / n
        return weights

    def compute_portfolio_returns(
        self,
        weights_series: np.ndarray,     # (T, N)
        asset_returns: np.ndarray,       # (T, N)
    ) -> np.ndarray:
        """Compute portfolio return series with transaction costs."""
        T, N = weights_series.shape
        port_rets = np.zeros(T)
        for t in range(T):
            # Raw return
            port_rets[t] = (weights_series[t] * asset_returns[t]).sum()
            # Transaction cost
            if t > 0:
                turnover = np.abs(weights_series[t] - weights_series[t - 1]).sum()
                port_rets[t] -= self.transaction_cost * turnover
        return port_rets

    def evaluate(
        self,
        model_signals: np.ndarray,      # (T, N) model predictions per asset
        asset_returns: np.ndarray,       # (T, N) actual returns per asset
    ) -> Dict[str, Any]:
        """Run full portfolio benchmark."""
        T, N = asset_returns.shape
        results = {}

        # Lumina portfolio
        lumina_weights = np.zeros((T, N))
        ew_weights = np.ones((T, N)) / N
        mvo_weights = np.zeros((T, N))
        mom_weights = np.zeros((T, N))

        for t in range(T):
            if t % self.rebalance_freq == 0 or t == 0:
                lumina_weights[t] = self.signals_to_weights(model_signals[t], method="rank")
                ew_weights[t] = np.ones(N) / N

                if t >= 60:
                    mvo = MarkowitzBaseline(lookback=60)
                    mvo_weights[t] = mvo.predict_weights(asset_returns, t)
                    # Momentum
                    mom_scores = asset_returns[max(0, t-20):t].mean(axis=0)
                    mom_weights[t] = self.signals_to_weights(mom_scores, method="rank")
                else:
                    mvo_weights[t] = np.ones(N) / N
                    mom_weights[t] = np.ones(N) / N
            else:
                lumina_weights[t] = lumina_weights[t - 1]
                ew_weights[t] = ew_weights[t - 1]
                mvo_weights[t] = mvo_weights[t - 1]
                mom_weights[t] = mom_weights[t - 1]

        lumina_rets = self.compute_portfolio_returns(lumina_weights, asset_returns)
        ew_rets = self.compute_portfolio_returns(ew_weights, asset_returns)
        mvo_rets = self.compute_portfolio_returns(mvo_weights, asset_returns)
        mom_rets = self.compute_portfolio_returns(mom_weights, asset_returns)

        results["lumina"] = {
            **PerformanceMetrics.full_report(lumina_rets),
            "turnover": PerformanceMetrics.turnover(lumina_weights),
        }
        results["equal_weight"] = {
            **PerformanceMetrics.full_report(ew_rets),
            "turnover": PerformanceMetrics.turnover(ew_weights),
        }
        results["markowitz"] = {
            **PerformanceMetrics.full_report(mvo_rets),
            "turnover": PerformanceMetrics.turnover(mvo_weights),
        }
        results["momentum"] = {
            **PerformanceMetrics.full_report(mom_rets),
            "turnover": PerformanceMetrics.turnover(mom_weights),
        }

        # Rank by Sharpe
        sharpes = {k: v["sharpe_ratio"] for k, v in results.items()}
        results["sharpe_ranking"] = dict(sorted(sharpes.items(), key=lambda x: x[1], reverse=True))

        # Reality check: is Lumina significantly better?
        rc = WhiteRealityCheck()
        rc_pval, rc_best = rc.test(
            ew_rets,
            [lumina_rets, mvo_rets, mom_rets],
        )
        results["white_reality_check"] = {"p_value": rc_pval, "best_excess": rc_best}

        return results


# ---------------------------------------------------------------------------
# Comprehensive benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs all benchmarks and compiles a comprehensive report.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        transaction_cost: float = 0.001,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model = self.model.to(self.device)
        self.tc = transaction_cost

        self.direction_bench = DirectionPredictionBenchmark(transaction_cost=transaction_cost)
        self.vol_bench = VolatilityForecastBenchmark()
        self.crisis_bench = CrisisDetectionBenchmark()
        self.portfolio_bench = PortfolioOptimizationBenchmark(transaction_cost=transaction_cost)

    @torch.no_grad()
    def run_direction_benchmark(
        self,
        dataloader: torch.utils.data.DataLoader,
        prices: np.ndarray,
    ) -> Dict[str, Any]:
        """Run direction prediction benchmark."""
        self.model.eval()
        all_preds, all_rets = [], []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(self.device), batch[1].numpy()
            elif isinstance(batch, dict):
                x = batch.get("input_ids", batch.get("features")).to(self.device)
                y = batch.get("labels", torch.zeros(x.shape[0])).numpy()
            else:
                continue

            outputs = self.model(x)
            if isinstance(outputs, dict):
                preds = outputs.get("logits", outputs.get("output", torch.zeros(x.shape[0]))).cpu().numpy()
            else:
                preds = outputs.cpu().numpy()

            if preds.ndim > 1:
                preds = preds[:, 0]
            all_preds.append(preds)
            all_rets.append(y.flatten())

        if not all_preds:
            return {}

        preds = np.concatenate(all_preds)
        rets = np.concatenate(all_rets)
        n = min(len(preds), len(prices))

        return self.direction_bench.evaluate(preds[:n], rets[:n], prices[:n])

    @torch.no_grad()
    def run_full_benchmark(
        self,
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        data_arrays: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive report."""
        report = {}

        if "direction" in dataloaders and "prices" in data_arrays:
            report["direction"] = self.run_direction_benchmark(
                dataloaders["direction"], data_arrays["prices"]
            )

        if "vol" in data_arrays:
            # Run vol forecasting
            rets = data_arrays.get("returns", np.zeros(100))
            rv = data_arrays.get("realized_vol", np.abs(rets))
            # Model vol forecasts (stub - would come from model)
            model_vol = np.abs(rets) + np.random.randn(len(rets)) * 0.01
            report["volatility"] = self.vol_bench.evaluate(model_vol, rv, rets)

        if "crisis_labels" in data_arrays:
            model_scores = np.random.rand(len(data_arrays["crisis_labels"]))
            report["crisis"] = self.crisis_bench.evaluate(
                model_scores,
                data_arrays["crisis_labels"],
                vix=data_arrays.get("vix"),
                returns=data_arrays.get("returns"),
            )

        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        """Pretty-print benchmark results."""
        for benchmark_name, results in report.items():
            print(f"\n{'='*60}")
            print(f"BENCHMARK: {benchmark_name.upper()}")
            print('='*60)
            for model_name, metrics in results.items():
                if isinstance(metrics, dict) and "sharpe_ratio" in metrics:
                    print(f"\n  {model_name}:")
                    print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
                    print(f"    Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}")
                    print(f"    Hit Rate:     {metrics.get('hit_rate', 'N/A'):.2%}")
                elif isinstance(metrics, dict) and "rmse" in metrics:
                    print(f"\n  {model_name}:")
                    print(f"    RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                    print(f"    MAE:  {metrics.get('mae', 'N/A'):.4f}")


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Metrics
    "PerformanceMetrics",
    # Baselines
    "MomentumBaseline",
    "MeanReversionBaseline",
    "GARCHBaseline",
    "RealizedVolBaseline",
    "EqualWeightBaseline",
    "MarkowitzBaseline",
    "VIXBaseline",
    "CUSUMBaseline",
    # Statistical tests
    "DieboldMarianoTest",
    "WhiteRealityCheck",
    # Walk-forward validation
    "WalkForwardSplit",
    "walk_forward_splits",
    # Benchmarks
    "DirectionPredictionBenchmark",
    "VolatilityForecastBenchmark",
    "CrisisDetectionBenchmark",
    "PortfolioOptimizationBenchmark",
    # Runner
    "BenchmarkRunner",
]
