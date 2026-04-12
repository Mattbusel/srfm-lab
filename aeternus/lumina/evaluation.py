

# ============================================================
# Extended Evaluation Components - Part 2
# ============================================================

import math
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field


@dataclass
class BacktestResult:
    """Complete backtest result with all performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # in periods
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    hit_rate: float
    information_ratio: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    r_squared: Optional[float] = None
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "total_trades": self.total_trades,
            "hit_rate": self.hit_rate,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }


def compute_backtest_result(
    returns: np.ndarray,
    benchmark: Optional[np.ndarray] = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.02,
) -> BacktestResult:
    """Compute comprehensive backtest result from return series."""
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n == 0:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Basic metrics
    total_return = float(np.prod(1 + r) - 1)
    ann_return = float((1 + total_return) ** (periods_per_year / n) - 1)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))

    rfr_period = risk_free_rate / periods_per_year
    excess_r = r - rfr_period
    sharpe = float(np.mean(excess_r) / (np.std(excess_r, ddof=1) + 1e-10) * np.sqrt(periods_per_year))

    # Sortino
    downside = r[r < rfr_period] - rfr_period
    sortino_denom = float(np.sqrt(np.mean(downside ** 2) + 1e-10) * np.sqrt(periods_per_year))
    sortino = float(np.mean(excess_r) * periods_per_year / max(sortino_denom, 1e-10))

    # Max drawdown
    cumret = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumret)
    drawdown = (cumret - running_max) / (running_max + 1e-10)
    max_dd = float(drawdown.min())
    calmar = ann_return / (abs(max_dd) + 1e-10)

    # Max drawdown duration
    in_dd = drawdown < -1e-6
    dd_dur = 0
    max_dd_dur = 0
    for x in in_dd:
        if x:
            dd_dur += 1
            max_dd_dur = max(max_dd_dur, dd_dur)
        else:
            dd_dur = 0

    # Win/loss statistics
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = float(len(wins) / max(n, 1))
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    profit_factor = float(np.sum(wins) / (abs(np.sum(losses)) + 1e-10))

    # Benchmark metrics
    info_ratio = alpha = beta = r_sq = None
    if benchmark is not None:
        bm = np.asarray(benchmark, dtype=np.float64)
        bm = bm[:n]
        if len(bm) == n and np.std(bm) > 1e-10:
            cov_mat = np.cov(r, bm, ddof=1)
            beta_val = float(cov_mat[0, 1] / (cov_mat[1, 1] + 1e-10))
            alpha_val = float(np.mean(r) - beta_val * np.mean(bm))
            active_r = r - bm
            info_ratio = float(np.mean(active_r) / (np.std(active_r, ddof=1) + 1e-10) * np.sqrt(periods_per_year))
            # R-squared
            r_pred = alpha_val + beta_val * bm
            ss_res = np.sum((r - r_pred) ** 2)
            ss_tot = np.sum((r - np.mean(r)) ** 2)
            r_sq = float(1 - ss_res / (ss_tot + 1e-10))
            alpha = alpha_val
            beta = beta_val

    # Tail metrics
    var_95 = float(np.percentile(r, 5))
    cvar_95 = float(np.mean(r[r <= var_95]))

    # Moments
    skew = float(((r - np.mean(r)) ** 3).mean() / (np.std(r) + 1e-10) ** 3)
    kurt = float(((r - np.mean(r)) ** 4).mean() / (np.std(r) + 1e-10) ** 4 - 3)

    return BacktestResult(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_dur,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=n,
        hit_rate=win_rate,
        information_ratio=info_ratio,
        alpha=alpha,
        beta=beta,
        r_squared=r_sq,
        var_95=var_95,
        cvar_95=cvar_95,
        skewness=skew,
        kurtosis=kurt,
    )


class StrategyEvaluationSuite:
    """Complete evaluation suite for trading strategy comparison."""

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self._results: Dict[str, BacktestResult] = {}

    def add_strategy(
        self,
        name: str,
        returns: np.ndarray,
        benchmark: Optional[np.ndarray] = None,
    ) -> BacktestResult:
        result = compute_backtest_result(
            returns, benchmark, self.periods_per_year, self.risk_free_rate
        )
        self._results[name] = result
        return result

    def rank_strategies(self, metric: str = "sharpe_ratio") -> List[Tuple[str, float]]:
        """Rank strategies by given metric (higher is better)."""
        ranked = []
        for name, result in self._results.items():
            val = getattr(result, metric, None)
            if val is not None:
                ranked.append((name, float(val)))
        ranked.sort(key=lambda x: -x[1])
        return ranked

    def pairwise_comparison(self, s1: str, s2: str) -> Dict[str, Any]:
        """Compare two strategies head-to-head."""
        r1 = self._results.get(s1)
        r2 = self._results.get(s2)
        if r1 is None or r2 is None:
            return {}
        return {
            "better_sharpe": s1 if r1.sharpe_ratio > r2.sharpe_ratio else s2,
            "better_sortino": s1 if r1.sortino_ratio > r2.sortino_ratio else s2,
            "better_calmar": s1 if r1.calmar_ratio > r2.calmar_ratio else s2,
            "better_max_drawdown": s1 if r1.max_drawdown > r2.max_drawdown else s2,
            "sharpe_diff": r1.sharpe_ratio - r2.sharpe_ratio,
            "return_diff": r1.annualized_return - r2.annualized_return,
            "vol_diff": r1.annualized_volatility - r2.annualized_volatility,
        }

    def full_report(self) -> Dict[str, Dict[str, Any]]:
        return {name: result.to_dict() for name, result in self._results.items()}


class MLModelEvaluator:
    """Evaluates ML model predictions for financial forecasting tasks."""

    def __init__(self):
        self._metrics: Dict[str, List[float]] = {}

    def compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Comprehensive regression metrics for return prediction."""
        r = y_true
        p = y_pred
        n = len(r)

        if sample_weight is None:
            w = np.ones(n)
        else:
            w = sample_weight / sample_weight.sum()

        # Weighted metrics
        wmse = float(np.sum(w * (r - p) ** 2))
        wmae = float(np.sum(w * np.abs(r - p)))
        r2 = float(1 - np.sum((r - p) ** 2) / (np.sum((r - np.mean(r)) ** 2) + 1e-10))

        # Correlation metrics
        ic = float(np.corrcoef(r, p)[0, 1]) if np.std(r) > 1e-10 and np.std(p) > 1e-10 else 0.0
        rank_ic = float(np.corrcoef(np.argsort(np.argsort(r)), np.argsort(np.argsort(p)))[0, 1])

        # Directional accuracy
        direction_acc = float(np.mean((r > 0) == (p > 0)))

        # Quantile metrics
        q_returns = []
        n_quantiles = 5
        pred_ranks = np.argsort(np.argsort(p))
        for q in range(n_quantiles):
            q_mask = (pred_ranks >= q * n // n_quantiles) & (pred_ranks < (q + 1) * n // n_quantiles)
            if q_mask.any():
                q_returns.append(float(r[q_mask].mean()))
            else:
                q_returns.append(0.0)

        q_spread = q_returns[-1] - q_returns[0] if len(q_returns) >= 2 else 0.0

        return {
            "mse": wmse,
            "mae": wmae,
            "r2": r2,
            "ic": ic,
            "rank_ic": rank_ic,
            "direction_accuracy": direction_acc,
            "quantile_spread": q_spread,
            "quantile_returns": q_returns,
        }

    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Classification metrics for direction prediction."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        y_true_bin = (y_true > 0).astype(int)

        tp = int(((y_pred == 1) & (y_true_bin == 1)).sum())
        tn = int(((y_pred == 0) & (y_true_bin == 0)).sum())
        fp = int(((y_pred == 1) & (y_true_bin == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_bin == 1)).sum())

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        specificity = tn / max(tn + fp, 1)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }

    def information_coefficient_series(
        self,
        forecasts: List[np.ndarray],
        realizations: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compute IC across time periods and return summary stats."""
        ics = []
        for f, r in zip(forecasts, realizations):
            if len(f) > 1 and np.std(f) > 1e-10 and np.std(r) > 1e-10:
                ic = np.corrcoef(f, r)[0, 1]
                ics.append(float(ic))

        if not ics:
            return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "ic_positive_frac": 0.0}

        ic_arr = np.array(ics)
        return {
            "ic_mean": float(np.mean(ic_arr)),
            "ic_std": float(np.std(ic_arr)),
            "icir": float(np.mean(ic_arr) / (np.std(ic_arr) + 1e-10)),
            "ic_positive_frac": float((ic_arr > 0).mean()),
            "ic_t_stat": float(np.mean(ic_arr) / (np.std(ic_arr) / max(np.sqrt(len(ic_arr)), 1))),
            "ic_series": ics,
        }


class PortfolioConstructionEvaluator:
    """Evaluates portfolio construction algorithms."""

    def __init__(self, n_assets: int, risk_free_rate: float = 0.02):
        self.n_assets = n_assets
        self.risk_free_rate = risk_free_rate

    def evaluate_weights(
        self,
        weights: np.ndarray,
        asset_returns: np.ndarray,
        benchmark_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate portfolio weights against asset return matrix.

        weights: (n_assets,) or (T, n_assets)
        asset_returns: (T, n_assets)
        """
        if weights.ndim == 1:
            w = np.broadcast_to(weights[None, :], asset_returns.shape)
        else:
            w = weights[:len(asset_returns)]
            asset_returns = asset_returns[:len(w)]

        port_returns = (w * asset_returns).sum(-1)
        result = compute_backtest_result(port_returns, risk_free_rate=self.risk_free_rate)

        metrics = result.to_dict()

        # Weight-specific metrics
        metrics["weight_herfindahl"] = float((weights.flatten() ** 2).sum())
        metrics["effective_n_assets"] = float(1 / max(metrics["weight_herfindahl"], 1e-10))
        metrics["max_weight"] = float(abs(weights).max())
        metrics["long_exposure"] = float(weights[weights > 0].sum() if (weights > 0).any() else 0.0)
        metrics["short_exposure"] = float(abs(weights[weights < 0].sum()) if (weights < 0).any() else 0.0)
        metrics["gross_exposure"] = metrics["long_exposure"] + metrics["short_exposure"]
        metrics["net_exposure"] = metrics["long_exposure"] - metrics["short_exposure"]

        if benchmark_weights is not None:
            active_w = weights.flatten()[:len(benchmark_weights.flatten())] - benchmark_weights.flatten()
            metrics["tracking_error_ex_ante"] = float(np.sqrt((active_w ** 2).sum()))

        return metrics

    def mean_variance_efficiency(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        gamma: float = 1.0,
    ) -> float:
        """Compute mean-variance utility: E[r] - gamma/2 * Var[r]."""
        mu = float(expected_returns @ weights)
        var = float(weights @ cov_matrix @ weights)
        return mu - gamma / 2 * var

    def turnover_cost(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        transaction_cost_bps: float = 10.0,
    ) -> float:
        """Estimate transaction cost from rebalancing."""
        turnover = float(np.abs(weights_after - weights_before).sum() / 2)
        return turnover * transaction_cost_bps / 10000.0
