"""
research/portfolio_lab/rebalancing.py

Rebalancing strategy analysis for SRFM-Lab.

Analyses the impact of different rebalancing policies on portfolio
performance, transaction costs, and tracking error.

Classes:
    RebalancingAnalyzer — all rebalancing strategies
    RebalResult         — standardised result dataclass
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# RebalResult
# ---------------------------------------------------------------------------


@dataclass
class RebalResult:
    """Standardised result for a rebalancing strategy backtest."""

    strategy: str
    equity_curve: np.ndarray
    weights_history: np.ndarray           # (T, N) actual weights over time
    rebalance_dates: list[int]            # step indices where rebalances occurred
    n_rebalances: int
    total_cost: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    tracking_error: float                  # vs. no-rebalance buy-and-hold
    turnover: float                        # average absolute weight change per rebalance
    asset_names: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RebalancingAnalyzer
# ---------------------------------------------------------------------------


class RebalancingAnalyzer:
    """
    Analyse rebalancing strategies on historical return data.

    Args:
        ann_factor       : Annualisation factor (252 for daily).
        verbose          : Print progress.
    """

    def __init__(
        self,
        ann_factor: int = 252,
        verbose: bool = False,
    ) -> None:
        self.ann_factor = ann_factor
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Calendar rebalancing
    # ------------------------------------------------------------------

    def calendar_rebalance(
        self,
        target_weights: dict[str, float] | np.ndarray,
        returns: pd.DataFrame,
        freq: str = "monthly",
        transaction_cost: float = 0.0002,
        benchmark_weights: Optional[dict[str, float]] = None,
    ) -> RebalResult:
        """
        Simulate calendar-based rebalancing.

        Args:
            target_weights   : Target portfolio weights (dict or array).
            returns          : Returns DataFrame (T, N).
            freq             : 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annually'.
            transaction_cost : Round-trip cost per unit of turnover.
            benchmark_weights: Buy-and-hold benchmark for tracking error.

        Returns:
            RebalResult.
        """
        freq_map = {
            "daily": 1,
            "weekly": 5,
            "monthly": 21,
            "quarterly": 63,
            "annually": 252,
            "biweekly": 10,
        }
        n_steps = freq_map.get(freq.lower(), 21)

        w0, assets, R = self._prepare(target_weights, returns)
        T = len(R)

        return self._run_backtest(
            w0=w0.copy(),
            target_w=w0.copy(),
            R=R,
            rebalance_condition=lambda t, w, hist: t % n_steps == 0 and t > 0,
            transaction_cost=transaction_cost,
            strategy=f"calendar_{freq}",
            assets=assets,
            params={"freq": freq, "n_steps": n_steps},
            benchmark_weights=self._resolve_weights(benchmark_weights, assets)
            if benchmark_weights else w0.copy(),
        )

    # ------------------------------------------------------------------
    # Threshold rebalancing
    # ------------------------------------------------------------------

    def threshold_rebalance(
        self,
        target_weights: dict[str, float] | np.ndarray,
        returns: pd.DataFrame,
        threshold: float = 0.05,
        transaction_cost: float = 0.0002,
    ) -> RebalResult:
        """
        Rebalance whenever any asset drifts more than `threshold` from target.

        Args:
            target_weights   : Target weights.
            returns          : Returns DataFrame.
            threshold        : Absolute drift threshold (e.g. 0.05 = 5% drift).
            transaction_cost : Cost per unit turnover.

        Returns:
            RebalResult.
        """
        w0, assets, R = self._prepare(target_weights, returns)

        def condition(t: int, w: np.ndarray, hist: list) -> bool:
            if t == 0:
                return False
            drift = np.max(np.abs(w - w0))
            return drift > threshold

        return self._run_backtest(
            w0=w0.copy(),
            target_w=w0.copy(),
            R=R,
            rebalance_condition=condition,
            transaction_cost=transaction_cost,
            strategy=f"threshold_{threshold:.2f}",
            assets=assets,
            params={"threshold": threshold},
            benchmark_weights=w0.copy(),
        )

    # ------------------------------------------------------------------
    # Signal-driven rebalancing
    # ------------------------------------------------------------------

    def signal_driven_rebalance(
        self,
        delta_scores: pd.DataFrame,
        returns: pd.DataFrame,
        min_shift: float = 0.03,
        transaction_cost: float = 0.0002,
        base_weights: Optional[dict[str, float]] = None,
    ) -> RebalResult:
        """
        Rebalance based on BH delta_scores signals.

        Delta scores drive shifts in target weights. When the aggregate
        signal shift exceeds min_shift, a rebalance is triggered using
        delta-score-normalised target weights.

        Args:
            delta_scores     : DataFrame of delta scores, shape (T, N),
                               aligned with returns.
            returns          : Returns DataFrame (T, N).
            min_shift        : Minimum signal-implied weight shift to trigger rebalance.
            transaction_cost : Cost per unit turnover.
            base_weights     : Starting equal-weight if None.

        Returns:
            RebalResult.
        """
        assets = [c for c in returns.columns if c in delta_scores.columns]
        if len(assets) == 0:
            raise ValueError("No matching columns between delta_scores and returns.")

        R = returns[assets].values.astype(np.float64)
        D = delta_scores[assets].values.astype(np.float64)
        T, N = R.shape

        if base_weights is None:
            w0 = np.ones(N) / N
        else:
            w0 = np.array([base_weights.get(a, 1.0 / N) for a in assets])
            w0 = np.maximum(w0, 0.0)
            w0 /= w0.sum()

        current_w = w0.copy()
        equity = 1.0
        equity_curve = [equity]
        weights_history = [current_w.copy()]
        rebalance_dates = []
        total_cost = 0.0
        prev_target = w0.copy()

        for t in range(T):
            # Signal-derived target weights (delta_score based)
            scores_t = D[t]
            # Normalise: positive score = overweight, negative = underweight
            score_shifted = scores_t - scores_t.mean()
            target_w = w0 + score_shifted * 0.5  # scale shift
            target_w = np.maximum(target_w, 0.0)
            s = target_w.sum()
            if s > 1e-12:
                target_w /= s
            else:
                target_w = w0.copy()

            # Check if shift is large enough
            shift = float(np.sum(np.abs(target_w - prev_target)))
            if shift >= min_shift:
                turnover = float(np.sum(np.abs(target_w - current_w)))
                cost = turnover * transaction_cost * equity
                total_cost += cost
                equity -= cost
                current_w = target_w.copy()
                prev_target = target_w.copy()
                rebalance_dates.append(t)

            # Apply returns
            gross = float(current_w @ R[t])
            equity *= 1.0 + gross

            # Update weights for drift
            if np.any(current_w > 0):
                new_w = current_w * (1.0 + R[t])
                total_val = new_w.sum()
                if total_val > 1e-12:
                    current_w = new_w / total_val

            equity_curve.append(equity)
            weights_history.append(current_w.copy())

        eq_arr = np.array(equity_curve)
        stats = _compute_stats(eq_arr, self.ann_factor)

        bh_equity = self._buy_and_hold(w0, R)
        te = _tracking_error_from_equity(eq_arr[1:], bh_equity[1:])
        turnover_avg = float(sum(
            np.sum(np.abs(weights_history[i+1] - weights_history[i]))
            for i in rebalance_dates if i + 1 < len(weights_history)
        ) / max(1, len(rebalance_dates)))

        return RebalResult(
            strategy="signal_driven",
            equity_curve=eq_arr,
            weights_history=np.array(weights_history),
            rebalance_dates=rebalance_dates,
            n_rebalances=len(rebalance_dates),
            total_cost=total_cost,
            total_return=stats["total_return"],
            sharpe_ratio=stats["sharpe"],
            sortino_ratio=stats["sortino"],
            calmar_ratio=stats["calmar"],
            max_drawdown=stats["max_dd"],
            tracking_error=te,
            turnover=turnover_avg,
            asset_names=assets,
            params={"min_shift": min_shift},
        )

    # ------------------------------------------------------------------
    # Optimal rebalancing frequency
    # ------------------------------------------------------------------

    def optimal_rebalance_frequency(
        self,
        target_weights: dict[str, float] | np.ndarray,
        returns: pd.DataFrame,
        transaction_cost: float = 0.0002,
        candidate_freqs: Optional[list[str]] = None,
    ) -> str:
        """
        Find the rebalancing frequency that maximises net-of-cost Sharpe ratio.

        Args:
            target_weights   : Target portfolio weights.
            returns          : Returns DataFrame.
            transaction_cost : Cost per unit turnover.
            candidate_freqs  : List of frequencies to test.

        Returns:
            Best frequency string.
        """
        if candidate_freqs is None:
            candidate_freqs = ["weekly", "monthly", "quarterly", "annually"]

        best_sharpe = -math.inf
        best_freq = "monthly"

        for freq in candidate_freqs:
            result = self.calendar_rebalance(
                target_weights, returns, freq=freq,
                transaction_cost=transaction_cost,
            )
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_freq = freq

        if self.verbose:
            print(f"Optimal rebalancing frequency: {best_freq} (Sharpe={best_sharpe:.3f})")

        return best_freq

    # ------------------------------------------------------------------
    # Cost analysis
    # ------------------------------------------------------------------

    def rebalance_cost_analysis(
        self,
        target_weights: dict[str, float] | np.ndarray,
        returns: pd.DataFrame,
        costs: dict[str, float],
    ) -> pd.DataFrame:
        """
        Compare rebalancing costs across multiple frequencies and cost levels.

        Args:
            target_weights : Target weights.
            returns        : Returns DataFrame.
            costs          : Dict mapping cost scenario name to transaction cost value.

        Returns:
            DataFrame with columns: freq, cost_scenario, n_rebalances,
                total_cost, net_return, sharpe.
        """
        freqs = ["weekly", "monthly", "quarterly", "annually"]
        rows = []

        for freq in freqs:
            for cost_name, cost_val in costs.items():
                result = self.calendar_rebalance(
                    target_weights, returns, freq=freq,
                    transaction_cost=cost_val,
                )
                rows.append({
                    "frequency": freq,
                    "cost_scenario": cost_name,
                    "transaction_cost": cost_val,
                    "n_rebalances": result.n_rebalances,
                    "total_cost": result.total_cost,
                    "net_return": result.total_return,
                    "sharpe": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                    "turnover": result.turnover,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_backtest(
        self,
        w0: np.ndarray,
        target_w: np.ndarray,
        R: np.ndarray,
        rebalance_condition,
        transaction_cost: float,
        strategy: str,
        assets: list[str],
        params: dict,
        benchmark_weights: np.ndarray,
    ) -> RebalResult:
        """
        Core rebalancing backtest engine.

        Args:
            w0                   : Starting weights.
            target_w             : Target weights to rebalance toward.
            R                    : Returns matrix (T, N).
            rebalance_condition  : Callable(t, current_w, history) -> bool.
            transaction_cost     : Cost per unit turnover.
            strategy             : Strategy name string.
            assets               : Asset names.
            params               : Strategy parameters dict.
            benchmark_weights    : Buy-and-hold benchmark.

        Returns:
            RebalResult.
        """
        T, N = R.shape
        current_w = w0.copy()
        equity = 1.0
        equity_curve = [equity]
        weights_history = [current_w.copy()]
        rebalance_dates = []
        total_cost = 0.0

        for t in range(T):
            if rebalance_condition(t, current_w, weights_history):
                turnover = float(np.sum(np.abs(target_w - current_w)))
                cost = turnover * transaction_cost * equity
                total_cost += cost
                equity -= cost
                current_w = target_w.copy()
                rebalance_dates.append(t)

            gross = float(current_w @ R[t])
            equity *= 1.0 + gross

            # Drift
            new_w = current_w * (1.0 + R[t])
            total_val = new_w.sum()
            if total_val > 1e-12:
                current_w = new_w / total_val

            equity_curve.append(equity)
            weights_history.append(current_w.copy())

        eq_arr = np.array(equity_curve)
        stats = _compute_stats(eq_arr, self.ann_factor)

        # Benchmark
        bh_equity = self._buy_and_hold(benchmark_weights, R)
        te = _tracking_error_from_equity(eq_arr[1:], bh_equity[1:])

        n_reb = len(rebalance_dates)
        if n_reb > 1:
            turnovers = []
            for i in range(len(rebalance_dates) - 1):
                ri = rebalance_dates[i]
                ri1 = rebalance_dates[i + 1]
                if ri < len(weights_history) and ri1 < len(weights_history):
                    turnovers.append(float(np.sum(np.abs(weights_history[ri1] - weights_history[ri]))))
            turnover_avg = float(np.mean(turnovers)) if turnovers else 0.0
        else:
            turnover_avg = 0.0

        return RebalResult(
            strategy=strategy,
            equity_curve=eq_arr,
            weights_history=np.array(weights_history),
            rebalance_dates=rebalance_dates,
            n_rebalances=n_reb,
            total_cost=total_cost,
            total_return=stats["total_return"],
            sharpe_ratio=stats["sharpe"],
            sortino_ratio=stats["sortino"],
            calmar_ratio=stats["calmar"],
            max_drawdown=stats["max_dd"],
            tracking_error=te,
            turnover=turnover_avg,
            asset_names=assets,
            params=params,
        )

    def _buy_and_hold(self, w0: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Compute buy-and-hold equity curve with given starting weights."""
        T = len(R)
        current_w = w0.copy()
        equity = 1.0
        curve = [equity]
        for t in range(T):
            gross = float(current_w @ R[t])
            equity *= 1.0 + gross
            new_w = current_w * (1.0 + R[t])
            total_val = new_w.sum()
            if total_val > 1e-12:
                current_w = new_w / total_val
            curve.append(equity)
        return np.array(curve)

    @staticmethod
    def _prepare(
        weights: dict | np.ndarray,
        returns: pd.DataFrame,
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        assets = list(returns.columns)
        if isinstance(weights, dict):
            w = np.array([weights.get(a, 0.0) for a in assets], dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64)
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s < 1e-12:
            w = np.ones(len(assets)) / len(assets)
        else:
            w /= s
        R = returns[assets].fillna(0.0).values.astype(np.float64)
        return w, assets, R

    @staticmethod
    def _resolve_weights(weights: dict, assets: list[str]) -> np.ndarray:
        w = np.array([weights.get(a, 0.0) for a in assets], dtype=np.float64)
        w = np.maximum(w, 0.0)
        s = w.sum()
        return w / s if s > 1e-12 else np.ones(len(assets)) / len(assets)

    def compare_strategies(
        self,
        target_weights: dict[str, float] | np.ndarray,
        returns: pd.DataFrame,
        transaction_cost: float = 0.0002,
    ) -> pd.DataFrame:
        """
        Compare all rebalancing strategies on the same data.

        Returns a summary DataFrame sorted by Sharpe ratio.
        """
        strategies: list[RebalResult] = []

        for freq in ["weekly", "monthly", "quarterly", "annually"]:
            r = self.calendar_rebalance(target_weights, returns, freq=freq, transaction_cost=transaction_cost)
            strategies.append(r)

        for threshold in [0.03, 0.05, 0.10]:
            r = self.threshold_rebalance(target_weights, returns, threshold=threshold, transaction_cost=transaction_cost)
            strategies.append(r)

        rows = []
        for r in strategies:
            rows.append({
                "strategy": r.strategy,
                "total_return": r.total_return,
                "sharpe": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown,
                "n_rebalances": r.n_rebalances,
                "total_cost": r.total_cost,
                "tracking_error": r.tracking_error,
                "turnover": r.turnover,
            })
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _compute_stats(equity_curve: np.ndarray, ann_factor: int) -> dict:
    """Compute key performance statistics from an equity curve."""
    if len(equity_curve) < 2:
        return {"total_return": 0.0, "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_dd": 0.0}

    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-12)
    total_return = float((equity_curve[-1] - equity_curve[0]) / (equity_curve[0] + 1e-12))
    ann_ret = float(np.mean(returns) * ann_factor)
    ann_vol = float(np.std(returns) * math.sqrt(ann_factor)) + 1e-12
    sharpe = ann_ret / ann_vol

    downside = returns[returns < 0.0]
    std_d = float(np.std(downside) * math.sqrt(ann_factor)) + 1e-12 if len(downside) > 0 else ann_vol
    sortino = ann_ret / std_d

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / (running_max + 1e-12)
    max_dd = float(np.max(drawdowns)) + 1e-12
    calmar = ann_ret / max_dd

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": float(np.max(drawdowns)),
    }


def _tracking_error_from_equity(
    port_equity: np.ndarray,
    bench_equity: np.ndarray,
    ann_factor: int = 252,
) -> float:
    """Compute tracking error from two equity series."""
    min_len = min(len(port_equity), len(bench_equity))
    if min_len < 2:
        return 0.0
    p = port_equity[:min_len]
    b = bench_equity[:min_len]
    p_ret = np.diff(p) / (p[:-1] + 1e-12)
    b_ret = np.diff(b) / (b[:-1] + 1e-12)
    diff = p_ret - b_ret
    return float(np.std(diff) * math.sqrt(ann_factor))
