"""
RL strategy backtester with:
- Walk-forward evaluation
- Regime breakdown analysis
- Comparison vs buy-and-hold baseline
- Full performance metrics
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .environment import (
    TradingEnv, TradingConfig, RegimeTradingEnv, Instrument,
    make_trading_env, generate_synthetic_data,
)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, annualize: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / annualize
    return float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(annualize))


def sortino_ratio(returns: np.ndarray, risk_free: float = 0.0, annualize: int = 252) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / annualize
    downside = excess[excess < 0]
    downside_std = float(downside.std() + 1e-8) if len(downside) > 0 else 1e-8
    return float(excess.mean() / downside_std * np.sqrt(annualize))


def calmar_ratio(returns: np.ndarray, annualize: int = 252) -> float:
    ann_return = float(returns.mean() * annualize)
    wealth = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(wealth)
    dd     = (wealth / peak - 1).min()
    max_dd = float(abs(dd))
    if max_dd < 1e-8:
        return 0.0
    return float(ann_return / max_dd)


def max_drawdown(returns: np.ndarray) -> float:
    wealth = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(wealth)
    dd     = (wealth / peak - 1)
    return float(abs(dd.min()))


def profit_factor(returns: np.ndarray) -> float:
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return float(gains / (losses + 1e-8))


def win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Value at Risk and Conditional VaR at given alpha level."""
    if len(returns) < 2:
        return 0.0, 0.0
    var  = float(np.percentile(returns, alpha * 100))
    cvar = float(returns[returns <= var].mean()) if (returns <= var).any() else var
    return var, cvar


def information_ratio(returns: np.ndarray, benchmark: np.ndarray) -> float:
    if len(returns) != len(benchmark) or len(returns) < 2:
        return 0.0
    excess = returns - benchmark
    return float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(252))


def compute_all_metrics(returns: np.ndarray, benchmark: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute comprehensive performance metrics."""
    if len(returns) == 0:
        return {}

    var, cvar = var_cvar(returns)
    metrics = {
        "total_return": float(np.cumprod(1 + returns)[-1] - 1.0),
        "annualized_return": float(returns.mean() * 252),
        "annualized_vol": float(returns.std() * np.sqrt(252)),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "profit_factor": profit_factor(returns),
        "win_rate": win_rate(returns),
        "var_5pct": var,
        "cvar_5pct": cvar,
        "skewness": float(pd.Series(returns).skew()),
        "kurtosis": float(pd.Series(returns).kurtosis()),
        "n_days": len(returns),
    }
    if benchmark is not None:
        metrics["information_ratio"] = information_ratio(returns, benchmark)
        metrics["alpha"] = float((returns - benchmark).mean() * 252)
        bh_total = float(np.cumprod(1 + benchmark)[-1] - 1.0)
        metrics["bh_total_return"] = bh_total
        metrics["excess_return"] = metrics["total_return"] - bh_total

    return metrics


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    # Walk-forward
    use_walk_forward: bool = True
    train_window: int = 252          # bars in each training window
    test_window: int = 63            # bars in each test (out-of-sample) window
    step_size: int = 21              # advance by this many bars each fold

    # Transaction costs
    slippage_bps: float = 2.0
    commission_bps: float = 5.0
    market_impact_bps: float = 1.0

    # Position limits
    max_position: float = 1.0
    max_leverage: float = 2.0
    min_trade_size: float = 0.02     # ignore trades smaller than 2% change

    # BH comparison
    bh_rebalance_freq: int = 21      # rebalance BH every 21 days (monthly)

    # Regime breakdown
    compute_regime_breakdown: bool = True

    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.02

    # Output
    save_dir: Optional[str] = None
    verbose: bool = True


# ---------------------------------------------------------------------------
# BH baseline strategy
# ---------------------------------------------------------------------------

class BuyAndHoldStrategy:
    """Equal-weight buy-and-hold across all instruments."""

    def __init__(self, n_assets: int, initial_capital: float = 100_000.0):
        self.n = n_assets
        self.capital = initial_capital
        self.weights = np.ones(n_assets) / n_assets
        self.entry_prices: Optional[np.ndarray] = None
        self.values: List[float] = []
        self.returns: List[float] = []

    def reset(self, initial_prices: np.ndarray) -> None:
        self.entry_prices = initial_prices.copy()
        self.values = [self.capital]
        self.returns = []

    def step(self, current_prices: np.ndarray) -> float:
        assert self.entry_prices is not None
        price_returns = (current_prices - self.entry_prices) / (self.entry_prices + 1e-8)
        portfolio_return = float(np.dot(self.weights, price_returns))
        value = self.capital * (1.0 + portfolio_return)
        self.values.append(value)
        step_ret = (value - self.values[-2]) / max(self.values[-2], 1e-8)
        self.returns.append(step_ret)
        return step_ret


# ---------------------------------------------------------------------------
# Walk-forward evaluator
# ---------------------------------------------------------------------------

class WalkForwardEvaluator:
    """
    Walk-forward out-of-sample evaluation.
    Splits the data into training and test windows, rolling forward.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def split(self, total_len: int) -> List[Tuple[int, int, int, int]]:
        """
        Returns list of (train_start, train_end, test_start, test_end).
        """
        splits = []
        train_w = self.config.train_window
        test_w  = self.config.test_window
        step    = self.config.step_size

        start = 0
        while start + train_w + test_w <= total_len:
            train_start = start
            train_end   = start + train_w
            test_start  = train_end
            test_end    = min(test_start + test_w, total_len)
            splits.append((train_start, train_end, test_start, test_end))
            start += step

        return splits

    def evaluate_fold(
        self,
        agent: Any,
        data: Dict[str, np.ndarray],   # symbol -> (T, 5) OHLCV
        test_start: int,
        test_end: int,
        env_config: TradingConfig,
        obs_normalizer: Any = None,
        agent_type: str = "ppo",
    ) -> Dict[str, Any]:
        """
        Evaluate agent on one test fold.
        """
        # Slice data to test window
        fold_data = {}
        symbols = list(data.keys())
        for sym in symbols:
            arr = data[sym][test_start:test_end]
            fold_data[sym] = pd.DataFrame(
                arr, columns=["open", "high", "low", "close", "volume"]
            )

        env_config_fold = TradingConfig(
            initial_capital=self.config.initial_capital,
            max_episode_steps=test_end - test_start,
            window_size=min(30, test_end - test_start - 1),
            use_bh_features=True,
        )

        env = make_trading_env(fold_data, config=env_config_fold)
        obs, _ = env.reset()
        if obs_normalizer:
            obs = obs_normalizer.normalize(obs)

        if agent_type == "ppo":
            agent.reset_lstm()
        elif agent_type == "transformer":
            agent.reset_sequence()

        agent_returns = []
        agent_positions = []
        portfolio_values = []

        done = False
        while not done:
            if agent_type in ("ppo",):
                action, _, _ = agent.collect_action(obs)
            elif agent_type == "sac":
                action = agent.select_action(obs, deterministic=True)
            else:
                action = np.zeros(env.act_dim)

            obs, reward, terminated, truncated, info = env.step(action)
            if obs_normalizer:
                obs = obs_normalizer.normalize(obs)

            done = terminated or truncated
            agent_returns.append(info.get("step_return", 0.0))
            agent_positions.append(info.get("positions", [0.0]))
            portfolio_values.append(info.get("portfolio_value", self.config.initial_capital))

        env.close()

        # BH baseline for this fold
        bh = BuyAndHoldStrategy(len(symbols), self.config.initial_capital)
        first_prices = np.array([data[s][test_start, 3] for s in symbols])
        bh.reset(first_prices)
        bh_returns = []
        for t in range(test_start, test_end):
            prices = np.array([data[s][t, 3] for s in symbols])
            bh_ret = bh.step(prices)
            bh_returns.append(bh_ret)

        agent_arr = np.array(agent_returns)
        bh_arr    = np.array(bh_returns[:len(agent_arr)])

        return {
            "fold_test_start": test_start,
            "fold_test_end":   test_end,
            "agent_returns":   agent_arr,
            "bh_returns":      bh_arr,
            "portfolio_values": np.array(portfolio_values),
            "agent_metrics":   compute_all_metrics(agent_arr, bh_arr),
            "bh_metrics":      compute_all_metrics(bh_arr),
        }


# ---------------------------------------------------------------------------
# Regime breakdown
# ---------------------------------------------------------------------------

class RegimeBreakdown:
    """Compute performance metrics by market regime."""

    REGIME_NAMES = ["bull", "bear", "sideways", "high_vol", "low_vol"]

    def __init__(self, vol_threshold_high: float = 0.02, vol_threshold_low: float = 0.005):
        self.vth = vol_threshold_high
        self.vtl = vol_threshold_low

    def label_regimes(self, returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Label each bar with a regime index."""
        T = len(returns)
        labels = np.zeros(T, dtype=np.int32)
        for t in range(window, T):
            r_w = returns[t - window: t]
            cum_ret = float(np.cumprod(1 + r_w)[-1] - 1.0)
            vol = float(r_w.std())

            if vol > self.vth:
                labels[t] = 3   # high_vol
            elif vol < self.vtl:
                labels[t] = 4   # low_vol
            elif cum_ret > 0.05:
                labels[t] = 0   # bull
            elif cum_ret < -0.05:
                labels[t] = 1   # bear
            else:
                labels[t] = 2   # sideways

        return labels

    def compute_breakdown(
        self,
        agent_returns: np.ndarray,
        bh_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return per-regime metrics."""
        labels = self.label_regimes(bh_returns)
        result = {}

        for regime_idx, regime_name in enumerate(self.REGIME_NAMES):
            mask = labels == regime_idx
            if mask.sum() < 5:
                continue

            ar = agent_returns[mask]
            br = bh_returns[mask]

            result[regime_name] = {
                "n_days": int(mask.sum()),
                **compute_all_metrics(ar, br),
            }

        return result


# ---------------------------------------------------------------------------
# Main Backtester
# ---------------------------------------------------------------------------

class RLBacktester:
    """
    Full RL strategy backtester.

    Usage:
        bt = RLBacktester(config)
        result = bt.run(agent, data, agent_type="ppo")
        bt.print_report(result)
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.wf_evaluator = WalkForwardEvaluator(config)
        self.regime_breakdown = RegimeBreakdown()

    def run(
        self,
        agent: Any,
        data: Union[Dict[str, pd.DataFrame], Dict[str, np.ndarray]],
        agent_type: str = "ppo",
        obs_normalizer: Any = None,
    ) -> Dict[str, Any]:
        """
        Full backtest run.

        Returns:
            dict with keys: walk_forward_results, aggregate_metrics, regime_breakdown,
            bh_comparison, equity_curve
        """
        # Normalize data
        np_data = self._normalize_data(data)
        total_len = min(arr.shape[0] for arr in np_data.values())
        symbols = list(np_data.keys())
        n_assets = len(symbols)

        env_config = TradingConfig(
            initial_capital=self.config.initial_capital,
            max_episode_steps=self.config.test_window,
        )

        if self.config.use_walk_forward:
            splits = self.wf_evaluator.split(total_len)
            if self.config.verbose:
                print(f"Walk-forward: {len(splits)} folds, "
                      f"train={self.config.train_window} bars, test={self.config.test_window} bars")

            fold_results = []
            for i, (_, _, test_start, test_end) in enumerate(splits):
                if self.config.verbose:
                    print(f"  Fold {i+1}/{len(splits)}: test [{test_start}:{test_end}]", end="")

                fold = self.wf_evaluator.evaluate_fold(
                    agent, np_data, test_start, test_end, env_config,
                    obs_normalizer=obs_normalizer, agent_type=agent_type,
                )
                fold_results.append(fold)

                if self.config.verbose:
                    sh = fold["agent_metrics"].get("sharpe", 0.0)
                    dd = fold["agent_metrics"].get("max_drawdown", 0.0)
                    print(f" | Sharpe: {sh:+.3f}, DD: {dd:.2%}")

            # Stitch together equity curves
            all_agent_returns = np.concatenate([f["agent_returns"] for f in fold_results])
            all_bh_returns    = np.concatenate([f["bh_returns"] for f in fold_results])

        else:
            # Single full backtest
            fold = self.wf_evaluator.evaluate_fold(
                agent, np_data, 0, total_len, env_config,
                obs_normalizer=obs_normalizer, agent_type=agent_type,
            )
            fold_results = [fold]
            all_agent_returns = fold["agent_returns"]
            all_bh_returns    = fold["bh_returns"]

        # Aggregate metrics
        agg_metrics = compute_all_metrics(all_agent_returns, all_bh_returns)

        # Regime breakdown
        regime_metrics = {}
        if self.config.compute_regime_breakdown:
            regime_metrics = self.regime_breakdown.compute_breakdown(
                all_agent_returns, all_bh_returns
            )

        # Equity curves
        agent_equity = np.cumprod(1 + all_agent_returns) * self.config.initial_capital
        bh_equity    = np.cumprod(1 + all_bh_returns)    * self.config.initial_capital

        # Drawdown series
        agent_dd = self._drawdown_series(all_agent_returns)
        bh_dd    = self._drawdown_series(all_bh_returns)

        result = {
            "walk_forward_folds": fold_results,
            "aggregate_metrics": agg_metrics,
            "bh_metrics": compute_all_metrics(all_bh_returns),
            "regime_breakdown": regime_metrics,
            "agent_returns": all_agent_returns,
            "bh_returns": all_bh_returns,
            "agent_equity": agent_equity,
            "bh_equity": bh_equity,
            "agent_drawdown_series": agent_dd,
            "bh_drawdown_series": bh_dd,
            "n_folds": len(fold_results),
        }

        if self.config.save_dir:
            self._save_results(result)

        return result

    def _normalize_data(self, data: Any) -> Dict[str, np.ndarray]:
        """Convert DataFrames to np arrays if needed."""
        result = {}
        for sym, val in data.items():
            if isinstance(val, pd.DataFrame):
                cols = [c.lower() for c in val.columns]
                val = val.copy()
                val.columns = cols
                result[sym] = val[["open", "high", "low", "close", "volume"]].values.astype(np.float32)
            else:
                result[sym] = np.asarray(val, dtype=np.float32)
        return result

    def _drawdown_series(self, returns: np.ndarray) -> np.ndarray:
        wealth = np.cumprod(1 + returns)
        peak   = np.maximum.accumulate(wealth)
        return (wealth / peak - 1).astype(np.float32)

    def print_report(self, result: Dict[str, Any]) -> None:
        """Print formatted performance report."""
        print("\n" + "=" * 70)
        print("RL BACKTEST REPORT")
        print("=" * 70)

        am = result["aggregate_metrics"]
        bm = result["bh_metrics"]

        print(f"\n{'Metric':<25} {'Agent':>12} {'Buy&Hold':>12} {'Alpha':>12}")
        print("-" * 62)

        metrics_to_show = [
            ("Total Return",       "total_return",       ".2%"),
            ("Annualized Return",  "annualized_return",  ".2%"),
            ("Annualized Vol",     "annualized_vol",     ".2%"),
            ("Sharpe Ratio",       "sharpe",             ".3f"),
            ("Sortino Ratio",      "sortino",            ".3f"),
            ("Calmar Ratio",       "calmar",             ".3f"),
            ("Max Drawdown",       "max_drawdown",       ".2%"),
            ("Win Rate",           "win_rate",           ".2%"),
            ("Profit Factor",      "profit_factor",      ".3f"),
            ("VaR 5%",             "var_5pct",           ".3%"),
            ("CVaR 5%",            "cvar_5pct",          ".3%"),
        ]

        for label, key, fmt in metrics_to_show:
            agent_val = am.get(key, float("nan"))
            bh_val    = bm.get(key, float("nan"))
            try:
                alpha_val = agent_val - bh_val
                print(f"  {label:<23} {format(agent_val, fmt):>12} {format(bh_val, fmt):>12} {format(alpha_val, fmt):>12}")
            except Exception:
                print(f"  {label:<23} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

        if result["regime_breakdown"]:
            print(f"\n{'REGIME BREAKDOWN':}")
            print(f"{'Regime':<15} {'N Days':>8} {'Sharpe':>10} {'Return':>12} {'MaxDD':>10}")
            print("-" * 58)
            for regime, rm in result["regime_breakdown"].items():
                print(f"  {regime:<13} {rm.get('n_days', 0):>8} "
                      f"{rm.get('sharpe', 0.0):>10.3f} "
                      f"{rm.get('total_return', 0.0):>12.2%} "
                      f"{rm.get('max_drawdown', 0.0):>10.2%}")

        print(f"\nWalk-forward folds: {result['n_folds']}")
        if result["walk_forward_folds"]:
            fold_sharpes = [f["agent_metrics"].get("sharpe", 0.0) for f in result["walk_forward_folds"]]
            print(f"Mean fold Sharpe: {np.mean(fold_sharpes):.3f} ± {np.std(fold_sharpes):.3f}")
            print(f"% positive folds: {(np.array(fold_sharpes) > 0).mean():.1%}")

        print("=" * 70)

    def _save_results(self, result: Dict[str, Any]) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Save metrics as JSON
        metrics_path = os.path.join(self.config.save_dir, "backtest_metrics.json")
        save_obj = {
            "aggregate_metrics": result["aggregate_metrics"],
            "bh_metrics": result["bh_metrics"],
            "regime_breakdown": result["regime_breakdown"],
            "n_folds": result["n_folds"],
        }
        with open(metrics_path, "w") as f:
            json.dump(save_obj, f, indent=2, default=str)

        # Save equity curves
        np.save(os.path.join(self.config.save_dir, "agent_equity.npy"), result["agent_equity"])
        np.save(os.path.join(self.config.save_dir, "bh_equity.npy"), result["bh_equity"])
        np.save(os.path.join(self.config.save_dir, "agent_returns.npy"), result["agent_returns"])

        if self.config.verbose:
            print(f"Backtest results saved to {self.config.save_dir}")


# ---------------------------------------------------------------------------
# Fold statistics aggregator
# ---------------------------------------------------------------------------

def aggregate_fold_stats(fold_results: List[Dict]) -> Dict[str, float]:
    """Compute statistics across walk-forward folds."""
    if not fold_results:
        return {}

    sharpes   = [f["agent_metrics"].get("sharpe", 0.0) for f in fold_results]
    returns   = [f["agent_metrics"].get("total_return", 0.0) for f in fold_results]
    drawdowns = [f["agent_metrics"].get("max_drawdown", 0.0) for f in fold_results]

    return {
        "mean_fold_sharpe": float(np.mean(sharpes)),
        "std_fold_sharpe":  float(np.std(sharpes)),
        "min_fold_sharpe":  float(np.min(sharpes)),
        "max_fold_sharpe":  float(np.max(sharpes)),
        "pct_positive_folds": float((np.array(sharpes) > 0).mean()),
        "mean_fold_return": float(np.mean(returns)),
        "mean_fold_drawdown": float(np.mean(drawdowns)),
        "n_folds": len(fold_results),
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing RL Backtester...")

    data = generate_synthetic_data(n_assets=2, n_days=400)
    bt_config = BacktestConfig(
        use_walk_forward=True,
        train_window=100,
        test_window=50,
        step_size=25,
        initial_capital=100_000,
        compute_regime_breakdown=True,
        verbose=True,
    )

    # Create a random agent that generates random actions
    class RandomAgent:
        def __init__(self, act_dim):
            self.act_dim = act_dim
        def collect_action(self, obs):
            action = np.random.uniform(-1, 1, self.act_dim)
            return action, 0.0, 0.0
        def reset_lstm(self):
            pass

    from .environment import make_trading_env as _make_env
    tmp_env = _make_env(data)
    agent = RandomAgent(tmp_env.act_dim)

    bt = RLBacktester(bt_config)
    result = bt.run(agent, data, agent_type="ppo")
    bt.print_report(result)

    fold_agg = aggregate_fold_stats(result["walk_forward_folds"])
    print(f"\nFold aggregates: {fold_agg}")
    print("Backtester self-test passed.")
