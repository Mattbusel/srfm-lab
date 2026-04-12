"""
AETERNUS Real-Time Execution Layer (RTEL)
backtest_runner.py — Full backtesting orchestration

Integrates: DataPipeline, SignalEngine, PortfolioOptimizer, ExperimentOrchestrator
to run full-stack backtests with realistic market simulation.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data_pipeline import DataPipeline, SyntheticDataSource, RawTick
from .signal_engine import SignalEngine
from .portfolio_optimizer import PortfolioOptimizationEngine

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Backtest configuration
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    n_assets:           int   = 10
    n_steps:            int   = 1000
    initial_capital:    float = 1_000_000.0
    commission_bps:     float = 5.0
    slippage_bps:       float = 2.0
    bar_duration_s:     float = 1.0
    signal_lookback:    int   = 20
    opt_method:         str   = "erc"          # erc | mvo | kelly | min_var
    kelly_fraction:     float = 0.25
    max_position_pct:   float = 0.15
    rebalance_every:    int   = 5
    seed:               int   = 42
    sigma:              float = 0.01
    mu:                 float = 0.0001         # daily drift
    verbose:            bool  = False


# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

class Portfolio:
    def __init__(self, initial_capital: float, n_assets: int):
        self.cash        = initial_capital
        self.initial     = initial_capital
        self.positions   = np.zeros(n_assets)
        self.avg_costs   = np.zeros(n_assets)
        self.realized_pnl= 0.0
        self.commissions = 0.0
        self.n_trades    = 0

    def equity(self, prices: np.ndarray) -> float:
        return self.cash + float(self.positions @ prices)

    def weights(self, prices: np.ndarray) -> np.ndarray:
        eq = self.equity(prices)
        if eq < _EPS:
            return np.zeros(len(self.positions))
        return self.positions * prices / eq

    def apply_trade(self, asset_idx: int, qty: float, price: float,
                    commission: float) -> None:
        """qty > 0 = buy, qty < 0 = sell."""
        cur = self.positions[asset_idx]
        cur_cost = self.avg_costs[asset_idx]

        if cur == 0.0:
            self.positions[asset_idx]  = qty
            self.avg_costs[asset_idx]  = price
        elif cur * qty > 0.0:
            # Same direction
            new_qty = cur + qty
            self.avg_costs[asset_idx] = (
                (cur_cost * abs(cur) + price * abs(qty)) / abs(new_qty)
            )
            self.positions[asset_idx] = new_qty
        else:
            # Reducing / flipping
            close_qty = min(abs(cur), abs(qty))
            sign      = 1.0 if cur > 0 else -1.0
            self.realized_pnl += sign * close_qty * (price - cur_cost)
            new_qty = cur + qty
            if abs(new_qty) < _EPS:
                self.positions[asset_idx] = 0.0
                self.avg_costs[asset_idx] = 0.0
            else:
                self.positions[asset_idx] = new_qty
                if new_qty * cur < 0:
                    self.avg_costs[asset_idx] = price

        self.cash -= qty * price + commission
        self.commissions += commission
        self.n_trades    += 1

    def unrealized_pnl(self, prices: np.ndarray) -> float:
        pnl = 0.0
        for i, qty in enumerate(self.positions):
            if abs(qty) < _EPS:
                continue
            sign = 1.0 if qty > 0 else -1.0
            pnl += sign * abs(qty) * (prices[i] - self.avg_costs[i])
        return pnl

    def total_pnl(self, prices: np.ndarray) -> float:
        return self.realized_pnl + self.unrealized_pnl(prices) - self.commissions


# ---------------------------------------------------------------------------
# GBM price simulation
# ---------------------------------------------------------------------------

class MultiAssetGBM:
    """Correlated GBM simulation for n assets."""

    def __init__(self, n_assets: int, mu: float, sigma: float,
                 dt: float = 1.0/252.0, seed: int = 42,
                 correlation: float = 0.3):
        self.n_assets = n_assets
        self.mu       = mu
        self.dt       = dt
        self.rng      = np.random.default_rng(seed)

        # Per-asset sigma with some variation
        self.sigmas = sigma * (1.0 + 0.2 * np.arange(n_assets) / max(n_assets - 1, 1))

        # Correlation matrix (factor model: common factor correlation)
        self.corr = np.eye(n_assets) * (1 - correlation) + correlation
        self.chol = np.linalg.cholesky(self.corr)

        self.prices = 100.0 * (1.0 + 0.1 * np.arange(n_assets))

        # GARCH-like vol state
        self.current_vols = self.sigmas.copy()

    def step(self) -> np.ndarray:
        """Return new price vector."""
        z_ind  = self.rng.standard_normal(self.n_assets)
        z_corr = self.chol @ z_ind  # correlated shocks

        # Update vols (GARCH-like)
        self.current_vols = (0.9 * self.current_vols +
                             0.1 * np.abs(self.sigmas * z_ind))
        self.current_vols = np.maximum(self.current_vols, self.sigmas * 0.1)

        drift = (self.mu - 0.5 * self.current_vols**2) * self.dt
        shock = self.current_vols * math.sqrt(self.dt) * z_corr
        self.prices *= np.exp(drift + shock)
        return self.prices.copy()

    def make_ticks(self, spread_bps: float = 5.0) -> List[RawTick]:
        prices = self.step()
        ticks  = []
        for i, price in enumerate(prices):
            half_spread = price * spread_bps / 2e4
            ticks.append(RawTick(
                asset_id   = i,
                timestamp  = time.time(),
                bid        = price - half_spread,
                ask        = price + half_spread,
                bid_size   = float(abs(self.rng.normal(1000, 200))),
                ask_size   = float(abs(self.rng.normal(1000, 200))),
                last_price = price,
                volume     = float(abs(self.rng.normal(500, 100))),
            ))
        return ticks


# ---------------------------------------------------------------------------
# Backtest statistics
# ---------------------------------------------------------------------------

@dataclass
class BacktestStats:
    total_return:     float = 0.0
    annual_return:    float = 0.0
    annual_vol:       float = 0.0
    sharpe:           float = 0.0
    sortino:          float = 0.0
    max_drawdown:     float = 0.0
    calmar:           float = 0.0
    win_rate:         float = 0.0
    avg_win:          float = 0.0
    avg_loss:         float = 0.0
    profit_factor:    float = 0.0
    n_trades:         int   = 0
    total_commission: float = 0.0
    n_steps:          int   = 0
    equity_curve:     List[float] = field(default_factory=list)
    drawdown_series:  List[float] = field(default_factory=list)

    @classmethod
    def from_equity_curve(cls, equity: List[float], n_trades: int,
                           commissions: float,
                           trade_pnls: List[float],
                           steps_per_year: float = 252.0) -> "BacktestStats":
        s    = cls()
        n    = len(equity)
        s.n_steps = n
        s.equity_curve = equity
        s.total_commission = commissions
        s.n_trades = n_trades
        if n < 2:
            return s

        s.total_return = (equity[-1] - equity[0]) / equity[0]
        n_years        = n / steps_per_year
        s.annual_return= (1.0 + s.total_return) ** (1.0/max(n_years, _EPS)) - 1.0

        rets = np.diff(equity) / np.array(equity[:-1])
        s.annual_vol   = float(rets.std() * math.sqrt(steps_per_year))
        s.sharpe       = s.annual_return / (s.annual_vol + _EPS)

        # Sortino
        neg_rets = rets[rets < 0]
        if len(neg_rets) > 0:
            down_vol = float(neg_rets.std() * math.sqrt(steps_per_year))
            s.sortino = s.annual_return / (down_vol + _EPS)

        # Max drawdown
        peak = equity[0]
        dds  = []
        for e in equity:
            peak = max(peak, e)
            dds.append((peak - e) / peak if peak > _EPS else 0.0)
        s.drawdown_series = dds
        s.max_drawdown    = max(dds)
        s.calmar          = s.annual_return / (s.max_drawdown + _EPS)

        # Trade stats
        if trade_pnls:
            wins  = [p for p in trade_pnls if p > 0]
            losses= [p for p in trade_pnls if p < 0]
            s.win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
            s.avg_win  = float(np.mean(wins)) if wins else 0.0
            s.avg_loss = float(np.mean(losses)) if losses else 0.0
            if s.avg_loss < -_EPS and s.win_rate < 1.0:
                s.profit_factor = (s.win_rate * s.avg_win) / (
                    (1.0 - s.win_rate) * abs(s.avg_loss))

        return s

    def print_summary(self) -> None:
        print("=== AETERNUS Backtest Results ===")
        print(f"  Total return:     {self.total_return*100:.2f}%")
        print(f"  Annual return:    {self.annual_return*100:.2f}%")
        print(f"  Annual vol:       {self.annual_vol*100:.2f}%")
        print(f"  Sharpe ratio:     {self.sharpe:.3f}")
        print(f"  Sortino ratio:    {self.sortino:.3f}")
        print(f"  Max drawdown:     {self.max_drawdown*100:.2f}%")
        print(f"  Calmar ratio:     {self.calmar:.3f}")
        print(f"  Win rate:         {self.win_rate*100:.1f}%")
        print(f"  Profit factor:    {self.profit_factor:.3f}")
        print(f"  N trades:         {self.n_trades}")
        print(f"  Commission:       ${self.total_commission:.2f}")
        print(f"  N steps:          {self.n_steps}")


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """
    Full-stack backtest orchestrator.
    Connects: GBM market simulation → DataPipeline → SignalEngine
              → PortfolioOptimizer → Portfolio → Metrics
    """

    def __init__(self, config: BacktestConfig):
        self.cfg  = config
        self.n    = config.n_assets

        # Components
        self.market    = MultiAssetGBM(
            n_assets    = config.n_assets,
            mu          = config.mu,
            sigma       = config.sigma,
            dt          = 1.0 / 252.0,
            seed        = config.seed,
        )
        self.pipeline  = DataPipeline(config.n_assets, config.bar_duration_s)
        self.signals   = SignalEngine(config.n_assets, lookback_short=5,
                                      lookback_long=config.signal_lookback)
        self.optimizer = PortfolioOptimizationEngine(
            config.n_assets, method=config.opt_method,
            kelly_fraction=config.kelly_fraction)
        self.portfolio = Portfolio(config.initial_capital, config.n_assets)

        # Tracking
        self.equity_curve  : List[float] = [config.initial_capital]
        self.trade_pnls    : List[float] = []
        self.step          = 0
        self.current_prices= np.ones(self.n) * 100.0

    def run(self) -> BacktestStats:
        """Run the full backtest. Returns performance statistics."""
        logger.info("Starting backtest: %d assets, %d steps, method=%s",
                    self.n, self.cfg.n_steps, self.cfg.opt_method)
        t0 = time.perf_counter()

        for step in range(self.cfg.n_steps):
            self._step(step)

        elapsed = time.perf_counter() - t0
        logger.info("Backtest complete in %.2fs (%.0f steps/s)",
                    elapsed, self.cfg.n_steps / max(elapsed, _EPS))

        stats = BacktestStats.from_equity_curve(
            equity       = self.equity_curve,
            n_trades     = self.portfolio.n_trades,
            commissions  = self.portfolio.commissions,
            trade_pnls   = self.trade_pnls,
            steps_per_year = 252.0,
        )
        if self.cfg.verbose:
            stats.print_summary()
        return stats

    def _step(self, step: int) -> None:
        # 1. Generate market ticks
        ticks = self.market.make_ticks(spread_bps=self.cfg.slippage_bps)

        # 2. Extract new prices
        prices = np.array([t.mid() for t in ticks])
        self.current_prices = prices

        # 3. Process through data pipeline
        self.pipeline.process_batch(ticks)

        # 4. Update signals
        price_dict = {i: float(prices[i]) for i in range(self.n)}
        self.signals.update_prices(price_dict)
        self.signals.update_forward_returns(price_dict)

        # Update returns for covariance estimator
        if step > 0:
            prev_eq = self.equity_curve[-1] if self.equity_curve else self.cfg.initial_capital
            rets = np.zeros(self.n)
            for i in range(self.n):
                p_prev = self.market.prices[i] if step == 0 else prices[i]
                # Approximate: use log returns from GBM
                rets[i] = math.log(max(prices[i], _EPS) / max(prices[i] * 0.99, _EPS))
            self.optimizer.update_returns(rets)

        # 5. Rebalance if needed
        if step % self.cfg.rebalance_every == 0 and step > self.cfg.signal_lookback:
            self._rebalance(prices)

        # 6. Record equity
        equity = self.portfolio.equity(prices)
        self.equity_curve.append(equity)
        self.step += 1

    def _rebalance(self, prices: np.ndarray) -> None:
        """Compute target weights and execute trades."""
        # Get combined signals
        signal_dict = self.signals.get_normalized_signal()
        signals     = np.array([signal_dict.get(i, 0.0) for i in range(self.n)])

        # Compute target weights
        target_weights, trades_usd = self.optimizer.rebalance(
            signals, prices, self.portfolio.equity(prices))

        # Execute trades
        for i in range(self.n):
            trade_usd = float(trades_usd[i])
            if abs(trade_usd) < 100.0:  # minimum trade size
                continue

            price      = float(prices[i])
            if price < _EPS:
                continue
            qty        = trade_usd / price
            commission = abs(trade_usd) * self.cfg.commission_bps / 1e4

            prev_equity = self.portfolio.equity(prices)
            self.portfolio.apply_trade(i, qty, price, commission)
            new_equity  = self.portfolio.equity(prices)
            self.trade_pnls.append(new_equity - prev_equity)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

@dataclass
class GridSearchResult:
    config:    BacktestConfig
    stats:     BacktestStats
    sharpe:    float
    max_dd:    float
    ann_ret:   float

    def __repr__(self) -> str:
        return (f"GridSearchResult(method={self.config.opt_method}, "
                f"lookback={self.config.signal_lookback}, "
                f"sharpe={self.sharpe:.3f}, max_dd={self.max_dd*100:.1f}%)")


def run_grid_search(base_config: BacktestConfig,
                    methods: List[str] = None,
                    lookbacks: List[int] = None,
                    kelly_fractions: List[float] = None,
                    verbose: bool = False) -> List[GridSearchResult]:
    """
    Grid search over strategy parameters.
    Returns sorted list of results (best Sharpe first).
    """
    if methods is None:
        methods = ["erc", "mvo", "min_var"]
    if lookbacks is None:
        lookbacks = [10, 20, 40]
    if kelly_fractions is None:
        kelly_fractions = [0.25]

    results = []
    n_runs  = len(methods) * len(lookbacks) * len(kelly_fractions)
    run_idx = 0

    for method in methods:
        for lookback in lookbacks:
            for kf in kelly_fractions:
                cfg = BacktestConfig(
                    n_assets        = base_config.n_assets,
                    n_steps         = base_config.n_steps,
                    initial_capital = base_config.initial_capital,
                    commission_bps  = base_config.commission_bps,
                    slippage_bps    = base_config.slippage_bps,
                    signal_lookback = lookback,
                    opt_method      = method,
                    kelly_fraction  = kf,
                    seed            = base_config.seed,
                    sigma           = base_config.sigma,
                    mu              = base_config.mu,
                    verbose         = False,
                )
                runner = BacktestRunner(cfg)
                stats  = runner.run()
                result = GridSearchResult(
                    config  = cfg,
                    stats   = stats,
                    sharpe  = stats.sharpe,
                    max_dd  = stats.max_drawdown,
                    ann_ret = stats.annual_return,
                )
                results.append(result)
                run_idx += 1
                if verbose:
                    print(f"[{run_idx}/{n_runs}] {result}")

    results.sort(key=lambda r: r.sharpe, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Walk-forward analysis
# ---------------------------------------------------------------------------

class WalkForwardAnalyzer:
    """
    Walks forward through time, re-fitting strategy parameters
    on in-sample window and evaluating on out-of-sample.
    """

    def __init__(self, total_steps: int, is_window: int, oos_window: int,
                 base_config: BacktestConfig):
        self.total_steps = total_steps
        self.is_window   = is_window
        self.oos_window  = oos_window
        self.base_config = base_config
        self.folds: List[dict] = []

    def run(self) -> dict:
        """Returns walk-forward statistics."""
        oos_equities = []
        oos_sharpes  = []
        fold_idx     = 0

        offset = 0
        while offset + self.is_window + self.oos_window <= self.total_steps:
            # In-sample grid search
            is_cfg = BacktestConfig(
                n_assets    = self.base_config.n_assets,
                n_steps     = self.is_window,
                initial_capital = self.base_config.initial_capital,
                seed        = self.base_config.seed + fold_idx,
            )
            is_results = run_grid_search(is_cfg, verbose=False)
            best_cfg   = is_results[0].config if is_results else self.base_config

            # Out-of-sample evaluation with best parameters
            oos_cfg = BacktestConfig(
                n_assets    = best_cfg.n_assets,
                n_steps     = self.oos_window,
                initial_capital = self.base_config.initial_capital,
                opt_method  = best_cfg.opt_method,
                signal_lookback = best_cfg.signal_lookback,
                kelly_fraction  = best_cfg.kelly_fraction,
                seed        = self.base_config.seed + fold_idx + 1000,
            )
            oos_runner = BacktestRunner(oos_cfg)
            oos_stats  = oos_runner.run()
            oos_sharpes.append(oos_stats.sharpe)
            oos_equities.extend(oos_stats.equity_curve)

            fold = {
                "fold":        fold_idx,
                "is_best_method": best_cfg.opt_method,
                "is_best_sharpe": is_results[0].sharpe if is_results else 0.0,
                "oos_sharpe":  oos_stats.sharpe,
                "oos_return":  oos_stats.total_return,
                "oos_max_dd":  oos_stats.max_drawdown,
            }
            self.folds.append(fold)
            fold_idx += 1
            offset   += self.oos_window

        return {
            "n_folds":       fold_idx,
            "mean_oos_sharpe": float(np.mean(oos_sharpes)) if oos_sharpes else 0.0,
            "std_oos_sharpe":  float(np.std(oos_sharpes))  if oos_sharpes else 0.0,
            "folds":         self.folds,
        }
