"""
Core backtesting engine for the idea engine.

Bar-by-bar event loop, multi-asset support, fill simulation, cost models,
position tracking, performance analytics, walk-forward, Monte Carlo,
regime breakdown, factor attribution, and reporting.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class RebalanceFreq(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class BacktestConfig:
    strategy_name: str = "unnamed"
    universe: List[str] = field(default_factory=list)
    start_date: int = 0
    end_date: int = -1
    initial_capital: float = 1_000_000.0
    rebalance_freq: RebalanceFreq = RebalanceFreq.DAILY
    rebalance_days: Optional[List[int]] = None  # for CUSTOM
    cost_model: str = "proportional"  # fixed, proportional, sqrt, almgren_chriss
    cost_params: Dict[str, float] = field(default_factory=lambda: {"bps": 5.0})
    slippage_bps: float = 1.0
    max_leverage: float = 2.0
    max_position_pct: float = 0.10
    margin_requirement: float = 0.5


# ---------------------------------------------------------------------------
# Signal to Position Mapping
# ---------------------------------------------------------------------------

class SignalToPosition:
    """Convert raw signals to target positions."""

    @staticmethod
    def threshold(signals: np.ndarray, long_thresh: float = 0.5,
                  short_thresh: float = -0.5,
                  position_size: float = 1.0) -> np.ndarray:
        positions = np.zeros_like(signals)
        positions[signals > long_thresh] = position_size
        positions[signals < short_thresh] = -position_size
        return positions

    @staticmethod
    def proportional(signals: np.ndarray, scale: float = 1.0,
                     cap: float = 1.0) -> np.ndarray:
        positions = signals * scale
        return np.clip(positions, -cap, cap)

    @staticmethod
    def rank_based(signals: np.ndarray, long_pct: float = 0.2,
                   short_pct: float = 0.2) -> np.ndarray:
        n = len(signals)
        ranks = np.argsort(np.argsort(signals))  # 0-indexed ranks
        positions = np.zeros(n)
        long_cutoff = int(n * (1 - long_pct))
        short_cutoff = int(n * short_pct)
        positions[ranks >= long_cutoff] = 1.0
        positions[ranks < short_cutoff] = -1.0
        # Normalize to net zero
        long_count = np.sum(positions > 0)
        short_count = np.sum(positions < 0)
        if long_count > 0:
            positions[positions > 0] /= long_count
        if short_count > 0:
            positions[positions < 0] /= short_count
        return positions


# ---------------------------------------------------------------------------
# Cost Model
# ---------------------------------------------------------------------------

class CostModel:
    """Transaction cost models."""

    def __init__(self, model_type: str = "proportional", params: Optional[Dict[str, float]] = None):
        self.model_type = model_type
        self.params = params or {"bps": 5.0}

    def compute(self, trade_value: float, adv: float = 1e8) -> float:
        if self.model_type == "fixed":
            return self.params.get("fixed_cost", 1.0) * (1 if abs(trade_value) > 0 else 0)
        elif self.model_type == "proportional":
            return abs(trade_value) * self.params.get("bps", 5.0) * 1e-4
        elif self.model_type == "sqrt":
            sigma = self.params.get("sigma", 0.02)
            eta = self.params.get("eta", 0.1)
            participation = abs(trade_value) / (adv + 1e-10)
            return eta * sigma * abs(trade_value) * np.sqrt(participation)
        elif self.model_type == "almgren_chriss":
            sigma = self.params.get("sigma", 0.02)
            gamma = self.params.get("gamma", 0.1)
            eta = self.params.get("eta", 0.05)
            T = self.params.get("T", 1.0)
            permanent = gamma * abs(trade_value)
            temporary = eta * sigma * abs(trade_value) / np.sqrt(T + 1e-10)
            return permanent + temporary
        return 0.0


# ---------------------------------------------------------------------------
# Fill Simulator
# ---------------------------------------------------------------------------

@dataclass
class Order:
    asset_idx: int
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


@dataclass
class Fill:
    asset_idx: int
    quantity: float
    fill_price: float
    cost: float
    slippage: float
    timestamp: int = 0


class FillSimulator:
    """Simulate order fills with slippage and partial fills."""

    def __init__(self, slippage_bps: float = 1.0, partial_fill_prob: float = 0.0,
                 min_fill_pct: float = 0.5, rng: Optional[np.random.Generator] = None):
        self.slippage_bps = slippage_bps
        self.partial_fill_prob = partial_fill_prob
        self.min_fill_pct = min_fill_pct
        self.rng = rng or np.random.default_rng(42)

    def simulate_fill(self, order: Order, bar_open: float, bar_high: float,
                      bar_low: float, bar_close: float, bar_volume: float,
                      cost_model: CostModel, timestamp: int = 0) -> Optional[Fill]:
        if order.order_type == OrderType.MARKET:
            mid = (bar_open + bar_close) / 2
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return None
            if order.quantity > 0 and bar_low > order.limit_price:
                return None
            if order.quantity < 0 and bar_high < order.limit_price:
                return None
            mid = order.limit_price
        elif order.order_type == OrderType.STOP:
            if order.stop_price is None:
                return None
            if order.quantity > 0 and bar_high < order.stop_price:
                return None
            if order.quantity < 0 and bar_low > order.stop_price:
                return None
            mid = order.stop_price
        else:
            return None
        # Slippage
        direction = np.sign(order.quantity)
        slip = mid * self.slippage_bps * 1e-4 * direction
        fill_price = mid + slip
        # Partial fill
        fill_qty = order.quantity
        if self.rng.random() < self.partial_fill_prob:
            fill_pct = self.min_fill_pct + self.rng.random() * (1 - self.min_fill_pct)
            fill_qty = order.quantity * fill_pct
        trade_value = abs(fill_qty * fill_price)
        cost = cost_model.compute(trade_value)
        return Fill(asset_idx=order.asset_idx, quantity=fill_qty,
                    fill_price=fill_price, cost=cost, slippage=abs(slip * fill_qty),
                    timestamp=timestamp)


# ---------------------------------------------------------------------------
# Position Tracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """Track positions, cash, margin, and P&L."""

    def __init__(self, n_assets: int, initial_capital: float = 1_000_000.0):
        self.n_assets = n_assets
        self.positions = np.zeros(n_assets)
        self.cash = initial_capital
        self.initial_capital = initial_capital
        self.avg_entry_prices = np.zeros(n_assets)
        self.realized_pnl = 0.0
        self.total_costs = 0.0
        self.total_slippage = 0.0

    def apply_fill(self, fill: Fill) -> None:
        idx = fill.asset_idx
        old_pos = self.positions[idx]
        new_pos = old_pos + fill.quantity
        # Realized P&L on reducing/closing
        if old_pos != 0 and np.sign(fill.quantity) != np.sign(old_pos):
            closed_qty = min(abs(fill.quantity), abs(old_pos)) * np.sign(old_pos)
            self.realized_pnl += closed_qty * (fill.fill_price - self.avg_entry_prices[idx])
        # Update average entry price
        if np.sign(new_pos) == np.sign(old_pos) or old_pos == 0:
            if abs(new_pos) > 1e-12:
                total_cost = self.avg_entry_prices[idx] * abs(old_pos) + fill.fill_price * abs(fill.quantity)
                self.avg_entry_prices[idx] = total_cost / abs(new_pos)
        elif abs(new_pos) > 1e-12:
            self.avg_entry_prices[idx] = fill.fill_price
        else:
            self.avg_entry_prices[idx] = 0.0
        self.positions[idx] = new_pos
        self.cash -= fill.quantity * fill.fill_price
        self.cash -= fill.cost
        self.total_costs += fill.cost
        self.total_slippage += fill.slippage

    def mark_to_market(self, prices: np.ndarray) -> float:
        unrealized = np.sum(self.positions * prices)
        return self.cash + unrealized

    def leverage(self, prices: np.ndarray) -> float:
        nav = self.mark_to_market(prices)
        gross = np.sum(np.abs(self.positions * prices))
        return gross / (nav + 1e-10)

    def position_weights(self, prices: np.ndarray) -> np.ndarray:
        nav = self.mark_to_market(prices)
        return self.positions * prices / (nav + 1e-10)


# ---------------------------------------------------------------------------
# Performance Calculator
# ---------------------------------------------------------------------------

class PerformanceCalculator:
    """Compute strategy performance metrics from equity curve."""

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
        excess = returns - rf / periods
        if np.std(excess) < 1e-12:
            return 0.0
        return float(np.mean(excess) / np.std(excess) * np.sqrt(periods))

    @staticmethod
    def sortino_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
        excess = returns - rf / periods
        downside = returns[returns < 0]
        if len(downside) == 0 or np.std(downside) < 1e-12:
            return 0.0
        return float(np.mean(excess) / np.std(downside) * np.sqrt(periods))

    @staticmethod
    def calmar_ratio(returns: np.ndarray, periods: int = 252) -> float:
        ann_ret = np.mean(returns) * periods
        cum = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / (running_max + 1e-10)
        max_dd = abs(np.min(dd))
        return float(ann_ret / (max_dd + 1e-10))

    @staticmethod
    def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
        gains = np.sum(np.maximum(returns - threshold, 0))
        losses = np.sum(np.maximum(threshold - returns, 0))
        return float(gains / (losses + 1e-10))

    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        cum = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / (running_max + 1e-10)
        return float(abs(np.min(dd)))

    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        if len(returns) == 0:
            return 0.0
        return float(np.mean(returns > 0))

    @staticmethod
    def profit_factor(returns: np.ndarray) -> float:
        gains = np.sum(returns[returns > 0])
        losses = abs(np.sum(returns[returns < 0]))
        return float(gains / (losses + 1e-10))

    @staticmethod
    def all_metrics(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> Dict[str, float]:
        pc = PerformanceCalculator
        return {
            "ann_return": float(np.mean(returns) * periods),
            "ann_volatility": float(np.std(returns) * np.sqrt(periods)),
            "sharpe": pc.sharpe_ratio(returns, rf, periods),
            "sortino": pc.sortino_ratio(returns, rf, periods),
            "calmar": pc.calmar_ratio(returns, periods),
            "omega": pc.omega_ratio(returns),
            "max_drawdown": pc.max_drawdown(returns),
            "win_rate": pc.win_rate(returns),
            "profit_factor": pc.profit_factor(returns),
            "skewness": float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-10)) ** 3)),
            "kurtosis": float(np.mean(((returns - returns.mean()) / (returns.std() + 1e-10)) ** 4) - 3),
            "n_observations": len(returns),
        }


# ---------------------------------------------------------------------------
# Equity Curve
# ---------------------------------------------------------------------------

class EquityCurve:
    """Daily NAV, drawdown series, rolling metrics."""

    def __init__(self, nav_series: np.ndarray):
        self.nav = nav_series
        self.returns = np.diff(nav_series) / (nav_series[:-1] + 1e-10)

    def drawdown_series(self) -> np.ndarray:
        running_max = np.maximum.accumulate(self.nav)
        return (self.nav - running_max) / (running_max + 1e-10)

    def rolling_sharpe(self, window: int = 63) -> np.ndarray:
        out = np.full(len(self.returns), np.nan)
        for i in range(window, len(self.returns)):
            r = self.returns[i - window:i]
            if np.std(r) > 1e-12:
                out[i] = np.mean(r) / np.std(r) * np.sqrt(252)
        return out

    def rolling_volatility(self, window: int = 21) -> np.ndarray:
        out = np.full(len(self.returns), np.nan)
        for i in range(window, len(self.returns)):
            out[i] = np.std(self.returns[i - window:i]) * np.sqrt(252)
        return out

    def underwater_periods(self) -> List[Dict[str, Any]]:
        dd = self.drawdown_series()
        periods = []
        in_dd = False
        start = 0
        for i in range(len(dd)):
            if dd[i] < -1e-6 and not in_dd:
                in_dd = True
                start = i
            elif (dd[i] >= -1e-6 or i == len(dd) - 1) and in_dd:
                in_dd = False
                periods.append({
                    "start": start,
                    "end": i,
                    "duration": i - start,
                    "depth": float(np.min(dd[start:i + 1])),
                })
        return periods


# ---------------------------------------------------------------------------
# Trade Log
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    asset_idx: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    cost: float
    hold_duration: int


class TradeLog:
    """Record and analyze individual trades."""

    def __init__(self):
        self.trades: List[TradeRecord] = []
        self._open_trades: Dict[int, List[Dict]] = {}

    def open_trade(self, asset_idx: int, time: int, price: float,
                   quantity: float, cost: float) -> None:
        if asset_idx not in self._open_trades:
            self._open_trades[asset_idx] = []
        self._open_trades[asset_idx].append({
            "time": time, "price": price, "quantity": quantity, "cost": cost
        })

    def close_trade(self, asset_idx: int, time: int, price: float,
                    quantity: float, cost: float) -> None:
        if asset_idx not in self._open_trades or not self._open_trades[asset_idx]:
            return
        entry = self._open_trades[asset_idx].pop(0)
        closed_qty = min(abs(quantity), abs(entry["quantity"]))
        direction = np.sign(entry["quantity"])
        pnl = direction * closed_qty * (price - entry["price"]) - entry["cost"] - cost
        self.trades.append(TradeRecord(
            asset_idx=asset_idx,
            entry_time=entry["time"],
            exit_time=time,
            entry_price=entry["price"],
            exit_price=price,
            quantity=direction * closed_qty,
            pnl=pnl,
            cost=entry["cost"] + cost,
            hold_duration=time - entry["time"],
        ))

    def summary(self) -> Dict[str, Any]:
        if not self.trades:
            return {"n_trades": 0}
        pnls = np.array([t.pnl for t in self.trades])
        durations = np.array([t.hold_duration for t in self.trades])
        costs = np.array([t.cost for t in self.trades])
        winners = pnls[pnls > 0]
        losers = pnls[pnls <= 0]
        return {
            "n_trades": len(self.trades),
            "total_pnl": float(np.sum(pnls)),
            "avg_pnl": float(np.mean(pnls)),
            "win_rate": float(len(winners) / len(pnls)),
            "avg_winner": float(np.mean(winners)) if len(winners) > 0 else 0.0,
            "avg_loser": float(np.mean(losers)) if len(losers) > 0 else 0.0,
            "profit_factor": float(np.sum(winners) / (abs(np.sum(losers)) + 1e-10)),
            "avg_hold_duration": float(np.mean(durations)),
            "total_costs": float(np.sum(costs)),
            "max_win": float(np.max(pnls)),
            "max_loss": float(np.min(pnls)),
        }


# ---------------------------------------------------------------------------
# Backtest Engine: bar-by-bar event loop
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Bar-by-bar backtesting engine with multi-asset support."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cost_model = CostModel(config.cost_model, config.cost_params)
        self.fill_sim = FillSimulator(slippage_bps=config.slippage_bps)

    def run(self, prices: np.ndarray, signals: np.ndarray,
            volumes: Optional[np.ndarray] = None,
            signal_mapper: str = "proportional",
            signal_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        prices: (T, N) close prices
        signals: (T, N) raw signals
        volumes: (T, N) optional volume data
        """
        T, N = prices.shape
        config = self.config
        start = config.start_date
        end = config.end_date if config.end_date > 0 else T
        tracker = PositionTracker(N, config.initial_capital)
        trade_log = TradeLog()
        nav_series = [config.initial_capital]
        daily_returns = []
        signal_params = signal_params or {}
        # Determine rebalance days
        rebal_days = set()
        if config.rebalance_freq == RebalanceFreq.DAILY:
            rebal_days = set(range(start, end))
        elif config.rebalance_freq == RebalanceFreq.WEEKLY:
            rebal_days = set(range(start, end, 5))
        elif config.rebalance_freq == RebalanceFreq.MONTHLY:
            rebal_days = set(range(start, end, 21))
        elif config.rebalance_freq == RebalanceFreq.CUSTOM and config.rebalance_days:
            rebal_days = set(config.rebalance_days)
        for t in range(start + 1, end):
            # Mark to market with today's prices
            current_nav = tracker.mark_to_market(prices[t])
            if t in rebal_days:
                # Compute target positions
                raw_signals = signals[t]
                if signal_mapper == "threshold":
                    target_weights = SignalToPosition.threshold(raw_signals, **signal_params)
                elif signal_mapper == "rank":
                    target_weights = SignalToPosition.rank_based(raw_signals, **signal_params)
                else:
                    target_weights = SignalToPosition.proportional(raw_signals, **signal_params)
                # Cap position sizes
                target_weights = np.clip(target_weights, -config.max_position_pct, config.max_position_pct)
                target_positions = target_weights * current_nav / (prices[t] + 1e-10)
                # Generate orders
                for i in range(N):
                    delta = target_positions[i] - tracker.positions[i]
                    if abs(delta) < 1e-6:
                        continue
                    order = Order(asset_idx=i, quantity=delta, order_type=OrderType.MARKET)
                    fill = self.fill_sim.simulate_fill(
                        order, prices[t], prices[t] * 1.01, prices[t] * 0.99,
                        prices[t], volumes[t, i] if volumes is not None else 1e8,
                        self.cost_model, timestamp=t)
                    if fill is not None:
                        # Trade log
                        old_pos = tracker.positions[i]
                        if np.sign(delta) != np.sign(old_pos) and abs(old_pos) > 1e-6:
                            trade_log.close_trade(i, t, fill.fill_price,
                                                   delta, fill.cost)
                        if abs(tracker.positions[i] + delta) > 1e-6:
                            trade_log.open_trade(i, t, fill.fill_price,
                                                  delta, fill.cost)
                        tracker.apply_fill(fill)
                # Leverage check
                lev = tracker.leverage(prices[t])
                if lev > config.max_leverage:
                    scale = config.max_leverage / lev
                    for i in range(N):
                        reduce = tracker.positions[i] * (1 - scale)
                        if abs(reduce) > 1e-6:
                            order = Order(asset_idx=i, quantity=-reduce)
                            fill = self.fill_sim.simulate_fill(
                                order, prices[t], prices[t] * 1.01, prices[t] * 0.99,
                                prices[t], 1e8, self.cost_model, t)
                            if fill:
                                tracker.apply_fill(fill)
            new_nav = tracker.mark_to_market(prices[t])
            prev_nav = nav_series[-1]
            daily_ret = (new_nav - prev_nav) / (prev_nav + 1e-10)
            daily_returns.append(daily_ret)
            nav_series.append(new_nav)
        nav_arr = np.array(nav_series)
        ret_arr = np.array(daily_returns)
        eq_curve = EquityCurve(nav_arr)
        perf = PerformanceCalculator.all_metrics(ret_arr)
        perf["total_costs"] = tracker.total_costs
        perf["total_slippage"] = tracker.total_slippage
        perf["realized_pnl"] = tracker.realized_pnl
        return {
            "config": config.__dict__,
            "nav_series": nav_arr,
            "returns": ret_arr,
            "equity_curve": eq_curve,
            "performance": perf,
            "trade_log": trade_log.summary(),
            "final_positions": tracker.positions.copy(),
        }


# ---------------------------------------------------------------------------
# Walk-Forward Mode
# ---------------------------------------------------------------------------

class WalkForwardMode:
    """Walk-forward backtesting with train/test splits."""

    def __init__(self, train_window: int = 252, test_window: int = 63,
                 step: int = 21):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def generate_splits(self, T: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        splits = []
        start = 0
        while start + self.train_window + self.test_window <= T:
            train = (start, start + self.train_window)
            test = (start + self.train_window, start + self.train_window + self.test_window)
            splits.append((train, test))
            start += self.step
        return splits

    def run(self, prices: np.ndarray, signal_fn: Callable,
            config: BacktestConfig) -> Dict[str, Any]:
        """
        signal_fn: (train_prices) -> (test_signals) callable.
        Returns combined out-of-sample results.
        """
        T, N = prices.shape
        splits = self.generate_splits(T)
        all_returns = []
        split_results = []
        for (tr_s, tr_e), (te_s, te_e) in splits:
            train_prices = prices[tr_s:tr_e]
            test_prices = prices[te_s:te_e]
            signals = signal_fn(train_prices)
            if signals.shape[0] != test_prices.shape[0]:
                min_len = min(signals.shape[0], test_prices.shape[0])
                signals = signals[:min_len]
                test_prices = test_prices[:min_len]
            cfg = BacktestConfig(**{**config.__dict__,
                                    "start_date": 0, "end_date": len(test_prices)})
            engine = BacktestEngine(cfg)
            result = engine.run(test_prices, signals)
            all_returns.extend(result["returns"].tolist())
            split_results.append(result["performance"])
        combined_returns = np.array(all_returns)
        return {
            "combined_performance": PerformanceCalculator.all_metrics(combined_returns),
            "split_performances": split_results,
            "n_splits": len(splits),
            "combined_returns": combined_returns,
        }


# ---------------------------------------------------------------------------
# Monte Carlo Bootstrap
# ---------------------------------------------------------------------------

class MonteCarloBootstrap:
    """Resample trades or returns for confidence intervals."""

    def __init__(self, n_simulations: int = 1000, rng: Optional[np.random.Generator] = None):
        self.n_simulations = n_simulations
        self.rng = rng or np.random.default_rng(42)

    def bootstrap_returns(self, returns: np.ndarray) -> Dict[str, Any]:
        n = len(returns)
        sharpes = []
        max_dds = []
        ann_rets = []
        for _ in range(self.n_simulations):
            idx = self.rng.integers(0, n, size=n)
            sample = returns[idx]
            sharpes.append(PerformanceCalculator.sharpe_ratio(sample))
            max_dds.append(PerformanceCalculator.max_drawdown(sample))
            ann_rets.append(float(np.mean(sample) * 252))
        return {
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_ci_95": (float(np.percentile(sharpes, 2.5)),
                             float(np.percentile(sharpes, 97.5))),
            "max_dd_mean": float(np.mean(max_dds)),
            "max_dd_ci_95": (float(np.percentile(max_dds, 2.5)),
                             float(np.percentile(max_dds, 97.5))),
            "ann_return_ci_95": (float(np.percentile(ann_rets, 2.5)),
                                 float(np.percentile(ann_rets, 97.5))),
        }

    def block_bootstrap(self, returns: np.ndarray, block_size: int = 21) -> Dict[str, Any]:
        n = len(returns)
        n_blocks = n // block_size
        sharpes = []
        for _ in range(self.n_simulations):
            blocks = self.rng.integers(0, n - block_size, size=n_blocks)
            sample = np.concatenate([returns[b:b + block_size] for b in blocks])
            sharpes.append(PerformanceCalculator.sharpe_ratio(sample))
        return {
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_ci_95": (float(np.percentile(sharpes, 2.5)),
                             float(np.percentile(sharpes, 97.5))),
        }


# ---------------------------------------------------------------------------
# Regime Breakdown
# ---------------------------------------------------------------------------

class RegimeBreakdown:
    """Performance breakdown per regime label."""

    @staticmethod
    def compute(returns: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Dict[str, float]]:
        unique_regimes = np.unique(regime_labels)
        breakdown = {}
        for regime in unique_regimes:
            mask = regime_labels == regime
            r = returns[mask]
            if len(r) > 1:
                breakdown[str(regime)] = PerformanceCalculator.all_metrics(r)
            else:
                breakdown[str(regime)] = {"n_observations": len(r)}
        return breakdown


# ---------------------------------------------------------------------------
# Factor Attribution
# ---------------------------------------------------------------------------

class FactorAttribution:
    """Decompose returns into factor exposures + alpha."""

    @staticmethod
    def compute(returns: np.ndarray, factor_returns: np.ndarray,
                factor_names: Optional[List[str]] = None) -> Dict[str, Any]:
        T = len(returns)
        K = factor_returns.shape[1]
        if factor_names is None:
            factor_names = [f"factor_{i}" for i in range(K)]
        # OLS: returns = alpha + beta * factors + epsilon
        X = np.column_stack([np.ones(T), factor_returns])
        try:
            beta = np.linalg.lstsq(X, returns, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(K + 1)
        alpha = beta[0]
        betas = beta[1:]
        fitted = X @ beta
        residual = returns - fitted
        ss_tot = np.sum((returns - returns.mean()) ** 2)
        ss_res = np.sum(residual ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        factor_contributions = {}
        for i, name in enumerate(factor_names):
            contrib = betas[i] * np.mean(factor_returns[:, i]) * 252
            factor_contributions[name] = {
                "beta": float(betas[i]),
                "ann_contribution": float(contrib),
            }
        return {
            "alpha_ann": float(alpha * 252),
            "r_squared": float(r_squared),
            "residual_vol": float(np.std(residual) * np.sqrt(252)),
            "factor_contributions": factor_contributions,
            "betas": {name: float(betas[i]) for i, name in enumerate(factor_names)},
        }


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

def generate_report(backtest_result: Dict[str, Any],
                    factor_returns: Optional[np.ndarray] = None,
                    regime_labels: Optional[np.ndarray] = None,
                    factor_names: Optional[List[str]] = None,
                    n_bootstrap: int = 1000) -> Dict[str, Any]:
    """Generate comprehensive backtest report."""
    report: Dict[str, Any] = {}
    report["performance"] = backtest_result["performance"]
    report["trade_summary"] = backtest_result["trade_log"]
    # Monte Carlo
    mc = MonteCarloBootstrap(n_simulations=n_bootstrap)
    report["monte_carlo"] = mc.bootstrap_returns(backtest_result["returns"])
    # Regime breakdown
    if regime_labels is not None:
        n = min(len(backtest_result["returns"]), len(regime_labels))
        report["regime_breakdown"] = RegimeBreakdown.compute(
            backtest_result["returns"][:n], regime_labels[:n])
    # Factor attribution
    if factor_returns is not None:
        n = min(len(backtest_result["returns"]), len(factor_returns))
        report["factor_attribution"] = FactorAttribution.compute(
            backtest_result["returns"][:n], factor_returns[:n], factor_names)
    return report
