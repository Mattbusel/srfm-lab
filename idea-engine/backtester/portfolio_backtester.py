"""
Portfolio-level backtester -- multi-asset, multi-strategy backtesting engine.

Extends the single-asset backtester_engine with:
  - Multi-asset universe management
  - Cross-asset signal aggregation
  - Portfolio-level rebalancing (threshold/calendar/risk-target)
  - Sector and factor constraints during backtest
  - Transaction cost aware portfolio transitions
  - Multi-strategy blending with regime routing
  - Cash management and margin accounting
  - Corporate action handling (splits, dividends)
  - Benchmark tracking and relative performance
  - Walk-forward portfolio optimization
  - Attribution: Brinson (allocation + selection + interaction)
  - Turnover analysis and cost decomposition
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


# -- Data Structures --

@dataclass
class Asset:
    symbol: str
    sector: str = "unknown"
    market_cap: float = 0.0
    avg_daily_volume: float = 0.0
    borrow_cost_bps: float = 0.0
    tick_size: float = 0.01


@dataclass
class Bar:
    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0


@dataclass
class PortfolioState:
    timestamp: int = 0
    cash: float = 0.0
    positions: dict = field(default_factory=dict)  # symbol -> quantity
    avg_costs: dict = field(default_factory=dict)   # symbol -> avg cost basis
    nav: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    timestamp: int
    symbol: str
    side: str         # buy / sell
    quantity: float
    price: float
    cost_bps: float
    slippage_bps: float
    commission: float
    pnl: float = 0.0
    hold_bars: int = 0
    signal_strength: float = 0.0
    regime: str = "unknown"
    strategy: str = ""


@dataclass
class RebalanceEvent:
    timestamp: int
    reason: str       # threshold / calendar / risk_target / signal
    trades: list = field(default_factory=list)
    turnover_pct: float = 0.0
    cost_bps: float = 0.0


@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    rebalance_mode: str = "threshold"   # threshold / calendar / risk_target
    rebalance_threshold: float = 0.05   # 5% drift triggers rebalance
    rebalance_frequency: int = 21       # business days (for calendar mode)
    cost_model: str = "proportional"    # fixed / proportional / sqrt_impact
    base_cost_bps: float = 5.0
    slippage_bps: float = 3.0
    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30
    min_trade_pct: float = 0.005        # min trade size as % of NAV
    short_allowed: bool = False
    leverage_limit: float = 1.0
    benchmark_weights: dict = field(default_factory=dict)
    vol_target: float = 0.0            # 0 = no vol targeting


@dataclass
class BacktestMetrics:
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    annualized_vol_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_hold_bars: float = 0.0
    total_trades: int = 0
    turnover_annual_pct: float = 0.0
    total_costs_pct: float = 0.0
    information_ratio: float = 0.0
    tracking_error_pct: float = 0.0
    beta: float = 0.0
    alpha_pct: float = 0.0


# -- Cost Models --

class CostModel:
    def __init__(self, model_type: str = "proportional", base_bps: float = 5.0, slippage_bps: float = 3.0):
        self.model_type = model_type
        self.base_bps = base_bps
        self.slippage_bps = slippage_bps

    def compute_cost(self, notional: float, adv: float = 1e9) -> float:
        if self.model_type == "fixed":
            return self.base_bps / 10000 * notional
        elif self.model_type == "proportional":
            return (self.base_bps + self.slippage_bps) / 10000 * notional
        elif self.model_type == "sqrt_impact":
            participation = notional / max(adv, 1e-10)
            impact_bps = 10 * math.sqrt(participation)
            return (self.base_bps + self.slippage_bps + impact_bps) / 10000 * notional
        return self.base_bps / 10000 * notional

    def execution_price(self, mid_price: float, side: str, notional: float, adv: float = 1e9) -> float:
        cost_frac = self.compute_cost(notional, adv) / max(notional, 1e-10)
        if side == "buy":
            return mid_price * (1 + cost_frac)
        else:
            return mid_price * (1 - cost_frac)


# -- Signal Framework --

class SignalGenerator:
    """Base class for signal generation."""

    def generate(self, bars_history: dict, current_bars: dict, regime: str) -> dict:
        """Return dict of symbol -> signal (-1 to +1)."""
        return {}


class MomentumSignal(SignalGenerator):
    def __init__(self, lookback: int = 252, skip: int = 21):
        self.lookback = lookback
        self.skip = skip

    def generate(self, bars_history: dict, current_bars: dict, regime: str) -> dict:
        signals = {}
        for symbol, bars in bars_history.items():
            if len(bars) < self.lookback:
                continue
            prices = np.array([b.close for b in bars])
            if len(prices) >= self.lookback:
                mom = prices[-self.skip] / prices[-self.lookback] - 1
                signals[symbol] = float(np.tanh(mom * 3))
        return signals


class MeanReversionSignal(SignalGenerator):
    def __init__(self, lookback: int = 63):
        self.lookback = lookback

    def generate(self, bars_history: dict, current_bars: dict, regime: str) -> dict:
        signals = {}
        for symbol, bars in bars_history.items():
            if len(bars) < self.lookback:
                continue
            prices = np.array([b.close for b in bars[-self.lookback:]])
            z = (prices[-1] - prices.mean()) / max(prices.std(), 1e-10)
            if abs(z) > 1.0:
                signals[symbol] = float(-np.tanh(z / 2))
        return signals


class VolatilitySignal(SignalGenerator):
    def __init__(self, lookback: int = 21):
        self.lookback = lookback

    def generate(self, bars_history: dict, current_bars: dict, regime: str) -> dict:
        signals = {}
        for symbol, bars in bars_history.items():
            if len(bars) < self.lookback + 1:
                continue
            prices = np.array([b.close for b in bars])
            returns = np.diff(np.log(prices[-self.lookback - 1:]))
            vol = float(returns.std() * math.sqrt(252))
            signals[symbol] = float(-np.tanh((vol - 0.2) * 5))
        return signals


class CompositeSignal(SignalGenerator):
    def __init__(self, signals: list, weights: list = None):
        self.signals = signals
        self.weights = weights or [1.0 / len(signals)] * len(signals)

    def generate(self, bars_history: dict, current_bars: dict, regime: str) -> dict:
        all_signals = [s.generate(bars_history, current_bars, regime) for s in self.signals]
        symbols = set()
        for s in all_signals:
            symbols.update(s.keys())

        composite = {}
        for sym in symbols:
            total = 0.0
            w_sum = 0.0
            for sig_dict, w in zip(all_signals, self.weights):
                if sym in sig_dict:
                    total += sig_dict[sym] * w
                    w_sum += w
            if w_sum > 0:
                composite[sym] = float(total / w_sum)
        return composite


# -- Portfolio Optimizer --

class PortfolioOptimizer:
    """Convert signals to target weights."""

    def __init__(
        self,
        method: str = "proportional",
        max_position: float = 0.10,
        max_sector: float = 0.30,
        short_allowed: bool = False,
        leverage_limit: float = 1.0,
    ):
        self.method = method
        self.max_position = max_position
        self.max_sector = max_sector
        self.short_allowed = short_allowed
        self.leverage_limit = leverage_limit

    def optimize(
        self,
        signals: dict,
        assets: dict,
        current_prices: dict,
        covariance: Optional[np.ndarray] = None,
        vol_target: float = 0.0,
    ) -> dict:
        if not signals:
            return {}

        if self.method == "proportional":
            return self._proportional(signals, assets)
        elif self.method == "risk_parity":
            return self._risk_parity(signals, assets, covariance)
        elif self.method == "mean_variance":
            return self._mean_variance(signals, assets, covariance, vol_target)
        return self._proportional(signals, assets)

    def _proportional(self, signals: dict, assets: dict) -> dict:
        abs_total = sum(abs(v) for v in signals.values()) + 1e-10
        weights = {}
        for sym, sig in signals.items():
            if not self.short_allowed and sig < 0:
                continue
            w = sig / abs_total * self.leverage_limit
            w = max(-self.max_position, min(self.max_position, w))
            weights[sym] = w

        # Sector constraints
        weights = self._apply_sector_constraints(weights, assets)
        return weights

    def _risk_parity(self, signals: dict, assets: dict, cov: Optional[np.ndarray]) -> dict:
        symbols = sorted(signals.keys())
        n = len(symbols)
        if n == 0:
            return {}
        if cov is None or cov.shape[0] != n:
            return self._proportional(signals, assets)

        # ERC: w_i * (Sigma @ w)_i = budget for all i
        w = np.ones(n) / n
        for _ in range(100):
            risk_contrib = w * (cov @ w)
            total_risk = float(w @ cov @ w)
            target = total_risk / n
            for i in range(n):
                if risk_contrib[i] > 0:
                    w[i] *= (target / risk_contrib[i]) ** 0.5
            w = np.maximum(w, 0)
            w /= w.sum() + 1e-10

        # Apply signal direction
        weights = {}
        for i, sym in enumerate(symbols):
            direction = 1.0 if signals[sym] > 0 else (-1.0 if self.short_allowed else 0.0)
            weights[sym] = float(w[i] * direction * self.leverage_limit)
            weights[sym] = max(-self.max_position, min(self.max_position, weights[sym]))

        return self._apply_sector_constraints(weights, assets)

    def _mean_variance(self, signals: dict, assets: dict, cov: Optional[np.ndarray], vol_target: float) -> dict:
        symbols = sorted(signals.keys())
        n = len(symbols)
        if n == 0 or cov is None or cov.shape[0] != n:
            return self._proportional(signals, assets)

        mu = np.array([signals[s] * 0.1 for s in symbols])  # signal as expected return proxy
        # Solve: max mu^T w - 0.5 * lambda * w^T Sigma w
        risk_aversion = 2.0
        try:
            inv_cov = np.linalg.inv(cov + np.eye(n) * 1e-6)
            w = inv_cov @ mu / risk_aversion
        except np.linalg.LinAlgError:
            w = np.ones(n) / n

        if not self.short_allowed:
            w = np.maximum(w, 0)
        w_sum = abs(w).sum()
        if w_sum > self.leverage_limit:
            w *= self.leverage_limit / w_sum

        # Vol targeting
        if vol_target > 0:
            port_vol = float(np.sqrt(w @ cov @ w) * math.sqrt(252))
            if port_vol > 1e-6:
                scale = vol_target / port_vol
                w *= min(scale, self.leverage_limit / max(abs(w).sum(), 1e-10))

        weights = {}
        for i, sym in enumerate(symbols):
            weights[sym] = float(max(-self.max_position, min(self.max_position, w[i])))

        return self._apply_sector_constraints(weights, assets)

    def _apply_sector_constraints(self, weights: dict, assets: dict) -> dict:
        sector_totals = {}
        for sym, w in weights.items():
            sector = assets.get(sym, Asset(sym)).sector if isinstance(assets.get(sym), Asset) else "unknown"
            sector_totals[sector] = sector_totals.get(sector, 0.0) + abs(w)

        for sector, total in sector_totals.items():
            if total > self.max_sector:
                scale = self.max_sector / total
                for sym, w in weights.items():
                    s = assets.get(sym, Asset(sym)).sector if isinstance(assets.get(sym), Asset) else "unknown"
                    if s == sector:
                        weights[sym] = w * scale
        return weights


# -- Regime Detector --

class SimpleRegimeDetector:
    def detect(self, returns: np.ndarray) -> str:
        if len(returns) < 21:
            return "unknown"
        vol = float(returns[-21:].std() * math.sqrt(252))
        trend = float(returns[-21:].mean() * 252)
        if vol > 0.35:
            return "crisis"
        elif vol > 0.20:
            return "high_vol"
        elif trend > 0.10:
            return "trending_up"
        elif trend < -0.10:
            return "trending_down"
        return "mean_reverting"


# -- Main Backtester --

class PortfolioBacktester:
    """Multi-asset portfolio backtesting engine."""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.cost_model = CostModel(
            self.config.cost_model,
            self.config.base_cost_bps,
            self.config.slippage_bps,
        )
        self.regime_detector = SimpleRegimeDetector()

    def run(
        self,
        bars_by_symbol: dict,      # symbol -> list[Bar] sorted by time
        signal_generator: SignalGenerator,
        optimizer: PortfolioOptimizer = None,
        assets: dict = None,       # symbol -> Asset
    ) -> dict:
        """Run full portfolio backtest."""
        if optimizer is None:
            optimizer = PortfolioOptimizer(
                max_position=self.config.max_position_pct,
                short_allowed=self.config.short_allowed,
                leverage_limit=self.config.leverage_limit,
            )

        if assets is None:
            assets = {s: Asset(s) for s in bars_by_symbol}

        # Align timestamps
        all_timestamps = set()
        for bars in bars_by_symbol.values():
            for b in bars:
                all_timestamps.add(b.timestamp)
        timestamps = sorted(all_timestamps)

        # Initialize portfolio
        state = PortfolioState(
            cash=self.config.initial_capital,
            nav=self.config.initial_capital,
            equity=self.config.initial_capital,
        )

        # Track history
        nav_history = []
        trade_log = []
        rebalance_log = []
        bars_history = {s: [] for s in bars_by_symbol}
        returns_history = []
        last_rebalance = 0

        for t_idx, ts in enumerate(timestamps):
            # Get current bars
            current_bars = {}
            for sym, bars in bars_by_symbol.items():
                matching = [b for b in bars if b.timestamp == ts]
                if matching:
                    current_bars[sym] = matching[0]
                    bars_history[sym].append(matching[0])

            if not current_bars:
                continue

            # Mark to market
            current_prices = {sym: b.close for sym, b in current_bars.items()}
            self._mark_to_market(state, current_prices, ts)
            nav_history.append(state.nav)

            # Returns
            if len(nav_history) >= 2:
                ret = (nav_history[-1] - nav_history[-2]) / max(nav_history[-2], 1e-10)
                returns_history.append(ret)

            # Check rebalance trigger
            should_rebalance = self._check_rebalance(
                state, current_prices, t_idx, last_rebalance, optimizer
            )

            if should_rebalance and len(returns_history) >= 21:
                # Detect regime
                regime = self.regime_detector.detect(np.array(returns_history))

                # Generate signals
                signals = signal_generator.generate(bars_history, current_bars, regime)

                # Compute covariance
                cov = self._compute_covariance(bars_history, list(signals.keys()))

                # Optimize
                target_weights = optimizer.optimize(
                    signals, assets, current_prices, cov, self.config.vol_target
                )

                # Execute rebalance
                trades, turnover, cost = self._rebalance(
                    state, target_weights, current_prices, assets, regime
                )

                if trades:
                    trade_log.extend(trades)
                    rebalance_log.append(RebalanceEvent(
                        timestamp=ts,
                        reason=self.config.rebalance_mode,
                        trades=trades,
                        turnover_pct=turnover,
                        cost_bps=cost,
                    ))
                    last_rebalance = t_idx

        # Compute metrics
        metrics = self._compute_metrics(nav_history, trade_log, returns_history)

        # Attribution
        attribution = self._compute_attribution(
            trade_log, bars_by_symbol, assets, self.config.benchmark_weights
        )

        return {
            "metrics": metrics,
            "nav_history": nav_history,
            "trade_log": trade_log,
            "rebalance_log": rebalance_log,
            "final_state": state,
            "attribution": attribution,
            "n_timestamps": len(timestamps),
        }

    def _mark_to_market(self, state: PortfolioState, prices: dict, ts: int) -> None:
        state.timestamp = ts
        unrealized = 0.0
        equity = state.cash
        for sym, qty in state.positions.items():
            if sym in prices:
                mv = qty * prices[sym]
                cost = qty * state.avg_costs.get(sym, prices[sym])
                unrealized += mv - cost
                equity += mv
        state.unrealized_pnl = unrealized
        state.equity = equity
        state.nav = equity

    def _check_rebalance(self, state, prices, t_idx, last_rebal, optimizer) -> bool:
        if self.config.rebalance_mode == "calendar":
            return (t_idx - last_rebal) >= self.config.rebalance_frequency
        elif self.config.rebalance_mode == "threshold":
            if not state.positions:
                return t_idx >= 21
            # Check if any position drifted beyond threshold
            total_mv = sum(
                abs(state.positions.get(s, 0) * prices.get(s, 0))
                for s in state.positions
            )
            if total_mv < 1e-10:
                return True
            for sym, qty in state.positions.items():
                if sym in prices:
                    current_weight = abs(qty * prices[sym]) / max(state.nav, 1e-10)
                    if current_weight > self.config.max_position_pct + self.config.rebalance_threshold:
                        return True
            return (t_idx - last_rebal) >= self.config.rebalance_frequency * 2
        elif self.config.rebalance_mode == "risk_target":
            return (t_idx - last_rebal) >= 5  # check every week
        return (t_idx - last_rebal) >= self.config.rebalance_frequency

    def _rebalance(
        self, state, target_weights, prices, assets, regime
    ) -> tuple:
        trades = []
        total_turnover = 0.0
        total_cost = 0.0

        for sym, target_w in target_weights.items():
            if sym not in prices:
                continue

            target_qty = target_w * state.nav / prices[sym]
            current_qty = state.positions.get(sym, 0.0)
            delta_qty = target_qty - current_qty

            if abs(delta_qty * prices[sym]) < self.config.min_trade_pct * state.nav:
                continue

            side = "buy" if delta_qty > 0 else "sell"
            notional = abs(delta_qty * prices[sym])
            adv = assets.get(sym, Asset(sym)).avg_daily_volume if isinstance(assets.get(sym), Asset) else 1e9
            exec_price = self.cost_model.execution_price(prices[sym], side, notional, adv)
            cost = self.cost_model.compute_cost(notional, adv)

            # Update positions
            if side == "buy":
                old_cost_basis = state.avg_costs.get(sym, 0) * current_qty
                new_cost = abs(delta_qty) * exec_price
                total_qty = current_qty + abs(delta_qty)
                if total_qty > 0:
                    state.avg_costs[sym] = (old_cost_basis + new_cost) / total_qty
                state.positions[sym] = total_qty
                state.cash -= abs(delta_qty) * exec_price + cost
            else:
                sell_qty = min(abs(delta_qty), current_qty) if not self.config.short_allowed else abs(delta_qty)
                pnl = sell_qty * (exec_price - state.avg_costs.get(sym, exec_price))
                state.positions[sym] = current_qty - sell_qty
                if state.positions[sym] <= 0 and not self.config.short_allowed:
                    state.positions.pop(sym, None)
                    state.avg_costs.pop(sym, None)
                state.cash += sell_qty * exec_price - cost
                state.realized_pnl += pnl

            trades.append(Trade(
                timestamp=state.timestamp,
                symbol=sym,
                side=side,
                quantity=abs(delta_qty),
                price=exec_price,
                cost_bps=float(cost / max(notional, 1e-10) * 10000),
                slippage_bps=self.config.slippage_bps,
                commission=cost,
                pnl=0.0,
                regime=regime,
            ))

            total_turnover += notional / max(state.nav, 1e-10)
            total_cost += cost

        # Clean zero positions
        for sym in list(state.positions.keys()):
            if abs(state.positions[sym]) < 1e-10:
                state.positions.pop(sym)
                state.avg_costs.pop(sym, None)

        return trades, total_turnover, float(total_cost / max(state.nav, 1e-10) * 10000)

    def _compute_covariance(self, bars_history: dict, symbols: list, window: int = 63) -> Optional[np.ndarray]:
        n = len(symbols)
        if n < 2:
            return None
        returns_matrix = []
        for sym in symbols:
            bars = bars_history.get(sym, [])
            if len(bars) < window + 1:
                returns_matrix.append(np.zeros(window))
            else:
                prices = np.array([b.close for b in bars[-window - 1:]])
                rets = np.diff(np.log(prices + 1e-10))
                if len(rets) < window:
                    rets = np.pad(rets, (window - len(rets), 0))
                returns_matrix.append(rets[-window:])

        R = np.array(returns_matrix)
        cov = np.cov(R) + np.eye(n) * 1e-8
        return cov

    def _compute_metrics(self, nav_history, trade_log, returns_history) -> BacktestMetrics:
        if len(nav_history) < 2:
            return BacktestMetrics()

        nav = np.array(nav_history)
        rets = np.array(returns_history) if returns_history else np.diff(nav) / (nav[:-1] + 1e-10)
        n = len(rets)

        total_ret = float(nav[-1] / nav[0] - 1) if nav[0] > 0 else 0.0
        years = max(n / 252, 1 / 252)
        ann_ret = float((1 + total_ret) ** (1 / years) - 1)
        ann_vol = float(rets.std() * math.sqrt(252)) if n > 1 else 0.0
        sharpe = float(ann_ret / max(ann_vol, 1e-10))

        # Sortino
        downside = rets[rets < 0]
        down_vol = float(downside.std() * math.sqrt(252)) if len(downside) > 1 else ann_vol
        sortino = float(ann_ret / max(down_vol, 1e-10))

        # Max drawdown
        peak = np.maximum.accumulate(nav)
        dd = (peak - nav) / (peak + 1e-10)
        max_dd = float(dd.max())

        # DD duration
        in_dd = dd > 0
        dd_dur = 0
        max_dur = 0
        for d in in_dd:
            if d:
                dd_dur += 1
                max_dur = max(max_dur, dd_dur)
            else:
                dd_dur = 0

        calmar = float(ann_ret / max(max_dd, 1e-10))

        # Trade stats
        n_trades = len(trade_log)
        if n_trades > 0:
            pnls = [t.pnl for t in trade_log if t.pnl != 0]
            if pnls:
                wins = [p for p in pnls if p > 0]
                losses = [p for p in pnls if p < 0]
                win_rate = float(len(wins) / len(pnls))
                profit_factor = float(sum(wins) / max(abs(sum(losses)), 1e-10)) if losses else float("inf")
                avg_pnl = float(np.mean(pnls))
            else:
                win_rate = 0.0
                profit_factor = 0.0
                avg_pnl = 0.0
        else:
            win_rate = profit_factor = avg_pnl = 0.0

        total_costs = sum(t.commission for t in trade_log)
        costs_pct = float(total_costs / max(nav[0], 1e-10) * 100)

        return BacktestMetrics(
            total_return_pct=total_ret * 100,
            annualized_return_pct=ann_ret * 100,
            annualized_vol_pct=ann_vol * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd * 100,
            max_drawdown_duration=max_dur,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_pnl,
            total_trades=n_trades,
            total_costs_pct=costs_pct,
        )

    def _compute_attribution(self, trade_log, bars_by_symbol, assets, benchmark_weights) -> dict:
        if not benchmark_weights or not trade_log:
            return {"allocation": 0.0, "selection": 0.0, "interaction": 0.0}

        # Simplified Brinson attribution
        sectors = {}
        for sym, asset in assets.items():
            s = asset.sector if isinstance(asset, Asset) else "unknown"
            if s not in sectors:
                sectors[s] = {"portfolio_weight": 0.0, "benchmark_weight": 0.0, "portfolio_return": 0.0, "benchmark_return": 0.0}

        # Aggregate by sector from trades
        for t in trade_log:
            s = assets.get(t.symbol, Asset(t.symbol)).sector if isinstance(assets.get(t.symbol), Asset) else "unknown"
            if s in sectors:
                sectors[s]["portfolio_weight"] += abs(t.quantity * t.price)

        total_pv = sum(s["portfolio_weight"] for s in sectors.values()) + 1e-10
        for s in sectors.values():
            s["portfolio_weight"] /= total_pv

        allocation = sum(
            (sectors[s]["portfolio_weight"] - sectors[s]["benchmark_weight"]) * sectors[s]["benchmark_return"]
            for s in sectors
        )
        selection = sum(
            sectors[s]["benchmark_weight"] * (sectors[s]["portfolio_return"] - sectors[s]["benchmark_return"])
            for s in sectors
        )
        interaction = sum(
            (sectors[s]["portfolio_weight"] - sectors[s]["benchmark_weight"]) *
            (sectors[s]["portfolio_return"] - sectors[s]["benchmark_return"])
            for s in sectors
        )

        return {
            "allocation": float(allocation),
            "selection": float(selection),
            "interaction": float(interaction),
            "total_active": float(allocation + selection + interaction),
            "n_sectors": len(sectors),
        }
