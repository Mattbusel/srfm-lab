"""
shadow_portfolio.py
===================
Shadow (paper) portfolio tracking for the idea-engine.

Run strategies in simulation alongside live trading, compare performance,
and promote shadow strategies when they demonstrate edge.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FillModel(Enum):
    IMMEDIATE = "immediate"
    LATENCY = "latency"
    PARTIAL = "partial"
    REALISTIC = "realistic"


class PromotionStatus(Enum):
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    DEMOTED = "demoted"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShadowOrder:
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    symbol: str = ""
    side: str = "buy"
    quantity: float = 0.0
    limit_price: Optional[float] = None
    order_time: datetime = field(default_factory=datetime.utcnow)
    strategy_id: str = ""


@dataclass
class ShadowFill:
    order_id: str = ""
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fill_time: datetime = field(default_factory=datetime.utcnow)
    slippage: float = 0.0
    latency_ms: float = 0.0
    commission: float = 0.0


@dataclass
class ShadowPosition:
    symbol: str = ""
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_mark_price: float = 0.0
    strategy_id: str = ""

    def mark_to_market(self, price: float) -> float:
        self.last_mark_price = price
        self.unrealized_pnl = (price - self.avg_entry_price) * self.quantity
        return self.unrealized_pnl

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.avg_entry_price)


@dataclass
class ShadowSnapshot:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    equity: float = 0.0
    positions: Dict[str, ShadowPosition] = field(default_factory=dict)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    n_trades: int = 0


@dataclass
class DivergenceMetrics:
    return_divergence: float = 0.0
    correlation: float = 0.0
    tracking_error: float = 0.0
    max_divergence: float = 0.0
    mean_abs_divergence: float = 0.0


# ---------------------------------------------------------------------------
# Simulated execution
# ---------------------------------------------------------------------------

class SimulatedExecution:
    """Model fills with slippage, latency, and partial fills."""

    def __init__(
        self,
        slippage_bps: float = 5.0,
        latency_ms: float = 50.0,
        fill_rate: float = 0.95,
        commission_bps: float = 2.0,
        partial_fill_prob: float = 0.1,
        rng_seed: int = 42,
    ):
        self.slippage_bps = slippage_bps
        self.latency_ms = latency_ms
        self.fill_rate = fill_rate
        self.commission_bps = commission_bps
        self.partial_fill_prob = partial_fill_prob
        self.rng = np.random.default_rng(rng_seed)

    def execute(self, order: ShadowOrder, market_price: float) -> List[ShadowFill]:
        """Simulate execution of an order. May return 0, 1, or 2 fills (partial)."""
        if self.rng.random() > self.fill_rate:
            logger.debug("Order %s not filled (fill rate miss)", order.order_id)
            return []

        # Slippage
        slip_frac = self.slippage_bps / 10_000.0
        slip_sign = 1.0 if order.side == "buy" else -1.0
        slip = slip_sign * slip_frac * market_price * (0.5 + self.rng.random())
        fill_price = market_price + slip

        # Latency jitter
        latency = self.latency_ms * (0.5 + self.rng.random())

        # Commission
        commission = self.commission_bps / 10_000.0 * abs(order.quantity * fill_price)

        # Partial fill?
        if self.rng.random() < self.partial_fill_prob and order.quantity > 1:
            frac = 0.3 + self.rng.random() * 0.5
            q1 = order.quantity * frac
            q2 = order.quantity - q1
            fill1 = ShadowFill(
                order_id=order.order_id, fill_price=fill_price,
                fill_quantity=q1, slippage=abs(slip) * q1,
                latency_ms=latency, commission=commission * frac,
            )
            slip2 = slip_sign * slip_frac * market_price * (0.5 + self.rng.random())
            fill2 = ShadowFill(
                order_id=order.order_id, fill_price=market_price + slip2,
                fill_quantity=q2, slippage=abs(slip2) * q2,
                latency_ms=latency + 20 + self.rng.random() * 30,
                commission=commission * (1 - frac),
            )
            return [fill1, fill2]

        fill = ShadowFill(
            order_id=order.order_id, fill_price=fill_price,
            fill_quantity=order.quantity, slippage=abs(slip) * order.quantity,
            latency_ms=latency, commission=commission,
        )
        return [fill]


# ---------------------------------------------------------------------------
# Shadow portfolio
# ---------------------------------------------------------------------------

class ShadowPortfolio:
    """Track positions and P&L without real execution."""

    def __init__(
        self,
        portfolio_id: str = "",
        strategy_id: str = "",
        initial_capital: float = 100_000.0,
        execution: Optional[SimulatedExecution] = None,
    ):
        self.portfolio_id = portfolio_id or uuid.uuid4().hex[:8]
        self.strategy_id = strategy_id
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, ShadowPosition] = {}
        self.execution = execution or SimulatedExecution()
        self.fills: List[ShadowFill] = []
        self.snapshots: List[ShadowSnapshot] = []
        self.n_trades = 0

    def submit_order(self, order: ShadowOrder, market_price: float) -> List[ShadowFill]:
        fills = self.execution.execute(order, market_price)
        for fill in fills:
            self._apply_fill(order, fill)
            self.fills.append(fill)
        return fills

    def _apply_fill(self, order: ShadowOrder, fill: ShadowFill) -> None:
        sym = order.symbol
        qty = fill.fill_quantity if order.side == "buy" else -fill.fill_quantity
        cost = fill.fill_price * abs(fill.fill_quantity) * (1 if order.side == "buy" else -1)
        self.cash -= cost + fill.commission

        if sym not in self.positions:
            self.positions[sym] = ShadowPosition(
                symbol=sym, quantity=0, avg_entry_price=0, strategy_id=self.strategy_id
            )
        pos = self.positions[sym]
        old_qty = pos.quantity
        new_qty = old_qty + qty

        if abs(new_qty) > abs(old_qty):
            # Adding to position
            total_cost = pos.avg_entry_price * abs(old_qty) + fill.fill_price * abs(qty)
            pos.avg_entry_price = total_cost / abs(new_qty) if abs(new_qty) > 0 else 0
        elif new_qty * old_qty < 0 or abs(new_qty) < 1e-10:
            # Closing / flipping
            closed_qty = min(abs(old_qty), abs(qty))
            pnl = (fill.fill_price - pos.avg_entry_price) * closed_qty * (1 if old_qty > 0 else -1)
            pos.realized_pnl += pnl
            if abs(new_qty) > 1e-10:
                pos.avg_entry_price = fill.fill_price

        pos.quantity = new_qty
        self.n_trades += 1

        if abs(pos.quantity) < 1e-12:
            del self.positions[sym]

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        total_unrealized = 0.0
        for sym, pos in self.positions.items():
            if sym in prices:
                total_unrealized += pos.mark_to_market(prices[sym])
        return total_unrealized

    @property
    def equity(self) -> float:
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + unrealized

    @property
    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values()) + (self.cash - self.initial_capital)

    def snapshot(self) -> ShadowSnapshot:
        snap = ShadowSnapshot(
            equity=self.equity,
            positions=dict(self.positions),
            realized_pnl=self.total_realized_pnl,
            unrealized_pnl=sum(p.unrealized_pnl for p in self.positions.values()),
            n_trades=self.n_trades,
        )
        self.snapshots.append(snap)
        return snap

    def equity_curve(self) -> List[float]:
        return [s.equity for s in self.snapshots]

    def drawdown_series(self) -> List[float]:
        eq = self.equity_curve()
        if not eq:
            return []
        arr = np.array(eq)
        peak = np.maximum.accumulate(arr)
        return list(arr - peak)

    def max_drawdown(self) -> float:
        dd = self.drawdown_series()
        return float(min(dd)) if dd else 0.0

    def sharpe_ratio(self) -> float:
        eq = self.equity_curve()
        if len(eq) < 3:
            return 0.0
        returns = np.diff(eq) / np.array(eq[:-1])
        s = float(np.std(returns, ddof=1))
        return float(np.mean(returns)) / s if s > 1e-10 else 0.0

    def var_95(self) -> float:
        eq = self.equity_curve()
        if len(eq) < 10:
            return 0.0
        returns = np.diff(eq) / np.array(eq[:-1])
        return float(np.percentile(returns, 5))

    def reset(self) -> None:
        self.cash = self.initial_capital
        self.positions.clear()
        self.fills.clear()
        self.snapshots.clear()
        self.n_trades = 0


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

class ComparisonEngine:
    """Compare shadow vs live portfolio."""

    @staticmethod
    def divergence(shadow_equity: List[float], live_equity: List[float]) -> DivergenceMetrics:
        n = min(len(shadow_equity), len(live_equity))
        if n < 2:
            return DivergenceMetrics()
        s = np.array(shadow_equity[:n])
        l = np.array(live_equity[:n])
        s_ret = np.diff(s) / s[:-1]
        l_ret = np.diff(l) / l[:-1]
        diff = s_ret - l_ret
        corr_val = 0.0
        sm = s_ret - np.mean(s_ret)
        lm = l_ret - np.mean(l_ret)
        d = np.sqrt(np.sum(sm ** 2) * np.sum(lm ** 2))
        if d > 1e-15:
            corr_val = float(np.sum(sm * lm) / d)
        return DivergenceMetrics(
            return_divergence=float(np.mean(diff)),
            correlation=corr_val,
            tracking_error=float(np.std(diff, ddof=1)),
            max_divergence=float(np.max(np.abs(diff))),
            mean_abs_divergence=float(np.mean(np.abs(diff))),
        )

    @staticmethod
    def ab_test(
        shadow_a: ShadowPortfolio,
        shadow_b: ShadowPortfolio,
        min_observations: int = 30,
    ) -> Dict[str, Any]:
        eq_a = shadow_a.equity_curve()
        eq_b = shadow_b.equity_curve()
        n = min(len(eq_a), len(eq_b))
        if n < min_observations:
            return {"conclusive": False, "reason": "insufficient_data", "n": n}
        ret_a = np.diff(eq_a[:n]) / np.array(eq_a[:n - 1])
        ret_b = np.diff(eq_b[:n]) / np.array(eq_b[:n - 1])
        diff = ret_a - ret_b
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))
        if std_diff < 1e-15:
            return {"conclusive": False, "reason": "zero_variance"}
        t_stat = mean_diff / (std_diff / math.sqrt(len(diff)))
        # Approximate p from t
        p_approx = math.exp(-0.5 * t_stat ** 2) * 2 if abs(t_stat) < 10 else 0.0
        winner = shadow_a.portfolio_id if mean_diff > 0 else shadow_b.portfolio_id
        return {
            "conclusive": abs(t_stat) > 1.96,
            "t_stat": t_stat,
            "p_approx": p_approx,
            "mean_diff": mean_diff,
            "winner": winner,
            "sharpe_a": shadow_a.sharpe_ratio(),
            "sharpe_b": shadow_b.sharpe_ratio(),
        }


# ---------------------------------------------------------------------------
# Promotion criteria
# ---------------------------------------------------------------------------

class PromotionEvaluator:
    """Decide when a shadow strategy should be promoted to live."""

    def __init__(
        self,
        min_trades: int = 50,
        min_sharpe: float = 0.5,
        max_drawdown_pct: float = -0.15,
        min_win_rate: float = 0.45,
        min_profit_factor: float = 1.2,
        outperformance_threshold: float = 0.001,
    ):
        self.min_trades = min_trades
        self.min_sharpe = min_sharpe
        self.max_drawdown_pct = max_drawdown_pct
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.outperformance_threshold = outperformance_threshold

    def evaluate(
        self,
        shadow: ShadowPortfolio,
        live_equity: Optional[List[float]] = None,
    ) -> Tuple[PromotionStatus, Dict[str, Any]]:
        reasons: Dict[str, Any] = {}

        if shadow.n_trades < self.min_trades:
            reasons["insufficient_trades"] = shadow.n_trades
            return PromotionStatus.SHADOW, reasons

        sharpe = shadow.sharpe_ratio()
        reasons["sharpe"] = sharpe
        if sharpe < self.min_sharpe:
            reasons["fail_sharpe"] = True
            return PromotionStatus.SHADOW, reasons

        dd = shadow.max_drawdown()
        eq = shadow.equity_curve()
        peak = max(eq) if eq else shadow.initial_capital
        dd_pct = dd / abs(peak) if abs(peak) > 0 else 0
        reasons["max_dd_pct"] = dd_pct
        if dd_pct < self.max_drawdown_pct:
            reasons["fail_drawdown"] = True
            return PromotionStatus.DEMOTED, reasons

        # Win rate from fills
        wins = sum(1 for s in shadow.snapshots if s.realized_pnl > 0)
        total = len(shadow.snapshots)
        wr = wins / total if total > 0 else 0
        reasons["win_rate"] = wr

        # Outperformance vs live
        if live_equity and len(live_equity) > 10:
            div = ComparisonEngine.divergence(shadow.equity_curve(), live_equity)
            reasons["outperformance"] = div.return_divergence
            if div.return_divergence > self.outperformance_threshold:
                reasons["outperforms_live"] = True
                return PromotionStatus.CANDIDATE, reasons

        if sharpe >= self.min_sharpe:
            return PromotionStatus.CANDIDATE, reasons

        return PromotionStatus.SHADOW, reasons


# ---------------------------------------------------------------------------
# Historical shadow replay
# ---------------------------------------------------------------------------

class HistoricalShadowReplay:
    """Reconstruct what a shadow portfolio would have done historically."""

    def __init__(self, execution: Optional[SimulatedExecution] = None):
        self.execution = execution or SimulatedExecution()

    def replay(
        self,
        signals: List[Dict[str, Any]],
        prices: Dict[str, List[float]],
        initial_capital: float = 100_000.0,
    ) -> ShadowPortfolio:
        """
        signals: list of dicts with keys {timestamp, symbol, side, quantity}
        prices: {symbol: [price_t0, price_t1, ...]}
        """
        portfolio = ShadowPortfolio(
            strategy_id="replay", initial_capital=initial_capital, execution=self.execution
        )
        price_idx: Dict[str, int] = defaultdict(int)

        for sig in signals:
            sym = sig["symbol"]
            idx = price_idx.get(sym, 0)
            price_list = prices.get(sym, [])
            if idx >= len(price_list):
                continue
            mkt_price = price_list[idx]
            order = ShadowOrder(
                symbol=sym, side=sig.get("side", "buy"),
                quantity=sig.get("quantity", 1.0), strategy_id="replay",
            )
            portfolio.submit_order(order, mkt_price)
            price_idx[sym] = idx + 1

            # Mark to market
            current_prices = {}
            for s, pl in prices.items():
                pi = min(price_idx.get(s, 0), len(pl) - 1)
                current_prices[s] = pl[pi]
            portfolio.mark_to_market(current_prices)
            portfolio.snapshot()

        return portfolio


# ---------------------------------------------------------------------------
# Multi-shadow runner
# ---------------------------------------------------------------------------

class MultiShadowRunner:
    """Run N strategies in parallel as shadow portfolios."""

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        self.shadows: Dict[str, ShadowPortfolio] = {}
        self.promotion_eval = PromotionEvaluator()

    def add_shadow(
        self,
        strategy_id: str,
        execution: Optional[SimulatedExecution] = None,
    ) -> ShadowPortfolio:
        sp = ShadowPortfolio(
            strategy_id=strategy_id,
            initial_capital=self.initial_capital,
            execution=execution or SimulatedExecution(),
        )
        self.shadows[strategy_id] = sp
        return sp

    def remove_shadow(self, strategy_id: str) -> None:
        self.shadows.pop(strategy_id, None)

    def submit_order(
        self, strategy_id: str, order: ShadowOrder, market_price: float
    ) -> List[ShadowFill]:
        sp = self.shadows.get(strategy_id)
        if not sp:
            logger.warning("No shadow for strategy %s", strategy_id)
            return []
        return sp.submit_order(order, market_price)

    def mark_all(self, prices: Dict[str, float]) -> Dict[str, float]:
        equities: Dict[str, float] = {}
        for sid, sp in self.shadows.items():
            sp.mark_to_market(prices)
            sp.snapshot()
            equities[sid] = sp.equity
        return equities

    def rank_by_sharpe(self) -> List[Tuple[str, float]]:
        ranking = [(sid, sp.sharpe_ratio()) for sid, sp in self.shadows.items()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def rank_by_pnl(self) -> List[Tuple[str, float]]:
        ranking = [(sid, sp.equity - sp.initial_capital) for sid, sp in self.shadows.items()]
        return sorted(ranking, key=lambda x: x[1], reverse=True)

    def evaluate_promotions(
        self, live_equity: Optional[List[float]] = None
    ) -> Dict[str, Tuple[PromotionStatus, Dict[str, Any]]]:
        results: Dict[str, Tuple[PromotionStatus, Dict[str, Any]]] = {}
        for sid, sp in self.shadows.items():
            status, reasons = self.promotion_eval.evaluate(sp, live_equity)
            results[sid] = (status, reasons)
            if status == PromotionStatus.CANDIDATE:
                logger.info("Strategy %s is a CANDIDATE for promotion: %s", sid, reasons)
        return results

    def ab_test(self, strategy_a: str, strategy_b: str) -> Dict[str, Any]:
        sp_a = self.shadows.get(strategy_a)
        sp_b = self.shadows.get(strategy_b)
        if not sp_a or not sp_b:
            return {"error": "strategy not found"}
        return ComparisonEngine.ab_test(sp_a, sp_b)

    def summary(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for sid, sp in self.shadows.items():
            out[sid] = {
                "equity": sp.equity,
                "pnl": sp.equity - sp.initial_capital,
                "n_trades": sp.n_trades,
                "sharpe": sp.sharpe_ratio(),
                "max_dd": sp.max_drawdown(),
                "var_95": sp.var_95(),
                "n_positions": len(sp.positions),
            }
        return out

    def reset_all(self) -> None:
        for sp in self.shadows.values():
            sp.reset()
