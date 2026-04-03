"""
broker.py — Lightweight SimulatedBroker for arena/tournament backtests.

No LEAN, no Docker, no QC. Pure Python + numpy.
Designed to run thousands of backtests per minute on synthetic or real price bars.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional


class SimulatedBroker:
    """
    Minimal execution model for synthetic backtesting.

    Position is stored as a fraction of current equity (-1.0 to +1.0).
    A set_position() call incurs a flat fee and slippage.

    Usage:
        broker = SimulatedBroker(cash=1_000_000)
        for bar_return in returns:
            broker.update(bar_return)
            new_pos = strategy_signal(...)
            broker.set_position(new_pos)
        stats = broker.stats()
    """

    def __init__(
        self,
        cash:              float = 1_000_000.0,
        fee_per_trade:     float = 50.0,
        slippage_pct:      float = 0.0002,   # 2bps per trade
        max_leverage:      float = 1.0,
        margin_buffer:     float = 0.10,     # 10% margin buffer before forced unwind
    ):
        self.initial_equity  = cash
        self.equity          = cash
        self.fee             = fee_per_trade
        self.slippage        = slippage_pct
        self.max_lev         = max_leverage
        self.margin_buffer   = margin_buffer

        self.position:       float = 0.0    # current position fraction
        self.trade_count:    int   = 0

        self.equity_curve:   List[float] = [cash]
        self.trade_log:      List[dict]  = []
        self._peak_equity:   float = cash
        self._max_dd:        float = 0.0

    # ------------------------------------------------------------------
    def update(self, bar_return: float):
        """Apply one bar's return to current position. Call before set_position."""
        pnl = self.equity * self.position * bar_return
        self.equity += pnl
        self.equity = max(self.equity, 1.0)   # prevent ruin

        if self.equity > self._peak_equity:
            self._peak_equity = self.equity
        dd = (self._peak_equity - self.equity) / self._peak_equity
        if dd > self._max_dd:
            self._max_dd = dd

        self.equity_curve.append(self.equity)

    def set_position(self, target: float):
        """
        Set target position (fraction of equity, clipped to ±max_lev).
        Incurs fee + slippage if the position changes meaningfully.
        """
        target = float(np.clip(target, -self.max_lev, self.max_lev))
        delta  = abs(target - self.position)
        if delta > 0.02:
            cost = self.fee + self.equity * delta * self.slippage
            self.equity -= cost
            self.trade_log.append({
                "from": self.position, "to": target, "cost": cost, "equity": self.equity
            })
            self.trade_count += 1
            self.position = target

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        curve = np.array(self.equity_curve)
        total_return = (curve[-1] / curve[0] - 1) * 100.0
        rets = np.diff(curve) / curve[:-1]
        sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(2080)  # annualised hourly
        return {
            "total_return_pct":  round(total_return, 2),
            "max_drawdown_pct":  round(self._max_dd * 100, 2),
            "sharpe":            round(float(sharpe), 3),
            "trade_count":       self.trade_count,
            "final_equity":      round(self.equity, 2),
            "win_rate":          self._win_rate(),
        }

    def _win_rate(self) -> float:
        if len(self.trade_log) < 2:
            return 0.0
        equities = [t["equity"] for t in self.trade_log]
        wins = sum(1 for i in range(1, len(equities)) if equities[i] > equities[i-1])
        return round(wins / (len(equities) - 1), 3)

    def reset(self, cash: Optional[float] = None):
        c = cash or self.initial_equity
        self.__init__(
            cash=c,
            fee_per_trade=self.fee,
            slippage_pct=self.slippage,
            max_leverage=self.max_lev,
            margin_buffer=self.margin_buffer,
        )
