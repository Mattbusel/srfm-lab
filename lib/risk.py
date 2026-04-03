"""
risk.py — Portfolio risk management for SRFM strategies.

Handles position sizing, stop-loss management, circuit breakers,
and daily/drawdown loss limits.  Designed to plug into any LEAN algorithm.
"""

from __future__ import annotations
from collections import deque
from typing import Optional


class RiskManager:
    """
    Centralised risk management.  Call check() before every entry;
    it returns False if any limit is breached.

    Parameters
    ----------
    max_daily_loss_pct : float
        Maximum allowed daily P&L loss as a fraction of portfolio value.
    max_drawdown_pct : float
        Maximum drawdown from equity peak before circuit breaker triggers.
    stop_loss_pct : float
        Per-trade stop loss as a fraction of entry price.
    take_profit_pct : float
        Per-trade take profit as a fraction of entry price.
    max_position_fraction : float
        Maximum fraction of portfolio in any single position.
    cooldown_bars : int
        Bars to wait after a stop-loss hit before re-entering.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.02,
        max_drawdown_pct: float = 0.10,
        stop_loss_pct: float = 0.008,
        take_profit_pct: float = 0.025,
        max_position_fraction: float = 0.20,
        cooldown_bars: int = 5,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_fraction = max_position_fraction
        self.cooldown_bars = cooldown_bars

        self._peak_equity: float = 0.0
        self._day_start_equity: float = 0.0
        self._current_equity: float = 0.0
        self._cooldown_remaining: int = 0
        self._circuit_broken: bool = False

        self._daily_pnl_history: deque = deque(maxlen=252)

        # Per-trade tracking
        self.entry_price: Optional[float] = None
        self.position_direction: int = 0  # +1 long, -1 short, 0 flat

    # ------------------------------------------------------------------
    def update_equity(self, equity: float):
        if self._peak_equity == 0.0:
            self._peak_equity = equity
            self._day_start_equity = equity
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

    def start_new_day(self, equity: float):
        if self._day_start_equity > 0:
            self._daily_pnl_history.append(
                (equity - self._day_start_equity) / self._day_start_equity
            )
        self._day_start_equity = equity

    # ------------------------------------------------------------------
    def check(self, equity: float) -> bool:
        """
        Returns True if it is safe to enter a new position.
        Checks circuit breaker, drawdown, daily loss, and cooldown.
        """
        self.update_equity(equity)

        if self._circuit_broken:
            return False

        if self._cooldown_remaining > 0:
            return False

        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        if drawdown >= self.max_drawdown_pct:
            self._circuit_broken = True
            return False

        daily_loss = (self._day_start_equity - equity) / self._day_start_equity if self._day_start_equity > 0 else 0.0
        if daily_loss >= self.max_daily_loss_pct:
            return False

        return True

    def reset_circuit_breaker(self):
        """Manual override — use with caution."""
        self._circuit_broken = False

    # ------------------------------------------------------------------
    def on_entry(self, entry_price: float, direction: int):
        self.entry_price = entry_price
        self.position_direction = direction

    def on_exit(self, hit_stop: bool = False):
        if hit_stop:
            self._cooldown_remaining = self.cooldown_bars
        self.entry_price = None
        self.position_direction = 0

    # ------------------------------------------------------------------
    def stop_price(self) -> Optional[float]:
        if self.entry_price is None:
            return None
        offset = self.entry_price * self.stop_loss_pct
        return self.entry_price - offset * self.position_direction

    def target_price(self) -> Optional[float]:
        if self.entry_price is None:
            return None
        offset = self.entry_price * self.take_profit_pct
        return self.entry_price + offset * self.position_direction

    def should_stop(self, current_price: float) -> bool:
        stop = self.stop_price()
        if stop is None:
            return False
        if self.position_direction == 1:
            return current_price <= stop
        if self.position_direction == -1:
            return current_price >= stop
        return False

    def should_take_profit(self, current_price: float) -> bool:
        target = self.target_price()
        if target is None:
            return False
        if self.position_direction == 1:
            return current_price >= target
        if self.position_direction == -1:
            return current_price <= target
        return False

    # ------------------------------------------------------------------
    def position_size(
        self,
        equity: float,
        price: float,
        contract_value: float = 1.0,
        hawking_scalar: float = 1.0,
    ) -> int:
        """
        Compute number of contracts to trade.

        hawking_scalar: 0.0–1.0 from HawkingMonitor.size_scalar()
        """
        max_dollar_risk = equity * self.max_position_fraction * hawking_scalar
        if price <= 0 or contract_value <= 0:
            return 0
        return max(0, int(max_dollar_risk / (price * contract_value)))

    # ------------------------------------------------------------------
    @property
    def drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity

    @property
    def daily_pnl(self) -> float:
        if self._day_start_equity <= 0:
            return 0.0
        return (self._current_equity - self._day_start_equity) / self._day_start_equity

    @property
    def is_circuit_broken(self) -> bool:
        return self._circuit_broken
