"""
execution/oms/risk_guard.py
============================
Synchronous pre-trade risk checks.  Every check must pass before an order
is submitted to the broker.

Design principles
-----------------
- All checks are *pure functions* of the inputs — no network calls.
- Each check returns ``(passed: bool, reason: str)``.
- ``run_all_checks`` short-circuits on the first failure and logs it.
- Frequency tracking uses a sliding deque per symbol — no external state.

Constants mirror ``live_trader_alpaca.py`` so risk limits stay consistent:
  CRYPTO_CAP_FRAC = 0.40  (max single crypto position)
  BLOCKED_ENTRY_HOURS_UTC = {1, 13, 14, 15, 17, 18}

All rejections are written to the module logger at WARNING level.  The
AuditLog independently captures them via the OrderManager.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("execution.risk_guard")

# ---------------------------------------------------------------------------
# Risk parameter defaults (override via RiskGuard constructor kwargs)
# ---------------------------------------------------------------------------

CRYPTO_CAP_FRAC:       float = 0.40   # max single-instrument fraction
MAX_LEVERAGE:          float = 1.50   # total gross notional / equity
DAILY_LOSS_LIMIT_FRAC: float = 0.05   # halt if daily loss > 5 %
ORDER_FREQ_LIMIT:      int   = 10     # max orders/minute per symbol
FAT_FINGER_HIGH:       float = 2.00   # reject if price > 2x last_price
FAT_FINGER_LOW:        float = 0.50   # reject if price < 0.5x last_price
CORR_WARN_THRESHOLD:   float = 0.80   # warn if portfolio correlation exceeds this


# ---------------------------------------------------------------------------
# RiskGuard
# ---------------------------------------------------------------------------

class RiskGuard:
    """
    Pre-trade risk checker — synchronous, O(1) per check.

    Parameters
    ----------
    crypto_cap_frac : float
        Maximum allowed fraction in any single crypto instrument.
    max_leverage : float
        Maximum portfolio leverage (gross notional / equity).
    daily_loss_limit_frac : float
        Fraction of equity; halt trading if exceeded.
    order_freq_limit : int
        Max orders per symbol per 60-second sliding window.
    fat_finger_high : float
        Multiplier above which a price is considered erroneous.
    fat_finger_low : float
        Multiplier below which a price is considered erroneous.
    corr_warn_threshold : float
        Correlation level at which a warning is issued.
    blocked_hours_utc : set[int]
        UTC hours during which new entries are forbidden.
    """

    def __init__(
        self,
        crypto_cap_frac:       float    = CRYPTO_CAP_FRAC,
        max_leverage:          float    = MAX_LEVERAGE,
        daily_loss_limit_frac: float    = DAILY_LOSS_LIMIT_FRAC,
        order_freq_limit:      int      = ORDER_FREQ_LIMIT,
        fat_finger_high:       float    = FAT_FINGER_HIGH,
        fat_finger_low:        float    = FAT_FINGER_LOW,
        corr_warn_threshold:   float    = CORR_WARN_THRESHOLD,
        blocked_hours_utc:     set[int] | None = None,
    ) -> None:
        self._crypto_cap_frac       = crypto_cap_frac
        self._max_leverage          = max_leverage
        self._daily_loss_limit_frac = daily_loss_limit_frac
        self._order_freq_limit      = order_freq_limit
        self._fat_finger_high       = fat_finger_high
        self._fat_finger_low        = fat_finger_low
        self._corr_warn_threshold   = corr_warn_threshold
        self._blocked_hours         = blocked_hours_utc or {1, 13, 14, 15, 17, 18}

        # sliding window: symbol -> deque of timestamps (float epoch seconds)
        self._order_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self._lock = threading.RLock()

        # daily reference equity (set at session start)
        self._day_start_equity: float = 0.0

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset_daily_counters(self, equity: float) -> None:
        """Call once per trading day to reset the daily loss baseline."""
        with self._lock:
            self._day_start_equity = equity
            log.info("RiskGuard: daily counters reset, equity=%.2f", equity)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_position_limit(
        self,
        symbol: str,
        new_frac: float,
    ) -> tuple[bool, str]:
        """
        Reject if the requested fraction exceeds CRYPTO_CAP_FRAC.

        We apply the cap to all instruments (not just crypto) as a catch-all
        concentration guard.
        """
        if new_frac > self._crypto_cap_frac:
            reason = (
                f"Position limit: {symbol} requested {new_frac:.2%} > "
                f"cap {self._crypto_cap_frac:.2%}"
            )
            log.warning("RISK_REJECT %s", reason)
            return False, reason
        return True, ""

    def check_portfolio_leverage(
        self,
        new_exposure: float,
        current_gross: float,
        equity: float,
    ) -> tuple[bool, str]:
        """
        Reject if adding *new_exposure* pushes total leverage above MAX_LEVERAGE.

        Parameters
        ----------
        new_exposure : float
            Notional USD of the proposed new position.
        current_gross : float
            Current total gross notional before this order.
        equity : float
            Current portfolio equity.
        """
        if equity <= 0:
            return True, ""
        projected_leverage = (current_gross + new_exposure) / equity
        if projected_leverage > self._max_leverage:
            reason = (
                f"Leverage: projected {projected_leverage:.2f}x > "
                f"limit {self._max_leverage:.2f}x"
            )
            log.warning("RISK_REJECT %s", reason)
            return False, reason
        return True, ""

    def check_daily_loss_limit(
        self,
        current_equity: float,
    ) -> tuple[bool, str]:
        """
        Halt trading if daily drawdown exceeds the configured limit.

        Requires ``reset_daily_counters`` to have been called with the
        session-start equity.
        """
        baseline = self._day_start_equity
        if baseline <= 0:
            return True, ""
        daily_pnl_frac = (current_equity - baseline) / baseline
        if daily_pnl_frac < -self._daily_loss_limit_frac:
            reason = (
                f"Daily loss limit: {daily_pnl_frac:.2%} < "
                f"-{self._daily_loss_limit_frac:.2%}"
            )
            log.warning("RISK_REJECT %s", reason)
            return False, reason
        return True, ""

    def check_order_frequency(self, symbol: str) -> tuple[bool, str]:
        """
        Reject if more than ``order_freq_limit`` orders have been submitted
        for *symbol* in the last 60 seconds.
        """
        with self._lock:
            now = datetime.now(timezone.utc).timestamp()
            dq  = self._order_times[symbol]
            # Prune entries older than 60 s
            while dq and dq[0] < now - 60.0:
                dq.popleft()
            if len(dq) >= self._order_freq_limit:
                reason = (
                    f"Order frequency: {symbol} has {len(dq)} orders in last 60s "
                    f"(limit={self._order_freq_limit})"
                )
                log.warning("RISK_REJECT %s", reason)
                return False, reason
            dq.append(now)
            return True, ""

    def check_fat_finger(
        self,
        price: float,
        symbol: str,
        last_price: float,
    ) -> tuple[bool, str]:
        """
        Reject orders where the submitted price deviates wildly from the
        last known price (fat-finger protection).

        Skips the check if *last_price* is zero (unknown).
        """
        if last_price <= 0 or price <= 0:
            return True, ""
        ratio = price / last_price
        if ratio > self._fat_finger_high or ratio < self._fat_finger_low:
            reason = (
                f"Fat finger: {symbol} price {price:.4f} vs last {last_price:.4f} "
                f"ratio={ratio:.3f} (allowed [{self._fat_finger_low}, "
                f"{self._fat_finger_high}])"
            )
            log.warning("RISK_REJECT %s", reason)
            return False, reason
        return True, ""

    def check_correlation_risk(
        self,
        new_symbol: str,
        new_frac: float,
        current_positions: dict,
    ) -> tuple[bool, str]:
        """
        Warn (but do not reject) if the portfolio already has high
        crypto concentration and the new position would add to it.

        Current implementation uses a simple heuristic: count the number
        of positions with fraction > 10 % as a correlation proxy.  A full
        correlation matrix would require historical price data not available
        in this synchronous check; that lives in the TCA layer.

        Returns (False, reason) only when concentration is extreme.
        """
        n_large = sum(
            1 for sym, pos in current_positions.items()
            if sym != new_symbol and abs(getattr(pos, "quantity", 0)) > 0
        )
        # Rough proxy: if we have ≥ 5 positions each at ~cap, warn
        if n_large >= 5 and new_frac > 0.10:
            reason = (
                f"Correlation risk: {new_symbol} new_frac={new_frac:.2%} with "
                f"{n_large} existing positions — portfolio may be over-correlated"
            )
            log.warning("RISK_WARN %s", reason)
            # Warning only — do not block
        return True, ""

    def check_blocked_hours(self) -> tuple[bool, str]:
        """
        Reject orders submitted during BLOCKED_ENTRY_HOURS_UTC.

        Mirrors ``live_trader_alpaca.py`` logic so that live and OMS paths
        enforce the same entry-hour restriction.
        """
        hour = datetime.now(timezone.utc).hour
        if hour in self._blocked_hours:
            reason = f"Blocked hour: UTC hour {hour} is in blocked set {sorted(self._blocked_hours)}"
            log.warning("RISK_REJECT %s", reason)
            return False, reason
        return True, ""

    # ------------------------------------------------------------------
    # Composite check
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        symbol: str,
        new_frac: float,
        price: float,
        order,                         # Order object
        equity: float,
        positions: dict,
        last_price: Optional[float] = None,
        current_gross: float = 0.0,
        enforce_hour_block: bool = True,
    ) -> tuple[bool, str]:
        """
        Run all pre-trade checks in priority order.  Short-circuits on first
        failure.

        Parameters
        ----------
        symbol : str
            Instrument being ordered.
        new_frac : float
            Target fraction of equity.
        price : float
            Order price (or current mid for market orders).
        order : Order
            The order being evaluated (used for metadata).
        equity : float
            Current portfolio equity in USD.
        positions : dict
            Current PositionTracker.positions dict.
        last_price : float | None
            Last known price for fat-finger check.
        current_gross : float
            Current total gross notional (for leverage check).
        enforce_hour_block : bool
            If True, also apply blocked-hours check.

        Returns
        -------
        (True, "") on pass, or (False, reason) on first failure.
        """
        checks = [
            lambda: self.check_position_limit(symbol, new_frac),
            lambda: self.check_portfolio_leverage(new_frac * equity, current_gross, equity),
            lambda: self.check_daily_loss_limit(equity),
            lambda: self.check_order_frequency(symbol),
            lambda: self.check_fat_finger(price, symbol, last_price or price),
            lambda: self.check_correlation_risk(symbol, new_frac, positions),
        ]
        if enforce_hour_block:
            checks.append(self.check_blocked_hours)

        for check in checks:
            passed, reason = check()
            if not passed:
                return False, reason

        return True, ""
