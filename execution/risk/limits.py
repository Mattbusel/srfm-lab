"""
execution/risk/limits.py
========================
Risk limit enforcement for the SRFM Lab trading system.

Limits are loaded from config/risk_limits.yaml and checked against the
current portfolio state on each update cycle. A proposed trade is tested
before submission; its size may be reduced or blocked entirely.

DrawdownGuard integrates with the LiveTrader circuit-breaker pattern: it
calls back into the CircuitBreaker when equity falls through the halt
threshold and re-enables entries when equity recovers.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request as _urllib_request

import numpy as np
import yaml

log = logging.getLogger("execution.risk.limits")

_CONFIG_PATH = Path(__file__).parents[2] / "config" / "risk_limits.yaml"
_DB_PATH = Path(__file__).parents[2] / "execution" / "live_trades.db"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LimitType(str, Enum):
    VAR          = "VaR"
    NOTIONAL     = "notional"
    CONCENTRATION = "concentration"
    DRAWDOWN     = "drawdown"
    DAILY_LOSS   = "daily_loss"
    GROSS_EXPOSURE = "gross_exposure"


class LimitAction(str, Enum):
    WARN   = "WARN"    # log warning only
    REDUCE = "REDUCE"  # reduce position size
    HALT   = "HALT"    # block new entries entirely


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RiskLimit:
    """A single named risk limit with its current state."""
    name: str
    limit_type: LimitType
    threshold: float            # the numeric threshold
    action: LimitAction
    current_value: float = 0.0  # latest measured value
    is_breached: bool = False

    @property
    def severity(self) -> str:
        if not self.is_breached:
            return "OK"
        return self.action.value

    def check(self, value: float) -> "RiskLimit":
        """Return a copy of self with updated current_value and is_breached flag."""
        import copy
        updated = copy.copy(self)
        updated.current_value = value
        updated.is_breached = value > self.threshold
        return updated


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

class RiskLimitConfig:
    """
    Loads risk limits from config/risk_limits.yaml and exposes them as
    RiskLimit objects.

    Per-instrument notional caps and the portfolio-level limits from
    risk_limits.yaml are both surfaced here.
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        self.config_path = config_path
        self._raw: Dict = {}
        self._limits: List[RiskLimit] = []
        self.load()

    def load(self) -> None:
        """(Re-)read config file and rebuild limit list."""
        if not self.config_path.exists():
            log.warning("risk_limits.yaml not found at %s; using defaults", self.config_path)
            self._raw = {}
        else:
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    self._raw = yaml.safe_load(fh) or {}
            except Exception as exc:
                log.error("Failed to load risk_limits.yaml: %s", exc)
                self._raw = {}
        self._limits = self._build_limits()

    def _build_limits(self) -> List[RiskLimit]:
        limits: List[RiskLimit] = []
        port = self._raw.get("portfolio", {})

        # Drawdown limit (maps to DD_HALT_PCT = 10% in live_trader)
        max_dd = port.get("max_drawdown_pct", 0.15)
        limits.append(RiskLimit(
            name="portfolio_max_drawdown",
            limit_type=LimitType.DRAWDOWN,
            threshold=max_dd,
            action=LimitAction.HALT,
        ))

        # Daily loss limit
        max_daily_loss = port.get("max_daily_loss_pct", 0.02)
        limits.append(RiskLimit(
            name="portfolio_daily_loss",
            limit_type=LimitType.DAILY_LOSS,
            threshold=max_daily_loss,
            action=LimitAction.HALT,
        ))

        # Gross exposure limit
        max_gross = port.get("max_gross_exposure", 1.50)
        limits.append(RiskLimit(
            name="portfolio_gross_exposure",
            limit_type=LimitType.GROSS_EXPOSURE,
            threshold=max_gross,
            action=LimitAction.REDUCE,
        ))

        # VaR limit: 5% of equity at 99% confidence
        limits.append(RiskLimit(
            name="portfolio_var99_pct",
            limit_type=LimitType.VAR,
            threshold=0.05,
            action=LimitAction.WARN,
        ))

        # Per-instrument concentration: default_max_frac from config
        inst = self._raw.get("per_instrument", {})
        default_max_frac = inst.get("default_max_frac", 0.65)
        limits.append(RiskLimit(
            name="instrument_max_concentration",
            limit_type=LimitType.CONCENTRATION,
            threshold=default_max_frac,
            action=LimitAction.REDUCE,
        ))

        # Per-instrument notional caps
        notional_caps: Dict = inst.get("notional_caps", {})
        for sym, cap in notional_caps.items():
            limits.append(RiskLimit(
                name=f"notional_cap_{sym}",
                limit_type=LimitType.NOTIONAL,
                threshold=float(cap),
                action=LimitAction.REDUCE,
            ))

        return limits

    @property
    def limits(self) -> List[RiskLimit]:
        return list(self._limits)

    def portfolio_limits(self) -> List[RiskLimit]:
        return [lim for lim in self._limits if not lim.name.startswith("notional_cap_")]

    def instrument_limits(self, symbol: str) -> List[RiskLimit]:
        """Return limits specific to a symbol."""
        return [lim for lim in self._limits if lim.name.endswith(f"_{symbol}")]

    def max_frac(self) -> float:
        inst = self._raw.get("per_instrument", {})
        return float(inst.get("default_max_frac", 0.65))

    def notional_cap(self, symbol: str) -> Optional[float]:
        inst = self._raw.get("per_instrument", {})
        caps = inst.get("notional_caps", {})
        return float(caps[symbol]) if symbol in caps else None

    def equity_floor_pct(self) -> float:
        cb = self._raw.get("circuit_breakers", {})
        return float(cb.get("equity_floor_pct", 0.70))

    def max_open_positions(self) -> int:
        port = self._raw.get("portfolio", {})
        return int(port.get("max_open_positions", 8))


# ---------------------------------------------------------------------------
# Limit Checker
# ---------------------------------------------------------------------------

class LimitChecker:
    """
    Checks all configured limits against the current portfolio state.

    Emits log warnings and optional webhook calls on breach.
    """

    def __init__(
        self,
        config: Optional[RiskLimitConfig] = None,
        webhook_url: Optional[str] = None,
    ) -> None:
        self.config = config or RiskLimitConfig()
        self.webhook_url = webhook_url

    def check_portfolio(
        self,
        equity: float,
        initial_equity: float,
        daily_pnl: float,
        gross_exposure_frac: float,
        var99_frac: float,
        max_position_frac: float,
    ) -> List[RiskLimit]:
        """
        Check portfolio-level limits.

        Parameters
        ----------
        equity              : current account equity in USD
        initial_equity      : equity at session start (for daily loss calc)
        daily_pnl           : today's P&L in USD
        gross_exposure_frac : sum(abs(notional)) / equity
        var99_frac          : current VaR99 / equity
        max_position_frac   : largest single-position notional / equity

        Returns
        -------
        List of breached RiskLimit objects (empty if all clear).
        """
        breached: List[RiskLimit] = []

        if initial_equity > 0:
            daily_loss_frac = -daily_pnl / initial_equity  # positive = loss
        else:
            daily_loss_frac = 0.0

        # drawdown relative to initial_equity (peak tracking is in DrawdownGuard)
        drawdown_frac = max(0.0, (initial_equity - equity) / max(initial_equity, 1.0))

        checks = {
            "portfolio_max_drawdown": drawdown_frac,
            "portfolio_daily_loss": daily_loss_frac,
            "portfolio_gross_exposure": gross_exposure_frac,
            "portfolio_var99_pct": var99_frac,
            "instrument_max_concentration": max_position_frac,
        }

        for lim in self.config.limits:
            if lim.name not in checks:
                continue
            updated = lim.check(checks[lim.name])
            if updated.is_breached:
                self._emit(updated)
                breached.append(updated)

        return breached

    def check_symbol_notional(
        self, symbol: str, proposed_notional: float
    ) -> Optional[RiskLimit]:
        """Check if proposed_notional would breach this symbol's cap."""
        cap = self.config.notional_cap(symbol)
        if cap is None:
            return None
        lim = RiskLimit(
            name=f"notional_cap_{symbol}",
            limit_type=LimitType.NOTIONAL,
            threshold=cap,
            action=LimitAction.REDUCE,
        )
        updated = lim.check(abs(proposed_notional))
        if updated.is_breached:
            self._emit(updated)
            return updated
        return None

    def _emit(self, lim: RiskLimit) -> None:
        log.warning(
            "RISK LIMIT BREACH [%s] %s: current=%.4f threshold=%.4f action=%s",
            lim.limit_type.value, lim.name, lim.current_value,
            lim.threshold, lim.action.value,
        )
        if self.webhook_url:
            self._post_webhook(lim)

    def _post_webhook(self, lim: RiskLimit) -> None:
        payload = {
            "text": (
                f"[RISK] {lim.action.value}: {lim.name} "
                f"current={lim.current_value:.4f} threshold={lim.threshold:.4f}"
            )
        }
        try:
            data = str(payload).encode("utf-8")
            req = _urllib_request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            _urllib_request.urlopen(req, timeout=3)
        except Exception as exc:
            log.debug("Webhook post failed: %s", exc)

    def all_limit_states(
        self,
        equity: float,
        initial_equity: float,
        daily_pnl: float,
        gross_exposure_frac: float,
        var99_frac: float,
        max_position_frac: float,
        per_symbol_notionals: Optional[Dict[str, float]] = None,
    ) -> List[RiskLimit]:
        """
        Return all limits (breached and non-breached) with current_value filled.
        Useful for the risk API endpoint.
        """
        if initial_equity > 0:
            daily_loss_frac = -daily_pnl / initial_equity
        else:
            daily_loss_frac = 0.0
        drawdown_frac = max(0.0, (initial_equity - equity) / max(initial_equity, 1.0))

        state_map = {
            "portfolio_max_drawdown": drawdown_frac,
            "portfolio_daily_loss": daily_loss_frac,
            "portfolio_gross_exposure": gross_exposure_frac,
            "portfolio_var99_pct": var99_frac,
            "instrument_max_concentration": max_position_frac,
        }

        result: List[RiskLimit] = []
        for lim in self.config.limits:
            if lim.name in state_map:
                result.append(lim.check(state_map[lim.name]))
            elif per_symbol_notionals and lim.limit_type == LimitType.NOTIONAL:
                # Extract symbol from name like notional_cap_NQ
                sym = lim.name.replace("notional_cap_", "")
                if sym in per_symbol_notionals:
                    result.append(lim.check(abs(per_symbol_notionals[sym])))
                else:
                    result.append(lim)
            else:
                result.append(lim)
        return result


# ---------------------------------------------------------------------------
# Position Limiter
# ---------------------------------------------------------------------------

@dataclass
class PositionLimitResult:
    """Result of a pre-trade position size check."""
    requested_qty: float
    allowed_qty: float
    is_blocked: bool
    breach_reason: Optional[str]

    @property
    def is_reduced(self) -> bool:
        return (not self.is_blocked) and (self.allowed_qty < self.requested_qty - 1e-9)


class PositionLimiter:
    """
    Pre-trade check: given a proposed trade, determines the maximum
    allowed quantity and whether the order should be blocked.

    Checks applied in order:
        1. Per-symbol notional cap
        2. Single-position concentration limit (fraction of equity)
        3. Gross exposure limit
        4. Max open positions count
    """

    def __init__(
        self,
        config: Optional[RiskLimitConfig] = None,
        checker: Optional[LimitChecker] = None,
    ) -> None:
        self.config = config or RiskLimitConfig()
        self.checker = checker or LimitChecker(config=self.config)

    def check(
        self,
        symbol: str,
        requested_qty: float,
        price: float,
        equity: float,
        current_gross_notional: float,
        n_open_positions: int,
        current_symbol_notional: float = 0.0,
    ) -> PositionLimitResult:
        """
        Evaluate whether a trade of requested_qty @ price is permissible.

        Parameters
        ----------
        symbol                  : instrument ticker
        requested_qty           : unsigned number of units to buy/sell
        price                   : current price per unit
        equity                  : current account equity in USD
        current_gross_notional  : current total abs(notional) before this trade
        n_open_positions        : current number of open positions
        current_symbol_notional : current notional already held in this symbol
        """
        if equity <= 0:
            return PositionLimitResult(requested_qty, 0.0, True, "equity <= 0")

        proposed_notional = abs(requested_qty) * price
        total_symbol_notional = current_symbol_notional + proposed_notional

        # 1. Max open positions
        max_pos = self.config.max_open_positions()
        if n_open_positions >= max_pos and current_symbol_notional == 0.0:
            return PositionLimitResult(
                requested_qty, 0.0, True,
                f"max_open_positions {max_pos} reached",
            )

        allowed_notional = proposed_notional

        # 2. Per-symbol notional cap
        sym_cap = self.config.notional_cap(symbol)
        if sym_cap is not None:
            remaining_cap = max(0.0, sym_cap - current_symbol_notional)
            if remaining_cap <= 0:
                return PositionLimitResult(
                    requested_qty, 0.0, True,
                    f"notional cap {sym_cap:,.0f} already exhausted for {symbol}",
                )
            allowed_notional = min(allowed_notional, remaining_cap)

        # 3. Concentration limit
        max_frac = self.config.max_frac()
        max_notional_for_frac = max_frac * equity
        headroom_conc = max(0.0, max_notional_for_frac - current_symbol_notional)
        allowed_notional = min(allowed_notional, headroom_conc)

        # 4. Gross exposure limit
        raw = self.config._raw.get("portfolio", {})
        max_gross = float(raw.get("max_gross_exposure", 1.50))
        max_gross_usd = max_gross * equity
        headroom_gross = max(0.0, max_gross_usd - current_gross_notional)
        allowed_notional = min(allowed_notional, headroom_gross)

        if allowed_notional < 1e-6:
            return PositionLimitResult(
                requested_qty, 0.0, True, "no headroom under exposure limits"
            )

        allowed_qty = allowed_notional / max(price, 1e-9)
        allowed_qty = min(allowed_qty, abs(requested_qty))

        # Preserve sign of requested_qty
        sign = 1.0 if requested_qty >= 0 else -1.0
        allowed_qty *= sign

        breach_reason: Optional[str] = None
        if abs(allowed_qty) < abs(requested_qty) - 1e-9:
            breach_reason = "quantity reduced due to risk limits"

        return PositionLimitResult(
            requested_qty=requested_qty,
            allowed_qty=allowed_qty,
            is_blocked=False,
            breach_reason=breach_reason,
        )


# ---------------------------------------------------------------------------
# Drawdown Guard
# ---------------------------------------------------------------------------

class DrawdownGuard:
    """
    Tracks rolling max equity and enforces drawdown-based circuit breakers.

    Thresholds (taken from live_trader_alpaca.py constants):
        DD_HALT_PCT   = 10% drawdown from peak -> halt new entries
        DD_RESUME_PCT =  5% remaining drawdown -> resume entries

    The guard integrates with execution.monitoring.CircuitBreaker if
    provided; otherwise it manages a simple internal halted flag.
    """

    DD_HALT_PCT   = 0.10
    DD_RESUME_PCT = 0.05

    def __init__(
        self,
        initial_equity: float,
        circuit_breaker=None,   # optional CircuitBreaker instance
        halt_pct: float = DD_HALT_PCT,
        resume_pct: float = DD_RESUME_PCT,
    ) -> None:
        self.peak_equity = initial_equity
        self.initial_equity = initial_equity
        self.circuit_breaker = circuit_breaker
        self.halt_pct = halt_pct
        self.resume_pct = resume_pct
        self._halted: bool = False
        self._halt_time: Optional[datetime] = None

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def current_drawdown(self) -> float:
        """Current drawdown fraction from peak (0 = at peak, 0.10 = 10% below)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self._last_equity) / self.peak_equity)

    def update(self, equity: float) -> bool:
        """
        Update equity; trigger or release halt as appropriate.

        Parameters
        ----------
        equity : current account equity in USD

        Returns
        -------
        True if trading should be halted, False if trading is permitted.
        """
        self._last_equity = equity

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity

        dd = self.current_drawdown

        if not self._halted and dd >= self.halt_pct:
            self._halted = True
            self._halt_time = datetime.now(timezone.utc)
            log.warning(
                "DrawdownGuard: HALT triggered. Drawdown=%.2f%% (threshold %.2f%%)",
                dd * 100, self.halt_pct * 100,
            )
            if self.circuit_breaker is not None:
                try:
                    self.circuit_breaker.trigger(
                        "DRAWDOWN_HALT",
                        f"DrawdownGuard: DD={dd:.2%} >= halt threshold {self.halt_pct:.2%}",
                    )
                except Exception as exc:
                    log.error("CircuitBreaker.trigger failed: %s", exc)

        elif self._halted and dd <= self.resume_pct:
            self._halted = False
            log.info(
                "DrawdownGuard: RESUME. Drawdown recovered to %.2f%% (resume threshold %.2f%%)",
                dd * 100, self.resume_pct * 100,
            )
            if self.circuit_breaker is not None:
                try:
                    # Only clear the DRAWDOWN_HALT if the CB is still in that state
                    if hasattr(self.circuit_breaker, "halt_reason"):
                        if "DRAWDOWN" in (self.circuit_breaker.halt_reason or ""):
                            self.circuit_breaker.clear()
                except Exception as exc:
                    log.debug("CircuitBreaker.clear failed: %s", exc)

        return self._halted

    def status(self) -> Dict:
        """Return a dict summary suitable for logging or the risk API."""
        return {
            "peak_equity": round(self.peak_equity, 2),
            "current_equity": round(getattr(self, "_last_equity", self.initial_equity), 2),
            "drawdown_pct": round(self.current_drawdown * 100, 4),
            "is_halted": self._halted,
            "halt_pct": self.halt_pct,
            "resume_pct": self.resume_pct,
            "halt_time": self._halt_time.isoformat() if self._halt_time else None,
        }
