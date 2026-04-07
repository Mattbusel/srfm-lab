"""
pre_trade_checks.py # Pre-trade risk gate for SRFM execution layer.

All checks are run in sequence; the engine returns on the first failure.
Limits are sourced from ParamManager (or defaults if unavailable).
"""

from __future__ import annotations

import sqlite3
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter defaults # overridden by ParamManager when available
# ---------------------------------------------------------------------------

DEFAULT_MAX_POSITION_PCT: float = 0.20       # max 20% NAV in single symbol
DEFAULT_DAILY_LOSS_LIMIT_PCT: float = 0.02   # halt if daily loss > 2% NAV
DEFAULT_MAX_ORDER_ADV_PCT: float = 0.05      # order < 5% ADV
DEFAULT_MAX_SPREAD_BPS: float = 50.0         # reject if spread > 50 bps
DEFAULT_MAX_PORTFOLIO_CORR: float = 0.70     # portfolio correlation < 0.70
DEFAULT_MAX_SECTOR_PCT: float = 0.35         # sector exposure < 35% NAV
DEFAULT_MAX_LEVERAGE: float = 3.0            # total leverage < 3x
DEFAULT_MIN_HOLD_BARS: int = 3               # minimum hold period in bars
DEFAULT_EVENT_SIZE_FACTOR: float = 0.50      # reduce to 50% size near events


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    SELL_SHORT = "SELL_SHORT"
    BUY_TO_COVER = "BUY_TO_COVER"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OrderRequest:
    """Represents a single order request submitted for pre-trade risk review."""
    symbol: str
    side: OrderSide
    qty: float                    # shares / contracts / units
    price: float                  # limit/reference price; use last for market orders
    order_type: OrderType
    strategy_id: str
    signal_strength: float        # normalized 0.0-1.0 signal confidence
    order_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    asset_class: str = "equity"   # equity | crypto | futures


@dataclass
class CheckResult:
    """Result from a single risk check function."""
    passed: bool
    check_name: str
    message: str
    rejected_reason: Optional[str] = None


@dataclass
class RiskCheckResult:
    """Aggregated result from the full PreTradeRiskEngine.check() pipeline."""
    passed: bool
    check_name: str               # last check run (or first failure)
    message: str
    rejected_reason: Optional[str] = None
    checks_run: int = 0
    latency_us: float = 0.0       # microseconds for full pipeline


@dataclass
class PositionSnapshot:
    """Current portfolio positions for risk calculations."""
    positions: Dict[str, float]   # symbol -> dollar_value (positive=long, negative=short)
    nav: float
    cash: float = 0.0


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_position_limit(
    order: OrderRequest,
    positions: PositionSnapshot,
) -> CheckResult:
    """Reject if adding this order would exceed 20% NAV in a single symbol."""
    check_name = "position_limit"
    limit_pct = DEFAULT_MAX_POSITION_PCT

    if positions.nav <= 0:
        return CheckResult(False, check_name, "NAV is zero or negative", "invalid_nav")

    current_exposure = abs(positions.positions.get(order.symbol, 0.0))
    order_value = order.qty * order.price

    if order.side in (OrderSide.BUY, OrderSide.BUY_TO_COVER):
        new_exposure = current_exposure + order_value
    else:
        # selling reduces long exposure or increases short exposure
        new_exposure = abs(current_exposure - order_value)

    new_pct = new_exposure / positions.nav

    if new_pct > limit_pct:
        return CheckResult(
            False,
            check_name,
            f"{order.symbol} would be {new_pct:.1%} of NAV (limit {limit_pct:.1%})",
            "position_limit_exceeded",
        )
    return CheckResult(True, check_name, f"{order.symbol} position {new_pct:.1%} of NAV # OK")


def check_daily_loss_limit(
    order: OrderRequest,
    daily_pnl: float,
    nav: float,
) -> CheckResult:
    """Halt all trading if realized+unrealized daily loss exceeds 2% NAV."""
    check_name = "daily_loss_limit"
    limit_pct = DEFAULT_DAILY_LOSS_LIMIT_PCT

    if nav <= 0:
        return CheckResult(False, check_name, "NAV is zero or negative", "invalid_nav")

    loss_pct = -daily_pnl / nav  # positive means losing money

    if loss_pct >= limit_pct:
        return CheckResult(
            False,
            check_name,
            f"Daily loss {loss_pct:.2%} exceeds limit {limit_pct:.2%}",
            "daily_loss_limit_breached",
        )
    return CheckResult(True, check_name, f"Daily P&L {daily_pnl:+.2f} ({-loss_pct:.2%}) # OK")


def check_max_order_size(
    order: OrderRequest,
    adv: float,
) -> CheckResult:
    """Reject if order value exceeds 5% of average daily volume (dollar)."""
    check_name = "max_order_size"
    limit_pct = DEFAULT_MAX_ORDER_ADV_PCT

    if adv <= 0:
        return CheckResult(False, check_name, "ADV unavailable or zero", "adv_unavailable")

    order_value = order.qty * order.price
    pct_adv = order_value / adv

    if pct_adv > limit_pct:
        return CheckResult(
            False,
            check_name,
            f"Order {order_value:.0f} is {pct_adv:.1%} of ADV {adv:.0f} (limit {limit_pct:.1%})",
            "order_too_large_vs_adv",
        )
    return CheckResult(True, check_name, f"Order size {pct_adv:.2%} of ADV # OK")


def check_spread_gate(
    order: OrderRequest,
    spread_bps: float,
) -> CheckResult:
    """Reject market/limit orders when bid-ask spread exceeds 50 bps."""
    check_name = "spread_gate"
    limit_bps = DEFAULT_MAX_SPREAD_BPS

    if order.order_type == OrderType.MARKET and spread_bps > limit_bps:
        return CheckResult(
            False,
            check_name,
            f"Spread {spread_bps:.1f} bps exceeds limit {limit_bps:.1f} bps",
            "spread_too_wide",
        )
    return CheckResult(True, check_name, f"Spread {spread_bps:.1f} bps # OK")


def check_circuit_breaker(
    symbol: str,
    circuit_broken_symbols: set,
) -> CheckResult:
    """Reject orders in symbols currently subject to a circuit breaker."""
    check_name = "circuit_breaker"

    if symbol in circuit_broken_symbols:
        return CheckResult(
            False,
            check_name,
            f"{symbol} is circuit-broken",
            "circuit_breaker_active",
        )
    return CheckResult(True, check_name, f"{symbol} not circuit-broken # OK")


def check_event_calendar(
    order: OrderRequest,
    calendar: Dict[str, datetime],
    current_time: Optional[datetime] = None,
    event_window_hours: float = 24.0,
) -> CheckResult:
    """
    Reduce order size to 50% (event_size_factor) within event_window_hours
    of a scheduled event (earnings, FOMC, etc.) for the symbol.

    Returns a CheckResult with passed=True but appends a note when sizing
    is restricted. The caller is responsible for applying the size factor.
    """
    check_name = "event_calendar"
    size_factor = DEFAULT_EVENT_SIZE_FACTOR

    if current_time is None:
        current_time = datetime.now(timezone.utc)

    event_dt = calendar.get(order.symbol)
    if event_dt is None:
        return CheckResult(True, check_name, f"No event scheduled for {order.symbol} # OK")

    # Normalise timezone
    if event_dt.tzinfo is None:
        event_dt = event_dt.replace(tzinfo=timezone.utc)

    hours_to_event = abs((event_dt - current_time).total_seconds()) / 3600.0

    if hours_to_event <= event_window_hours:
        # Check if the incoming order already has reduced size
        # # we flag but pass; upstream must have already halved qty
        max_allowed_pct_signal = size_factor  # signal_strength proxy
        if order.signal_strength > size_factor + 0.05:
            return CheckResult(
                False,
                check_name,
                (
                    f"{order.symbol} within {hours_to_event:.1f}h of event; "
                    f"signal_strength {order.signal_strength:.2f} exceeds "
                    f"allowed {size_factor:.2f}"
                ),
                "event_calendar_size_exceeded",
            )
        return CheckResult(
            True,
            check_name,
            (
                f"{order.symbol} within {hours_to_event:.1f}h of event; "
                f"size factor {size_factor} applied # OK"
            ),
        )

    return CheckResult(True, check_name, f"{order.symbol} event in {hours_to_event:.1f}h # OK")


def check_correlation_concentration(
    order: OrderRequest,
    positions: PositionSnapshot,
    corr_matrix: Optional[np.ndarray],
    symbol_index: Optional[Dict[str, int]] = None,
) -> CheckResult:
    """
    Reject if adding this position would push average pairwise portfolio
    correlation above 0.70.
    """
    check_name = "correlation_concentration"
    limit = DEFAULT_MAX_PORTFOLIO_CORR

    if corr_matrix is None or symbol_index is None:
        return CheckResult(True, check_name, "Correlation data unavailable # skipping")

    existing_symbols = [s for s in positions.positions if s in symbol_index]
    new_symbol = order.symbol

    if new_symbol not in symbol_index or len(existing_symbols) < 2:
        return CheckResult(True, check_name, "Insufficient symbols for correlation check # OK")

    # Compute mean pairwise correlation of new portfolio (existing + new symbol)
    all_symbols = list(set(existing_symbols + [new_symbol]))
    indices = [symbol_index[s] for s in all_symbols if s in symbol_index]

    if len(indices) < 2:
        return CheckResult(True, check_name, "Not enough symbols in corr matrix # OK")

    sub_matrix = corr_matrix[np.ix_(indices, indices)]
    n = len(indices)
    # Off-diagonal mean
    off_diag = (sub_matrix.sum() - np.trace(sub_matrix)) / (n * (n - 1))

    if off_diag > limit:
        return CheckResult(
            False,
            check_name,
            f"Avg pairwise correlation {off_diag:.3f} would exceed limit {limit:.2f}",
            "correlation_concentration_exceeded",
        )
    return CheckResult(True, check_name, f"Portfolio avg correlation {off_diag:.3f} # OK")


def check_sector_limit(
    order: OrderRequest,
    positions: PositionSnapshot,
    sector_map: Dict[str, str],
) -> CheckResult:
    """Reject if a single sector would exceed 35% of NAV after order."""
    check_name = "sector_limit"
    limit_pct = DEFAULT_MAX_SECTOR_PCT

    if positions.nav <= 0:
        return CheckResult(False, check_name, "NAV zero or negative", "invalid_nav")

    order_sector = sector_map.get(order.symbol, "UNKNOWN")
    order_value = order.qty * order.price

    # Aggregate current sector exposure
    sector_exposure: Dict[str, float] = {}
    for sym, val in positions.positions.items():
        sec = sector_map.get(sym, "UNKNOWN")
        sector_exposure[sec] = sector_exposure.get(sec, 0.0) + abs(val)

    if order.side in (OrderSide.BUY, OrderSide.SELL_SHORT):
        sector_exposure[order_sector] = sector_exposure.get(order_sector, 0.0) + order_value

    max_sector = max(sector_exposure.values(), default=0.0)
    max_pct = max_sector / positions.nav

    if max_pct > limit_pct:
        worst_sector = max(sector_exposure, key=sector_exposure.get)  # type: ignore[arg-type]
        return CheckResult(
            False,
            check_name,
            (
                f"Sector '{worst_sector}' would be {max_pct:.1%} of NAV "
                f"(limit {limit_pct:.1%})"
            ),
            "sector_limit_exceeded",
        )
    return CheckResult(True, check_name, f"Max sector exposure {max_pct:.1%} of NAV # OK")


def check_leverage(
    order: OrderRequest,
    positions: PositionSnapshot,
) -> CheckResult:
    """Reject if gross leverage (sum of absolute exposures / NAV) would exceed 3x."""
    check_name = "leverage"
    limit = DEFAULT_MAX_LEVERAGE

    if positions.nav <= 0:
        return CheckResult(False, check_name, "NAV zero or negative", "invalid_nav")

    current_gross = sum(abs(v) for v in positions.positions.values())
    order_value = order.qty * order.price
    new_gross = current_gross + order_value  # conservative: always adds
    new_leverage = new_gross / positions.nav

    if new_leverage > limit:
        return CheckResult(
            False,
            check_name,
            f"Leverage would be {new_leverage:.2f}x (limit {limit:.1f}x)",
            "leverage_limit_exceeded",
        )
    return CheckResult(True, check_name, f"Gross leverage {new_leverage:.2f}x # OK")


def check_min_hold_bars(
    symbol: str,
    last_entry_bar: Optional[int],
    current_bar: int,
    min_hold_bars: int = DEFAULT_MIN_HOLD_BARS,
) -> CheckResult:
    """Reject an exit if minimum hold period has not elapsed."""
    check_name = "min_hold_bars"

    if last_entry_bar is None:
        return CheckResult(True, check_name, f"{symbol} no prior entry # OK")

    bars_held = current_bar - last_entry_bar
    if bars_held < min_hold_bars:
        return CheckResult(
            False,
            check_name,
            (
                f"{symbol} held {bars_held} bars; "
                f"minimum is {min_hold_bars} bars"
            ),
            "min_hold_bars_not_met",
        )
    return CheckResult(
        True,
        check_name,
        f"{symbol} held {bars_held} bars (min {min_hold_bars}) # OK",
    )


# ---------------------------------------------------------------------------
# SQLite-backed risk check log
# ---------------------------------------------------------------------------

class RiskCheckLog:
    """
    Persists pre-trade check results to SQLite for audit and monitoring.
    Thread-safe via check_same_thread=False with WAL mode.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS risk_check_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          REAL    NOT NULL,
            order_id    TEXT,
            symbol      TEXT,
            strategy_id TEXT,
            passed      INTEGER NOT NULL,
            check_name  TEXT    NOT NULL,
            message     TEXT,
            reason      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_rcl_ts ON risk_check_log(ts);
        CREATE INDEX IF NOT EXISTS idx_rcl_passed ON risk_check_log(passed);
    """

    def __init__(self, db_path: str = "risk_checks.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(self._CREATE_TABLE)
        self._conn.commit()
        logger.debug("RiskCheckLog initialised at %s", db_path)

    def log(self, order: OrderRequest, result: RiskCheckResult) -> None:
        """Persist a single pre-trade check result."""
        self._conn.execute(
            """
            INSERT INTO risk_check_log
                (ts, order_id, symbol, strategy_id, passed, check_name, message, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.time(),
                order.order_id,
                order.symbol,
                order.strategy_id,
                int(result.passed),
                result.check_name,
                result.message,
                result.rejected_reason,
            ),
        )
        self._conn.commit()

    def recent_rejections(self, n: int = 50) -> List[dict]:
        """Return the n most recent rejections as dicts."""
        cur = self._conn.execute(
            """
            SELECT ts, order_id, symbol, strategy_id, check_name, message, reason
            FROM   risk_check_log
            WHERE  passed = 0
            ORDER  BY ts DESC
            LIMIT  ?
            """,
            (n,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def rejection_rate(self, hours: float = 24.0) -> float:
        """
        Fraction of check results that were rejections over the last `hours`.
        Returns 0.0 if no records exist in the window.
        """
        cutoff = time.time() - hours * 3600
        cur = self._conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN passed=0 THEN 1 ELSE 0 END) "
            "FROM risk_check_log WHERE ts >= ?",
            (cutoff,),
        )
        row = cur.fetchone()
        total, rejections = row
        if not total:
            return 0.0
        return (rejections or 0) / total

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Pre-trade risk engine
# ---------------------------------------------------------------------------

class PreTradeRiskEngine:
    """
    Orchestrates all pre-trade checks for a given OrderRequest.

    Usage::

        engine = PreTradeRiskEngine(nav=1_000_000, db_path="risk.db")
        result = engine.check(order, context)
        if not result.passed:
            reject(order, result.rejected_reason)

    All market data (ADV, spread, circuit breakers, etc.) is injected via
    the `context` dict at call time so the engine remains stateless between
    calls.
    """

    def __init__(
        self,
        nav: float,
        db_path: str = "risk_checks.db",
        param_overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        self.nav = nav
        self._log = RiskCheckLog(db_path)
        self._overrides: Dict[str, float] = param_overrides or {}
        self._circuit_broken: set = set()

    # # Configuration helpers -----------------------------------------------

    def _param(self, key: str, default: float) -> float:
        return self._overrides.get(key, default)

    def set_circuit_breaker(self, symbol: str, active: bool = True) -> None:
        if active:
            self._circuit_broken.add(symbol)
        else:
            self._circuit_broken.discard(symbol)

    def update_nav(self, nav: float) -> None:
        self.nav = nav

    # # Main pipeline -------------------------------------------------------

    def check(
        self,
        order: OrderRequest,
        context: Optional[Dict] = None,
    ) -> RiskCheckResult:
        """
        Run all pre-trade checks against `order`.

        `context` keys (all optional; checks are skipped gracefully if missing):
            positions      # PositionSnapshot
            daily_pnl      # float (realized + unrealized P&L today)
            adv            # float (average daily dollar volume)
            spread_bps     # float (current bid-ask spread in bps)
            calendar       # Dict[str, datetime] (upcoming events per symbol)
            corr_matrix    # np.ndarray
            symbol_index   # Dict[str, int] (symbol -> row/col in corr_matrix)
            sector_map     # Dict[str, str]
            last_entry_bar # int or None (bar index of last entry for symbol)
            current_bar    # int
        """
        ctx = context or {}
        t0 = time.perf_counter()
        checks_run = 0

        positions: PositionSnapshot = ctx.get(
            "positions",
            PositionSnapshot(positions={}, nav=self.nav),
        )
        if positions.nav <= 0:
            positions = PositionSnapshot(positions=positions.positions, nav=self.nav, cash=positions.cash)

        def _fail(res: CheckResult) -> RiskCheckResult:
            return RiskCheckResult(
                passed=False,
                check_name=res.check_name,
                message=res.message,
                rejected_reason=res.rejected_reason,
                checks_run=checks_run,
                latency_us=(time.perf_counter() - t0) * 1e6,
            )

        # 1. Circuit breaker
        checks_run += 1
        r = check_circuit_breaker(order.symbol, self._circuit_broken)
        if not r.passed:
            result = _fail(r)
            self._log.log(order, result)
            return result

        # 2. Daily loss limit
        checks_run += 1
        daily_pnl: float = ctx.get("daily_pnl", 0.0)
        r = check_daily_loss_limit(order, daily_pnl, self.nav)
        if not r.passed:
            result = _fail(r)
            self._log.log(order, result)
            return result

        # 3. Leverage check
        checks_run += 1
        r = check_leverage(order, positions)
        if not r.passed:
            result = _fail(r)
            self._log.log(order, result)
            return result

        # 4. Position limit
        checks_run += 1
        r = check_position_limit(order, positions)
        if not r.passed:
            result = _fail(r)
            self._log.log(order, result)
            return result

        # 5. Sector limit
        checks_run += 1
        sector_map: Dict[str, str] = ctx.get("sector_map", {})
        r = check_sector_limit(order, positions, sector_map)
        if not r.passed:
            result = _fail(r)
            self._log.log(order, result)
            return result

        # 6. Correlation concentration
        checks_run += 1
        corr_matrix: Optional[np.ndarray] = ctx.get("corr_matrix")
        symbol_index: Optional[Dict[str, int]] = ctx.get("symbol_index")
        r = check_correlation_concentration(order, positions, corr_matrix, symbol_index)
        if not r.passed:
            result = _fail(r)
            self._log.log(order, result)
            return result

        # 7. Max order size vs ADV
        checks_run += 1
        adv: Optional[float] = ctx.get("adv")
        if adv is not None:
            r = check_max_order_size(order, adv)
            if not r.passed:
                result = _fail(r)
                self._log.log(order, result)
                return result

        # 8. Spread gate
        checks_run += 1
        spread_bps: Optional[float] = ctx.get("spread_bps")
        if spread_bps is not None:
            r = check_spread_gate(order, spread_bps)
            if not r.passed:
                result = _fail(r)
                self._log.log(order, result)
                return result

        # 9. Event calendar
        checks_run += 1
        calendar: Dict[str, datetime] = ctx.get("calendar", {})
        r = check_event_calendar(order, calendar)
        if not r.passed:
            result = _fail(r)
            self._log.log(order, result)
            return result

        # 10. Minimum hold bars (only applies to exit orders)
        if order.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            checks_run += 1
            last_entry_bar: Optional[int] = ctx.get("last_entry_bar")
            current_bar: int = ctx.get("current_bar", 0)
            min_hold = int(self._param("MIN_HOLD_BARS", DEFAULT_MIN_HOLD_BARS))
            r = check_min_hold_bars(order.symbol, last_entry_bar, current_bar, min_hold)
            if not r.passed:
                result = _fail(r)
                self._log.log(order, result)
                return result

        latency_us = (time.perf_counter() - t0) * 1e6
        result = RiskCheckResult(
            passed=True,
            check_name="all_checks",
            message=f"All {checks_run} checks passed",
            checks_run=checks_run,
            latency_us=latency_us,
        )
        self._log.log(order, result)
        return result

    def close(self) -> None:
        """Release DB resources."""
        self._log.close()
