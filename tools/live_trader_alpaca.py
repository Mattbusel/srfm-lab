"""
tools/live_trader_alpaca.py
===========================
LARSA v17 — Multi-Asset Live Trader (Alpaca Paper)

Streams 15-minute bars for crypto, stocks/ETFs, and trades options
using Black-Hole physics + GARCH vol scaling + OU mean reversion
across three timeframes (15m, 1h, 4h).

Asset classes:
  • crypto  — BTC, ETH, XRP, … (24/7, GTC orders, fractional)
  • equity  — SPY, QQQ, GLD, … (RTH only, DAY orders, fractional)
  • option  — ATM directional overlay on equity BH signals (tf≥6),
              ~35 DTE, rolls 7 days before expiry

New features vs v16:
  • StockDataStream runs alongside CryptoDataStream
  • OptionOverlay selects ATM contracts and manages expiry
  • Market-hours gate for equity/option orders
  • Per-asset-class TF_CAP and sizing caps

Run:
  python tools/live_trader_alpaca.py

Environment (tools/.env or shell):
  ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER=true
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import sys
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("live_trader")

# ── Repo paths ─────────────────────────────────────────────────────────────────
_REPO_ROOT      = Path(__file__).parents[1]
_ENV_FILE       = Path(__file__).parent / ".env"
_OVERRIDES_FILE = _REPO_ROOT / "config" / "signal_overrides.json"
_DB_PATH        = _REPO_ROOT / "execution" / "live_trades.db"

STRATEGY_VERSION = "larsa_v17"

# ── Strategy constants ─────────────────────────────────────────────────────────
# asset_class: "crypto" | "equity"
# options_overlay: True → OptionOverlay will trade ATM options on this ticker
#                  when BH fires at tf>=6 (in addition to the equity position)
INSTRUMENTS = {
    # ── Crypto (24/7, GTC, fractional) ───────────────────────────────────────
    # Removed penny/micro-cap tokens: SHIB, CRV, SUSHI, BAT, UNI, DOT, DOGE
    # — sub-cent prices cause runaway qty math and catastrophic fill churn
    "BTC":   {"ticker": "BTC/USD",   "asset_class": "crypto", "cf_4h": 0.016, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "ETH":   {"ticker": "ETH/USD",   "asset_class": "crypto", "cf_4h": 0.012, "cf_15m": 0.007, "cf_1h": 0.020, "cf_1d": 0.07},
    "XRP":   {"ticker": "XRP/USD",   "asset_class": "crypto", "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "AVAX":  {"ticker": "AVAX/USD",  "asset_class": "crypto", "cf_4h": 0.010, "cf_15m": 0.006, "cf_1h": 0.018, "cf_1d": 0.06},
    "LINK":  {"ticker": "LINK/USD",  "asset_class": "crypto", "cf_4h": 0.010, "cf_15m": 0.006, "cf_1h": 0.018, "cf_1d": 0.06},
    "AAVE":  {"ticker": "AAVE/USD",  "asset_class": "crypto", "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "LTC":   {"ticker": "LTC/USD",   "asset_class": "crypto", "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "BCH":   {"ticker": "BCH/USD",   "asset_class": "crypto", "cf_4h": 0.020, "cf_15m": 0.012, "cf_1h": 0.035, "cf_1d": 0.12},
    "MKR":   {"ticker": "MKR/USD",   "asset_class": "crypto", "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "YFI":   {"ticker": "YFI/USD",   "asset_class": "crypto", "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},

    # ── Equities / ETFs (RTH only, DAY orders, fractional) ───────────────────
    "SPY":   {"ticker": "SPY",  "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.003, "cf_15m": 0.0003, "cf_1h": 0.001,  "cf_1d": 0.005},
    "QQQ":   {"ticker": "QQQ",  "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.004, "cf_15m": 0.0004, "cf_1h": 0.0012, "cf_1d": 0.006},
    "IWM":   {"ticker": "IWM",  "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.005, "cf_15m": 0.0005, "cf_1h": 0.0015, "cf_1d": 0.007},
    "GLD":   {"ticker": "GLD",  "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.004, "cf_15m": 0.0004, "cf_1h": 0.0012, "cf_1d": 0.005},
    "TLT":   {"ticker": "TLT",  "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.003, "cf_15m": 0.0003, "cf_1h": 0.0010, "cf_1d": 0.004},
    "SLV":   {"ticker": "SLV",  "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.006, "cf_15m": 0.0006, "cf_1h": 0.0018, "cf_1d": 0.008},
    "USO":   {"ticker": "USO",  "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.008, "cf_15m": 0.0008, "cf_1h": 0.0025, "cf_1d": 0.010},
    "NVDA":  {"ticker": "NVDA", "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.012, "cf_15m": 0.0012, "cf_1h": 0.004,  "cf_1d": 0.015},
    "AAPL":  {"ticker": "AAPL", "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.006, "cf_15m": 0.0006, "cf_1h": 0.002,  "cf_1d": 0.008},
    "TSLA":  {"ticker": "TSLA", "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.015, "cf_15m": 0.0015, "cf_1h": 0.005,  "cf_1d": 0.018},
    "MSFT":  {"ticker": "MSFT", "asset_class": "equity", "options_overlay": True,  "cf_4h": 0.005, "cf_15m": 0.0005, "cf_1h": 0.0016, "cf_1d": 0.007},
}
N_INST = len(INSTRUMENTS)

BH_FORM     = 1.92
BH_CTL_MIN  = 3
BH_DECAY    = 0.924
BH_COLLAPSE = 0.992
MIN_HOLD_MINUTES = 240        # minimum hold = 4 hours wall-clock time (replaces bar-count gate)

DAILY_RISK              = 0.05
CORR_NORMAL             = 0.25
CORR_STRESS             = 0.60
CORR_STRESS_THRESHOLD   = 0.60
CRYPTO_CAP_FRAC         = 0.40
MIN_TRADE_FRAC          = 0.003

# ── Drawdown circuit breaker ───────────────────────────────────────────────────
DD_HALT_PCT  = 0.10   # halt ALL new entries if portfolio drops >10% from peak
DD_RESUME_PCT = 0.05  # resume once equity recovers to within 5% of peak
GARCH_TARGET_VOL        = 0.90
OU_FRAC                 = 0.08
DELTA_MAX_FRAC          = 0.40
STALE_15M_MOVE          = 0.002
WINNER_PROTECTION_PCT   = 0.005
BLOCKED_ENTRY_HOURS_UTC = {1, 13, 14, 15, 17, 18}
BOOST_ENTRY_HOURS_UTC   = {3, 9, 16, 19}
HOUR_BOOST_MULTIPLIER   = 1.25
OU_DISABLED_SYMS        = {"AVAX", "DOT", "LINK"}

TF_CAP = {7: 1.0, 6: 1.0, 4: 0.60, 3: 0.50, 2: 0.40, 1: 0.20, 0: 0.0}

# ── Equity / options constants ─────────────────────────────────────────────────
EQUITY_CAP_FRAC         = 0.20   # max fraction of portfolio per equity position
EQUITY_LONG_SHORT       = True   # allow short equities (sells via negative fraction)
# RTH window (US Eastern) — equities only trade inside this
RTH_OPEN_ET  = (9, 30)           # (hour, minute)
RTH_CLOSE_ET = (16, 0)

# Options overlay
OPT_TARGET_DTE          = 35     # aim for ~35 days to expiry when opening
OPT_ROLL_DTE            = 7      # roll/close when DTE drops below this
OPT_NOTIONAL_FRAC       = 0.015  # fraction of equity per option position (smaller = more concurrent)
OPT_MIN_TF              = 2      # minimum tf score to OPEN  (low = aggressive, fires on any BH signal)
OPT_EXIT_TF             = 0      # close when tf drops AT OR BELOW this (0 = only close when fully off)
OPT_MAX_HOLD_BARS       = 96     # hard liquidation after N x 15m bars (96 = 24 hours), keeps turnover high

MAX_ORDER_NOTIONAL = 195_000.0   # split into slices above this
OVERRIDES_TTL_SECS = 300         # re-read signal_overrides.json every 5 min


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ema(prev: float | None, val: float, alpha: float) -> float:
    return val if prev is None else alpha * val + (1.0 - alpha) * prev


def _alpha(n: int) -> float:
    return 2.0 / (n + 1)


class BHState:
    """Black Hole physics — single timeframe."""

    def __init__(self, cf: float, bh_form: float = BH_FORM) -> None:
        self.cf       = cf
        self.bh_form  = bh_form
        self.cf_scale = 1.0
        self.mass     = 0.0
        self.active   = False
        self.bh_dir   = 0
        self.ctl      = 0
        self.prices: deque[float] = deque(maxlen=25)

    def update(self, price: float) -> None:
        self.prices.append(float(price))
        if len(self.prices) < 2:
            return
        px   = list(self.prices)
        beta = abs(px[-1] - px[-2]) / (px[-2] + 1e-9) / (self.cf * self.cf_scale + 1e-9)
        was  = self.active
        if beta < 1.0:
            self.ctl  += 1
            self.mass  = self.mass * 0.97 + 0.03 * min(2.0, 1.0 + self.ctl * 0.1)
        else:
            self.ctl   = 0
            self.mass *= BH_DECAY
        if not was:
            self.active = self.mass > self.bh_form and self.ctl >= BH_CTL_MIN
        else:
            self.active = self.mass > BH_COLLAPSE and self.ctl >= BH_CTL_MIN
        if not was and self.active:
            lb = min(20, len(px) - 1)
            self.bh_dir = 1 if px[-1] > px[-1 - lb] else -1
        elif was and not self.active:
            self.bh_dir = 0


class ATRTracker:
    """14-bar EMA of ATR."""

    def __init__(self) -> None:
        self.atr: float | None    = None
        self.prev_c: float | None = None

    def update(self, h: float, l: float, c: float) -> None:
        tr = (h - l) if self.prev_c is None else max(h - l, abs(h - self.prev_c), abs(l - self.prev_c))
        self.atr    = _ema(self.atr, tr, _alpha(14))
        self.prev_c = c


class BullScale:
    """Trend alignment based on EMA stack."""

    def __init__(self) -> None:
        self.e12 = self.e26 = self.e50 = self.e200 = self.last = None

    def update(self, p: float) -> None:
        for attr, n in [("e12", 12), ("e26", 26), ("e50", 50), ("e200", 200)]:
            setattr(self, attr, _ema(getattr(self, attr), p, _alpha(n)))
        self.last = p

    @property
    def scale(self) -> float:
        if any(x is None for x in [self.e12, self.e26, self.e50, self.e200]):
            return 1.0
        return 3.0 if (self.last > self.e200 and self.e12 > self.e26 and self.e26 > self.e50) else 1.0


class GARCHTracker:
    """Online GARCH(1,1) vol forecaster — annualised."""

    def __init__(self, omega: float = 1e-6, alpha: float = 0.10,
                 beta: float = 0.85, warmup: int = 30) -> None:
        self.omega   = omega
        self.alpha   = alpha
        self.beta    = beta
        self._warmup = warmup
        self._var: float | None       = None
        self._returns: deque[float]   = deque(maxlen=100)
        self.vol: float | None        = None

    def update(self, ret: float) -> None:
        self._returns.append(ret)
        if len(self._returns) < self._warmup:
            return
        if self._var is None:
            self._var = float(np.var(list(self._returns))) + 1e-12
        else:
            self._var = self.omega + self.alpha * ret ** 2 + self.beta * self._var
        self.vol = math.sqrt(max(self._var, 1e-12) * 365)

    @property
    def vol_scale(self) -> float:
        if self.vol is None or self.vol <= 0:
            return 1.0
        return min(2.0, max(0.3, GARCH_TARGET_VOL / self.vol))


class OUDetector:
    """Ornstein-Uhlenbeck mean-reversion detector."""

    def __init__(self, window: int = 50, entry_z: float = 1.5, exit_z: float = 0.05) -> None:
        self.window    = window
        self.entry_z   = entry_z
        self.exit_z    = exit_z
        self._prices: deque[float] = deque(maxlen=window)
        self.mean = self.std = self.zscore = self.half_life = None

    def update(self, price: float) -> None:
        self._prices.append(math.log(price + 1e-12))
        if len(self._prices) < 20:
            return
        px         = np.array(self._prices)
        self.mean  = float(np.mean(px))
        self.std   = float(np.std(px)) + 1e-9
        self.zscore = (px[-1] - self.mean) / self.std
        y, x = px[1:], px[:-1]
        if len(x) > 5:
            rho = float(np.corrcoef(x, y)[0, 1])
            rho = max(-0.9999, min(0.9999, rho))
            self.half_life = -math.log(2) / math.log(abs(rho) + 1e-12) if rho < 1 else 999

    @property
    def long_signal(self) -> bool:
        return (self.zscore is not None and self.zscore < -self.entry_z
                and self.half_life is not None and 2 < self.half_life < 120)

    @property
    def short_signal(self) -> bool:
        return (self.zscore is not None and self.zscore > self.entry_z
                and self.half_life is not None and 2 < self.half_life < 120)

    @property
    def exit_signal(self) -> bool:
        return self.zscore is not None and abs(self.zscore) < self.exit_z


# ─────────────────────────────────────────────────────────────────────────────
# Per-instrument state container
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentState:
    """All per-instrument live state."""

    def __init__(self, sym: str) -> None:
        self.sym  = sym
        cfg       = INSTRUMENTS[sym]
        # BH per timeframe
        self.bh_15m = BHState(cfg["cf_15m"])
        self.bh_1h  = BHState(cfg["cf_1h"])
        self.bh_4h  = BHState(cfg["cf_4h"])
        # Indicators
        self.atr_1h  = ATRTracker()
        self.atr_4h  = ATRTracker()
        self.bull     = BullScale()
        self.garch    = GARCHTracker()
        self.ou       = OUDetector()
        # Aggregation buffers for 1h / 4h from 15m bars
        self._h1_buf: list[dict]  = []
        self._h4_buf: list[dict]  = []
        self._last_h1_close: datetime | None = None
        self._last_h4_close: datetime | None = None
        # Position tracking
        self.last_frac: float             = 0.0
        self.entry_time: datetime | None  = None   # wall-clock entry time for MIN_HOLD_MINUTES
        self.ou_pos: float                = 0.0
        self.pos_floor: float             = 0.0
        self.entry_px: float | None       = None
        self.last_15m_px: float | None = None
        self.warmup_done: bool    = False
        self._bar_count: int      = 0
        # FIFO queue for P&L tracking: list of (qty, price, time)
        self._fifo: list[tuple[float, float, str]] = []
        # Previous bar close — used for single-bar GARCH log-return
        self._prev_close: float | None = None


# ─────────────────────────────────────────────────────────────────────────────
# SQLite helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol           TEXT    NOT NULL,
            side             TEXT    NOT NULL,
            qty              REAL    NOT NULL,
            price            REAL    NOT NULL,
            notional         REAL    NOT NULL,
            fill_time        TEXT    NOT NULL,
            order_id         TEXT,
            strategy_version TEXT    DEFAULT 'larsa_v17'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_pnl (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT,
            entry_time   TEXT,
            exit_time    TEXT,
            entry_price  REAL,
            exit_price   REAL,
            qty          REAL,
            pnl          REAL,
            hold_bars    INTEGER
        )
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Market-hours helper
# ─────────────────────────────────────────────────────────────────────────────

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore

_ET = ZoneInfo("America/New_York")

def _is_rth(ts: datetime) -> bool:
    """Return True if ts falls within regular trading hours (Mon–Fri, 9:30–16:00 ET)."""
    et = ts.astimezone(_ET)
    if et.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    open_mins  = RTH_OPEN_ET[0]  * 60 + RTH_OPEN_ET[1]
    close_mins = RTH_CLOSE_ET[0] * 60 + RTH_CLOSE_ET[1]
    now_mins   = et.hour * 60 + et.minute
    return open_mins <= now_mins < close_mins


# ─────────────────────────────────────────────────────────────────────────────
# OptionOverlay — ATM directional options on equity underlyings
# ─────────────────────────────────────────────────────────────────────────────

class OptionPosition:
    """Tracks a single open option position."""
    __slots__ = ("sym", "contract_sym", "option_type", "strike", "expiry",
                 "qty", "entry_price", "entry_time", "bars_held")

    def __init__(self, sym: str, contract_sym: str, option_type: str,
                 strike: float, expiry: "date", qty: int,
                 entry_price: float, entry_time: datetime) -> None:
        self.sym           = sym
        self.contract_sym  = contract_sym
        self.option_type   = option_type   # "call" or "put"
        self.strike        = strike
        self.expiry        = expiry
        self.qty           = qty
        self.entry_price   = entry_price
        self.entry_time    = entry_time
        self.bars_held     = 0             # incremented every 15m bar


class OptionOverlay:
    """
    Aggressive ATM directional options overlay on all 11 equity instruments.

    Entry : tf >= OPT_MIN_TF (2)  — any single BH timeframe active
    Exits : signal fully off  |  direction flip  |  24h max hold  |  DTE roll
    Sizing: OPT_NOTIONAL_FRAC (1.5%) of equity per underlying

    IMPORTANT: update() never calls submit_order directly — it returns a
    pending action tuple that _act_on_targets dispatches via asyncio.to_thread,
    keeping all HTTP calls off the event loop.
    """

    def __init__(self, trading_client: Any) -> None:
        self._client: Any = trading_client
        self._positions: dict[str, OptionPosition] = {}   # sym → open position
        # sym → contract_sym mapping so fills can be routed back to the underlying
        self.contract_to_underlying: dict[str, str] = {}
        # Symbols with an in-flight open order — prevents double-open between bars
        self._pending_opens: set[str] = set()

    # ── Find ATM contract ──────────────────────────────────────────────────────

    def _find_contract(self, underlying: str, direction: int, spot: float) -> Any | None:
        """Return the best option contract object from Alpaca, or None on failure."""
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType
        from datetime import date, timedelta

        target_expiry = date.today() + timedelta(days=OPT_TARGET_DTE)
        opt_type = ContractType.CALL if direction > 0 else ContractType.PUT

        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date_gte=str(target_expiry - timedelta(days=7)),
                expiration_date_lte=str(target_expiry + timedelta(days=7)),
                type=opt_type,
                limit=50,
            )
            contracts = self._client.get_option_contracts(req).option_contracts
        except Exception as exc:
            log.warning("OptionOverlay: could not fetch contracts for %s: %s", underlying, exc)
            return None

        if not contracts:
            return None

        # Pick contract with strike closest to spot
        return min(contracts, key=lambda c: abs(float(c.strike_price) - spot))

    # ── Submit open (blocking — called via asyncio.to_thread) ─────────────────

    def _submit_open(self, sym: str, direction: int, spot: float, equity: float) -> None:
        """Blocking HTTP call — must be run in a thread, not on the event loop."""
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        from datetime import date
        try:
            if sym in self._positions:
                return

            contract = self._find_contract(sym, direction, spot)
            if contract is None:
                return

            notional  = equity * OPT_NOTIONAL_FRAC
            opt_price = float(getattr(contract, "close_price", None) or
                              getattr(contract, "last_price", None) or 1.0)
            if opt_price <= 0:
                opt_price = 1.0
            qty = max(1, int(notional / (opt_price * 100)))

            req = MarketOrderRequest(
                symbol        = contract.symbol,
                qty           = qty,
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY,
            )
            self._client.submit_order(req)
            expiry = contract.expiration_date
            if isinstance(expiry, str):
                expiry = date.fromisoformat(expiry)
            pos = OptionPosition(
                sym          = sym,
                contract_sym = contract.symbol,
                option_type  = "call" if direction > 0 else "put",
                strike       = float(contract.strike_price),
                expiry       = expiry,
                qty          = qty,
                entry_price  = opt_price,
                entry_time   = datetime.now(timezone.utc),
            )
            self._positions[sym] = pos
            self.contract_to_underlying[contract.symbol] = sym
            log.info(
                "OPT OPEN: %s %s x%d  strike=%.2f  expiry=%s  contract=%s  ~$%.0f notional",
                sym, pos.option_type.upper(), qty, pos.strike, pos.expiry,
                contract.symbol, qty * opt_price * 100,
            )
        except Exception as exc:
            log.error("OPT OPEN failed [%s]: %s", sym, exc)
        finally:
            self._pending_opens.discard(sym)

    # ── Submit close (blocking — called via asyncio.to_thread) ────────────────

    def _submit_close(self, sym: str, reason: str) -> None:
        """Blocking HTTP call — must be run in a thread, not on the event loop."""
        pos = self._positions.pop(sym, None)
        if pos is None:
            return
        self.contract_to_underlying.pop(pos.contract_sym, None)

        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol        = pos.contract_sym,
            qty           = pos.qty,
            side          = OrderSide.SELL,
            time_in_force = TimeInForce.DAY,
        )
        try:
            self._client.submit_order(req)
            log.info("OPT CLOSE: %s %s x%d  contract=%s  reason=%s",
                     sym, pos.option_type.upper(), pos.qty, pos.contract_sym, reason)
        except Exception as exc:
            log.error("OPT CLOSE failed [%s]: %s", sym, exc)
            # Re-insert so we try again next bar
            self._positions[sym] = pos
            self.contract_to_underlying[pos.contract_sym] = sym

    # ── Roll check ────────────────────────────────────────────────────────────

    def pending_rolls(self, now: datetime) -> list[str]:
        """Return list of syms that need rolling (DTE < OPT_ROLL_DTE). Caller submits."""
        from datetime import date
        today = now.date() if hasattr(now, "date") else now
        due = []
        for sym, pos in self._positions.items():
            if (pos.expiry - today).days < OPT_ROLL_DTE:
                due.append(sym)
        return due

    # ── Update — returns pending action for caller to dispatch async ──────────

    def update(self, sym: str, tf: int, direction: int, spot: float,
               equity: float, now: datetime) -> tuple[str, Any] | None:
        """
        Decide what to do with the option position for this underlying.
        Returns one of:
          ("open",  (sym, direction, spot, equity))
          ("close", (sym, reason))
          None  — no action needed

        NEVER calls submit_order — caller dispatches via asyncio.to_thread.
        """
        if not _is_rth(now):
            return None

        # Check rolls — schedule close if due
        for roll_sym in self.pending_rolls(now):
            if roll_sym == sym:
                log.info("OPT ROLL queued: %s", sym)
                return ("close", (sym, "roll"))

        has_pos = sym in self._positions

        if has_pos:
            pos = self._positions[sym]
            pos.bars_held += 1

            if pos.bars_held >= OPT_MAX_HOLD_BARS:
                return ("close", (sym, f"max_hold_{pos.bars_held}bars"))

            if tf <= OPT_EXIT_TF:
                return ("close", (sym, "signal_off"))

            if direction != 0:
                expected = "call" if direction > 0 else "put"
                if pos.option_type != expected:
                    return ("close", (sym, "direction_flip"))

            return None   # hold

        # No position — open if signal present and no in-flight open
        if tf >= OPT_MIN_TF and direction != 0 and sym not in self._pending_opens:
            self._pending_opens.add(sym)   # mark pending before returning so next bar skips
            return ("open", (sym, direction, spot, equity))

        if tf < OPT_MIN_TF or direction == 0:
            log.debug("OPT SKIP %s: tf=%d dir=%+d", sym, tf, direction)

        return None


# ─────────────────────────────────────────────────────────────────────────────
# LiveTrader
# ─────────────────────────────────────────────────────────────────────────────

class LiveTrader:
    """LARSA v17 multi-asset live trading engine."""

    def __init__(self) -> None:
        self._load_env()
        self._setup_alpaca()
        self._db: sqlite3.Connection = _init_db(_DB_PATH)
        log.info("SQLite DB ready at %s", _DB_PATH)

        # Option overlay
        self._opt_overlay: OptionOverlay = OptionOverlay(self._trading_client)

        # Pending order guard — symbols with an in-flight order; cleared on fill or error
        self._pending_orders: set[str] = set()

        # Instrument state
        self._states: dict[str, InstrumentState] = {
            sym: InstrumentState(sym) for sym in INSTRUMENTS
        }

        # Dynamic correlation
        self._daily_returns: dict[str, deque] = {
            sym: deque(maxlen=30) for sym in INSTRUMENTS
        }
        self._dynamic_corr: float       = CORR_NORMAL
        self._last_daily_px: dict[str, float | None] = {sym: None for sym in INSTRUMENTS}

        # Signal overrides cache
        self._overrides_cache: dict[str, Any]  = {}
        self._overrides_loaded_at: float        = 0.0

        # BTC 200-day EMA for Mayer Multiple
        self._btc_e200: float | None = None

        # Current estimated equity (updated from account API periodically)
        self._equity: float = 100_000.0
        self._equity_updated_at: float = 0.0

        # Price snapshot per symbol for order sizing
        self._last_price: dict[str, float] = {}

        # Set True during _bootstrap_history to suppress order placement
        self._bootstrapping: bool = False

        # Drawdown circuit breaker
        self._equity_peak: float  = 0.0   # populated on first equity fetch
        self._dd_halted:   bool   = False

        log.info("LiveTrader initialised — %d instruments", N_INST)

    # ── Environment / credentials ──────────────────────────────────────────────

    def _load_env(self) -> None:
        if _ENV_FILE.exists():
            for line in _ENV_FILE.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
        self._api_key    = os.environ["ALPACA_API_KEY"]
        self._secret_key = os.environ["ALPACA_SECRET_KEY"]
        self._paper      = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        log.info("Credentials loaded — paper=%s", self._paper)

    # ── Alpaca client setup ────────────────────────────────────────────────────

    def _setup_alpaca(self) -> None:
        from alpaca.trading.client import TradingClient
        from alpaca.data.live.crypto import CryptoDataStream
        from alpaca.data.live.stock import StockDataStream

        self._trading_client = TradingClient(
            api_key    = self._api_key,
            secret_key = self._secret_key,
            paper      = self._paper,
        )
        self._crypto_stream = CryptoDataStream(
            api_key    = self._api_key,
            secret_key = self._secret_key,
        )
        self._stock_stream = StockDataStream(
            api_key    = self._api_key,
            secret_key = self._secret_key,
        )
        # Keep backward-compat alias used elsewhere
        self._stream = self._crypto_stream
        log.info("Alpaca clients created (crypto + stock streams)")

    # ── Signal overrides ───────────────────────────────────────────────────────

    def _load_signal_overrides(self) -> dict[str, Any]:
        """Read and cache signal_overrides.json; refresh every OVERRIDES_TTL_SECS."""
        now = time.monotonic()
        if now - self._overrides_loaded_at < OVERRIDES_TTL_SECS and self._overrides_cache:
            return self._overrides_cache

        if not _OVERRIDES_FILE.exists():
            self._overrides_cache = {}
            self._overrides_loaded_at = now
            return {}

        try:
            raw = json.loads(_OVERRIDES_FILE.read_text())
        except Exception as exc:
            log.warning("Could not read signal_overrides.json: %s", exc)
            return self._overrides_cache

        # Support two formats:
        # Format A (signal_injector.py): keys = multipliers, blocked_hours, sizing_override, expires_at
        # Format B (prompt spec): keys = per-symbol dicts with size_multiplier+expiry, _global
        now_utc = datetime.now(timezone.utc)

        # Detect format by checking for "multipliers" key (format A)
        if "multipliers" in raw or "sizing_override" in raw:
            # Format A — normalize to internal dict
            exp_str = raw.get("expires_at")
            if exp_str:
                try:
                    exp_dt = datetime.fromisoformat(exp_str)
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                    if now_utc > exp_dt:
                        log.info("signal_overrides.json expired — ignoring")
                        self._overrides_cache = {}
                        self._overrides_loaded_at = now
                        return {}
                except Exception:
                    pass
            parsed: dict[str, Any] = {
                "_sizing_override": float(raw.get("sizing_override", 1.0)),
                "_blocked_hours":   set(raw.get("blocked_hours", [])),
                "_per_sym":         {k: float(v) for k, v in raw.get("multipliers", {}).items()},
            }
        else:
            # Format B — per-symbol keys
            parsed = {"_sizing_override": 1.0, "_blocked_hours": set(), "_per_sym": {}}
            for key, val in raw.items():
                if not isinstance(val, dict):
                    continue
                exp_str = val.get("expiry") or val.get("expires_at")
                if exp_str:
                    try:
                        exp_dt = datetime.fromisoformat(exp_str)
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                        if now_utc > exp_dt:
                            continue   # expired
                    except Exception:
                        pass
                mult = float(val.get("size_multiplier", 1.0))
                if key == "_global":
                    parsed["_sizing_override"] = mult
                    parsed["_blocked_hours"] = set(val.get("blocked_hours", []))
                else:
                    parsed["_per_sym"][key] = mult

        self._overrides_cache     = parsed
        self._overrides_loaded_at = now
        log.debug("Loaded signal overrides: %s", parsed)
        return parsed

    def _apply_signal_overrides(self, targets: dict[str, float]) -> dict[str, float]:
        """Apply loaded override multipliers to a targets dict (in-place, returns same dict)."""
        ov = self._load_signal_overrides()
        if not ov:
            return targets
        global_mult = ov.get("_sizing_override", 1.0)
        per_sym     = ov.get("_per_sym", {})
        for sym in list(targets.keys()):
            mult = global_mult * per_sym.get(sym, 1.0)
            targets[sym] = targets[sym] * mult
        return targets

    def _effective_blocked_hours(self) -> set[int]:
        ov = self._load_signal_overrides()
        extra = ov.get("_blocked_hours", set()) if ov else set()
        return BLOCKED_ENTRY_HOURS_UTC | set(extra)

    # ── Dynamic correlation ────────────────────────────────────────────────────

    def _on_daily_close(self, sym: str, close: float) -> None:
        """Call when a daily bar closes for a symbol."""
        prev = self._last_daily_px.get(sym)
        if prev and prev > 0:
            ret = math.log(close / prev)
            self._daily_returns[sym].append(ret)
        self._last_daily_px[sym] = close
        # Update BTC 200-day EMA
        if sym == "BTC":
            self._btc_e200 = _ema(self._btc_e200, close, _alpha(200))
        self._recompute_dynamic_corr()

    def _recompute_dynamic_corr(self) -> None:
        """Recompute average pairwise correlation from rolling 30 daily returns."""
        active = [list(self._daily_returns[s]) for s in INSTRUMENTS
                  if len(self._daily_returns[s]) >= 30]
        if len(active) < 2:
            return
        mat  = np.array(active)
        corr = np.corrcoef(mat)
        n    = corr.shape[0]
        avg  = (np.sum(corr) - n) / (n * (n - 1)) if n > 1 else 0.0
        self._dynamic_corr = CORR_STRESS if avg > CORR_STRESS_THRESHOLD else CORR_NORMAL
        log.info("Dynamic CORR updated: avg_pair=%.3f → CORR=%.2f", avg, self._dynamic_corr)

    # ── Bar processing ─────────────────────────────────────────────────────────

    def _ticker_to_sym(self, ticker: str) -> str | None:
        for sym, cfg in INSTRUMENTS.items():
            if cfg["ticker"] == ticker:
                return sym
        return None

    def on_bar(self, ticker: str, bar: Any) -> None:
        """
        Process an incoming 15-minute bar.
        Aggregates up to 1h and 4h, updates all indicators, recomputes targets.
        """
        sym = self._ticker_to_sym(ticker)
        if sym is None:
            return

        o  = float(bar.open)
        h  = float(bar.high)
        l  = float(bar.low)
        c  = float(bar.close)
        ts: datetime = bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        st = self._states[sym]
        st._bar_count += 1
        st.last_15m_px = c
        self._last_price[sym] = c

        # Update 15m BH
        st.bh_15m.update(c)

        # GARCH update — always use single-bar log return (prev close → this close)
        if st._prev_close and st._prev_close > 0:
            st.garch.update(math.log(c / st._prev_close))
        st._prev_close = c

        # OU update
        st.ou.update(c)

        # Aggregate to 1h (4 x 15m bars)
        st._h1_buf.append({"o": o, "h": h, "l": l, "c": c, "ts": ts})
        h1_bucket = ts.replace(minute=0, second=0, microsecond=0)
        if st._last_h1_close != h1_bucket:
            # Flush completed hour
            if len(st._h1_buf) >= 4 or st._last_h1_close is not None:
                self._flush_1h(sym)
            st._last_h1_close = h1_bucket
            st._h1_buf = [{"o": o, "h": h, "l": l, "c": c, "ts": ts}]

        # Aggregate to 4h (4 x 1h)
        h4_bucket = ts.replace(hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0)
        st._h4_buf.append({"o": o, "h": h, "l": l, "c": c, "ts": ts})
        if st._last_h4_close != h4_bucket:
            if st._last_h4_close is not None and len(st._h4_buf) >= 16:
                self._flush_4h(sym)
            st._last_h4_close = h4_bucket
            st._h4_buf = [{"o": o, "h": h, "l": l, "c": c, "ts": ts}]

        # Warmup: require at least 30 bars
        if st._bar_count >= 30:
            st.warmup_done = True

        # Check daily rollover (new UTC day)
        if ts.hour == 0 and ts.minute == 0:
            self._on_daily_close(sym, c)

        # BullScale update (daily proxy using close)
        st.bull.update(c)

        # Compute and act on targets
        self._act_on_targets(ts)

    def _flush_1h(self, sym: str) -> None:
        st  = self._states[sym]
        buf = st._h1_buf
        if not buf:
            return
        h1h = max(b["h"] for b in buf)
        h1l = min(b["l"] for b in buf)
        h1c = buf[-1]["c"]
        st.bh_1h.update(h1c)
        st.atr_1h.update(h1h, h1l, h1c)

    def _flush_4h(self, sym: str) -> None:
        st  = self._states[sym]
        buf = st._h4_buf
        if not buf:
            return
        h4h = max(b["h"] for b in buf)
        h4l = min(b["l"] for b in buf)
        h4c = buf[-1]["c"]
        st.bh_4h.update(h4c)
        st.atr_4h.update(h4h, h4l, h4c)

    # ── Target computation ─────────────────────────────────────────────────────

    def compute_targets(self, bar_time: datetime) -> dict[str, float]:
        """
        Full LARSA v16 target computation.
        Returns {sym: target_frac} with all rules applied.
        """
        n_eff        = N_INST
        corr_factor  = math.sqrt(n_eff + n_eff * (n_eff - 1) * self._dynamic_corr)
        per_inst_risk = DAILY_RISK / corr_factor

        bar_hour = bar_time.hour
        blocked  = bar_hour in self._effective_blocked_hours()
        boosted  = bar_hour in BOOST_ENTRY_HOURS_UTC

        # BTC lead boost check
        btc      = self._states["BTC"]
        btc_lead = btc.bh_4h.active and btc.bh_1h.active

        raw: dict[str, float] = {}

        is_rth_now = _is_rth(bar_time)

        for sym, st in self._states.items():
            asset_class = INSTRUMENTS[sym].get("asset_class", "crypto")

            # Equity instruments only trade during RTH
            if asset_class == "equity" and not is_rth_now:
                raw[sym] = 0.0
                continue

            if not st.warmup_done:
                raw[sym] = 0.0
                continue

            d_active = st.bh_4h.active
            h_active = st.bh_1h.active
            m_active = st.bh_15m.active
            tf       = (4 if d_active else 0) + (2 if h_active else 0) + (1 if m_active else 0)

            if asset_class == "equity":
                ceiling = min(TF_CAP.get(tf, 0.0), EQUITY_CAP_FRAC)
            else:
                ceiling = min(TF_CAP.get(tf, 0.0), CRYPTO_CAP_FRAC)

            if ceiling == 0.0:
                raw[sym] = 0.0
                continue

            # Direction — prefer 1h over 4h
            direction = 0
            if h_active and st.bh_1h.bh_dir:
                direction = st.bh_1h.bh_dir
            elif d_active and st.bh_4h.bh_dir:
                direction = st.bh_4h.bh_dir

            # Crypto: long-only.  Equities: long/short allowed.
            if asset_class == "crypto" and direction <= 0:
                raw[sym] = 0.0
                continue
            if direction == 0:
                raw[sym] = 0.0
                continue

            # Vol-adjusted base size
            atr = st.atr_1h.atr or st.atr_4h.atr
            cp  = self._last_price.get(sym, 1.0)
            vol = (atr / cp * math.sqrt(6.5)) if (atr and cp > 0) else 0.01
            base  = min(per_inst_risk / (vol + 1e-9), min(ceiling, DELTA_MAX_FRAC))
            raw[sym] = base * st.garch.vol_scale * direction

        # Mayer Multiple dampener — crypto only
        mayer_damp = 1.0
        btc_px = self._last_price.get("BTC", 0.0)
        if self._btc_e200 and self._btc_e200 > 0 and btc_px > 0:
            mayer = btc_px / self._btc_e200
            if mayer > 2.4:
                mayer_damp = max(0.5, 1.0 - (mayer - 2.4) / 2.2)
            elif mayer < 1.0:
                mayer_damp = min(1.2, 1.0 + (1.0 - mayer) * 0.3)
        for sym in raw:
            if INSTRUMENTS[sym].get("asset_class", "crypto") == "crypto":
                raw[sym] = raw.get(sym, 0.0) * mayer_damp

        # BTC cross-asset lead: 1.4x boost on other crypto only
        for sym, st in self._states.items():
            if sym == "BTC":
                continue
            if INSTRUMENTS[sym].get("asset_class", "crypto") != "crypto":
                continue
            if btc_lead and raw.get(sym, 0.0) > 0:
                raw[sym] *= 1.4

        # Hour boost — new entries only
        if boosted:
            for sym, st in self._states.items():
                if raw.get(sym, 0.0) > 0 and math.isclose(st.last_frac, 0.0):
                    raw[sym] *= HOUR_BOOST_MULTIPLIER

        # Blocked hours — suppress new entries
        if blocked:
            for sym, st in self._states.items():
                if math.isclose(st.last_frac, 0.0):
                    raw[sym] = 0.0

        # Stale 15m exit
        for sym, st in self._states.items():
            if raw.get(sym, 0.0) == 0.0:
                continue
            ep   = st.entry_px
            cp   = self._last_price.get(sym)
            px15 = st.last_15m_px
            if ep and cp and px15 and ep > 0 and px15 > 0:
                pnl_pct = (cp - ep) / ep
                move15  = abs(cp - px15) / px15
                if pnl_pct < 0 and move15 < STALE_15M_MOVE:
                    raw[sym] = 0.0
                elif pnl_pct > WINNER_PROTECTION_PCT:
                    raw[sym] = max(abs(raw[sym]), abs(st.last_frac)) * math.copysign(1, raw[sym])

        # OU mean reversion
        for sym, st in self._states.items():
            if not st.warmup_done:
                continue
            if st.bh_4h.active or st.bh_1h.active:
                continue
            if sym in OU_DISABLED_SYMS:
                continue
            if st.ou.long_signal and st.ou_pos <= 0:
                raw[sym] = raw.get(sym, 0.0) + OU_FRAC
                st.ou_pos = OU_FRAC
            elif st.ou.exit_signal and st.ou_pos > 0:
                st.ou_pos = 0.0
            elif st.ou.short_signal:
                st.ou_pos = 0.0

        # Position floor
        for sym, st in self._states.items():
            tgt      = raw.get(sym, 0.0)
            d_active = st.bh_4h.active
            h_active = st.bh_1h.active
            m_active = st.bh_15m.active
            tf       = (4 if d_active else 0) + (2 if h_active else 0) + (1 if m_active else 0)
            if tf >= 6 and abs(tgt) > 0.15 and st.bh_1h.ctl >= 5:
                st.pos_floor = max(st.pos_floor, 0.70 * abs(tgt))
            if st.pos_floor > 0 and tf >= 4 and not math.isclose(st.last_frac, 0.0):
                raw[sym] = math.copysign(max(abs(tgt), st.pos_floor), st.last_frac)
                st.pos_floor *= 0.95
            if tf < 4 or math.isclose(tgt, 0.0):
                st.pos_floor = 0.0
            if not d_active and not h_active:
                st.pos_floor = 0.0

        # Normalize if sum > 1
        total = sum(abs(v) for v in raw.values())
        scale = 1.0 / total if total > 1.0 else 1.0
        for sym in raw:
            raw[sym] *= scale

        # Apply signal overrides
        self._apply_signal_overrides(raw)

        return raw

    # ── Execution ──────────────────────────────────────────────────────────────

    def _act_on_targets(self, bar_time: datetime) -> None:
        """Compare targets to current positions, place orders for meaningful changes."""
        if self._bootstrapping:
            return   # indicator-only replay — no orders, no state mutation
        targets = self.compute_targets(bar_time)
        equity  = self._get_equity()

        # ── Drawdown circuit breaker ──────────────────────────────────────────
        if equity > self._equity_peak:
            self._equity_peak = equity
        if self._equity_peak > 0:
            dd = (self._equity_peak - equity) / self._equity_peak
            if not self._dd_halted and dd >= DD_HALT_PCT:
                self._dd_halted = True
                log.warning("CIRCUIT BREAKER: drawdown %.1f%% >= %.1f%% — halting new entries", dd*100, DD_HALT_PCT*100)
            elif self._dd_halted and dd <= DD_RESUME_PCT:
                self._dd_halted = False
                log.info("CIRCUIT BREAKER: drawdown recovered to %.1f%% — resuming", dd*100)

        # ── Option overlay ────────────────────────────────────────────────────
        for sym in list(targets.keys()):
            if not (INSTRUMENTS[sym].get("options_overlay") and
                    INSTRUMENTS[sym].get("asset_class") == "equity"):
                continue
            st        = self._states[sym]
            d_active  = st.bh_4h.active
            h_active  = st.bh_1h.active
            tf        = (4 if d_active else 0) + (2 if h_active else 0) + (1 if st.bh_15m.active else 0)
            direction = st.bh_1h.bh_dir or st.bh_4h.bh_dir
            spot      = self._last_price.get(sym, 0.0)
            if spot <= 0:
                continue
            action = self._opt_overlay.update(sym, tf, direction, spot, equity, bar_time)
            if action is None:
                continue
            kind, args = action
            if kind == "open":
                asyncio.create_task(asyncio.to_thread(self._opt_overlay._submit_open, *args))
            elif kind == "close":
                asyncio.create_task(asyncio.to_thread(self._opt_overlay._submit_close, *args))

        # ── Equity / crypto positions ─────────────────────────────────────────
        # Build order list; process SELLS before BUYS so proceeds free up cash
        now_utc = datetime.now(timezone.utc)

        order_items = []
        for sym, tgt_frac in targets.items():
            st = self._states[sym]
            if sym in self._pending_orders:
                continue

            # Time-based min-hold gate: no exit/reversal within MIN_HOLD_MINUTES
            held_minutes = 0.0
            if st.entry_time is not None:
                held_minutes = (now_utc - st.entry_time).total_seconds() / 60.0

            is_exit    = math.isclose(tgt_frac, 0.0) and not math.isclose(st.last_frac, 0.0)
            is_reversal = (not math.isclose(st.last_frac, 0.0) and
                           not math.isclose(tgt_frac, 0.0) and
                           math.copysign(1, tgt_frac) != math.copysign(1, st.last_frac))

            if (is_exit or is_reversal) and held_minutes < MIN_HOLD_MINUTES:
                continue   # hold — too soon to exit/reverse

            # Circuit breaker: no new entries when halted (exits/reduces still allowed)
            is_new_entry = math.isclose(st.last_frac, 0.0) and not math.isclose(tgt_frac, 0.0)
            if self._dd_halted and is_new_entry:
                continue

            delta = tgt_frac - st.last_frac
            if abs(delta) < MIN_TRADE_FRAC:
                continue
            cp = self._last_price.get(sym)
            if not cp or cp <= 0:
                log.warning("%s: no price — skipping order", sym)
                continue
            qty = delta * equity / cp
            # For sells, cap at 99.9% of held notional to avoid rounding over-sells
            if qty < 0:
                max_sell = abs(self._states[sym].last_frac) * equity / cp * 0.999
                qty = -min(abs(qty), max_sell)
            if abs(qty * cp) < 1.0:
                continue
            order_items.append((sym, tgt_frac, qty, cp))

        # Sells first (qty < 0), then buys
        order_items.sort(key=lambda x: x[2])   # ascending: negatives first

        for sym, tgt_frac, qty, cp in order_items:
            side = "buy" if qty > 0 else "sell"
            log.info(
                "%s %s %.6f units @ ~$%.2f  (frac %.3f→%.3f)",
                sym, side.upper(), abs(qty), cp, self._states[sym].last_frac, tgt_frac,
            )
            self._pending_orders.add(sym)
            asyncio.create_task(
                self._place_order_async(sym, side, abs(qty), cp, tgt_frac)
            )

    async def _place_order_async(
        self, sym: str, side: str, qty: float, price: float, new_frac: float
    ) -> None:
        """Place order(s) in a thread; always clears the pending guard when done."""
        try:
            await asyncio.to_thread(
                self._place_order, sym, side, qty, price, new_frac
            )
        except Exception as exc:
            log.error("%s order error: %s", sym, exc, exc_info=True)
        finally:
            self._pending_orders.discard(sym)

    def _place_order(
        self, sym: str, side: str, qty: float, price: float, new_frac: float
    ) -> None:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        ticker      = INSTRUMENTS[sym]["ticker"]
        asset_class = INSTRUMENTS[sym].get("asset_class", "crypto")
        max_slice   = MAX_ORDER_NOTIONAL
        remaining   = qty
        any_filled  = False

        # Equities: DAY TIF, 2dp fractional.  Crypto: GTC, 8dp fractional.
        if asset_class == "equity":
            tif       = TimeInForce.DAY
            qty_round = 2
            min_qty   = 0.01
        else:
            tif       = TimeInForce.GTC
            qty_round = 8
            min_qty   = 1e-7

        order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL

        while remaining > min_qty:
            slice_qty = min(remaining, max_slice / price)
            rounded   = round(slice_qty, qty_round)

            # Guard: rounding could produce 0 → break to avoid infinite loop
            if rounded <= 0:
                break

            slice_notional = rounded * price
            # Alpaca minimum notional check ($1)
            if slice_notional < 1.0:
                break

            req = MarketOrderRequest(
                symbol        = ticker,
                qty           = rounded,
                side          = order_side,
                time_in_force = tif,
            )
            try:
                resp = self._trading_client.submit_order(req)
                log.info(
                    "%s submitted: id=%s %s %.6f units notional=$%.0f",
                    sym, resp.id, side.upper(), float(rounded), float(slice_notional),
                )
                any_filled = True
            except Exception as exc:
                log.error("%s submit_order failed: %s", sym, exc)
                break

            remaining -= slice_qty

        # Only update state if at least one slice was accepted by the broker.
        # This prevents phantom positions from failed orders.
        if any_filled:
            st = self._states[sym]
            st.last_frac = new_frac
            if side == "buy":
                st.entry_px   = price
                st.entry_time = datetime.now(timezone.utc)
            elif math.isclose(new_frac, 0.0):
                st.entry_px   = None
                st.entry_time = None

    # ── Fill handling ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_side(raw_side: Any) -> str:
        """Normalize Alpaca OrderSide enum or string to 'buy' or 'sell'."""
        if hasattr(raw_side, "value"):          # OrderSide.BUY → "buy"
            return str(raw_side.value).lower()
        s = str(raw_side).lower()
        if "buy" in s:
            return "buy"
        if "sell" in s:
            return "sell"
        return s

    def _on_fill(self, fill_event: Any) -> None:
        """
        Handle a fill / trade_update event from Alpaca.
        Logs to live_trades and updates trade_pnl via FIFO matching.
        Handles equity, crypto, and option contract fills.
        """
        try:
            order  = fill_event.order if hasattr(fill_event, "order") else fill_event
            ticker = str(order.symbol)

            # Route: try equity/crypto lookup first, then option contract map
            sym = self._ticker_to_sym(ticker)
            if sym is None:
                sym = self._opt_overlay.contract_to_underlying.get(ticker)
            if sym is None:
                sym = ticker   # unknown — log it anyway

            side  = self._parse_side(order.side)
            qty   = float(order.filled_qty or 0)
            price = float(order.filled_avg_price or 0)
            if qty <= 0 or price <= 0:
                return
            notional  = qty * price
            order_id  = str(order.id) if hasattr(order, "id") else None
            fill_time = datetime.now(timezone.utc).isoformat()

            # Write to live_trades
            self._db.execute(
                """INSERT INTO live_trades
                   (symbol, side, qty, price, notional, fill_time, order_id, strategy_version)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (sym, side, qty, price, notional, fill_time, order_id, STRATEGY_VERSION),
            )
            self._db.commit()
            log.info("Fill logged: %s %s %.6f @ $%.4f  notional=$%.0f",
                     sym, side.upper(), qty, price, notional)

            # FIFO P&L tracking
            st = self._states.get(sym)
            if st is None:
                return

            if side == "buy":
                st._fifo.append((qty, price, fill_time))
                log.debug("%s FIFO push %.6f @ %.4f", sym, qty, price)
            elif side == "sell":
                remaining = qty
                while remaining > 1e-8 and st._fifo:
                    entry_qty, entry_price, entry_time = st._fifo[0]
                    matched = min(remaining, entry_qty)
                    pnl     = matched * (price - entry_price)
                    hold_mins = 0
                    if st.entry_time is not None:
                        hold_mins = int((datetime.now(timezone.utc) - st.entry_time).total_seconds() / 60)
                    self._db.execute(
                        """INSERT INTO trade_pnl
                           (symbol, entry_time, exit_time, entry_price, exit_price, qty, pnl, hold_bars)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (sym, entry_time, fill_time, entry_price, price, matched, round(pnl, 6), hold_mins),
                    )
                    self._db.commit()
                    log.info("P&L: %s  entry=%.4f exit=%.4f qty=%.6f pnl=$%.2f",
                             sym, entry_price, price, matched, pnl)
                    if matched >= entry_qty:
                        st._fifo.pop(0)
                    else:
                        st._fifo[0] = (entry_qty - matched, entry_price, entry_time)
                    remaining -= matched

        except Exception as exc:
            log.error("_on_fill error: %s", exc, exc_info=True)

    # ── Equity refresh ─────────────────────────────────────────────────────────

    def _get_equity(self) -> float:
        now = time.monotonic()
        if now - self._equity_updated_at > 60:
            try:
                acct = self._trading_client.get_account()
                self._equity = float(acct.equity)
                self._equity_updated_at = now
                log.info("Equity refreshed: $%.2f", self._equity)
            except Exception as exc:
                log.warning("Could not fetch equity: %s", exc)
        return self._equity

    # ── Stream entry point ─────────────────────────────────────────────────────

    async def _run_stream(self) -> None:
        from alpaca.trading.stream import TradingStream

        crypto_tickers = [cfg["ticker"] for sym, cfg in INSTRUMENTS.items()
                          if cfg.get("asset_class", "crypto") == "crypto"]
        equity_tickers = [cfg["ticker"] for sym, cfg in INSTRUMENTS.items()
                          if cfg.get("asset_class") == "equity"]

        # ── Shared 15-minute bar handler (works for both crypto and equity bars)
        async def bar_handler(bar: Any) -> None:
            try:
                self.on_bar(bar.symbol, bar)
            except Exception as exc:
                log.error("bar_handler error [%s]: %s", bar.symbol, exc, exc_info=True)

        # Subscribe crypto bars
        self._crypto_stream.subscribe_bars(bar_handler, *crypto_tickers)
        log.info("Subscribed to %d crypto bar feeds", len(crypto_tickers))

        # Subscribe equity bars
        if equity_tickers:
            self._stock_stream.subscribe_bars(bar_handler, *equity_tickers)
            log.info("Subscribed to %d equity bar feeds", len(equity_tickers))

        # ── Trade update handler (fills)
        trading_stream = TradingStream(
            api_key    = self._api_key,
            secret_key = self._secret_key,
            paper      = self._paper,
        )

        async def trade_update_handler(event: Any) -> None:
            try:
                event_type = str(getattr(event, "event", "")).lower()
                if event_type in ("fill", "partial_fill"):
                    self._on_fill(event)
            except Exception as exc:
                log.error("trade_update_handler error: %s", exc, exc_info=True)

        trading_stream.subscribe_trade_updates(trade_update_handler)
        log.info("Subscribed to trade_updates")

        # Run all three streams concurrently
        await asyncio.gather(
            self._crypto_stream._run_forever(),
            self._stock_stream._run_forever(),
            trading_stream._run_forever(),
        )

    # ── Historical bootstrap ───────────────────────────────────────────────────

    def _bootstrap_history(self) -> None:
        """
        Pre-warm BH/ATR/GARCH states by replaying the last 250 hourly bars
        from Alpaca's historical API before the live stream starts.

        Without this, bh_1h never activates (needs ~105 capture bars) and
        compute_targets always returns direction=0, so no trades ever fire.
        """
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.historical.stock  import StockHistoricalDataClient
        from alpaca.data.requests import (
            CryptoBarsRequest,
            StockBarsRequest,
        )
        from alpaca.data.timeframe import TimeFrame
        import pandas as pd

        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=15)   # 15d × 24h = 360 hourly bars > 250

        crypto_client = CryptoHistoricalDataClient(
            api_key    = self._api_key,
            secret_key = self._secret_key,
        )
        stock_client = StockHistoricalDataClient(
            api_key    = self._api_key,
            secret_key = self._secret_key,
        )

        crypto_tickers = [cfg["ticker"] for sym, cfg in INSTRUMENTS.items()
                          if cfg.get("asset_class", "crypto") == "crypto"]
        equity_tickers = [cfg["ticker"] for sym, cfg in INSTRUMENTS.items()
                          if cfg.get("asset_class") == "equity"]

        bars_by_ticker: dict[str, list] = {}

        def _iter_barset(resp) -> dict:
            """BarSet can be dict-like or have a .data attr — handle both."""
            if hasattr(resp, "data"):
                return resp.data
            if hasattr(resp, "items"):
                return dict(resp.items())
            return {}

        # Fetch crypto hourly bars
        try:
            resp = crypto_client.get_crypto_bars(
                CryptoBarsRequest(
                    symbol_or_symbols = crypto_tickers,
                    timeframe         = TimeFrame.Hour,
                    start             = start,
                    end               = end,
                )
            )
            for ticker, bar_list in _iter_barset(resp).items():
                bars_by_ticker[ticker] = sorted(bar_list, key=lambda b: b.timestamp)
            log.info("Bootstrap: fetched hourly bars for %d crypto tickers", len(bars_by_ticker))
        except Exception as exc:
            log.warning("Bootstrap crypto fetch failed: %s", exc)

        # Fetch equity hourly bars (IEX feed — available on free plan)
        try:
            resp = stock_client.get_stock_bars(
                StockBarsRequest(
                    symbol_or_symbols = equity_tickers,
                    timeframe         = TimeFrame.Hour,
                    start             = start,
                    end               = end,
                    feed              = "iex",
                )
            )
            n = 0
            for ticker, bar_list in _iter_barset(resp).items():
                bars_by_ticker[ticker] = sorted(bar_list, key=lambda b: b.timestamp)
                n += 1
            log.info("Bootstrap: fetched hourly bars for %d equity tickers", n)
        except Exception as exc:
            log.warning("Bootstrap equity fetch failed: %s", exc)

        if not bars_by_ticker:
            log.warning("Bootstrap: no historical bars fetched — BH states will warm up slowly")
            return

        # Merge all bars across all tickers into chronological order and replay
        all_bars: list[tuple] = []
        for ticker, bar_list in bars_by_ticker.items():
            for b in bar_list:
                all_bars.append((b.timestamp, ticker, b))
        all_bars.sort(key=lambda x: x[0])

        self._bootstrapping = True
        n_replayed = 0
        for ts, ticker, bar in all_bars:
            sym = self._ticker_to_sym(ticker)
            if sym is None:
                continue
            try:
                self.on_bar(ticker, bar)
                n_replayed += 1
            except Exception as exc:
                log.debug("Bootstrap bar error [%s]: %s", ticker, exc)
        self._bootstrapping = False

        # Mark all states as warmup_done after bootstrap
        active_count = 0
        for sym, st in self._states.items():
            st.warmup_done = True
            if st.bh_1h.active or st.bh_4h.active:
                active_count += 1

        log.info(
            "Bootstrap complete — %d bars replayed, %d/%d instruments have active 1h/4h BH signal",
            n_replayed, active_count, N_INST,
        )

        # Diagnostic: show equity signal states after bootstrap
        for sym, st in self._states.items():
            if INSTRUMENTS[sym].get("asset_class") != "equity":
                continue
            tf = (4 if st.bh_4h.active else 0) + (2 if st.bh_1h.active else 0) + (1 if st.bh_15m.active else 0)
            direction = st.bh_1h.bh_dir or st.bh_4h.bh_dir
            px = self._last_price.get(sym, 0.0)
            log.info(
                "EQUITY STATE: %s  tf=%d  dir=%+d  1h_active=%s  4h_active=%s  spot=%.2f",
                sym, tf, direction, st.bh_1h.active, st.bh_4h.active, px,
            )

        self._sync_positions_from_broker()

    def _sync_positions_from_broker(self) -> None:
        """
        Pull current broker positions and set last_frac / entry_px so the
        strategy knows what it already owns.  Without this, last_frac stays 0
        after a restart and the strategy never generates SELLs for existing
        positions — it just tries to buy more and hits insufficient buying power.
        """
        try:
            positions = self._trading_client.get_all_positions()
        except Exception as exc:
            log.warning("Could not fetch broker positions for sync: %s", exc)
            return

        if not positions:
            log.info("Position sync: no open positions at broker")
            return

        try:
            equity = float(self._trading_client.get_account().equity)
        except Exception:
            equity = self._equity

        synced = 0
        for pos in positions:
            raw_ticker = pos.symbol      # Alpaca may return "BTCUSD" or "BTC/USD"
            # Normalise: if no slash and ends with USD/USDT, try inserting slash
            if "/" not in raw_ticker and raw_ticker.endswith("USD"):
                base   = raw_ticker[:-3]
                ticker = f"{base}/USD"
            else:
                ticker = raw_ticker
            sym = self._ticker_to_sym(ticker)
            if sym is None:
                sym = self._ticker_to_sym(raw_ticker)   # fallback: try as-is
            if sym is None:
                continue
            st = self._states[sym]
            mkt_val  = float(pos.market_value)   # signed (neg for short)
            avg_px   = float(pos.avg_entry_price)
            frac     = mkt_val / equity if equity > 0 else 0.0
            st.last_frac  = frac
            st.entry_px   = avg_px
            st.dollar_pos = mkt_val
            # Treat synced positions as having just entered so MIN_HOLD_MINUTES applies
            st.entry_time = datetime.now(timezone.utc)
            if self._last_price.get(sym) is None:
                st._prev_close = avg_px
                self._last_price[sym] = avg_px
            synced += 1
            log.info(
                "Position sync: %s  frac=%.3f  avg_px=%.4f  mkt_val=$%.0f",
                sym, frac, avg_px, mkt_val,
            )

        log.info("Position sync complete — %d positions loaded", synced)

    def run(self) -> None:
        """Main entry point — blocks until interrupted."""
        log.info("LARSA v17 LiveTrader starting (strategy_version=%s)", STRATEGY_VERSION)
        self._bootstrap_history()
        while True:
            try:
                asyncio.run(self._run_stream())
            except KeyboardInterrupt:
                log.info("Shutdown requested — exiting.")
                break
            except Exception as exc:
                log.error("Stream crashed: %s — reconnecting in 10s", exc, exc_info=True)
                time.sleep(10)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()
