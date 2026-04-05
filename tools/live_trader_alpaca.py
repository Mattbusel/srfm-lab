"""
tools/live_trader_alpaca.py
===========================
LARSA v16 — Live Crypto Trader (Alpaca Paper)

Streams 15-minute bars for a universe of crypto instruments, runs
Black-Hole physics + GARCH vol scaling + OU mean reversion detection
across three timeframes (15m, 1h, 4h), computes target fractional
positions, and places orders when targets change meaningfully.

New features:
  • Reads bridge/signal_overrides.json every 5 minutes and applies
    per-symbol and global size multipliers / blocked hour overrides.
  • Logs every fill to execution/live_trades.db (SQLite) and maintains
    a trade P&L table with FIFO matching.

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

STRATEGY_VERSION = "larsa_v16"

# ── Strategy constants ─────────────────────────────────────────────────────────
INSTRUMENTS = {
    "BTC":   {"ticker": "BTC/USD",   "cf_4h": 0.016, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "ETH":   {"ticker": "ETH/USD",   "cf_4h": 0.012, "cf_15m": 0.007, "cf_1h": 0.020, "cf_1d": 0.07},
    "XRP":   {"ticker": "XRP/USD",   "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "AVAX":  {"ticker": "AVAX/USD",  "cf_4h": 0.010, "cf_15m": 0.006, "cf_1h": 0.018, "cf_1d": 0.06},
    "LINK":  {"ticker": "LINK/USD",  "cf_4h": 0.010, "cf_15m": 0.006, "cf_1h": 0.018, "cf_1d": 0.06},
    "DOT":   {"ticker": "DOT/USD",   "cf_4h": 0.010, "cf_15m": 0.006, "cf_1h": 0.018, "cf_1d": 0.06},
    "UNI":   {"ticker": "UNI/USD",   "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "AAVE":  {"ticker": "AAVE/USD",  "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "LTC":   {"ticker": "LTC/USD",   "cf_4h": 0.018, "cf_15m": 0.010, "cf_1h": 0.030, "cf_1d": 0.10},
    "BCH":   {"ticker": "BCH/USD",   "cf_4h": 0.020, "cf_15m": 0.012, "cf_1h": 0.035, "cf_1d": 0.12},
    "DOGE":  {"ticker": "DOGE/USD",  "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "SHIB":  {"ticker": "SHIB/USD",  "cf_4h": 0.035, "cf_15m": 0.025, "cf_1h": 0.075, "cf_1d": 0.25},
    "BAT":   {"ticker": "BAT/USD",   "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "CRV":   {"ticker": "CRV/USD",   "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "SUSHI": {"ticker": "SUSHI/USD", "cf_4h": 0.030, "cf_15m": 0.020, "cf_1h": 0.060, "cf_1d": 0.20},
    "MKR":   {"ticker": "MKR/USD",   "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
    "YFI":   {"ticker": "YFI/USD",   "cf_4h": 0.022, "cf_15m": 0.015, "cf_1h": 0.045, "cf_1d": 0.15},
}
N_INST = len(INSTRUMENTS)

BH_FORM     = 1.92
BH_CTL_MIN  = 3
BH_DECAY    = 0.924
BH_COLLAPSE = 0.992
MIN_HOLD    = 8

DAILY_RISK              = 0.05
CORR_NORMAL             = 0.25
CORR_STRESS             = 0.60
CORR_STRESS_THRESHOLD   = 0.60
CRYPTO_CAP_FRAC         = 0.40
MIN_TRADE_FRAC          = 0.003
GARCH_TARGET_VOL        = 0.90
OU_FRAC                 = 0.08
DELTA_MAX_FRAC          = 0.40
STALE_15M_MOVE          = 0.008
WINNER_PROTECTION_PCT   = 0.005
BLOCKED_ENTRY_HOURS_UTC = {1, 13, 14, 15, 17, 18}
BOOST_ENTRY_HOURS_UTC   = {3, 9, 16, 19}
HOUR_BOOST_MULTIPLIER   = 1.25
OU_DISABLED_SYMS        = {"AVAX", "DOT", "LINK"}

TF_CAP = {7: 1.0, 6: 1.0, 4: 0.60, 3: 0.50, 2: 0.40, 1: 0.20, 0: 0.0}

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

    def __init__(self, window: int = 50, entry_z: float = 1.5, exit_z: float = 0.3) -> None:
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
        self.last_frac: float     = 0.0
        self.bars_held: int       = 0
        self.ou_pos: float        = 0.0
        self.pos_floor: float     = 0.0
        self.entry_px: float | None = None
        self.last_15m_px: float | None = None
        self.warmup_done: bool    = False
        self._bar_count: int      = 0
        # FIFO queue for P&L tracking: list of (qty, price, time)
        self._fifo: list[tuple[float, float, str]] = []
        self.hold_bars: int = 0


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
            strategy_version TEXT    DEFAULT 'larsa_v16'
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
# LiveTrader
# ─────────────────────────────────────────────────────────────────────────────

class LiveTrader:
    """LARSA v16 live trading engine."""

    def __init__(self) -> None:
        self._load_env()
        self._setup_alpaca()
        self._db: sqlite3.Connection = _init_db(_DB_PATH)
        log.info("SQLite DB ready at %s", _DB_PATH)

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

        self._trading_client = TradingClient(
            api_key    = self._api_key,
            secret_key = self._secret_key,
            paper      = self._paper,
        )
        self._stream = CryptoDataStream(
            api_key    = self._api_key,
            secret_key = self._secret_key,
        )
        log.info("Alpaca clients created")

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

        # GARCH update
        if st.entry_px and st.entry_px > 0:
            st.garch.update(math.log(c / st.entry_px))
        elif len(st.bh_15m.prices) >= 2:
            px = list(st.bh_15m.prices)
            st.garch.update(math.log(px[-1] / (px[-2] + 1e-12)))

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

        for sym, st in self._states.items():
            if not st.warmup_done:
                raw[sym] = 0.0
                continue

            d_active = st.bh_4h.active
            h_active = st.bh_1h.active
            m_active = st.bh_15m.active
            tf       = (4 if d_active else 0) + (2 if h_active else 0) + (1 if m_active else 0)
            ceiling  = min(TF_CAP.get(tf, 0.0), CRYPTO_CAP_FRAC)

            if ceiling == 0.0:
                raw[sym] = 0.0
                continue

            # Direction — prefer 1h over 4h
            direction = 0
            if h_active and st.bh_1h.bh_dir:
                direction = st.bh_1h.bh_dir
            elif d_active and st.bh_4h.bh_dir:
                direction = st.bh_4h.bh_dir

            # Long-only: skip bearish
            if direction <= 0:
                raw[sym] = 0.0
                continue

            # Vol-adjusted base size
            atr = st.atr_1h.atr or st.atr_4h.atr
            cp  = self._last_price.get(sym, 1.0)
            vol = (atr / cp * math.sqrt(6.5)) if (atr and cp > 0) else 0.01
            base  = min(per_inst_risk / (vol + 1e-9), min(ceiling, DELTA_MAX_FRAC))
            raw[sym] = base * st.garch.vol_scale

        # Mayer Multiple dampener
        mayer_damp = 1.0
        btc_px = self._last_price.get("BTC", 0.0)
        if self._btc_e200 and self._btc_e200 > 0 and btc_px > 0:
            mayer = btc_px / self._btc_e200
            if mayer > 2.4:
                mayer_damp = max(0.5, 1.0 - (mayer - 2.4) / 2.2)
            elif mayer < 1.0:
                mayer_damp = min(1.2, 1.0 + (1.0 - mayer) * 0.3)
        for sym in raw:
            raw[sym] = raw.get(sym, 0.0) * mayer_damp

        # BTC cross-asset lead: 1.4x on confirmed lead
        for sym, st in self._states.items():
            if sym == "BTC":
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
        targets = self.compute_targets(bar_time)
        equity  = self._get_equity()

        for sym, tgt_frac in targets.items():
            st = self._states[sym]

            # Min-hold gate (no reversal within MIN_HOLD bars)
            if (not math.isclose(st.last_frac, 0.0) and
                    not math.isclose(tgt_frac, 0.0) and
                    math.copysign(1, tgt_frac) != math.copysign(1, st.last_frac) and
                    st.bars_held < MIN_HOLD):
                tgt_frac = st.last_frac

            delta = tgt_frac - st.last_frac
            if abs(delta) < MIN_TRADE_FRAC:
                st.bars_held += 1
                continue

            cp = self._last_price.get(sym)
            if not cp or cp <= 0:
                log.warning("%s: no price — skipping order", sym)
                continue

            tgt_dollar  = tgt_frac  * equity
            curr_dollar = st.last_frac * equity
            delta_dollar = tgt_dollar - curr_dollar
            qty = delta_dollar / cp

            if abs(qty * cp) < 1.0:
                continue

            side = "buy" if qty > 0 else "sell"
            log.info(
                "%s → %s %.6f units @ ~$%.2f  (frac %.3f→%.3f)",
                sym, side.upper(), abs(qty), cp, st.last_frac, tgt_frac,
            )
            asyncio.create_task(
                self._place_order_async(sym, side, abs(qty), cp, tgt_frac)
            )

    async def _place_order_async(
        self, sym: str, side: str, qty: float, price: float, new_frac: float
    ) -> None:
        """Place order(s), splitting if notional > MAX_ORDER_NOTIONAL."""
        try:
            await asyncio.to_thread(
                self._place_order, sym, side, qty, price, new_frac
            )
        except Exception as exc:
            log.error("%s order error: %s", sym, exc, exc_info=True)

    def _place_order(
        self, sym: str, side: str, qty: float, price: float, new_frac: float
    ) -> None:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        ticker      = INSTRUMENTS[sym]["ticker"]
        notional    = qty * price
        max_slice   = MAX_ORDER_NOTIONAL
        remaining   = qty

        while remaining > 1e-8:
            slice_qty     = min(remaining, max_slice / price)
            slice_notional = slice_qty * price

            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol       = ticker,
                qty          = round(slice_qty, 8),
                side         = order_side,
                time_in_force= TimeInForce.GTC,
            )
            try:
                resp = self._trading_client.submit_order(req)
                log.info(
                    "%s order submitted: id=%s side=%s qty=%.6f notional=$%.0f",
                    sym, resp.id, side, float(slice_qty), float(slice_notional),
                )
            except Exception as exc:
                log.error("%s submit_order failed: %s", sym, exc)
                break

            remaining -= slice_qty

        # Optimistically update last_frac (fills will confirm via trade_updates)
        st = self._states[sym]
        st.last_frac = new_frac
        st.bars_held = 0
        if side == "buy":
            st.entry_px = price
        elif math.isclose(new_frac, 0.0):
            st.entry_px = None

    # ── Fill handling ──────────────────────────────────────────────────────────

    def _on_fill(self, fill_event: Any) -> None:
        """
        Handle a fill / trade_update event from Alpaca.
        Logs to live_trades and updates trade_pnl via FIFO matching.
        """
        try:
            order  = fill_event.order if hasattr(fill_event, "order") else fill_event
            ticker = str(order.symbol)
            sym    = self._ticker_to_sym(ticker) or ticker
            side   = str(order.side).lower().replace("orderside.", "")
            qty    = float(order.filled_qty or 0)
            price  = float(order.filled_avg_price or 0)
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
                    hold_b  = st.bars_held
                    self._db.execute(
                        """INSERT INTO trade_pnl
                           (symbol, entry_time, exit_time, entry_price, exit_price, qty, pnl, hold_bars)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (sym, entry_time, fill_time, entry_price, price, matched, round(pnl, 6), hold_b),
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
        from alpaca.data.live.crypto import CryptoDataStream
        from alpaca.trading.stream import TradingStream

        tickers = [cfg["ticker"] for cfg in INSTRUMENTS.values()]

        # ── 15-minute bar handler
        async def bar_handler(bar: Any) -> None:
            try:
                self.on_bar(bar.symbol, bar)
            except Exception as exc:
                log.error("bar_handler error [%s]: %s", bar.symbol, exc, exc_info=True)

        # Subscribe bars
        self._stream.subscribe_bars(bar_handler, *tickers)
        log.info("Subscribed to %d crypto bar feeds", len(tickers))

        # ── Trade update handler (fills)
        trading_stream = TradingStream(
            api_key    = self._api_key,
            secret_key = self._secret_key,
            paper      = self._paper,
        )

        async def trade_update_handler(event: Any) -> None:
            try:
                event_type = str(getattr(event, "event", "")).lower()
                if event_type == "fill":
                    self._on_fill(event)
            except Exception as exc:
                log.error("trade_update_handler error: %s", exc, exc_info=True)

        trading_stream.subscribe_trade_updates(trade_update_handler)
        log.info("Subscribed to trade_updates")

        # Run both streams concurrently
        await asyncio.gather(
            self._stream._run_forever(),
            trading_stream._run_forever(),
        )

    def run(self) -> None:
        """Main entry point — blocks until interrupted."""
        log.info("LARSA v16 LiveTrader starting (strategy_version=%s)", STRATEGY_VERSION)
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
