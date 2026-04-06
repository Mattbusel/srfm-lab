"""
tca.py — Transaction Cost Analysis engine.

Reads from execution/live_trades.db and produces per-symbol, per-side
cost breakdowns including implementation shortfall, VWAP deviation,
slippage components, intraday patterns, large-order scaling, and
BH-regime-conditioned analysis.

CLI
---
    python tca.py --since 2024-01-01 --report html
    python tca.py --since 2024-01-01 --report text
    python tca.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import yfinance as yf  # type: ignore
    _HAS_YF = True
except ImportError:
    _HAS_YF = False
    log.warning("yfinance not installed — VWAP deviation will be skipped")

try:
    from jinja2 import Environment, BaseLoader  # type: ignore
    _HAS_JINJA = True
except ImportError:
    _HAS_JINJA = False
    log.warning("jinja2 not installed — HTML report disabled; falling back to text")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_DB_PATH = _REPO_ROOT / "execution" / "live_trades.db"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SlippageRecord:
    """One trade's slippage breakdown in basis points."""
    trade_id: int
    symbol: str
    side: str
    qty: float
    fill_price: float
    arrival_price: float
    notional: float
    fill_time: datetime

    # computed
    impl_shortfall_bps: float = 0.0   # (fill - arrival) / arrival * 10000
    vwap_dev_bps: float = 0.0         # (fill - vwap) / vwap * 10000
    spread_cost_bps: float = 0.0      # half-spread estimate
    market_impact_bps: float = 0.0    # square-root model estimate
    timing_cost_bps: float = 0.0      # residual = impl_shortfall - spread - impact
    total_slippage_bps: float = 0.0


@dataclass
class SymbolSideReport:
    symbol: str
    side: str
    n_trades: int
    avg_slippage_bps: float
    p50_slippage_bps: float
    p95_slippage_bps: float
    p99_slippage_bps: float
    total_cost_usd: float
    avg_notional: float
    avg_spread_cost_bps: float
    avg_impact_bps: float
    avg_timing_cost_bps: float


@dataclass
class IntradayPattern:
    hour: int
    n_trades: int
    avg_slippage_bps: float
    p95_slippage_bps: float


@dataclass
class DowPattern:
    dow: int          # 0=Mon … 6=Sun
    dow_name: str
    n_trades: int
    avg_slippage_bps: float


@dataclass
class LargeOrderBucket:
    notional_low: float
    notional_high: float
    n_trades: int
    avg_slippage_bps: float
    p95_slippage_bps: float


@dataclass
class BHConditionedReport:
    tf: int
    n_trades: int
    avg_slippage_bps: float
    p95_slippage_bps: float
    total_cost_usd: float


@dataclass
class TCAReport:
    generated_at: str
    since: str
    until: str
    n_total_trades: int
    total_notional_usd: float
    total_cost_usd: float
    avg_slippage_bps: float
    p95_slippage_bps: float
    symbol_side_reports: List[SymbolSideReport] = field(default_factory=list)
    intraday_patterns: List[IntradayPattern] = field(default_factory=list)
    dow_patterns: List[DowPattern] = field(default_factory=list)
    large_order_buckets: List[LargeOrderBucket] = field(default_factory=list)
    bh_reports: List[BHConditionedReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VWAP helper
# ---------------------------------------------------------------------------

def _fetch_intraday_vwap(symbol: str, date: str) -> Optional[float]:
    """
    Fetch approximate intraday VWAP for *symbol* on *date* (YYYY-MM-DD)
    using yfinance 1-minute bars.  Returns None on any failure.
    """
    if not _HAS_YF:
        return None
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=date, end=date, interval="1m", auto_adjust=True)
        if df.empty:
            return None
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        vol = df["Volume"].replace(0, np.nan)
        vwap = (tp * vol).sum() / vol.sum()
        return float(vwap) if not math.isnan(vwap) else None
    except Exception as exc:
        log.debug("VWAP fetch failed for %s on %s: %s", symbol, date, exc)
        return None


# ---------------------------------------------------------------------------
# Spread / impact estimators
# ---------------------------------------------------------------------------

# Typical half-spread by asset type (bps) — used when L1 data unavailable
_CRYPTO_SPREAD_BPS: Dict[str, float] = {
    "BTC": 1.0, "ETH": 1.5, "SOL": 3.0,
}
_DEFAULT_CRYPTO_SPREAD_BPS = 5.0
_DEFAULT_EQUITY_SPREAD_BPS = 2.5

_CRYPTO_SYMS = {
    "BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK", "UNI", "AAVE",
    "CRV", "SUSHI", "BAT", "YFI", "DOT", "LTC", "BCH", "SHIB",
}


def _is_crypto(symbol: str) -> bool:
    return symbol.upper() in _CRYPTO_SYMS


def _spread_cost_bps(symbol: str) -> float:
    sym = symbol.upper()
    if _is_crypto(sym):
        return _CRYPTO_SPREAD_BPS.get(sym, _DEFAULT_CRYPTO_SPREAD_BPS)
    return _DEFAULT_EQUITY_SPREAD_BPS


def _market_impact_bps(notional: float, adv_usd: float, daily_vol: float) -> float:
    """
    Square-root impact model:  I = sigma * sqrt(Q / ADV)
    Returns impact in basis points.
    adv_usd  — Average Daily Volume in USD
    daily_vol — daily return volatility (e.g. 0.02 = 2%)
    """
    if adv_usd <= 0 or notional <= 0:
        return 0.0
    participation = notional / adv_usd
    impact = daily_vol * math.sqrt(participation)
    return impact * 10_000  # convert to bps


# ---------------------------------------------------------------------------
# ADV / vol cache
# ---------------------------------------------------------------------------

class _MarketDataCache:
    """Simple in-process cache for ADV and daily-vol lookups."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, float]] = {}  # sym -> (adv_usd, daily_vol)

    def get(self, symbol: str) -> Tuple[float, float]:
        sym = symbol.upper()
        if sym in self._cache:
            return self._cache[sym]
        adv, vol = self._fetch(sym)
        self._cache[sym] = (adv, vol)
        return adv, vol

    def _fetch(self, symbol: str) -> Tuple[float, float]:
        if not _HAS_YF:
            return self._defaults(symbol)
        try:
            ticker_sym = symbol if not _is_crypto(symbol) else f"{symbol}-USD"
            t = yf.Ticker(ticker_sym)
            hist = t.history(period="30d", interval="1d", auto_adjust=True)
            if hist.empty or len(hist) < 5:
                return self._defaults(symbol)
            adv_usd = float((hist["Close"] * hist["Volume"]).mean())
            rets = hist["Close"].pct_change().dropna()
            vol = float(rets.std())
            return adv_usd, vol
        except Exception as exc:
            log.debug("Market data fetch for %s failed: %s", symbol, exc)
            return self._defaults(symbol)

    @staticmethod
    def _defaults(symbol: str) -> Tuple[float, float]:
        if _is_crypto(symbol):
            return 50_000_000.0, 0.03   # $50M ADV, 3% vol
        return 200_000_000.0, 0.015     # $200M ADV, 1.5% vol


_mdc = _MarketDataCache()


# ---------------------------------------------------------------------------
# TCAEngine
# ---------------------------------------------------------------------------

class TCAEngine:
    """
    Transaction Cost Analysis engine.

    Parameters
    ----------
    db_path : str or Path
        Path to the live_trades SQLite database.
    since : datetime, optional
        Filter trades on or after this UTC datetime.
    until : datetime, optional
        Filter trades before this UTC datetime.  Defaults to now.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.since = since or datetime(2000, 1, 1, tzinfo=timezone.utc)
        self.until = until or datetime.now(timezone.utc)
        self._raw: Optional[pd.DataFrame] = None
        self._records: Optional[List[SlippageRecord]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self) -> "TCAEngine":
        """Load trades from DB into internal DataFrame."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
                SELECT id, symbol, side, qty, price, notional,
                       fill_time, order_id, strategy_version
                FROM live_trades
                WHERE fill_time >= ? AND fill_time < ?
                ORDER BY fill_time
            """
            df = pd.read_sql_query(
                query,
                conn,
                params=(self.since.isoformat(), self.until.isoformat()),
            )
        finally:
            conn.close()

        if df.empty:
            log.warning("No trades found in [%s, %s)", self.since, self.until)
            self._raw = df
            return self

        df["fill_time"] = pd.to_datetime(df["fill_time"], utc=True)
        df["hour"] = df["fill_time"].dt.hour
        df["dow"] = df["fill_time"].dt.dayofweek
        df["date"] = df["fill_time"].dt.date.astype(str)
        self._raw = df
        log.info("Loaded %d trades from %s", len(df), self.db_path)
        return self

    @property
    def raw(self) -> pd.DataFrame:
        if self._raw is None:
            self.load()
        return self._raw  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Slippage computation
    # ------------------------------------------------------------------

    def _compute_arrival_price(self, df: pd.DataFrame) -> pd.Series:
        """
        Approximate arrival price using the VWAP of the first 1-minute bar
        of the session, falling back to the first trade price in the group.

        For real usage this would come from an order management system.
        Here we estimate it as the mean price of the first N fills for a given
        order_id, or if order_id is missing, the symbol's open price for the day.
        """
        # group by order_id and take VWAP of first 3 fills as "arrival"
        arrival = {}
        for oid, grp in df.groupby("order_id"):
            first3 = grp.nsmallest(3, "fill_time")
            vwap_first = (first3["price"] * first3["qty"]).sum() / first3["qty"].sum()
            for idx in grp.index:
                arrival[idx] = float(vwap_first)
        return pd.Series(arrival, dtype=float)

    def compute_slippage(self) -> List[SlippageRecord]:
        """Compute per-trade slippage records."""
        if self._records is not None:
            return self._records

        df = self.raw
        if df.empty:
            self._records = []
            return []

        arrival_prices = self._compute_arrival_price(df)

        # VWAP cache per (symbol, date)
        vwap_cache: Dict[Tuple[str, str], Optional[float]] = {}

        records = []
        for idx, row in df.iterrows():
            sym = row["symbol"]
            side = row["side"].lower()
            qty = float(row["qty"])
            fill_px = float(row["price"])
            notional = float(row["notional"])
            arrival_px = arrival_prices.get(idx, fill_px)
            ft: datetime = row["fill_time"]

            # implementation shortfall
            if arrival_px > 0:
                if side == "buy":
                    is_bps = (fill_px - arrival_px) / arrival_px * 10_000
                else:
                    is_bps = (arrival_px - fill_px) / arrival_px * 10_000
            else:
                is_bps = 0.0

            # VWAP deviation
            date_str = row["date"]
            vwap_key = (sym, date_str)
            if vwap_key not in vwap_cache:
                vwap_cache[vwap_key] = _fetch_intraday_vwap(sym, date_str)
            vwap_px = vwap_cache[vwap_key]

            if vwap_px and vwap_px > 0:
                if side == "buy":
                    vwap_dev = (fill_px - vwap_px) / vwap_px * 10_000
                else:
                    vwap_dev = (vwap_px - fill_px) / vwap_px * 10_000
            else:
                vwap_dev = 0.0

            # spread cost
            spread_bps = _spread_cost_bps(sym)

            # market impact
            adv_usd, daily_vol = _mdc.get(sym)
            impact_bps = _market_impact_bps(notional, adv_usd, daily_vol)

            # timing cost = residual
            timing_bps = max(0.0, is_bps - spread_bps - impact_bps)

            total_bps = is_bps  # implementation shortfall is our best single number

            rec = SlippageRecord(
                trade_id=int(row["id"]),
                symbol=sym,
                side=side,
                qty=qty,
                fill_price=fill_px,
                arrival_price=arrival_px,
                notional=notional,
                fill_time=ft,
                impl_shortfall_bps=round(is_bps, 4),
                vwap_dev_bps=round(vwap_dev, 4),
                spread_cost_bps=round(spread_bps, 4),
                market_impact_bps=round(impact_bps, 4),
                timing_cost_bps=round(timing_bps, 4),
                total_slippage_bps=round(total_bps, 4),
            )
            records.append(rec)

        self._records = records
        return records

    # ------------------------------------------------------------------
    # Per-symbol / side reports
    # ------------------------------------------------------------------

    def symbol_side_report(self) -> List[SymbolSideReport]:
        """Aggregate per-symbol, per-side TCA stats."""
        records = self.compute_slippage()
        if not records:
            return []

        df = pd.DataFrame([asdict(r) for r in records])
        reports = []
        for (sym, side), grp in df.groupby(["symbol", "side"]):
            slips = grp["total_slippage_bps"].values
            cost_usd = (grp["notional"] * grp["total_slippage_bps"] / 10_000).sum()
            rep = SymbolSideReport(
                symbol=str(sym),
                side=str(side),
                n_trades=len(grp),
                avg_slippage_bps=float(np.mean(slips)),
                p50_slippage_bps=float(np.percentile(slips, 50)),
                p95_slippage_bps=float(np.percentile(slips, 95)),
                p99_slippage_bps=float(np.percentile(slips, 99)),
                total_cost_usd=float(cost_usd),
                avg_notional=float(grp["notional"].mean()),
                avg_spread_cost_bps=float(grp["spread_cost_bps"].mean()),
                avg_impact_bps=float(grp["market_impact_bps"].mean()),
                avg_timing_cost_bps=float(grp["timing_cost_bps"].mean()),
            )
            reports.append(rep)
        return sorted(reports, key=lambda r: abs(r.total_cost_usd), reverse=True)

    # ------------------------------------------------------------------
    # Intraday patterns
    # ------------------------------------------------------------------

    def intraday_patterns(self) -> List[IntradayPattern]:
        """Slippage broken down by hour-of-day (UTC)."""
        records = self.compute_slippage()
        if not records:
            return []
        df = pd.DataFrame([asdict(r) for r in records])
        df["hour"] = pd.to_datetime(df["fill_time"], utc=True).dt.hour
        out = []
        for hr in range(24):
            grp = df[df["hour"] == hr]
            if grp.empty:
                continue
            slips = grp["total_slippage_bps"].values
            out.append(IntradayPattern(
                hour=hr,
                n_trades=len(grp),
                avg_slippage_bps=float(np.mean(slips)),
                p95_slippage_bps=float(np.percentile(slips, 95)),
            ))
        return out

    def dow_patterns(self) -> List[DowPattern]:
        """Slippage broken down by day-of-week."""
        _DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        records = self.compute_slippage()
        if not records:
            return []
        df = pd.DataFrame([asdict(r) for r in records])
        df["dow"] = pd.to_datetime(df["fill_time"], utc=True).dt.dayofweek
        out = []
        for d in range(7):
            grp = df[df["dow"] == d]
            if grp.empty:
                continue
            slips = grp["total_slippage_bps"].values
            out.append(DowPattern(
                dow=d,
                dow_name=_DOW[d],
                n_trades=len(grp),
                avg_slippage_bps=float(np.mean(slips)),
            ))
        return out

    # ------------------------------------------------------------------
    # Large order analysis
    # ------------------------------------------------------------------

    def large_order_analysis(self, n_buckets: int = 5) -> List[LargeOrderBucket]:
        """
        Show how slippage scales with notional order size.
        Divides trades into quantile buckets by notional.
        """
        records = self.compute_slippage()
        if not records:
            return []
        df = pd.DataFrame([asdict(r) for r in records])
        df["bucket"] = pd.qcut(df["notional"], q=n_buckets, duplicates="drop")
        out = []
        for bucket, grp in df.groupby("bucket", observed=True):
            slips = grp["total_slippage_bps"].values
            out.append(LargeOrderBucket(
                notional_low=float(bucket.left),
                notional_high=float(bucket.right),
                n_trades=len(grp),
                avg_slippage_bps=float(np.mean(slips)),
                p95_slippage_bps=float(np.percentile(slips, 95)),
            ))
        return out

    # ------------------------------------------------------------------
    # BH-conditioned TCA
    # ------------------------------------------------------------------

    def bh_conditioned_report(self) -> List[BHConditionedReport]:
        """
        Slippage conditioned on BH timeframe signals (tf=4, 6, 7).

        Strategy version tag contains tf info when available.
        Falls back to hour-of-day proxy if not tagged:
          tf=4  → 4h bars  → fills in hours 0,4,8,12,16,20
          tf=6  → 1h bars  → fills in hours 0–5 (London open zone)
          tf=7  → 15m bars → fills in any hour
        """
        records = self.compute_slippage()
        if not records:
            return []
        df = pd.DataFrame([asdict(r) for r in records])
        raw = self.raw.reset_index(drop=True)
        df["strategy_version"] = raw["strategy_version"].values if "strategy_version" in raw.columns else "unknown"
        df["fill_dt"] = pd.to_datetime(df["fill_time"], utc=True)
        df["hour"] = df["fill_dt"].dt.hour

        def _assign_tf(row: pd.Series) -> Optional[int]:
            sv = str(row.get("strategy_version", ""))
            if "tf4" in sv or "_4h" in sv:
                return 4
            if "tf6" in sv or "_1h" in sv:
                return 6
            if "tf7" in sv or "_15m" in sv:
                return 7
            # hour-of-day proxy
            hr = int(row["hour"])
            if hr % 4 == 0:
                return 4
            if 0 <= hr < 6:
                return 6
            return 7

        df["tf"] = df.apply(_assign_tf, axis=1)
        out = []
        for tf_val in [4, 6, 7]:
            grp = df[df["tf"] == tf_val]
            if grp.empty:
                continue
            slips = grp["total_slippage_bps"].values
            cost_usd = float((grp["notional"] * grp["total_slippage_bps"] / 10_000).sum())
            out.append(BHConditionedReport(
                tf=tf_val,
                n_trades=len(grp),
                avg_slippage_bps=float(np.mean(slips)),
                p95_slippage_bps=float(np.percentile(slips, 95)),
                total_cost_usd=cost_usd,
            ))
        return out

    # ------------------------------------------------------------------
    # Full report assembly
    # ------------------------------------------------------------------

    def build_report(self) -> TCAReport:
        """Assemble the complete TCA report."""
        records = self.compute_slippage()
        if not records:
            return TCAReport(
                generated_at=datetime.now(timezone.utc).isoformat(),
                since=self.since.isoformat(),
                until=self.until.isoformat(),
                n_total_trades=0,
                total_notional_usd=0.0,
                total_cost_usd=0.0,
                avg_slippage_bps=0.0,
                p95_slippage_bps=0.0,
            )

        slips = np.array([r.total_slippage_bps for r in records])
        notionals = np.array([r.notional for r in records])
        total_notional = float(notionals.sum())
        total_cost = float((notionals * slips / 10_000).sum())

        return TCAReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            since=self.since.isoformat(),
            until=self.until.isoformat(),
            n_total_trades=len(records),
            total_notional_usd=total_notional,
            total_cost_usd=total_cost,
            avg_slippage_bps=float(np.mean(slips)),
            p95_slippage_bps=float(np.percentile(slips, 95)),
            symbol_side_reports=self.symbol_side_report(),
            intraday_patterns=self.intraday_patterns(),
            dow_patterns=self.dow_patterns(),
            large_order_buckets=self.large_order_analysis(),
            bh_reports=self.bh_conditioned_report(),
        )

    # ------------------------------------------------------------------
    # Report rendering
    # ------------------------------------------------------------------

    def render_text(self, report: Optional[TCAReport] = None) -> str:
        """Render the TCA report as plain text."""
        rpt = report or self.build_report()
        lines = [
            "=" * 72,
            "  TRANSACTION COST ANALYSIS REPORT",
            "=" * 72,
            f"  Generated : {rpt.generated_at}",
            f"  Period    : {rpt.since}  to  {rpt.until}",
            f"  Trades    : {rpt.n_total_trades:,}",
            f"  Notional  : ${rpt.total_notional_usd:,.2f}",
            f"  Total Cost: ${rpt.total_cost_usd:,.2f}",
            f"  Avg Slip  : {rpt.avg_slippage_bps:.2f} bps",
            f"  P95 Slip  : {rpt.p95_slippage_bps:.2f} bps",
            "",
            "── Per-Symbol / Side ──────────────────────────────────────────",
            f"  {'Symbol':<8} {'Side':<5} {'N':>7} {'Avg bps':>9} {'P95 bps':>9}"
            f" {'P99 bps':>9} {'Cost $':>12}",
        ]
        for r in rpt.symbol_side_reports:
            lines.append(
                f"  {r.symbol:<8} {r.side:<5} {r.n_trades:>7,} "
                f"{r.avg_slippage_bps:>9.2f} {r.p95_slippage_bps:>9.2f} "
                f"{r.p99_slippage_bps:>9.2f} {r.total_cost_usd:>12.2f}"
            )

        lines += [
            "",
            "── Intraday Pattern (hour UTC) ────────────────────────────────",
            f"  {'Hour':>5} {'N':>7} {'Avg bps':>9} {'P95 bps':>9}",
        ]
        for p in rpt.intraday_patterns:
            lines.append(
                f"  {p.hour:>5} {p.n_trades:>7,} {p.avg_slippage_bps:>9.2f} "
                f"{p.p95_slippage_bps:>9.2f}"
            )

        lines += [
            "",
            "── Day-of-Week Pattern ────────────────────────────────────────",
            f"  {'DOW':<6} {'N':>7} {'Avg bps':>9}",
        ]
        for p in rpt.dow_patterns:
            lines.append(f"  {p.dow_name:<6} {p.n_trades:>7,} {p.avg_slippage_bps:>9.2f}")

        lines += [
            "",
            "── Large Order Scaling ────────────────────────────────────────",
            f"  {'Notional Low':>14} {'Notional High':>14} {'N':>7} "
            f"{'Avg bps':>9} {'P95 bps':>9}",
        ]
        for b in rpt.large_order_buckets:
            lines.append(
                f"  {b.notional_low:>14,.0f} {b.notional_high:>14,.0f} "
                f"{b.n_trades:>7,} {b.avg_slippage_bps:>9.2f} "
                f"{b.p95_slippage_bps:>9.2f}"
            )

        lines += [
            "",
            "── BH-Regime Conditioned Slippage ─────────────────────────────",
            f"  {'TF':>4} {'N':>8} {'Avg bps':>10} {'P95 bps':>10} {'Cost $':>14}",
        ]
        for b in rpt.bh_reports:
            lines.append(
                f"  {b.tf:>4} {b.n_trades:>8,} {b.avg_slippage_bps:>10.2f} "
                f"{b.p95_slippage_bps:>10.2f} {b.total_cost_usd:>14.2f}"
            )
        lines.append("=" * 72)
        return "\n".join(lines)

    def render_html(self, report: Optional[TCAReport] = None) -> str:
        """Render the TCA report as a self-contained HTML page."""
        if not _HAS_JINJA:
            log.warning("Jinja2 not installed; falling back to text report")
            return "<pre>" + self.render_text(report) + "</pre>"

        rpt = report or self.build_report()
        tmpl_src = _HTML_TEMPLATE
        env = Environment(loader=BaseLoader())
        tmpl = env.from_string(tmpl_src)
        return tmpl.render(rpt=rpt)

    def save_report(
        self,
        fmt: str = "text",
        out_path: Optional[Path] = None,
    ) -> Path:
        """
        Save report to disk.

        Parameters
        ----------
        fmt      : 'text' or 'html'
        out_path : destination path; auto-generated if None
        """
        report = self.build_report()
        if fmt == "html":
            content = self.render_html(report)
            ext = ".html"
        else:
            content = self.render_text(report)
            ext = ".txt"

        if out_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = _HERE / f"tca_report_{ts}{ext}"

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        log.info("Report written to %s", out_path)
        return out_path


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>TCA Report</title>
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f1117; color: #e0e0e0; margin: 0; padding: 20px; }
  h1 { color: #7ec8e3; border-bottom: 1px solid #333; padding-bottom: 8px; }
  h2 { color: #a8d8ea; margin-top: 32px; }
  .meta { color: #888; font-size: 0.9em; }
  .kpi-grid { display: flex; flex-wrap: wrap; gap: 16px; margin: 20px 0; }
  .kpi { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px;
         padding: 16px 24px; min-width: 160px; }
  .kpi .label { font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: .05em; }
  .kpi .value { font-size: 1.6em; font-weight: 600; color: #7ec8e3; margin-top: 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 0.88em; }
  th { background: #1e2130; color: #7ec8e3; padding: 8px 12px; text-align: right; border-bottom: 1px solid #333; }
  th:first-child { text-align: left; }
  td { padding: 6px 12px; text-align: right; border-bottom: 1px solid #1e2130; }
  td:first-child { text-align: left; }
  tr:hover td { background: #1a1d27; }
  .warn { color: #f0a500; }
  .bad  { color: #e05c5c; }
</style>
</head>
<body>
<h1>Transaction Cost Analysis Report</h1>
<p class="meta">Generated: {{ rpt.generated_at }} &nbsp;|&nbsp; Period: {{ rpt.since }} → {{ rpt.until }}</p>

<div class="kpi-grid">
  <div class="kpi"><div class="label">Total Trades</div><div class="value">{{ "{:,}".format(rpt.n_total_trades) }}</div></div>
  <div class="kpi"><div class="label">Total Notional</div><div class="value">${{ "{:,.0f}".format(rpt.total_notional_usd) }}</div></div>
  <div class="kpi"><div class="label">Total Cost</div><div class="value">${{ "{:,.2f}".format(rpt.total_cost_usd) }}</div></div>
  <div class="kpi"><div class="label">Avg Slippage</div><div class="value">{{ "%.2f"|format(rpt.avg_slippage_bps) }} bps</div></div>
  <div class="kpi"><div class="label">P95 Slippage</div><div class="value">{{ "%.2f"|format(rpt.p95_slippage_bps) }} bps</div></div>
</div>

<h2>Per-Symbol / Side</h2>
<table>
<tr><th>Symbol</th><th>Side</th><th>N Trades</th><th>Avg bps</th><th>P50 bps</th><th>P95 bps</th><th>P99 bps</th><th>Cost $</th><th>Spread bps</th><th>Impact bps</th><th>Timing bps</th></tr>
{% for r in rpt.symbol_side_reports %}
<tr>
  <td>{{ r.symbol }}</td>
  <td>{{ r.side }}</td>
  <td>{{ "{:,}".format(r.n_trades) }}</td>
  <td {% if r.avg_slippage_bps > 20 %}class="bad"{% elif r.avg_slippage_bps > 10 %}class="warn"{% endif %}>{{ "%.2f"|format(r.avg_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(r.p50_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(r.p95_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(r.p99_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(r.total_cost_usd) }}</td>
  <td>{{ "%.2f"|format(r.avg_spread_cost_bps) }}</td>
  <td>{{ "%.2f"|format(r.avg_impact_bps) }}</td>
  <td>{{ "%.2f"|format(r.avg_timing_cost_bps) }}</td>
</tr>
{% endfor %}
</table>

<h2>Intraday Pattern (hour UTC)</h2>
<table>
<tr><th>Hour</th><th>N Trades</th><th>Avg bps</th><th>P95 bps</th></tr>
{% for p in rpt.intraday_patterns %}
<tr>
  <td>{{ "%02d:00"|format(p.hour) }}</td>
  <td>{{ "{:,}".format(p.n_trades) }}</td>
  <td>{{ "%.2f"|format(p.avg_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(p.p95_slippage_bps) }}</td>
</tr>
{% endfor %}
</table>

<h2>Day-of-Week Pattern</h2>
<table>
<tr><th>Day</th><th>N Trades</th><th>Avg bps</th></tr>
{% for p in rpt.dow_patterns %}
<tr>
  <td>{{ p.dow_name }}</td>
  <td>{{ "{:,}".format(p.n_trades) }}</td>
  <td>{{ "%.2f"|format(p.avg_slippage_bps) }}</td>
</tr>
{% endfor %}
</table>

<h2>Large Order Scaling (Notional Buckets)</h2>
<table>
<tr><th>Notional Low</th><th>Notional High</th><th>N Trades</th><th>Avg bps</th><th>P95 bps</th></tr>
{% for b in rpt.large_order_buckets %}
<tr>
  <td>${{ "{:,.0f}".format(b.notional_low) }}</td>
  <td>${{ "{:,.0f}".format(b.notional_high) }}</td>
  <td>{{ "{:,}".format(b.n_trades) }}</td>
  <td>{{ "%.2f"|format(b.avg_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(b.p95_slippage_bps) }}</td>
</tr>
{% endfor %}
</table>

<h2>BH-Regime Conditioned Slippage</h2>
<table>
<tr><th>TF</th><th>N Trades</th><th>Avg bps</th><th>P95 bps</th><th>Total Cost $</th></tr>
{% for b in rpt.bh_reports %}
<tr>
  <td>tf={{ b.tf }}</td>
  <td>{{ "{:,}".format(b.n_trades) }}</td>
  <td>{{ "%.2f"|format(b.avg_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(b.p95_slippage_bps) }}</td>
  <td>{{ "%.2f"|format(b.total_cost_usd) }}</td>
</tr>
{% endfor %}
</table>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TCA Engine — Transaction Cost Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--since",
        default="2020-01-01",
        help="Start date for analysis (YYYY-MM-DD), default: 2020-01-01",
    )
    p.add_argument(
        "--until",
        default=None,
        help="End date (YYYY-MM-DD), default: now",
    )
    p.add_argument(
        "--db",
        default=str(_DB_PATH),
        help=f"Path to live_trades.db, default: {_DB_PATH}",
    )
    p.add_argument(
        "--report",
        choices=["text", "html", "json"],
        default="text",
        help="Output format: text | html | json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output file path (auto-generated if omitted)",
    )
    p.add_argument(
        "--print",
        dest="print_stdout",
        action="store_true",
        help="Also print text report to stdout",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    until = (
        datetime.fromisoformat(args.until).replace(tzinfo=timezone.utc)
        if args.until
        else datetime.now(timezone.utc)
    )

    engine = TCAEngine(db_path=Path(args.db), since=since, until=until)
    engine.load()
    report = engine.build_report()

    if args.report == "json":
        content = json.dumps(asdict(report), indent=2, default=str)
        ext = ".json"
    elif args.report == "html":
        content = engine.render_html(report)
        ext = ".html"
    else:
        content = engine.render_text(report)
        ext = ".txt"

    out_path = Path(args.out) if args.out else (
        _HERE / f"tca_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"Report saved → {out_path}")

    if args.print_stdout or args.report == "text":
        print(engine.render_text(report))


if __name__ == "__main__":
    main()
