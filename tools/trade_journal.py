"""
trade_journal.py -- Production trade journal for LARSA v18 strategy.
Captures every trade with full market context, signal state, and post-trade metrics.
Persists to SQLite. Generates weekly reports in Markdown and HTML.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH_DEFAULT = Path(__file__).parent / "trade_journal.db"

VOL_REGIMES = ("low", "med", "high")
SESSIONS = ("asian", "london", "us", "overlap")
EXIT_REASONS = (
    "tp_hit", "sl_hit", "rl_exit", "time_exit", "nav_gate_close",
    "hurst_reversal", "event_filter", "manual", "eod_close", "unknown",
)

# ---------------------------------------------------------------------------
# JournalEntry dataclass -- 50+ fields
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    # -- Trade identity --
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    strategy_version: str = "LARSA_v18"

    # -- Timestamps --
    entry_ts: str = ""          # ISO-8601 string
    exit_ts: str = ""
    entry_bar_index: int = 0
    exit_bar_index: int = 0
    hold_bars: int = 0

    # -- Price & size --
    entry_price: float = 0.0
    exit_price: float = 0.0
    qty: float = 0.0
    side: str = "long"          # long / short

    # -- P&L --
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0

    # -- Market context at entry --
    entry_close: float = 0.0
    entry_volume: float = 0.0
    atr_14: float = 0.0
    spread_bps: float = 0.0
    vol_regime: str = "med"     # low / med / high
    session: str = "us"         # asian / london / us / overlap

    # -- BH signal context --
    bh_mass_15m: float = 0.0
    bh_mass_1h: float = 0.0
    bh_mass_4h: float = 0.0
    bh_active: bool = False

    # -- Cross-filter context --
    cf_direction: int = 0       # -1 / 0 / +1
    cf_alignment: float = 0.0   # 0-1 alignment score

    # -- Hurst context --
    hurst_h: float = 0.5
    hurst_regime: str = "neutral"   # trending / neutral / mean-reverting

    # -- QuatNav context --
    nav_omega: float = 0.0
    nav_geodesic: float = 0.0
    nav_quaternion: str = ""    # JSON list [w, x, y, z]

    # -- ML & Granger context --
    ml_signal: float = 0.0      # continuous score -1..+1
    ml_confidence: float = 0.0
    granger_btc_corr: float = 0.0
    granger_eth_corr: float = 0.0

    # -- Strategy gate flags --
    was_rl_exit: bool = False
    was_event_calendar_filtered: bool = False
    was_nav_gated: bool = False
    was_hurst_damped: bool = False
    was_cf_filtered: bool = False
    was_ml_filtered: bool = False

    # -- Post-trade excursion --
    mfe_pct: float = 0.0        # max favorable excursion %
    mae_pct: float = 0.0        # max adverse excursion % (positive = bad)
    mfe_price: float = 0.0
    mae_price: float = 0.0

    # -- Exit metadata --
    exit_reason: str = "unknown"
    exit_signal_value: float = 0.0

    # -- Portfolio context --
    portfolio_nav_at_entry: float = 0.0
    position_size_pct: float = 0.0

    # -- Notes --
    tags: str = ""              # comma-separated tags
    notes: str = ""

    # -- Computed helpers --
    def is_winner(self) -> bool:
        return self.net_pnl > 0.0

    def edge_ratio(self) -> float:
        """MFE / MAE ratio -- higher is better."""
        if self.mae_pct == 0.0:
            return float("inf") if self.mfe_pct > 0 else 0.0
        return self.mfe_pct / self.mae_pct

    def hold_duration_hours(self) -> float:
        try:
            entry = datetime.fromisoformat(self.entry_ts)
            exit_ = datetime.fromisoformat(self.exit_ts)
            return (exit_ - entry).total_seconds() / 3600.0
        except Exception:
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JournalEntry":
        # Accept extra keys gracefully
        valid = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# JournalStats
# ---------------------------------------------------------------------------

@dataclass
class JournalStats:
    total_trades: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0
    avg_pnl: float = 0.0
    avg_net_pnl: float = 0.0
    avg_hold_bars: float = 0.0
    avg_hold_hours: float = 0.0
    total_pnl: float = 0.0
    total_net_pnl: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0     # avg win * win_rate - avg_loss * loss_rate
    best_trade_pnl: float = 0.0
    worst_trade_pnl: float = 0.0
    best_trade_id: str = ""
    worst_trade_id: str = ""
    most_traded_symbol: str = ""
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0
    rl_exit_rate: float = 0.0
    nav_gated_rate: float = 0.0
    event_filtered_rate: float = 0.0
    by_symbol: Dict[str, int] = field(default_factory=dict)
    by_exit_reason: Dict[str, int] = field(default_factory=dict)
    by_vol_regime: Dict[str, float] = field(default_factory=dict)  # regime -> win_rate


# ---------------------------------------------------------------------------
# SQLite schema helpers
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trade_journal (
    trade_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    strategy_version TEXT,
    entry_ts TEXT,
    exit_ts TEXT,
    entry_bar_index INTEGER,
    exit_bar_index INTEGER,
    hold_bars INTEGER,
    entry_price REAL,
    exit_price REAL,
    qty REAL,
    side TEXT,
    pnl REAL,
    pnl_pct REAL,
    commission REAL,
    slippage REAL,
    net_pnl REAL,
    entry_close REAL,
    entry_volume REAL,
    atr_14 REAL,
    spread_bps REAL,
    vol_regime TEXT,
    session TEXT,
    bh_mass_15m REAL,
    bh_mass_1h REAL,
    bh_mass_4h REAL,
    bh_active INTEGER,
    cf_direction INTEGER,
    cf_alignment REAL,
    hurst_h REAL,
    hurst_regime TEXT,
    nav_omega REAL,
    nav_geodesic REAL,
    nav_quaternion TEXT,
    ml_signal REAL,
    ml_confidence REAL,
    granger_btc_corr REAL,
    granger_eth_corr REAL,
    was_rl_exit INTEGER,
    was_event_calendar_filtered INTEGER,
    was_nav_gated INTEGER,
    was_hurst_damped INTEGER,
    was_cf_filtered INTEGER,
    was_ml_filtered INTEGER,
    mfe_pct REAL,
    mae_pct REAL,
    mfe_price REAL,
    mae_price REAL,
    exit_reason TEXT,
    exit_signal_value REAL,
    portfolio_nav_at_entry REAL,
    position_size_pct REAL,
    tags TEXT,
    notes TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_INDEX_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_tj_symbol ON trade_journal(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_tj_entry_ts ON trade_journal(entry_ts);",
    "CREATE INDEX IF NOT EXISTS idx_tj_exit_reason ON trade_journal(exit_reason);",
]


def _row_to_entry(row: sqlite3.Row) -> JournalEntry:
    d = dict(row)
    # Convert INTEGER booleans back
    for bool_col in (
        "bh_active", "was_rl_exit", "was_event_calendar_filtered",
        "was_nav_gated", "was_hurst_damped", "was_cf_filtered", "was_ml_filtered",
    ):
        if bool_col in d and d[bool_col] is not None:
            d[bool_col] = bool(d[bool_col])
    # Drop extra columns not in dataclass
    d.pop("created_at", None)
    return JournalEntry.from_dict(d)


def _entry_to_row(e: JournalEntry) -> Dict[str, Any]:
    d = e.to_dict()
    # Store booleans as integers for SQLite
    for bool_col in (
        "bh_active", "was_rl_exit", "was_event_calendar_filtered",
        "was_nav_gated", "was_hurst_damped", "was_cf_filtered", "was_ml_filtered",
    ):
        d[bool_col] = int(d[bool_col])
    return d


# ---------------------------------------------------------------------------
# TradeJournal
# ---------------------------------------------------------------------------

class TradeJournal:
    """
    Persistent trade journal backed by SQLite.

    Usage::

        journal = TradeJournal()
        journal.add_entry(entry)
        stats = journal.get_stats()
    """

    def __init__(self, db_path: Path | str = DB_PATH_DEFAULT):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # -- Schema --

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEX_SQL:
            cur.execute(idx_sql)
        self._conn.commit()

    # -- Write --

    def add_entry(self, entry: JournalEntry) -> None:
        """Insert a JournalEntry into the database."""
        row = _entry_to_row(entry)
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        sql = f"INSERT OR REPLACE INTO trade_journal ({cols}) VALUES ({placeholders})"
        self._conn.execute(sql, list(row.values()))
        self._conn.commit()

    def update_entry(self, trade_id: str, **kwargs) -> None:
        """Update specific fields of an existing entry."""
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [trade_id]
        self._conn.execute(
            f"UPDATE trade_journal SET {sets} WHERE trade_id = ?", vals
        )
        self._conn.commit()

    def delete_entry(self, trade_id: str) -> None:
        self._conn.execute(
            "DELETE FROM trade_journal WHERE trade_id = ?", (trade_id,)
        )
        self._conn.commit()

    # -- Read --

    def get_recent(self, n: int = 20) -> List[JournalEntry]:
        """Return the n most recent entries by exit timestamp."""
        rows = self._conn.execute(
            "SELECT * FROM trade_journal ORDER BY exit_ts DESC LIMIT ?", (n,)
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def get_by_id(self, trade_id: str) -> Optional[JournalEntry]:
        row = self._conn.execute(
            "SELECT * FROM trade_journal WHERE trade_id = ?", (trade_id,)
        ).fetchone()
        return _row_to_entry(row) if row else None

    def get_by_symbol(
        self,
        symbol: str,
        since: Optional[str] = None,
    ) -> List[JournalEntry]:
        """Return all entries for symbol, optionally since an ISO-8601 timestamp."""
        if since:
            rows = self._conn.execute(
                "SELECT * FROM trade_journal WHERE symbol = ? AND entry_ts >= ? "
                "ORDER BY entry_ts ASC",
                (symbol, since),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM trade_journal WHERE symbol = ? ORDER BY entry_ts ASC",
                (symbol,),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def get_all(self) -> List[JournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM trade_journal ORDER BY entry_ts ASC"
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def get_between(self, start_ts: str, end_ts: str) -> List[JournalEntry]:
        rows = self._conn.execute(
            "SELECT * FROM trade_journal WHERE entry_ts >= ? AND entry_ts <= ? "
            "ORDER BY entry_ts ASC",
            (start_ts, end_ts),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def search(self, filter_fn: Callable[[JournalEntry], bool]) -> List[JournalEntry]:
        """
        Lambda-based search over all entries.

        Example::

            journal.search(lambda e: e.symbol == "BTC" and e.net_pnl > 100)
        """
        return [e for e in self.get_all() if filter_fn(e)]

    def count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM trade_journal"
        ).fetchone()[0]

    # -- Aggregates --

    def get_stats(self) -> JournalStats:
        """Compute aggregate statistics over all trades."""
        entries = self.get_all()
        if not entries:
            return JournalStats()

        total = len(entries)
        winners = [e for e in entries if e.net_pnl > 0]
        losers = [e for e in entries if e.net_pnl <= 0]
        win_rate = len(winners) / total
        loss_rate = 1.0 - win_rate

        avg_win = sum(e.net_pnl for e in winners) / len(winners) if winners else 0.0
        avg_loss = abs(sum(e.net_pnl for e in losers) / len(losers)) if losers else 0.0
        gross_profit = sum(e.net_pnl for e in winners)
        gross_loss = abs(sum(e.net_pnl for e in losers))

        best = max(entries, key=lambda e: e.net_pnl)
        worst = min(entries, key=lambda e: e.net_pnl)

        from collections import Counter
        sym_counts = Counter(e.symbol for e in entries)
        most_traded = sym_counts.most_common(1)[0][0] if sym_counts else ""

        by_symbol = dict(sym_counts)
        by_exit = dict(Counter(e.exit_reason for e in entries))

        # Win rate by vol regime
        by_vol: Dict[str, float] = {}
        for regime in VOL_REGIMES:
            sub = [e for e in entries if e.vol_regime == regime]
            if sub:
                by_vol[regime] = sum(1 for e in sub if e.net_pnl > 0) / len(sub)

        expectancy = avg_win * win_rate - avg_loss * loss_rate

        return JournalStats(
            total_trades=total,
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_pnl=sum(e.pnl for e in entries) / total,
            avg_net_pnl=sum(e.net_pnl for e in entries) / total,
            avg_hold_bars=sum(e.hold_bars for e in entries) / total,
            avg_hold_hours=sum(e.hold_duration_hours() for e in entries) / total,
            total_pnl=sum(e.pnl for e in entries),
            total_net_pnl=sum(e.net_pnl for e in entries),
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            expectancy=expectancy,
            best_trade_pnl=best.net_pnl,
            worst_trade_pnl=worst.net_pnl,
            best_trade_id=best.trade_id,
            worst_trade_id=worst.trade_id,
            most_traded_symbol=most_traded,
            avg_mfe_pct=sum(e.mfe_pct for e in entries) / total,
            avg_mae_pct=sum(e.mae_pct for e in entries) / total,
            rl_exit_rate=sum(1 for e in entries if e.was_rl_exit) / total,
            nav_gated_rate=sum(1 for e in entries if e.was_nav_gated) / total,
            event_filtered_rate=sum(
                1 for e in entries if e.was_event_calendar_filtered
            ) / total,
            by_symbol=by_symbol,
            by_exit_reason=by_exit,
            by_vol_regime=by_vol,
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# WeeklyReportData
# ---------------------------------------------------------------------------

@dataclass
class WeeklyReportData:
    week_start: str
    week_end: str
    total_trades: int
    net_pnl: float
    win_rate: float
    best_trades: List[JournalEntry]      # top 5 by net_pnl
    worst_trades: List[JournalEntry]     # bottom 5 by net_pnl
    # signal attribution: signal_name -> net_pnl_contribution
    signal_attribution: Dict[str, float]
    # regime breakdown: regime_label -> {"trades": int, "win_rate": float, "pnl": float}
    regime_breakdown: Dict[str, Dict[str, Any]]
    # day-by-day P&L within the week
    daily_pnl: Dict[str, float]
    # symbol breakdown
    symbol_pnl: Dict[str, float]
    avg_hold_bars: float
    avg_mfe_pct: float
    avg_mae_pct: float
    rl_exit_count: int
    nav_gated_count: int


# ---------------------------------------------------------------------------
# WeeklyReport
# ---------------------------------------------------------------------------

class WeeklyReport:
    """Generate weekly performance reports from a TradeJournal."""

    def __init__(self, journal: TradeJournal):
        self.journal = journal

    def generate(self, week_start: date) -> WeeklyReportData:
        """
        Build a WeeklyReportData for the 7-day period starting at week_start.
        week_start should be a Monday.
        """
        week_end = week_start + timedelta(days=6)
        start_ts = datetime.combine(week_start, datetime.min.time()).isoformat()
        end_ts = datetime.combine(week_end, datetime.max.time()).isoformat()

        entries = self.journal.get_between(start_ts, end_ts)

        if not entries:
            return WeeklyReportData(
                week_start=str(week_start),
                week_end=str(week_end),
                total_trades=0,
                net_pnl=0.0,
                win_rate=0.0,
                best_trades=[],
                worst_trades=[],
                signal_attribution={},
                regime_breakdown={},
                daily_pnl={},
                symbol_pnl={},
                avg_hold_bars=0.0,
                avg_mfe_pct=0.0,
                avg_mae_pct=0.0,
                rl_exit_count=0,
                nav_gated_count=0,
            )

        sorted_by_pnl = sorted(entries, key=lambda e: e.net_pnl, reverse=True)
        best_5 = sorted_by_pnl[:5]
        worst_5 = sorted_by_pnl[-5:]

        # Signal attribution -- simple bucketing by which gates fired
        sig_attr: Dict[str, float] = {
            "bh_only": 0.0,
            "cf_filtered": 0.0,
            "hurst_damped": 0.0,
            "nav_gated": 0.0,
            "ml_filtered": 0.0,
            "rl_exit": 0.0,
            "event_filtered": 0.0,
        }
        for e in entries:
            if e.was_cf_filtered:
                sig_attr["cf_filtered"] += e.net_pnl
            elif e.was_hurst_damped:
                sig_attr["hurst_damped"] += e.net_pnl
            elif e.was_nav_gated:
                sig_attr["nav_gated"] += e.net_pnl
            elif e.was_ml_filtered:
                sig_attr["ml_filtered"] += e.net_pnl
            elif e.was_event_calendar_filtered:
                sig_attr["event_filtered"] += e.net_pnl
            elif e.was_rl_exit:
                sig_attr["rl_exit"] += e.net_pnl
            else:
                sig_attr["bh_only"] += e.net_pnl

        # Regime breakdown by (vol_regime x hurst_regime)
        from collections import defaultdict
        regime_map: Dict[str, List[JournalEntry]] = defaultdict(list)
        for e in entries:
            key = f"{e.vol_regime}_vol / {e.hurst_regime}_hurst"
            regime_map[key].append(e)

        regime_breakdown: Dict[str, Dict[str, Any]] = {}
        for key, sub in regime_map.items():
            wr = sum(1 for x in sub if x.net_pnl > 0) / len(sub) if sub else 0.0
            regime_breakdown[key] = {
                "trades": len(sub),
                "win_rate": wr,
                "pnl": sum(x.net_pnl for x in sub),
            }

        # Daily P&L
        daily: Dict[str, float] = {}
        for e in entries:
            try:
                day = e.exit_ts[:10]
            except Exception:
                day = "unknown"
            daily[day] = daily.get(day, 0.0) + e.net_pnl

        # Symbol P&L
        sym_pnl: Dict[str, float] = {}
        for e in entries:
            sym_pnl[e.symbol] = sym_pnl.get(e.symbol, 0.0) + e.net_pnl

        total = len(entries)
        win_rate = sum(1 for e in entries if e.net_pnl > 0) / total

        return WeeklyReportData(
            week_start=str(week_start),
            week_end=str(week_end),
            total_trades=total,
            net_pnl=sum(e.net_pnl for e in entries),
            win_rate=win_rate,
            best_trades=best_5,
            worst_trades=worst_5,
            signal_attribution=sig_attr,
            regime_breakdown=regime_breakdown,
            daily_pnl=daily,
            symbol_pnl=sym_pnl,
            avg_hold_bars=sum(e.hold_bars for e in entries) / total,
            avg_mfe_pct=sum(e.mfe_pct for e in entries) / total,
            avg_mae_pct=sum(e.mae_pct for e in entries) / total,
            rl_exit_count=sum(1 for e in entries if e.was_rl_exit),
            nav_gated_count=sum(1 for e in entries if e.was_nav_gated),
        )

    # -- Formatters --

    def to_markdown(self, data: WeeklyReportData) -> str:
        """Render WeeklyReportData as a Markdown string."""
        lines: List[str] = []
        lines.append(f"# Weekly Trading Report: {data.week_start} to {data.week_end}")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- Total trades: {data.total_trades}")
        lines.append(f"- Net P&L: ${data.net_pnl:,.2f}")
        lines.append(f"- Win rate: {data.win_rate:.1%}")
        lines.append(f"- Avg hold: {data.avg_hold_bars:.1f} bars")
        lines.append(f"- Avg MFE: {data.avg_mfe_pct:.2%}  |  Avg MAE: {data.avg_mae_pct:.2%}")
        lines.append(f"- RL exits: {data.rl_exit_count}  |  NAV gated: {data.nav_gated_count}")
        lines.append("")

        # Daily P&L table
        lines.append("## Daily P&L")
        lines.append("| Date | P&L |")
        lines.append("|------|-----|")
        for day in sorted(data.daily_pnl):
            lines.append(f"| {day} | ${data.daily_pnl[day]:,.2f} |")
        lines.append("")

        # Symbol breakdown
        lines.append("## Symbol Breakdown")
        lines.append("| Symbol | Net P&L |")
        lines.append("|--------|---------|")
        for sym, pnl in sorted(data.symbol_pnl.items(), key=lambda x: -x[1]):
            lines.append(f"| {sym} | ${pnl:,.2f} |")
        lines.append("")

        # Best trades
        lines.append("## Top 5 Trades")
        lines.append("| ID | Symbol | P&L | Exit Reason | Hold Bars |")
        lines.append("|----|--------|-----|-------------|-----------|")
        for e in data.best_trades:
            short_id = e.trade_id[:8]
            lines.append(
                f"| {short_id} | {e.symbol} | ${e.net_pnl:,.2f} "
                f"| {e.exit_reason} | {e.hold_bars} |"
            )
        lines.append("")

        # Worst trades
        lines.append("## Bottom 5 Trades")
        lines.append("| ID | Symbol | P&L | Exit Reason | Hold Bars |")
        lines.append("|----|--------|-----|-------------|-----------|")
        for e in data.worst_trades:
            short_id = e.trade_id[:8]
            lines.append(
                f"| {short_id} | {e.symbol} | ${e.net_pnl:,.2f} "
                f"| {e.exit_reason} | {e.hold_bars} |"
            )
        lines.append("")

        # Signal attribution
        lines.append("## Signal Attribution")
        lines.append("| Layer | Net P&L |")
        lines.append("|-------|---------|")
        for layer, pnl in data.signal_attribution.items():
            lines.append(f"| {layer} | ${pnl:,.2f} |")
        lines.append("")

        # Regime breakdown
        lines.append("## Regime Breakdown")
        lines.append("| Regime | Trades | Win Rate | P&L |")
        lines.append("|--------|--------|----------|-----|")
        for regime, info in data.regime_breakdown.items():
            lines.append(
                f"| {regime} | {info['trades']} | {info['win_rate']:.1%} "
                f"| ${info['pnl']:,.2f} |"
            )
        lines.append("")

        return "\n".join(lines)

    def to_html(self, data: WeeklyReportData) -> str:
        """
        Render WeeklyReportData as HTML with embedded Plotly charts.
        Falls back gracefully if plotly is not installed.
        """
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            has_plotly = True
        except ImportError:
            has_plotly = False

        parts: List[str] = []
        parts.append(
            "<!DOCTYPE html><html><head>"
            "<meta charset='utf-8'>"
            f"<title>Weekly Report {data.week_start}</title>"
            "<style>"
            "body { font-family: Arial, sans-serif; margin: 40px; background: #0d1117; color: #c9d1d9; }"
            "table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }"
            "th, td { border: 1px solid #30363d; padding: 8px 12px; text-align: left; }"
            "th { background: #161b22; }"
            "h1, h2 { color: #58a6ff; }"
            ".positive { color: #3fb950; } .negative { color: #f85149; }"
            "</style>"
            "</head><body>"
        )
        parts.append(f"<h1>Weekly Trading Report: {data.week_start} -- {data.week_end}</h1>")

        # Summary grid
        parts.append("<h2>Summary</h2><table>")
        parts.append(
            f"<tr><th>Total Trades</th><td>{data.total_trades}</td>"
            f"<th>Net P&L</th>"
            f"<td class='{'positive' if data.net_pnl >= 0 else 'negative'}'>"
            f"${data.net_pnl:,.2f}</td></tr>"
        )
        parts.append(
            f"<tr><th>Win Rate</th><td>{data.win_rate:.1%}</td>"
            f"<th>Avg Hold</th><td>{data.avg_hold_bars:.1f} bars</td></tr>"
        )
        parts.append(
            f"<tr><th>Avg MFE</th><td>{data.avg_mfe_pct:.2%}</td>"
            f"<th>Avg MAE</th><td>{data.avg_mae_pct:.2%}</td></tr>"
        )
        parts.append(
            f"<tr><th>RL Exits</th><td>{data.rl_exit_count}</td>"
            f"<th>NAV Gated</th><td>{data.nav_gated_count}</td></tr>"
        )
        parts.append("</table>")

        # Daily P&L chart
        if has_plotly and data.daily_pnl:
            days = sorted(data.daily_pnl.keys())
            pnl_vals = [data.daily_pnl[d] for d in days]
            colors = ["#3fb950" if v >= 0 else "#f85149" for v in pnl_vals]
            fig = go.Figure(
                go.Bar(x=days, y=pnl_vals, marker_color=colors, name="Daily P&L")
            )
            fig.update_layout(
                title="Daily P&L",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                height=300,
            )
            parts.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))
        else:
            # Plain HTML table fallback
            parts.append("<h2>Daily P&L</h2><table><tr><th>Date</th><th>P&L</th></tr>")
            for day in sorted(data.daily_pnl):
                v = data.daily_pnl[day]
                cls = "positive" if v >= 0 else "negative"
                parts.append(f"<tr><td>{day}</td><td class='{cls}'>${v:,.2f}</td></tr>")
            parts.append("</table>")

        # Symbol P&L chart
        if has_plotly and data.symbol_pnl:
            syms = list(data.symbol_pnl.keys())
            sym_vals = [data.symbol_pnl[s] for s in syms]
            sym_colors = ["#3fb950" if v >= 0 else "#f85149" for v in sym_vals]
            fig2 = go.Figure(
                go.Bar(x=syms, y=sym_vals, marker_color=sym_colors, name="Symbol P&L")
            )
            fig2.update_layout(
                title="P&L by Symbol",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font_color="#c9d1d9",
                height=300,
            )
            parts.append(pio.to_html(fig2, full_html=False, include_plotlyjs=False))

        # Best / worst trades tables
        parts.append("<h2>Top 5 Trades</h2>")
        parts.append(
            "<table><tr><th>ID</th><th>Symbol</th><th>P&L</th>"
            "<th>Exit Reason</th><th>Hold Bars</th></tr>"
        )
        for e in data.best_trades:
            parts.append(
                f"<tr><td>{e.trade_id[:8]}</td><td>{e.symbol}</td>"
                f"<td class='positive'>${e.net_pnl:,.2f}</td>"
                f"<td>{e.exit_reason}</td><td>{e.hold_bars}</td></tr>"
            )
        parts.append("</table>")

        parts.append("<h2>Bottom 5 Trades</h2>")
        parts.append(
            "<table><tr><th>ID</th><th>Symbol</th><th>P&L</th>"
            "<th>Exit Reason</th><th>Hold Bars</th></tr>"
        )
        for e in data.worst_trades:
            cls = "negative" if e.net_pnl < 0 else "positive"
            parts.append(
                f"<tr><td>{e.trade_id[:8]}</td><td>{e.symbol}</td>"
                f"<td class='{cls}'>${e.net_pnl:,.2f}</td>"
                f"<td>{e.exit_reason}</td><td>{e.hold_bars}</td></tr>"
            )
        parts.append("</table>")

        # Signal attribution
        parts.append("<h2>Signal Attribution</h2>")
        parts.append("<table><tr><th>Layer</th><th>Net P&L</th></tr>")
        for layer, pnl in data.signal_attribution.items():
            cls = "positive" if pnl >= 0 else "negative"
            parts.append(
                f"<tr><td>{layer}</td><td class='{cls}'>${pnl:,.2f}</td></tr>"
            )
        parts.append("</table>")

        # Regime breakdown
        parts.append("<h2>Regime Breakdown</h2>")
        parts.append(
            "<table><tr><th>Regime</th><th>Trades</th>"
            "<th>Win Rate</th><th>P&L</th></tr>"
        )
        for regime, info in data.regime_breakdown.items():
            cls = "positive" if info["pnl"] >= 0 else "negative"
            parts.append(
                f"<tr><td>{regime}</td><td>{info['trades']}</td>"
                f"<td>{info['win_rate']:.1%}</td>"
                f"<td class='{cls}'>${info['pnl']:,.2f}</td></tr>"
            )
        parts.append("</table>")

        parts.append("</body></html>")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _make_sample_entry(
    symbol: str = "BTC",
    net_pnl: float = 250.0,
    hold_bars: int = 12,
    exit_reason: str = "tp_hit",
) -> JournalEntry:
    """Create a realistic sample entry for testing / demos."""
    now = datetime.utcnow()
    entry_ts = (now - timedelta(hours=hold_bars)).isoformat()
    exit_ts = now.isoformat()
    entry_price = 65_000.0 if symbol in ("BTC",) else 3_200.0
    exit_price = entry_price + (net_pnl / 0.01)  # assumes 0.01 BTC qty
    return JournalEntry(
        symbol=symbol,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        hold_bars=hold_bars,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=0.01,
        side="long",
        pnl=net_pnl + 5.0,
        pnl_pct=net_pnl / (entry_price * 0.01),
        commission=2.50,
        slippage=2.50,
        net_pnl=net_pnl,
        entry_close=entry_price,
        entry_volume=1_500.0,
        atr_14=800.0,
        spread_bps=3.5,
        vol_regime="med",
        session="us",
        bh_mass_15m=0.72,
        bh_mass_1h=0.68,
        bh_mass_4h=0.61,
        bh_active=True,
        cf_direction=1,
        cf_alignment=0.85,
        hurst_h=0.62,
        hurst_regime="trending",
        nav_omega=0.15,
        nav_geodesic=0.08,
        nav_quaternion=json.dumps([0.92, 0.21, 0.31, 0.12]),
        ml_signal=0.55,
        ml_confidence=0.78,
        granger_btc_corr=1.0,
        granger_eth_corr=0.72,
        was_rl_exit=(exit_reason == "rl_exit"),
        was_event_calendar_filtered=False,
        was_nav_gated=False,
        was_hurst_damped=False,
        was_cf_filtered=False,
        was_ml_filtered=False,
        mfe_pct=0.025,
        mae_pct=0.008,
        mfe_price=entry_price * 1.025,
        mae_price=entry_price * 0.992,
        exit_reason=exit_reason,
        exit_signal_value=0.0,
        portfolio_nav_at_entry=50_000.0,
        position_size_pct=0.02,
        tags="sample,demo",
        notes="",
    )


if __name__ == "__main__":
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_file = f.name

    with TradeJournal(db_file) as journal:
        for sym, pnl, bars in [
            ("BTC", 350.0, 8),
            ("ETH", -120.0, 5),
            ("SOL", 90.0, 14),
            ("BTC", 210.0, 10),
            ("AAPL", -50.0, 3),
        ]:
            journal.add_entry(_make_sample_entry(sym, pnl, bars))

        stats = journal.get_stats()
        print(f"Total trades: {stats.total_trades}")
        print(f"Win rate: {stats.win_rate:.1%}")
        print(f"Total net P&L: ${stats.total_net_pnl:,.2f}")
        print(f"Most traded: {stats.most_traded_symbol}")

        reporter = WeeklyReport(journal)
        week_start = date.today() - timedelta(days=date.today().weekday())
        report_data = reporter.generate(week_start)
        md = reporter.to_markdown(report_data)
        print("\n--- Weekly Report (Markdown excerpt) ---")
        print("\n".join(md.splitlines()[:20]))
