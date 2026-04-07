"""
tools/trade_replay.py
=====================
Replays historical trades for debugging and analysis.

Re-runs LARSA signal logic on bar data to reconstruct what happened
at entry and exit, finds similar trades by feature cosine similarity,
diagnoses losing trades, and plots trade windows.

Usage:
    python tools/trade_replay.py --replay-trade 42
    python tools/trade_replay.py --similar-trades 42 --n 10
    python tools/trade_replay.py --diagnose 42
    python tools/trade_replay.py --plot 42
    python tools/trade_replay.py --list-trades --symbol BTC
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("trade_replay")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

_REPO_ROOT = Path(__file__).parents[1]
_DB_PATH   = _REPO_ROOT / "execution" / "live_trades.db"

# BH physics constants (mirrored from live_trader_alpaca.py defaults)
_DEFAULT_CF_15M  = 0.010
_DEFAULT_BH_FORM = 1.5
_DEFAULT_DECAY   = 0.95
_HURST_WINDOW    = 50


# ─────────────────────────────────────────────────────────────────────────────
# BH mass reconstruction (lightweight, no live dependencies)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_bh_mass(
    closes: list[float],
    cf: float = _DEFAULT_CF_15M,
    form: float = _DEFAULT_BH_FORM,
    decay: float = _DEFAULT_DECAY,
) -> list[float]:
    """Reconstruct BH mass accumulation from a close price series."""
    masses: list[float] = []
    mass = 0.0
    prev = closes[0] if closes else 1.0
    for c in closes:
        if prev > 0:
            ret = (c - prev) / prev
        else:
            ret = 0.0
        impulse = cf * abs(ret) ** form
        mass = decay * mass + impulse
        masses.append(mass)
        prev = c
    return masses


def _parse_ts(ts_str: str) -> pd.Timestamp:
    """Parse an ISO timestamp string into a UTC-aware pd.Timestamp."""
    ts = pd.Timestamp(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _rs_hurst(log_prices: list[float]) -> float | None:
    """R/S Hurst exponent estimate. Returns None if too few points."""
    n = len(log_prices)
    if n < 20:
        return None
    prices = np.array(log_prices, dtype=float)
    chunk_sizes = [n // k for k in range(2, min(8, n // 4 + 1)) if n // k >= 4]
    if not chunk_sizes:
        return None
    rs_vals = []
    sizes   = []
    for size in chunk_sizes:
        rs_chunk = []
        for start in range(0, n - size + 1, size):
            chunk = prices[start: start + size]
            mean  = chunk.mean()
            dev   = np.cumsum(chunk - mean)
            r     = dev.max() - dev.min()
            s     = chunk.std()
            if s > 1e-12:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_vals.append(np.mean(rs_chunk))
            sizes.append(size)
    if len(rs_vals) < 2:
        return None
    log_rs   = np.log(rs_vals)
    log_n    = np.log(sizes)
    try:
        h, _ = np.polyfit(log_n, log_rs, 1)
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return None


def _nav_omega_approx(closes: list[float]) -> float:
    """Simplified angular velocity proxy from close price momentum."""
    if len(closes) < 4:
        return 0.0
    arr = np.array(closes[-8:], dtype=float)
    diffs = np.diff(arr)
    angles = np.arctan(diffs / (arr[:-1].clip(min=1e-9)))
    return float(np.abs(angles).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def _open_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def load_all_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    """Load all completed trades from trade_pnl."""
    try:
        df = pd.read_sql_query(
            "SELECT * FROM trade_pnl ORDER BY entry_time", conn
        )
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"]  = pd.to_datetime(df["exit_time"])
        return df
    except Exception as exc:
        log.warning("load_all_trades: %s", exc)
        return pd.DataFrame()


def load_bars_for_symbol(
    conn: sqlite3.Connection,
    symbol: str,
    start: datetime,
    end: datetime,
    padding_bars: int = 50,
) -> pd.DataFrame:
    """Load bar data for a symbol with padding before the window."""
    pad_start = start - timedelta(minutes=15 * padding_bars)
    try:
        df = pd.read_sql_query(
            "SELECT ts, open, high, low, close, volume FROM bar_data "
            "WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts",
            conn,
            params=(symbol, pad_start.isoformat(), end.isoformat()),
        )
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")
    except Exception:
        return pd.DataFrame()


def load_nav_state_for_trade(
    conn: sqlite3.Connection,
    symbol: str,
    entry_time: datetime,
    exit_time: datetime,
) -> pd.DataFrame:
    """Load nav_state rows for a trade window."""
    try:
        df = pd.read_sql_query(
            "SELECT ts, bh_mass_15m, bh_mass_1h, hurst_h, nav_omega, signal_strength "
            "FROM nav_state WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts",
            conn,
            params=(symbol, entry_time.isoformat(), exit_time.isoformat()),
        )
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts")
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Feature vector
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_vector(snapshot: dict) -> np.ndarray:
    """
    Build a normalised feature vector for cosine similarity comparison.
    Dimensions: [bh_mass_15m, bh_mass_1h, hurst_h, nav_omega, signal_strength,
                 entry_pnl_pct_proxy, hold_bars_norm]
    """
    fields = [
        snapshot.get("bh_mass_at_entry", 0.0),
        snapshot.get("bh_mass_1h_at_entry", 0.0),
        float(snapshot.get("hurst_h", 0.5) or 0.5),
        min(float(snapshot.get("nav_omega", 0.0) or 0.0), 1.0),
        float(snapshot.get("signal_strength", 0.0) or 0.0),
        float(snapshot.get("entry_price", 1.0) or 1.0) / 100000.0,
        min(float(snapshot.get("hold_bars", 1) or 1) / 100.0, 1.0),
    ]
    vec = np.array(fields, dtype=float)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TradeReplayer:
    """
    Replays and analyses historical trades from the SRFM SQLite database.

    Usage:
        replayer = TradeReplayer()
        info = replayer.replay_trade(42)
        similar = replayer.find_similar_trades(42, n=10)
        diag = replayer.diagnose_bad_trade(42)
        replayer.plot_trade(42)
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path   = Path(db_path) if db_path else _DB_PATH
        self._conn:    sqlite3.Connection | None = None
        self._trades:  pd.DataFrame | None = None
        self._feature_cache: dict[int, np.ndarray] = {}

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _open_db(self.db_path)
        return self._conn

    def _get_trades(self) -> pd.DataFrame:
        if self._trades is None:
            self._trades = load_all_trades(self._get_conn())
        return self._trades

    def _get_trade_row(self, trade_id: int) -> dict:
        trades = self._get_trades()
        if trades.empty:
            raise ValueError("No trades in database.")
        row = trades[trades["id"] == trade_id]
        if row.empty:
            raise ValueError(f"Trade {trade_id} not found.")
        return row.iloc[0].to_dict()

    # -- Core replay --

    def replay_trade(self, trade_id: int) -> dict:
        """
        Reconstruct the full signal context for a completed trade.

        Returns a dict with:
        - bh_mass_at_entry, bh_mass_1h_at_entry
        - hurst_h at entry
        - nav_omega at entry
        - signal_strength at entry
        - what triggered entry (signal source)
        - what triggered exit (reason)
        - reconstructed masses over trade window
        """
        trade = self._get_trade_row(trade_id)
        conn  = self._get_conn()
        sym   = trade.get("symbol", "")

        entry_time = _parse_ts(trade.get("entry_time") or "1970-01-01")
        exit_time  = _parse_ts(trade.get("exit_time")  or "1970-01-01")

        result: dict = {
            "trade_id":     trade_id,
            "symbol":       sym,
            "entry_time":   str(entry_time)[:19],
            "exit_time":    str(exit_time)[:19],
            "entry_price":  trade.get("entry_price"),
            "exit_price":   trade.get("exit_price"),
            "qty":          trade.get("qty"),
            "pnl":          trade.get("pnl"),
            "hold_bars":    trade.get("hold_bars"),
        }

        # Load nav_state records if available
        nav_df = load_nav_state_for_trade(conn, sym, entry_time.to_pydatetime(), exit_time.to_pydatetime())
        if not nav_df.empty:
            entry_snap = nav_df.iloc[0]
            result["bh_mass_at_entry"]    = float(entry_snap.get("bh_mass_15m", 0.0) or 0.0)
            result["bh_mass_1h_at_entry"] = float(entry_snap.get("bh_mass_1h", 0.0) or 0.0)
            result["hurst_h"]             = float(entry_snap.get("hurst_h", 0.5) or 0.5)
            result["nav_omega"]           = float(entry_snap.get("nav_omega", 0.0) or 0.0)
            result["signal_strength"]     = float(entry_snap.get("signal_strength", 0.0) or 0.0)
        else:
            # Reconstruct from bar data
            bars = load_bars_for_symbol(conn, sym, entry_time.to_pydatetime(), exit_time.to_pydatetime())
            if not bars.empty:
                closes = bars["close"].tolist()
                masses = _reconstruct_bh_mass(closes)
                # Find the bar at entry
                entry_idx = 0
                for i, ts in enumerate(bars.index):
                    if ts >= entry_time:
                        entry_idx = i
                        break
                result["bh_mass_at_entry"]    = masses[entry_idx] if entry_idx < len(masses) else 0.0
                result["bh_mass_1h_at_entry"] = masses[min(entry_idx + 3, len(masses) - 1)]
                result["reconstructed_masses"]= masses
                # Hurst from pre-entry window
                pre_closes = closes[:entry_idx] if entry_idx > 0 else closes
                h = _rs_hurst(pre_closes[-_HURST_WINDOW:])
                result["hurst_h"]  = h if h is not None else 0.5
                result["nav_omega"] = _nav_omega_approx(pre_closes[-8:])
                result["signal_strength"] = masses[entry_idx] if entry_idx < len(masses) else 0.0
            else:
                result["bh_mass_at_entry"]    = 0.0
                result["bh_mass_1h_at_entry"] = 0.0
                result["hurst_h"]             = 0.5
                result["nav_omega"]           = 0.0
                result["signal_strength"]     = 0.0

        # Infer entry/exit triggers
        bh = result.get("bh_mass_at_entry", 0.0)
        result["entry_trigger"] = _infer_entry_trigger(result)
        result["exit_trigger"]  = _infer_exit_trigger(trade, bh)

        return result

    def find_similar_trades(self, trade_id: int, n: int = 10) -> list[dict]:
        """
        Find the N most similar trades by cosine similarity of entry feature vector.

        Returns list of dicts with trade_id, symbol, pnl, similarity.
        """
        trades = self._get_trades()
        if trades.empty:
            return []

        # Get feature for reference trade
        ref_info = self.replay_trade(trade_id)
        ref_vec  = _build_feature_vector(ref_info)
        self._feature_cache[trade_id] = ref_vec

        results = []
        for _, row in trades.iterrows():
            tid = int(row.get("id", 0))
            if tid == trade_id:
                continue
            if tid in self._feature_cache:
                vec = self._feature_cache[tid]
            else:
                try:
                    info = self.replay_trade(tid)
                    vec  = _build_feature_vector(info)
                    self._feature_cache[tid] = vec
                except Exception:
                    continue
            sim = _cosine_similarity(ref_vec, vec)
            results.append({
                "trade_id":   tid,
                "symbol":     row.get("symbol", "?"),
                "entry_time": str(row.get("entry_time", ""))[:19],
                "pnl":        float(row.get("pnl", 0.0) or 0.0),
                "hold_bars":  int(row.get("hold_bars", 0) or 0),
                "similarity": sim,
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:n]

    def diagnose_bad_trade(self, trade_id: int) -> str:
        """
        For a losing trade, explain what went wrong.
        Returns a human-readable diagnostic string.
        """
        trade = self._get_trade_row(trade_id)
        pnl   = float(trade.get("pnl", 0.0) or 0.0)
        info  = self.replay_trade(trade_id)

        lines: list[str] = []
        lines.append(f"Trade #{trade_id}  {info.get('symbol','')}  P&L: {pnl:+.4f}")
        lines.append(f"Entry: {info.get('entry_time','')}  Exit: {info.get('exit_time','')}")
        lines.append(f"Hold: {info.get('hold_bars', 0)} bars")
        lines.append("")

        # Analyse entry quality
        bh15  = float(info.get("bh_mass_at_entry", 0.0) or 0.0)
        bh1h  = float(info.get("bh_mass_1h_at_entry", 0.0) or 0.0)
        hurst = float(info.get("hurst_h", 0.5) or 0.5)
        omega = float(info.get("nav_omega", 0.0) or 0.0)
        sig   = float(info.get("signal_strength", 0.0) or 0.0)
        hold  = int(info.get("hold_bars", 0) or 0)

        lines.append("ENTRY CONDITIONS:")
        bh15_quality = "STRONG" if bh15 > 0.6 else "MODERATE" if bh15 > 0.3 else "WEAK"
        lines.append(f"  BH mass 15m: {bh15:.3f}  ({bh15_quality})")
        bh1h_quality = "STRONG" if bh1h > 0.5 else "MODERATE" if bh1h > 0.2 else "WEAK"
        lines.append(f"  BH mass 1h:  {bh1h:.3f}  ({bh1h_quality})")
        hurst_label = "TRENDING" if hurst > 0.58 else "MEAN-REVERTING" if hurst < 0.42 else "RANDOM"
        lines.append(f"  Hurst H:     {hurst:.3f}  ({hurst_label})")
        lines.append(f"  NavOmega:    {omega:.5f}")
        lines.append(f"  Signal str:  {sig:.3f}")
        lines.append("")

        lines.append("DIAGNOSIS:")
        issues: list[str] = []

        if bh15 < 0.25:
            issues.append("-- Entry BH mass 15m was LOW (<0.25): signal was weak, "
                          "possibly entered on noise rather than true BH impulse.")
        if bh1h < 0.15:
            issues.append("-- BH mass on 1h timeframe was LOW: insufficient multi-TF confirmation.")
        if hurst < 0.42:
            issues.append("-- Market was MEAN-REVERTING (Hurst < 0.42) at entry: "
                          "trend-following BH signal has reduced edge in MR regimes.")
        if hurst > 0.58 and pnl < 0:
            issues.append("-- Market was TRENDING but trade lost: possible adverse selection "
                          "or entry timing was off (entered into exhausted trend).")
        if hold <= 2 and pnl < 0:
            issues.append("-- Very short hold (<= 2 bars): likely a stop-out from spread/slippage "
                          "or immediate adverse price action.")
        if hold > 50 and pnl < 0:
            issues.append("-- Very long hold (> 50 bars) with negative P&L: BH mass did not "
                          "sustain. Position held too long past signal deterioration.")
        if omega < 1e-5 and pnl < 0:
            issues.append("-- Nav angular velocity was near zero at entry: momentum was absent, "
                          "QuatNav gate should have blocked this entry.")

        if not issues:
            issues.append("-- No clear single failure mode. Loss may be within normal variance "
                          "for this signal type. Check macro regime and spread costs.")

        lines.extend(issues)
        lines.append("")
        lines.append(f"Exit trigger: {info.get('exit_trigger','unknown')}")
        lines.append(f"Entry trigger: {info.get('entry_trigger','unknown')}")

        return "\n".join(lines)

    def plot_trade(self, trade_id: int) -> None:
        """
        Plot price + BH mass + signal overlay for the trade window using matplotlib.
        Saves to trade_replay_<id>.png if display is unavailable.
        """
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            log.error("matplotlib not installed. Run: pip install matplotlib")
            return

        trade = self._get_trade_row(trade_id)
        conn  = self._get_conn()
        sym   = trade.get("symbol", "")
        entry_time = _parse_ts(trade.get("entry_time") or "1970-01-01")
        exit_time  = _parse_ts(trade.get("exit_time")  or "1970-01-01")
        pnl        = float(trade.get("pnl", 0.0) or 0.0)

        bars = load_bars_for_symbol(conn, sym, entry_time.to_pydatetime(), exit_time.to_pydatetime())
        if bars.empty:
            log.warning("No bar data found for trade %d (%s)", trade_id, sym)
            return

        closes = bars["close"].tolist()
        masses = _reconstruct_bh_mass(closes)

        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.5, 1]})
        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#ccc")
            ax.yaxis.label.set_color("#ccc")
            ax.xaxis.label.set_color("#ccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        ts_index = bars.index

        # -- Price panel --
        ax0 = axes[0]
        ax0.plot(ts_index, closes, color="#4fc3f7", linewidth=1.2, label="Close")
        # Entry/exit lines
        ax0.axvline(entry_time, color="#4caf50", linewidth=1.5, linestyle="--", label="Entry")
        ax0.axvline(exit_time,  color="#f44336", linewidth=1.5, linestyle="--", label="Exit")
        # Shade trade window
        ax0.axvspan(entry_time, exit_time, alpha=0.12, color="#4fc3f7")
        ax0.set_ylabel("Price")
        ax0.legend(loc="upper left", facecolor="#222", labelcolor="#ccc", fontsize=8)
        color_pnl = "#4caf50" if pnl >= 0 else "#f44336"
        ax0.set_title(
            f"Trade #{trade_id}  {sym}  P&L: {pnl:+.4f}  Hold: {trade.get('hold_bars',0)} bars",
            color=color_pnl, fontsize=11,
        )

        # -- BH mass panel --
        ax1 = axes[1]
        ax1.fill_between(ts_index, masses, alpha=0.5, color="#ab47bc", label="BH Mass 15m")
        ax1.axhline(0.3, color="#ffeb3b", linewidth=0.8, linestyle=":", alpha=0.7)
        ax1.axhline(0.6, color="#ff9800", linewidth=0.8, linestyle=":", alpha=0.7)
        ax1.axvline(entry_time, color="#4caf50", linewidth=1.5, linestyle="--")
        ax1.axvline(exit_time,  color="#f44336", linewidth=1.5, linestyle="--")
        ax1.set_ylabel("BH Mass")
        ax1.legend(loc="upper left", facecolor="#222", labelcolor="#ccc", fontsize=8)
        ax1.set_ylim(0, max(max(masses) * 1.2, 0.1))

        # -- Volume panel --
        ax2 = axes[2]
        volumes = bars.get("volume", pd.Series(dtype=float))
        if not volumes.empty:
            ax2.bar(ts_index, volumes.values, color="#607d8b", alpha=0.7, width=0.008)
            ax2.set_ylabel("Volume")
        ax2.axvline(entry_time, color="#4caf50", linewidth=1.5, linestyle="--")
        ax2.axvline(exit_time,  color="#f44336", linewidth=1.5, linestyle="--")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=30)

        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            out_path = Path(f"trade_replay_{trade_id}.png")
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            log.info("Plot saved to %s", out_path)
        plt.close(fig)

    def list_trades(
        self,
        symbol: str | None = None,
        losing_only: bool = False,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Return a summary DataFrame of trades, optionally filtered."""
        trades = self._get_trades()
        if trades.empty:
            return trades
        if symbol:
            trades = trades[trades["symbol"].str.upper() == symbol.upper()]
        if losing_only:
            trades = trades[trades["pnl"] < 0]
        cols = ["id", "symbol", "entry_time", "exit_time", "entry_price",
                "exit_price", "qty", "pnl", "hold_bars"]
        cols = [c for c in cols if c in trades.columns]
        return trades[cols].tail(limit).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry / exit trigger inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_entry_trigger(info: dict) -> str:
    bh   = float(info.get("bh_mass_at_entry", 0.0) or 0.0)
    bh1h = float(info.get("bh_mass_1h_at_entry", 0.0) or 0.0)
    h    = float(info.get("hurst_h", 0.5) or 0.5)
    sig  = float(info.get("signal_strength", 0.0) or 0.0)

    triggers = []
    if bh > 0.4:
        triggers.append("BH_15m_fire")
    if bh1h > 0.3:
        triggers.append("BH_1h_confirm")
    if h > 0.58:
        triggers.append("Hurst_trending")
    if sig > 0.5:
        triggers.append("ML_signal")
    return " + ".join(triggers) if triggers else "BH_15m_fire (baseline)"


def _infer_exit_trigger(trade: dict, bh_at_entry: float) -> str:
    pnl       = float(trade.get("pnl", 0.0) or 0.0)
    hold_bars = int(trade.get("hold_bars", 0) or 0)
    ep        = float(trade.get("entry_price", 1.0) or 1.0)
    xp        = float(trade.get("exit_price", ep) or ep)

    pnl_pct = (xp - ep) / ep if ep > 0 else 0.0

    if pnl_pct <= -0.05:
        return "stop_loss"
    if pnl_pct >= 0.10:
        return "take_profit"
    if hold_bars >= 96:  # 24h
        return "max_hold_bars"
    if bh_at_entry > 0 and pnl_pct < 0:
        return "bh_mass_decay"
    return "signal_reversal"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LARSA trade replay and diagnostics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--replay-trade", type=int, metavar="ID",
                       help="Replay a single trade by ID")
    group.add_argument("--similar-trades", type=int, metavar="ID",
                       help="Find trades similar to the given trade")
    group.add_argument("--diagnose", type=int, metavar="ID",
                       help="Diagnose a losing trade")
    group.add_argument("--plot", type=int, metavar="ID",
                       help="Plot price + BH mass for a trade")
    group.add_argument("--list-trades", action="store_true",
                       help="List all trades")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of similar trades to return")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Filter by symbol")
    parser.add_argument("--losing-only", action="store_true",
                        help="Show only losing trades in --list-trades")
    parser.add_argument("--db", type=str, default=str(_DB_PATH))
    return parser.parse_args()


def main() -> None:
    args     = _parse_args()
    replayer = TradeReplayer(db_path=args.db)

    if args.replay_trade is not None:
        try:
            info = replayer.replay_trade(args.replay_trade)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"\nTrade #{args.replay_trade} Replay")
        print("=" * 50)
        for k, v in info.items():
            if k == "reconstructed_masses":
                masses = v
                print(f"  reconstructed_masses: [{masses[0]:.4f} ... {masses[-1]:.4f}]  ({len(masses)} bars)")
            else:
                print(f"  {k}: {v}")

    elif args.similar_trades is not None:
        try:
            similar = replayer.find_similar_trades(args.similar_trades, n=args.n)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"\nTop {args.n} trades similar to #{args.similar_trades}")
        print("=" * 60)
        print(f"  {'ID':>6}  {'Symbol':<8}  {'P&L':>10}  {'Bars':>5}  {'Similarity':>10}")
        for t in similar:
            print(
                f"  {t['trade_id']:>6}  {t['symbol']:<8}  "
                f"{t['pnl']:>+10.4f}  {t['hold_bars']:>5}  {t['similarity']:>10.4f}"
            )

    elif args.diagnose is not None:
        try:
            diag = replayer.diagnose_bad_trade(args.diagnose)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print("\n" + diag)

    elif args.plot is not None:
        try:
            replayer.plot_trade(args.plot)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    elif args.list_trades:
        try:
            df = replayer.list_trades(symbol=args.symbol, losing_only=args.losing_only)
        except Exception as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        if df.empty:
            print("No trades found.")
        else:
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
