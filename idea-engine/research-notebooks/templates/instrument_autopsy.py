"""
InstrumentAutopsyTemplate: deep-dive into a single instrument's performance.

Analysis sections:
  1. P&L by year / month / hour of day
  2. Hold duration distribution
  3. Entry signal quality (IC of entry signal vs forward return)
  4. Exit quality (did we exit near the local peak/trough?)
  5. Comparison vs passive buy-and-hold

Output: InstrumentReport dataclass.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InstrumentReport:
    """Complete autopsy report for one instrument."""

    symbol: str
    n_trades: int
    total_pnl: float
    win_rate: float
    avg_pnl_per_trade: float
    sharpe: float
    max_drawdown: float
    pnl_by_year: Dict[str, float]
    pnl_by_month: Dict[int, float]
    pnl_by_hour: Dict[int, float]
    win_rate_by_hour: Dict[int, float]
    hold_duration_stats: Dict[str, float]   # min, max, mean, median, p90
    entry_ic: float                          # spearman IC of entry vs fwd return
    exit_quality_score: float               # 0-1, 1 = perfect exit timing
    bah_return_pct: float                   # buy-and-hold return over same period
    strategy_return_pct: float
    alpha_vs_bah: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_trades": self.n_trades,
            "total_pnl": round(self.total_pnl, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 6),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "pnl_by_year": {k: round(v, 4) for k, v in self.pnl_by_year.items()},
            "pnl_by_month": {k: round(v, 4) for k, v in self.pnl_by_month.items()},
            "pnl_by_hour": {k: round(v, 4) for k, v in self.pnl_by_hour.items()},
            "win_rate_by_hour": {k: round(v, 4) for k, v in self.win_rate_by_hour.items()},
            "hold_duration_stats": {k: round(v, 2) for k, v in self.hold_duration_stats.items()},
            "entry_ic": round(self.entry_ic, 4),
            "exit_quality_score": round(self.exit_quality_score, 4),
            "bah_return_pct": round(self.bah_return_pct, 4),
            "strategy_return_pct": round(self.strategy_return_pct, 4),
            "alpha_vs_bah": round(self.alpha_vs_bah, 4),
            "notes": self.notes,
        }


class InstrumentAutopsyTemplate:
    """
    Deep-dive analysis for a single instrument's trade history.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Required columns: pnl, entry_time (datetime), exit_time (datetime)
        Optional: entry_signal (float), exit_price, entry_price,
                  hold_bars (int), max_favorable_price (float)
    prices_df : pd.DataFrame
        Optional buy-and-hold reference: index=datetime, columns=['close']

    Usage::

        template = InstrumentAutopsyTemplate()
        report   = template.run(trades_df=df, symbol="SOL", prices_df=prices)
    """

    def run(
        self,
        trades_df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        prices_df: Optional[pd.DataFrame] = None,
    ) -> InstrumentReport:
        df = trades_df.copy().dropna(subset=["pnl"])
        notes: List[str] = []
        n = len(df)

        if n < 5:
            notes.append("Fewer than 5 trades — statistics are not meaningful.")

        # Ensure datetime columns
        for col in ("entry_time", "exit_time"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # ── core metrics ─────────────────────────────────────────────────────────
        pnl = df["pnl"].values
        total_pnl = float(np.sum(pnl))
        win_rate = float((pnl > 0).mean())
        avg_pnl = float(np.mean(pnl))
        pnl_std = float(np.std(pnl, ddof=1)) if n > 1 else 0.0
        sharpe = (avg_pnl / pnl_std * math.sqrt(252 * 24)) if pnl_std > 1e-9 else 0.0
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = float(np.min(cumulative - running_max))

        # ── time-based breakdowns ────────────────────────────────────────────────
        pnl_by_year: Dict[str, float] = {}
        pnl_by_month: Dict[int, float] = {}
        pnl_by_hour: Dict[int, float] = {}
        win_rate_by_hour: Dict[int, float] = {}

        if "entry_time" in df.columns and df["entry_time"].notna().any():
            df["year"] = df["entry_time"].dt.year.astype(str)
            df["month"] = df["entry_time"].dt.month
            df["hour"] = df["entry_time"].dt.hour

            pnl_by_year = df.groupby("year")["pnl"].sum().to_dict()
            pnl_by_month = df.groupby("month")["pnl"].sum().to_dict()
            pnl_by_hour = df.groupby("hour")["pnl"].mean().to_dict()
            win_rate_by_hour = df.groupby("hour")["pnl"].apply(lambda x: (x > 0).mean()).to_dict()
        else:
            notes.append("No 'entry_time' column — skipping time-based breakdowns.")

        # ── hold duration ────────────────────────────────────────────────────────
        hold_duration_stats: Dict[str, float] = {}
        if "hold_bars" in df.columns:
            hd = df["hold_bars"].dropna().values
            if len(hd) > 0:
                hold_duration_stats = {
                    "min": float(np.min(hd)), "max": float(np.max(hd)),
                    "mean": float(np.mean(hd)), "median": float(np.median(hd)),
                    "p90": float(np.percentile(hd, 90)),
                }
        elif "entry_time" in df.columns and "exit_time" in df.columns:
            durations = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
            durations = durations.dropna()
            if len(durations) > 0:
                hold_duration_stats = {
                    "min_h": float(durations.min()), "max_h": float(durations.max()),
                    "mean_h": float(durations.mean()), "median_h": float(durations.median()),
                    "p90_h": float(durations.quantile(0.9)),
                }

        # ── entry IC ─────────────────────────────────────────────────────────────
        entry_ic = 0.0
        if "entry_signal" in df.columns and "forward_return" in df.columns:
            entry_ic = float(df["entry_signal"].corr(df["forward_return"], method="spearman"))
        elif "entry_signal" in df.columns:
            entry_ic = float(df["entry_signal"].corr(pd.Series(pnl), method="spearman"))
        else:
            notes.append("No 'entry_signal' column — entry IC not computed.")

        # ── exit quality ─────────────────────────────────────────────────────────
        exit_quality = 0.5  # default neutral
        if "exit_price" in df.columns and "max_favorable_price" in df.columns:
            # exit quality = what fraction of max_favorable_price did we capture?
            entry_p = df.get("entry_price", pd.Series(np.ones(n)))
            max_fav = df["max_favorable_price"]
            exit_p = df["exit_price"]
            available_gain = (max_fav - entry_p).clip(lower=0)
            actual_gain = (exit_p - entry_p).clip(lower=0)
            quality = (actual_gain / available_gain.replace(0, np.nan)).dropna()
            exit_quality = float(quality.mean()) if len(quality) > 0 else 0.5
        else:
            notes.append("No 'exit_price'/'max_favorable_price' — exit quality not computed.")

        # ── buy-and-hold comparison ───────────────────────────────────────────────
        bah_return = 0.0
        strategy_return = float(np.sum(pnl) / max(abs(np.sum(pnl)) + 1e-9, 1.0))  # placeholder
        if prices_df is not None and len(prices_df) > 1:
            closes = prices_df["close"].dropna()
            if len(closes) >= 2:
                bah_return = float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0])
        alpha = strategy_return - bah_return

        return InstrumentReport(
            symbol=symbol,
            n_trades=n,
            total_pnl=total_pnl,
            win_rate=win_rate,
            avg_pnl_per_trade=avg_pnl,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            pnl_by_year=pnl_by_year,
            pnl_by_month={int(k): v for k, v in pnl_by_month.items()},
            pnl_by_hour={int(k): v for k, v in pnl_by_hour.items()},
            win_rate_by_hour={int(k): v for k, v in win_rate_by_hour.items()},
            hold_duration_stats=hold_duration_stats,
            entry_ic=entry_ic,
            exit_quality_score=exit_quality,
            bah_return_pct=bah_return,
            strategy_return_pct=strategy_return,
            alpha_vs_bah=alpha,
            notes=notes,
        )

    @staticmethod
    def generate_synthetic_trades(n: int = 300, symbol: str = "SOL", seed: int = 99) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        entry_times = pd.date_range("2023-01-01", periods=n, freq="8h")
        hold_bars = rng.integers(1, 24, n)
        exit_times = entry_times + pd.to_timedelta(hold_bars * 1, unit="h")
        entry_signal = rng.normal(0, 1, n)
        pnl = 0.12 * entry_signal + rng.normal(0, 0.02, n)
        return pd.DataFrame({
            "pnl": pnl,
            "entry_time": entry_times,
            "exit_time": exit_times,
            "hold_bars": hold_bars,
            "entry_signal": entry_signal,
            "forward_return": pnl,
        })
