"""
latency.py — Order Latency Analysis for the srfm-lab Live Trader
================================================================

Measures and analyses order-submission-to-fill latency:
  - Per-order latency computation
  - By-symbol and by-hour-of-day breakdowns
  - Correlation between latency and execution cost
  - Stale-quote detection
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Summary statistics for a latency distribution (milliseconds)."""
    mean: float
    p50: float
    p95: float
    p99: float
    max: float
    min: float = 0.0
    n: int = 0
    std: float = 0.0

    def is_acceptable(
        self,
        p95_threshold_ms: float = 2000.0,
        p99_threshold_ms: float = 5000.0,
    ) -> bool:
        """Return True if p95 and p99 are within acceptable bounds."""
        return self.p95 <= p95_threshold_ms and self.p99 <= p99_threshold_ms

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_ms": self.mean,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "max_ms": self.max,
            "min_ms": self.min,
            "n": float(self.n),
            "std_ms": self.std,
        }


@dataclass
class LatencyCorrelation:
    """Correlation between latency and execution cost."""
    n_observations: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    high_latency_mean_is_bps: float   # mean IS for top-quartile latency trades
    low_latency_mean_is_bps: float    # mean IS for bottom-quartile latency trades
    is_significant: bool              # p < 0.05


@dataclass
class StaleQuoteAnalysis:
    """Results from stale-quote detection."""
    n_trades: int
    n_stale: int
    stale_rate: float                  # fraction 0–1
    threshold_ms: float
    mean_staleness_ms: float           # among stale trades
    stale_trade_symbols: list[str]
    stale_is_bps: float                # mean IS for stale trades
    fresh_is_bps: float                # mean IS for fresh trades


# ---------------------------------------------------------------------------
# Latency analyzer
# ---------------------------------------------------------------------------

class LatencyAnalyzer:
    """
    Analyzes order-submission to fill-confirmation latency.

    Input data is typically a DataFrame with columns:
      - sym: str
      - order_id: str
      - order_submitted_at: datetime
      - fill_confirmed_at: datetime
      - side: str
      - fill_price: float
      - decision_price: float (optional, for IS correlation)
      - actual_impact_bps: float (optional)
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Core computation
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_latency(
        order_submitted_at: datetime,
        fill_confirmed_at: datetime,
    ) -> float:
        """
        Compute round-trip latency in milliseconds.

        Parameters
        ----------
        order_submitted_at : datetime
        fill_confirmed_at : datetime

        Returns
        -------
        float
            Latency in milliseconds. Always non-negative.

        Raises
        ------
        ValueError
            If fill_confirmed_at is before order_submitted_at.
        """
        delta = (fill_confirmed_at - order_submitted_at).total_seconds() * 1000
        if delta < 0:
            raise ValueError(
                f"fill_confirmed_at ({fill_confirmed_at}) is before "
                f"order_submitted_at ({order_submitted_at}). "
                f"Delta = {delta:.1f} ms"
            )
        return delta

    @staticmethod
    def _compute_latency_stats(latencies_ms: list[float] | np.ndarray) -> LatencyStats:
        """Compute LatencyStats from an array of latency values in ms."""
        arr = np.asarray(latencies_ms, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return LatencyStats(mean=0, p50=0, p95=0, p99=0, max=0, min=0, n=0, std=0)
        return LatencyStats(
            mean=float(np.mean(arr)),
            p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            max=float(np.max(arr)),
            min=float(np.min(arr)),
            n=len(arr),
            std=float(np.std(arr)),
        )

    # -----------------------------------------------------------------------
    # Add latency column
    # -----------------------------------------------------------------------

    def compute_all_latencies(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'latency_ms' column to the trades DataFrame.

        Expects columns: order_submitted_at, fill_confirmed_at.
        Both can be strings or datetime objects.

        Parameters
        ----------
        trades : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Copy with 'latency_ms' column added.
        """
        df = trades.copy()

        for col in ("order_submitted_at", "fill_confirmed_at"):
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
            df[col] = pd.to_datetime(df[col])

        df["latency_ms"] = (
            (df["fill_confirmed_at"] - df["order_submitted_at"])
            .dt.total_seconds() * 1000
        )

        # Flag negative latencies (clock skew or data error)
        n_negative = (df["latency_ms"] < 0).sum()
        if n_negative > 0:
            logger.warning(
                "%d trades have negative latency (clock skew?). Setting to NaN.",
                n_negative,
            )
            df.loc[df["latency_ms"] < 0, "latency_ms"] = np.nan

        return df

    # -----------------------------------------------------------------------
    # By-symbol breakdown
    # -----------------------------------------------------------------------

    def latency_by_instrument(
        self,
        trades: pd.DataFrame,
    ) -> dict[str, LatencyStats]:
        """
        Compute LatencyStats per symbol.

        Parameters
        ----------
        trades : pd.DataFrame
            Must have: sym, latency_ms  (or order_submitted_at + fill_confirmed_at)

        Returns
        -------
        dict[str, LatencyStats]
        """
        if "latency_ms" not in trades.columns:
            trades = self.compute_all_latencies(trades)

        if "sym" not in trades.columns:
            raise ValueError("trades must have 'sym' column")

        result: dict[str, LatencyStats] = {}
        for sym, group in trades.groupby("sym"):
            lat = group["latency_ms"].dropna().tolist()
            if lat:
                result[str(sym)] = self._compute_latency_stats(lat)

        logger.info("latency_by_instrument: computed stats for %d symbols", len(result))
        return result

    # -----------------------------------------------------------------------
    # By-hour-of-day breakdown
    # -----------------------------------------------------------------------

    def latency_by_time_of_day(
        self,
        trades: pd.DataFrame,
        tz: str = "UTC",
    ) -> dict[int, LatencyStats]:
        """
        Compute LatencyStats broken down by hour of day (0–23).

        Parameters
        ----------
        trades : pd.DataFrame
            Must have: order_submitted_at, latency_ms (or timestamps to compute from)
        tz : str
            Timezone for hour extraction.

        Returns
        -------
        dict[int, LatencyStats]
            Keys are integer hours (0–23).
        """
        if "latency_ms" not in trades.columns:
            trades = self.compute_all_latencies(trades)

        if "order_submitted_at" not in trades.columns:
            raise ValueError("trades must have 'order_submitted_at' column")

        df = trades.copy()
        df["order_submitted_at"] = pd.to_datetime(df["order_submitted_at"])
        df["hour"] = df["order_submitted_at"].dt.tz_localize("UTC").dt.tz_convert(tz).dt.hour

        result: dict[int, LatencyStats] = {}
        for hour, group in df.groupby("hour"):
            lat = group["latency_ms"].dropna().tolist()
            if lat:
                result[int(hour)] = self._compute_latency_stats(lat)

        return result

    # -----------------------------------------------------------------------
    # Latency vs impact correlation
    # -----------------------------------------------------------------------

    def latency_vs_impact(
        self,
        trades: pd.DataFrame,
    ) -> LatencyCorrelation:
        """
        Compute correlation between order latency and execution cost.

        High latency often leads to worse fills (stale quotes, market moved).

        Parameters
        ----------
        trades : pd.DataFrame
            Must have: latency_ms, actual_impact_bps

        Returns
        -------
        LatencyCorrelation
        """
        from scipy import stats as scipy_stats

        if "latency_ms" not in trades.columns:
            trades = self.compute_all_latencies(trades)

        if "actual_impact_bps" not in trades.columns:
            raise ValueError("trades must have 'actual_impact_bps' column")

        df = trades.dropna(subset=["latency_ms", "actual_impact_bps"])
        if len(df) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(df)}")

        lat = df["latency_ms"].values.astype(float)
        impact = df["actual_impact_bps"].values.astype(float)

        pr, pp = scipy_stats.pearsonr(lat, impact)
        sr, sp = scipy_stats.spearmanr(lat, impact)

        q75 = np.percentile(lat, 75)
        q25 = np.percentile(lat, 25)
        high_lat_is = float(df[df["latency_ms"] >= q75]["actual_impact_bps"].mean())
        low_lat_is = float(df[df["latency_ms"] <= q25]["actual_impact_bps"].mean())

        corr = LatencyCorrelation(
            n_observations=len(df),
            pearson_r=float(pr),
            pearson_p=float(pp),
            spearman_r=float(sr),
            spearman_p=float(sp),
            high_latency_mean_is_bps=high_lat_is,
            low_latency_mean_is_bps=low_lat_is,
            is_significant=pp < 0.05,
        )

        logger.info(
            "Latency vs impact: Pearson r=%.3f (p=%.4f), Spearman r=%.3f (p=%.4f)",
            pr, pp, sr, sp,
        )
        return corr

    # -----------------------------------------------------------------------
    # Stale quote analysis
    # -----------------------------------------------------------------------

    def stale_quote_analysis(
        self,
        trades: pd.DataFrame,
        quote_ts_col: str = "quote_ts",
        fill_ts_col: str = "fill_confirmed_at",
        threshold_ms: float = 100.0,
    ) -> StaleQuoteAnalysis:
        """
        Identify trades where the execution quote was stale.

        A quote is considered stale if the time between the quote timestamp
        and the fill timestamp exceeds `threshold_ms`.

        Parameters
        ----------
        trades : pd.DataFrame
        quote_ts_col : str
            Column with timestamp of the quote used to size the order.
        fill_ts_col : str
            Column with fill confirmation timestamp.
        threshold_ms : float
            Staleness threshold in milliseconds.

        Returns
        -------
        StaleQuoteAnalysis
        """
        df = trades.copy()

        for col in (quote_ts_col, fill_ts_col):
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
            df[col] = pd.to_datetime(df[col])

        df["quote_age_ms"] = (
            (df[fill_ts_col] - df[quote_ts_col]).dt.total_seconds() * 1000
        )

        stale_mask = df["quote_age_ms"] > threshold_ms
        n_stale = int(stale_mask.sum())
        n_total = len(df)

        stale_df = df[stale_mask]
        fresh_df = df[~stale_mask]

        mean_staleness_ms = float(stale_df["quote_age_ms"].mean()) if n_stale > 0 else 0.0

        stale_syms: list[str] = []
        if "sym" in df.columns:
            stale_syms = stale_df["sym"].unique().tolist()

        stale_is = 0.0
        fresh_is = 0.0
        if "actual_impact_bps" in df.columns:
            stale_is = float(stale_df["actual_impact_bps"].mean()) if n_stale > 0 else 0.0
            fresh_is = float(fresh_df["actual_impact_bps"].mean()) if len(fresh_df) > 0 else 0.0

        logger.info(
            "Stale quote analysis: %d / %d trades stale (>%g ms). Rate = %.1f%%",
            n_stale, n_total, threshold_ms, (n_stale / max(n_total, 1)) * 100,
        )

        return StaleQuoteAnalysis(
            n_trades=n_total,
            n_stale=n_stale,
            stale_rate=n_stale / max(n_total, 1),
            threshold_ms=threshold_ms,
            mean_staleness_ms=mean_staleness_ms,
            stale_trade_symbols=stale_syms,
            stale_is_bps=stale_is,
            fresh_is_bps=fresh_is,
        )

    # -----------------------------------------------------------------------
    # Convenience: full report
    # -----------------------------------------------------------------------

    def full_latency_report(self, trades: pd.DataFrame) -> dict[str, Any]:
        """
        Run all latency analyses and return a structured report dict.

        Parameters
        ----------
        trades : pd.DataFrame

        Returns
        -------
        dict with keys:
          overall_stats, by_symbol, by_hour, vs_impact (if available)
        """
        df = self.compute_all_latencies(trades)

        overall = self._compute_latency_stats(df["latency_ms"].dropna().tolist())
        by_sym = self.latency_by_instrument(df)
        by_hour = self.latency_by_time_of_day(df)

        report: dict[str, Any] = {
            "overall_stats": overall.as_dict(),
            "by_symbol": {sym: s.as_dict() for sym, s in by_sym.items()},
            "by_hour": {hour: s.as_dict() for hour, s in by_hour.items()},
            "n_trades": len(df),
            "pct_above_1s": float((df["latency_ms"] > 1000).mean() * 100),
            "pct_above_5s": float((df["latency_ms"] > 5000).mean() * 100),
        }

        if "actual_impact_bps" in df.columns:
            try:
                corr = self.latency_vs_impact(df)
                report["vs_impact"] = {
                    "pearson_r": corr.pearson_r,
                    "pearson_p": corr.pearson_p,
                    "spearman_r": corr.spearman_r,
                    "high_latency_mean_is_bps": corr.high_latency_mean_is_bps,
                    "low_latency_mean_is_bps": corr.low_latency_mean_is_bps,
                    "is_significant": corr.is_significant,
                }
            except Exception as exc:
                report["vs_impact"] = {"error": str(exc)}

        return report

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_latency_distribution(
        self,
        trades: pd.DataFrame,
        save_path: str,
        title: str = "Order Latency Distribution",
    ) -> None:
        """Plot histogram of order latency."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return

        df = self.compute_all_latencies(trades)
        lat = df["latency_ms"].dropna().values

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: full distribution
        ax = axes[0]
        ax.hist(lat, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(np.percentile(lat, 95), color="red", linestyle="--", label="P95")
        ax.axvline(np.percentile(lat, 99), color="orange", linestyle="--", label="P99")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{title}\nFull distribution")
        ax.legend()

        # Right: by hour of day
        ax2 = axes[1]
        by_hour = self.latency_by_time_of_day(df)
        hours = sorted(by_hour.keys())
        mean_lats = [by_hour[h].mean for h in hours]
        p95_lats = [by_hour[h].p95 for h in hours]
        ax2.bar(hours, mean_lats, alpha=0.7, color="steelblue", label="Mean")
        ax2.plot(hours, p95_lats, "r--o", markersize=4, label="P95")
        ax2.set_xlabel("Hour of Day (UTC)")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency by Hour of Day")
        ax2.legend()

        fig.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Latency distribution plot saved to %s", save_path)
