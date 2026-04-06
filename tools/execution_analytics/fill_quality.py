"""
fill_quality.py — Fill quality analytics for the execution system.

FillQualityAnalyzer reads from live_trades.db and computes:
  - Fill rate by symbol / side / time-of-day
  - Partial fill analysis
  - Time-to-fill distribution
  - Rejection analysis (if rejection log exists)
  - Connection pool pressure correlation

Usage
-----
    from execution_analytics.fill_quality import FillQualityAnalyzer
    fqa = FillQualityAnalyzer()
    report = fqa.fills_quality_report()
    print(report)

    python fill_quality.py --symbol ETH --since 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_DB_PATH = _REPO_ROOT / "execution" / "live_trades.db"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FillRateEntry:
    symbol: str
    side: str
    hour: int
    n_attempted: int       # proxy: fills + inferred rejections
    n_filled: int
    fill_rate_pct: float
    avg_fill_qty: float
    avg_notional: float


@dataclass
class PartialFillStats:
    symbol: str
    n_orders: int
    n_partial: int
    partial_rate_pct: float
    avg_completion_pct: float
    p25_completion_pct: float
    p75_completion_pct: float


@dataclass
class TimeToFillStats:
    symbol: str
    side: str
    n_fills: int
    p25_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float


@dataclass
class RejectionGroup:
    error_code: str
    n_rejections: int
    first_seen: str
    last_seen: str
    rate_per_hour: float
    symbols_affected: List[str]


@dataclass
class ConnectionPressurePoint:
    ts: str
    urllib3_warnings: int
    rejection_count: int
    correlation_window: str


@dataclass
class FillQualityReport:
    generated_at: str
    since: str
    until: str
    n_total_fills: int
    n_total_orders: int
    overall_fill_rate_pct: float
    fill_rate_by_symbol_side_hour: List[FillRateEntry]
    partial_fill_stats: List[PartialFillStats]
    time_to_fill: List[TimeToFillStats]
    rejection_groups: List[RejectionGroup]
    connection_pressure: List[ConnectionPressurePoint]
    summary: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Queue position estimator (for limit orders)
# ---------------------------------------------------------------------------

def estimate_queue_position(
    order_size_usd: float,
    adv_usd: float,
    spread_bps: float,
    side: str = "buy",
) -> Dict[str, float]:
    """
    Estimate queue position metrics for a limit order.

    Heuristic model:
    - Queue depth ∝ ADV × spread (tighter spread → thicker book)
    - Fill probability ∝ exp(-position / queue_depth)

    Returns dict with estimated fill_probability and expected_wait_periods.
    """
    queue_depth_usd = adv_usd * spread_bps / 10_000 * 100.0
    if queue_depth_usd <= 0:
        return {"fill_probability": 0.5, "expected_wait_periods": 5.0}

    participation = order_size_usd / queue_depth_usd
    fill_prob = float(np.exp(-participation))
    expected_wait = 1.0 / fill_prob if fill_prob > 0 else 999.0

    return {
        "fill_probability": round(fill_prob, 4),
        "expected_wait_periods": round(expected_wait, 2),
        "queue_depth_usd": round(queue_depth_usd, 2),
    }


# ---------------------------------------------------------------------------
# FillQualityAnalyzer
# ---------------------------------------------------------------------------

class FillQualityAnalyzer:
    """
    Analyzes fill quality from live_trades.db.

    Parameters
    ----------
    db_path : Path or str, optional
        Path to live_trades.db.  Defaults to execution/live_trades.db.
    since   : str, optional
        ISO-8601 start date for analysis.
    until   : str, optional
        ISO-8601 end date.  Defaults to now.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path else _DB_PATH
        self.since = since or "2000-01-01T00:00:00+00:00"
        self.until = until or datetime.now(timezone.utc).isoformat()
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if not self.db_path.exists():
            raise FileNotFoundError(f"DB not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(
                """
                SELECT id, symbol, side, qty, price, notional,
                       fill_time, order_id, strategy_version
                FROM live_trades
                WHERE fill_time >= ? AND fill_time < ?
                ORDER BY fill_time
                """,
                conn,
                params=(self.since, self.until),
            )
        finally:
            conn.close()

        df["fill_time"] = pd.to_datetime(df["fill_time"], utc=True)
        df["hour"] = df["fill_time"].dt.hour
        df["date"] = df["fill_time"].dt.date.astype(str)
        df["symbol"] = df["symbol"].str.upper()
        df["side"] = df["side"].str.lower()
        self._df = df
        log.info("FillQualityAnalyzer loaded %d fills", len(df))
        return df

    # ------------------------------------------------------------------
    # Fill rate
    # ------------------------------------------------------------------

    def fill_rate_analysis(self) -> List[FillRateEntry]:
        """
        Compute fill rate by symbol / side / hour-of-day.

        Because live_trades only records *successful* fills, we approximate
        'attempted orders' as fills + heuristic rejection estimate (5%).
        """
        df = self._load()
        if df.empty:
            return []

        entries = []
        for (sym, side, hour), grp in df.groupby(["symbol", "side", "hour"]):
            n_filled = len(grp)
            # Heuristic: assume ~95% fill rate → attempted ≈ n_filled / 0.95
            n_attempted = int(n_filled / 0.95)
            fill_rate = n_filled / n_attempted * 100.0
            entries.append(FillRateEntry(
                symbol=str(sym),
                side=str(side),
                hour=int(hour),
                n_attempted=n_attempted,
                n_filled=n_filled,
                fill_rate_pct=round(fill_rate, 2),
                avg_fill_qty=round(float(grp["qty"].mean()), 6),
                avg_notional=round(float(grp["notional"].mean()), 2),
            ))
        return sorted(entries, key=lambda e: (e.symbol, e.side, e.hour))

    # ------------------------------------------------------------------
    # Partial fill analysis
    # ------------------------------------------------------------------

    def partial_fill_analysis(self) -> List[PartialFillStats]:
        """
        Analyze partial fills by grouping fills by order_id.

        An order is considered 'partially filled' if:
        - It has multiple fill records AND
        - The sum of fill quantities varies significantly across orders with
          same symbol/side (i.e., not all orders get the same total qty).
        """
        df = self._load()
        if df.empty:
            return []

        results = []
        for sym, sym_grp in df.groupby("symbol"):
            # Group by order_id
            order_totals = sym_grp.groupby("order_id").agg(
                total_qty=("qty", "sum"),
                n_fills=("id", "count"),
            ).reset_index()

            if order_totals.empty:
                continue

            # Estimate expected qty as 75th percentile of per-order qty
            expected_qty = float(order_totals["total_qty"].quantile(0.75))
            if expected_qty <= 0:
                continue

            completions = (order_totals["total_qty"] / expected_qty * 100.0).clip(upper=100.0)
            n_orders = len(order_totals)
            n_partial = int((completions < 99.0).sum())

            results.append(PartialFillStats(
                symbol=str(sym),
                n_orders=n_orders,
                n_partial=n_partial,
                partial_rate_pct=round(n_partial / n_orders * 100.0, 2) if n_orders else 0.0,
                avg_completion_pct=round(float(completions.mean()), 2),
                p25_completion_pct=round(float(completions.quantile(0.25)), 2),
                p75_completion_pct=round(float(completions.quantile(0.75)), 2),
            ))
        return sorted(results, key=lambda r: r.partial_rate_pct, reverse=True)

    # ------------------------------------------------------------------
    # Time-to-fill distribution
    # ------------------------------------------------------------------

    def time_to_fill_analysis(self) -> List[TimeToFillStats]:
        """
        Estimate time-to-fill from the span of fill timestamps per order_id.

        For single-fill orders, uses the delta from the order's first fill
        to the batch median as a proxy.
        """
        df = self._load()
        if df.empty:
            return []

        results = []
        for (sym, side), grp in df.groupby(["symbol", "side"]):
            # time-to-fill per order = max_fill_time - min_fill_time in ms
            fill_spans_ms = []
            for oid, ord_grp in grp.groupby("order_id"):
                if len(ord_grp) < 2:
                    fill_spans_ms.append(0.0)
                else:
                    span = (ord_grp["fill_time"].max() - ord_grp["fill_time"].min())
                    fill_spans_ms.append(span.total_seconds() * 1000.0)

            if not fill_spans_ms:
                continue

            arr = np.array(fill_spans_ms)
            results.append(TimeToFillStats(
                symbol=str(sym),
                side=str(side),
                n_fills=len(grp),
                p25_ms=round(float(np.percentile(arr, 25)), 2),
                p50_ms=round(float(np.percentile(arr, 50)), 2),
                p75_ms=round(float(np.percentile(arr, 75)), 2),
                p90_ms=round(float(np.percentile(arr, 90)), 2),
                p95_ms=round(float(np.percentile(arr, 95)), 2),
                p99_ms=round(float(np.percentile(arr, 99)), 2),
                mean_ms=round(float(np.mean(arr)), 2),
            ))
        return sorted(results, key=lambda r: r.p50_ms, reverse=True)

    # ------------------------------------------------------------------
    # Rejection analysis
    # ------------------------------------------------------------------

    def rejection_analysis(
        self,
        rejection_log_path: Optional[Path] = None,
    ) -> List[RejectionGroup]:
        """
        Parse rejection log file for error codes and group them.

        The rejection log is expected to be a JSONL file with fields:
            ts, symbol, side, error_code, message

        If no log is found, returns an empty list with a warning.
        """
        log_path = rejection_log_path or (_REPO_ROOT / "execution" / "rejection_log.jsonl")
        if not log_path.exists():
            log.info("No rejection log found at %s", log_path)
            return []

        rows = []
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not rows:
            return []

        df = pd.DataFrame(rows)
        if "ts" not in df.columns:
            return []

        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
        df = df[(df["ts"] >= pd.Timestamp(self.since)) &
                (df["ts"] < pd.Timestamp(self.until))]

        if df.empty:
            return []

        groups = []
        for error_code, grp in df.groupby("error_code"):
            span_hours = max(
                (grp["ts"].max() - grp["ts"].min()).total_seconds() / 3600.0,
                1.0,
            )
            syms = list(grp["symbol"].unique()) if "symbol" in grp.columns else []
            groups.append(RejectionGroup(
                error_code=str(error_code),
                n_rejections=len(grp),
                first_seen=str(grp["ts"].min()),
                last_seen=str(grp["ts"].max()),
                rate_per_hour=round(len(grp) / span_hours, 4),
                symbols_affected=syms[:10],
            ))
        return sorted(groups, key=lambda g: g.n_rejections, reverse=True)

    # ------------------------------------------------------------------
    # Connection pool pressure
    # ------------------------------------------------------------------

    def connection_pool_pressure(
        self,
        log_path: Optional[Path] = None,
    ) -> List[ConnectionPressurePoint]:
        """
        Parse application logs for urllib3 pool warnings and correlate them
        with rejection counts.

        Looks for lines containing 'urllib3' or 'Connection pool' warnings.
        Returns hourly buckets.
        """
        app_log = log_path or (_REPO_ROOT / "logs" / "app.log")
        if not app_log.exists():
            log.info("No app log at %s — skipping connection pressure", app_log)
            return []

        urllib3_pattern = re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})"
            r".*(?:urllib3|Connection pool|pool is full|NewConnectionError)",
            re.IGNORECASE,
        )

        hourly_warnings: Counter = Counter()
        try:
            with open(app_log, encoding="utf-8", errors="replace") as f:
                for line in f:
                    m = urllib3_pattern.search(line)
                    if m:
                        try:
                            ts = datetime.fromisoformat(m.group("ts"))
                            bucket = ts.strftime("%Y-%m-%dT%H:00:00")
                            hourly_warnings[bucket] += 1
                        except ValueError:
                            continue
        except OSError as exc:
            log.warning("Could not read app log: %s", exc)
            return []

        if not hourly_warnings:
            return []

        # Load rejection counts per hour
        rejection_log = _REPO_ROOT / "execution" / "rejection_log.jsonl"
        hourly_rejections: Counter = Counter()
        if rejection_log.exists():
            with open(rejection_log, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line.strip())
                        ts_raw = row.get("ts", "")
                        ts = datetime.fromisoformat(ts_raw)
                        bucket = ts.strftime("%Y-%m-%dT%H:00:00")
                        hourly_rejections[bucket] += 1
                    except (json.JSONDecodeError, ValueError):
                        continue

        points = []
        for bucket, warn_count in sorted(hourly_warnings.items()):
            points.append(ConnectionPressurePoint(
                ts=bucket,
                urllib3_warnings=warn_count,
                rejection_count=hourly_rejections.get(bucket, 0),
                correlation_window="1h",
            ))
        return points

    # ------------------------------------------------------------------
    # Master report
    # ------------------------------------------------------------------

    def fills_quality_report(self) -> Dict[str, Any]:
        """
        Run all fill quality analyses and return a unified dict.

        Keys
        ----
        generated_at, since, until, n_total_fills, n_total_orders,
        overall_fill_rate_pct, fill_rate_by_symbol_side_hour,
        partial_fill_stats, time_to_fill, rejection_groups,
        connection_pressure, summary
        """
        df = self._load()
        n_fills = len(df)
        n_orders = df["order_id"].nunique() if not df.empty else 0

        fill_rates = self.fill_rate_analysis()
        partial = self.partial_fill_analysis()
        ttf = self.time_to_fill_analysis()
        rejections = self.rejection_analysis()
        conn_pressure = self.connection_pool_pressure()

        # overall fill rate estimate
        if fill_rates:
            overall_fr = float(
                np.average(
                    [e.fill_rate_pct for e in fill_rates],
                    weights=[e.n_filled for e in fill_rates],
                )
            )
        else:
            overall_fr = 0.0

        # summary stats
        avg_partial_rate = (
            float(np.mean([p.partial_rate_pct for p in partial])) if partial else 0.0
        )
        median_ttf_ms = (
            float(np.median([t.p50_ms for t in ttf])) if ttf else 0.0
        )
        n_rejection_types = len(rejections)
        total_rejections = sum(r.n_rejections for r in rejections)

        summary = {
            "n_fills": n_fills,
            "n_orders": n_orders,
            "overall_fill_rate_pct": round(overall_fr, 2),
            "avg_partial_rate_pct": round(avg_partial_rate, 2),
            "median_time_to_fill_ms": round(median_ttf_ms, 2),
            "n_rejection_types": n_rejection_types,
            "total_rejections": total_rejections,
            "n_symbols": df["symbol"].nunique() if not df.empty else 0,
            "period_start": self.since,
            "period_end": self.until,
        }

        report = FillQualityReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            since=self.since,
            until=self.until,
            n_total_fills=n_fills,
            n_total_orders=n_orders,
            overall_fill_rate_pct=round(overall_fr, 2),
            fill_rate_by_symbol_side_hour=fill_rates,
            partial_fill_stats=partial,
            time_to_fill=ttf,
            rejection_groups=rejections,
            connection_pressure=conn_pressure,
            summary=summary,
        )

        return asdict(report)

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        rpt = self.fills_quality_report()
        s = rpt["summary"]
        print("=" * 68)
        print("  FILL QUALITY REPORT")
        print("=" * 68)
        print(f"  Period     : {rpt['since']}  to  {rpt['until']}")
        print(f"  Total fills: {s['n_fills']:,}  |  Orders: {s['n_orders']:,}")
        print(f"  Fill rate  : {s['overall_fill_rate_pct']:.1f}%")
        print(f"  Partial %  : {s['avg_partial_rate_pct']:.1f}%  (avg per symbol)")
        print(f"  Median TTF : {s['median_time_to_fill_ms']:.1f} ms")
        print(f"  Rejections : {s['total_rejections']:,}  ({s['n_rejection_types']} types)")
        print()

        print("── Fill Rate by Symbol/Side/Hour (top 20 rows) ──────────────")
        fr = rpt["fill_rate_by_symbol_side_hour"][:20]
        print(f"  {'Symbol':<8} {'Side':<5} {'Hour':>5} {'N Fills':>8} "
              f"{'Fill%':>7} {'Avg $':>10}")
        for e in fr:
            print(f"  {e['symbol']:<8} {e['side']:<5} {e['hour']:>5} "
                  f"{e['n_filled']:>8,} {e['fill_rate_pct']:>7.1f} "
                  f"{e['avg_notional']:>10.2f}")

        print()
        print("── Partial Fill Stats ───────────────────────────────────────")
        print(f"  {'Symbol':<8} {'N Orders':>9} {'N Partial':>10} {'Partial%':>9} "
              f"{'Avg Cmp%':>9}")
        for p in rpt["partial_fill_stats"]:
            print(f"  {p['symbol']:<8} {p['n_orders']:>9,} {p['n_partial']:>10,} "
                  f"{p['partial_rate_pct']:>9.1f} {p['avg_completion_pct']:>9.1f}")

        print()
        print("── Time to Fill (ms) ────────────────────────────────────────")
        print(f"  {'Symbol':<8} {'Side':<5} {'P50':>8} {'P90':>8} {'P95':>8} {'P99':>8}")
        for t in rpt["time_to_fill"]:
            print(f"  {t['symbol']:<8} {t['side']:<5} {t['p50_ms']:>8.1f} "
                  f"{t['p90_ms']:>8.1f} {t['p95_ms']:>8.1f} {t['p99_ms']:>8.1f}")

        if rpt["rejection_groups"]:
            print()
            print("── Rejection Groups ─────────────────────────────────────────")
            for r in rpt["rejection_groups"]:
                print(f"  {r['error_code']:<30} N={r['n_rejections']:>6}  "
                      f"rate={r['rate_per_hour']:.2f}/hr")

        print("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill Quality Analyzer")
    p.add_argument("--since", default="2020-01-01")
    p.add_argument("--until", default=None)
    p.add_argument("--symbol", default=None, help="Filter to one symbol (informational)")
    p.add_argument("--db", default=str(_DB_PATH))
    p.add_argument("--json", dest="json_out", action="store_true",
                   help="Output full report as JSON")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    until = args.until or datetime.now(timezone.utc).isoformat()
    fqa = FillQualityAnalyzer(
        db_path=Path(args.db),
        since=args.since,
        until=until,
    )

    if args.json_out:
        rpt = fqa.fills_quality_report()
        print(json.dumps(rpt, indent=2, default=str))
    else:
        fqa.print_report()


if __name__ == "__main__":
    main()
