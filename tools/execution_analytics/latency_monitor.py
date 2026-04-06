"""
latency_monitor.py — End-to-end latency tracking for the execution pipeline.

Tracks the full latency chain:
    signal_time → order_submit_time → ack_time → fill_received_time

Uses monotonic clock (time.perf_counter_ns) for nanosecond precision.
Categorizes by asset class (crypto vs equity) and fires spike alerts.

Optionally exports Prometheus histograms and writes latency_log.jsonl.

Usage
-----
    from execution_analytics.latency_monitor import LatencyMonitor

    mon = LatencyMonitor()
    token = mon.start_trace(order_id="abc123", symbol="ETH", side="buy")
    mon.mark(token, "order_submit")
    mon.mark(token, "fill_received")
    record = mon.finish_trace(token)
    print(record)

    python latency_monitor.py --tail --symbol ETH
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_DEFAULT_LOG = _HERE / "latency_log.jsonl"

# ---------------------------------------------------------------------------
# Optional Prometheus
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Histogram, Counter, Gauge  # type: ignore
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LATENCY_BUCKETS_MS = (10.0, 50.0, 100.0, 500.0)  # boundaries
_BUCKET_LABELS = ("<10ms", "10-50ms", "50-100ms", "100-500ms", ">500ms")

_SPIKE_WINDOW = 50        # rolling window for spike detection
_SPIKE_MULTIPLIER = 3.0   # alert if latency > 3× rolling median

_CRYPTO_SYMS = {
    "BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK", "UNI", "AAVE",
    "CRV", "SUSHI", "BAT", "YFI", "DOT", "LTC", "BCH", "SHIB",
}


def _asset_class(symbol: str) -> str:
    return "crypto" if symbol.upper() in _CRYPTO_SYMS else "equity"


def _bucket_label(latency_ms: float) -> str:
    lo, hi = _LATENCY_BUCKETS_MS[0], _LATENCY_BUCKETS_MS[-1]
    if latency_ms < lo:
        return _BUCKET_LABELS[0]
    if latency_ms < _LATENCY_BUCKETS_MS[1]:
        return _BUCKET_LABELS[1]
    if latency_ms < _LATENCY_BUCKETS_MS[2]:
        return _BUCKET_LABELS[2]
    if latency_ms <= hi:
        return _BUCKET_LABELS[3]
    return _BUCKET_LABELS[4]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LatencyRecord:
    """Complete latency record for one order lifecycle."""
    order_id: str
    symbol: str
    side: str
    asset_class: str
    wall_time: str                      # ISO-8601 UTC when trace started

    # nanosecond timestamps from perf_counter_ns (relative, monotonic)
    signal_ns: int = 0
    order_submit_ns: int = 0
    ack_ns: int = 0
    fill_received_ns: int = 0

    # computed latencies in milliseconds
    signal_to_submit_ms: float = 0.0    # decision → wire
    submit_to_ack_ms: float = 0.0       # wire → broker ack
    ack_to_fill_ms: float = 0.0         # ack → fill
    total_ms: float = 0.0               # signal → fill

    bucket: str = ""
    spike: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LatencyStats:
    """Rolling statistics per (symbol, asset_class)."""
    symbol: str
    asset_class: str
    n_records: int
    p25_ms: float
    p50_ms: float
    p75_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    spike_count: int
    bucket_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trace context
# ---------------------------------------------------------------------------

@dataclass
class _TraceContext:
    token: str
    order_id: str
    symbol: str
    side: str
    wall_time: str
    signal_ns: int
    marks: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LatencyMonitor
# ---------------------------------------------------------------------------

class LatencyMonitor:
    """
    End-to-end latency tracker.

    Thread-safe.  Writes completed records to a JSONL log file and
    optionally exports Prometheus histograms.

    Parameters
    ----------
    log_path : Path, optional
        JSONL output file.  Defaults to tools/execution_analytics/latency_log.jsonl.
    rolling_window : int
        Number of recent records per symbol kept in memory for stats.
    alert_callback : callable, optional
        Called with (record, rolling_median_ms) on latency spikes.
    prometheus_registry : optional
        Prometheus registry to register metrics on.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        rolling_window: int = _SPIKE_WINDOW,
        alert_callback: Optional[callable] = None,
        prometheus_registry=None,
    ) -> None:
        self.log_path = Path(log_path) if log_path else _DEFAULT_LOG
        self.rolling_window = rolling_window
        self.alert_callback = alert_callback or self._default_alert

        self._lock = threading.Lock()
        self._active_traces: Dict[str, _TraceContext] = {}
        # symbol → deque of total_ms
        self._windows: Dict[str, Deque[float]] = {}
        # symbol → list of LatencyRecord
        self._records: Dict[str, Deque[LatencyRecord]] = {}
        # spike counts
        self._spike_counts: Dict[str, int] = {}

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)

        self._init_prometheus(prometheus_registry)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_prometheus(self, registry) -> None:
        if not _HAS_PROM:
            self._prom = None
            return
        try:
            kw = {"registry": registry} if registry else {}
            self._prom_hist = Histogram(
                "order_latency_ms",
                "End-to-end order latency in milliseconds",
                ["symbol", "asset_class", "bucket"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 5000],
                **kw,
            )
            self._prom_spike_counter = Counter(
                "latency_spikes_total",
                "Latency spike count",
                ["symbol"],
                **kw,
            )
            self._prom_median_gauge = Gauge(
                "latency_rolling_median_ms",
                "Rolling median latency in ms",
                ["symbol"],
                **kw,
            )
            self._prom = True
        except Exception as exc:
            log.debug("Prometheus init failed: %s", exc)
            self._prom = None

    # ------------------------------------------------------------------
    # Public trace API
    # ------------------------------------------------------------------

    def start_trace(
        self,
        order_id: str,
        symbol: str,
        side: str = "buy",
    ) -> str:
        """
        Start a latency trace.

        Records the signal time (now) using perf_counter_ns.

        Returns
        -------
        token : str
            Unique trace token; pass to mark() and finish_trace().
        """
        token = f"{order_id}_{time.perf_counter_ns()}"
        ctx = _TraceContext(
            token=token,
            order_id=order_id,
            symbol=symbol.upper(),
            side=side.lower(),
            wall_time=datetime.now(timezone.utc).isoformat(),
            signal_ns=time.perf_counter_ns(),
        )
        with self._lock:
            self._active_traces[token] = ctx
        return token

    def mark(self, token: str, checkpoint: str) -> None:
        """
        Record a checkpoint within a trace.

        Checkpoints
        -----------
        "order_submit"    — when the order was sent to the broker
        "ack"             — when broker acknowledged the order
        "fill_received"   — when the fill notification arrived
        Any string key is also accepted for custom checkpoints.
        """
        ns = time.perf_counter_ns()
        with self._lock:
            ctx = self._active_traces.get(token)
            if ctx is None:
                log.warning("mark() called with unknown token %s", token)
                return
            ctx.marks[checkpoint] = ns

    def finish_trace(self, token: str) -> Optional[LatencyRecord]:
        """
        Finalize a trace and compute latency breakdowns.

        Returns the completed LatencyRecord or None if token unknown.
        """
        finish_ns = time.perf_counter_ns()
        with self._lock:
            ctx = self._active_traces.pop(token, None)
        if ctx is None:
            log.warning("finish_trace() unknown token: %s", token)
            return None

        m = ctx.marks
        submit_ns = m.get("order_submit", ctx.signal_ns)
        ack_ns = m.get("ack", submit_ns)
        fill_ns = m.get("fill_received", finish_ns)

        def _ns_to_ms(a: int, b: int) -> float:
            return max(0.0, (b - a) / 1_000_000.0)

        s2s = _ns_to_ms(ctx.signal_ns, submit_ns)
        s2a = _ns_to_ms(submit_ns, ack_ns)
        a2f = _ns_to_ms(ack_ns, fill_ns)
        total = _ns_to_ms(ctx.signal_ns, fill_ns)

        rec = LatencyRecord(
            order_id=ctx.order_id,
            symbol=ctx.symbol,
            side=ctx.side,
            asset_class=_asset_class(ctx.symbol),
            wall_time=ctx.wall_time,
            signal_ns=ctx.signal_ns,
            order_submit_ns=submit_ns,
            ack_ns=ack_ns,
            fill_received_ns=fill_ns,
            signal_to_submit_ms=round(s2s, 3),
            submit_to_ack_ms=round(s2a, 3),
            ack_to_fill_ms=round(a2f, 3),
            total_ms=round(total, 3),
            bucket=_bucket_label(total),
        )

        with self._lock:
            is_spike = self._update_and_check_spike(ctx.symbol, total, rec)
            rec.spike = is_spike

        self._write_log(rec)
        self._update_prometheus(rec)

        return rec

    def record_direct(
        self,
        order_id: str,
        symbol: str,
        side: str,
        signal_to_submit_ms: float,
        submit_to_ack_ms: float = 0.0,
        ack_to_fill_ms: float = 0.0,
        wall_time: Optional[str] = None,
    ) -> LatencyRecord:
        """
        Record a latency observation from externally measured timestamps
        (e.g., reconstructed from log files).
        """
        total = signal_to_submit_ms + submit_to_ack_ms + ack_to_fill_ms
        rec = LatencyRecord(
            order_id=order_id,
            symbol=symbol.upper(),
            side=side.lower(),
            asset_class=_asset_class(symbol),
            wall_time=wall_time or datetime.now(timezone.utc).isoformat(),
            signal_to_submit_ms=round(signal_to_submit_ms, 3),
            submit_to_ack_ms=round(submit_to_ack_ms, 3),
            ack_to_fill_ms=round(ack_to_fill_ms, 3),
            total_ms=round(total, 3),
            bucket=_bucket_label(total),
        )
        with self._lock:
            is_spike = self._update_and_check_spike(symbol.upper(), total, rec)
            rec.spike = is_spike
        self._write_log(rec)
        self._update_prometheus(rec)
        return rec

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, symbol: str) -> Optional[LatencyStats]:
        """Return rolling stats for a symbol.  None if no records."""
        sym = symbol.upper()
        with self._lock:
            window = self._windows.get(sym)
            if not window:
                return None
            arr = np.array(window)
            records = list(self._records.get(sym, []))
            spike_count = self._spike_counts.get(sym, 0)

        bucket_counts: Dict[str, int] = {label: 0 for label in _BUCKET_LABELS}
        for rec in records:
            bucket_counts[rec.bucket] = bucket_counts.get(rec.bucket, 0) + 1

        return LatencyStats(
            symbol=sym,
            asset_class=_asset_class(sym),
            n_records=len(arr),
            p25_ms=round(float(np.percentile(arr, 25)), 3),
            p50_ms=round(float(np.percentile(arr, 50)), 3),
            p75_ms=round(float(np.percentile(arr, 75)), 3),
            p90_ms=round(float(np.percentile(arr, 90)), 3),
            p95_ms=round(float(np.percentile(arr, 95)), 3),
            p99_ms=round(float(np.percentile(arr, 99)), 3),
            mean_ms=round(float(np.mean(arr)), 3),
            spike_count=spike_count,
            bucket_counts=bucket_counts,
        )

    def all_stats(self) -> List[LatencyStats]:
        with self._lock:
            syms = list(self._windows.keys())
        return [s for sym in syms if (s := self.stats(sym)) is not None]

    def asset_class_stats(self) -> Dict[str, Dict]:
        """Aggregated stats by asset class (crypto / equity)."""
        all_recs = []
        with self._lock:
            for dq in self._records.values():
                all_recs.extend(dq)

        result: Dict[str, Dict] = {}
        for ac in ("crypto", "equity"):
            recs = [r for r in all_recs if r.asset_class == ac]
            if not recs:
                continue
            arr = np.array([r.total_ms for r in recs])
            result[ac] = {
                "n_records": len(arr),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "mean_ms": round(float(np.mean(arr)), 3),
                "spike_count": sum(1 for r in recs if r.spike),
            }
        return result

    def recent_records(self, symbol: Optional[str] = None, n: int = 20) -> List[LatencyRecord]:
        """Return the N most recent LatencyRecords."""
        with self._lock:
            if symbol:
                recs = list(self._records.get(symbol.upper(), []))
            else:
                recs = []
                for dq in self._records.values():
                    recs.extend(dq)
        recs.sort(key=lambda r: r.wall_time, reverse=True)
        return recs[:n]

    # ------------------------------------------------------------------
    # Log reader
    # ------------------------------------------------------------------

    @staticmethod
    def read_log(
        log_path: Optional[Path] = None,
        symbol: Optional[str] = None,
        n: int = 1000,
    ) -> List[Dict]:
        """Read records from the JSONL log file."""
        path = log_path or _DEFAULT_LOG
        if not path.exists():
            return []
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if symbol and row.get("symbol") != symbol.upper():
                        continue
                    rows.append(row)
                except json.JSONDecodeError:
                    continue
        return rows[-n:]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_and_check_spike(
        self, sym: str, total_ms: float, rec: LatencyRecord
    ) -> bool:
        if sym not in self._windows:
            self._windows[sym] = deque(maxlen=self.rolling_window)
            self._records[sym] = deque(maxlen=self.rolling_window)
            self._spike_counts[sym] = 0

        window = self._windows[sym]
        is_spike = False

        if len(window) >= 5:
            median = float(np.median(window))
            if total_ms > median * _SPIKE_MULTIPLIER and total_ms > _LATENCY_BUCKETS_MS[0]:
                is_spike = True
                self._spike_counts[sym] += 1

        window.append(total_ms)
        self._records[sym].append(rec)

        if is_spike:
            median_now = float(np.median(window))
            # must release lock before callback — but we're called inside lock
            # so we schedule the alert outside via threading
            threading.Thread(
                target=self.alert_callback,
                args=(rec, median_now),
                daemon=True,
            ).start()

        return is_spike

    def _write_log(self, rec: LatencyRecord) -> None:
        try:
            line = json.dumps(rec.to_dict(), default=str)
            self._log_file.write(line + "\n")
        except OSError as exc:
            log.error("Failed to write latency log: %s", exc)

    def _update_prometheus(self, rec: LatencyRecord) -> None:
        if not self._prom:
            return
        try:
            self._prom_hist.labels(
                symbol=rec.symbol,
                asset_class=rec.asset_class,
                bucket=rec.bucket,
            ).observe(rec.total_ms)

            with self._lock:
                window = self._windows.get(rec.symbol)
                if window:
                    med = float(np.median(window))
                    self._prom_median_gauge.labels(symbol=rec.symbol).set(med)

            if rec.spike:
                self._prom_spike_counter.labels(symbol=rec.symbol).inc()
        except Exception:
            pass

    @staticmethod
    def _default_alert(record: LatencyRecord, rolling_median_ms: float) -> None:
        log.warning(
            "LATENCY SPIKE: %s %s  total=%.1f ms  3× rolling_median=%.1f ms  "
            "bucket=%s",
            record.symbol, record.order_id,
            record.total_ms, rolling_median_ms,
            record.bucket,
        )

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._log_file.close()
        except OSError:
            pass

    def __enter__(self) -> "LatencyMonitor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LatencyMonitor — view latency log stats")
    p.add_argument("--log", default=str(_DEFAULT_LOG), help="Path to latency_log.jsonl")
    p.add_argument("--symbol", default=None, help="Filter to one symbol")
    p.add_argument("--tail", action="store_true", help="Print last 20 records")
    p.add_argument("--stats", action="store_true", help="Print per-symbol stats from log")
    p.add_argument("--demo", action="store_true", help="Run a demo trace")
    p.add_argument("-n", type=int, default=20)
    return p.parse_args()


def _run_demo(mon: LatencyMonitor) -> None:
    """Simulate a few order lifecycles for demo purposes."""
    import random
    symbols = ["ETH", "BTC", "TSLA", "AAPL", "DOT"]
    print("Running demo traces...")
    for i in range(30):
        sym = random.choice(symbols)
        token = mon.start_trace(order_id=f"demo_{i}", symbol=sym)
        time.sleep(random.uniform(0.001, 0.05))   # signal → submit
        mon.mark(token, "order_submit")
        time.sleep(random.uniform(0.001, 0.02))   # submit → ack
        mon.mark(token, "ack")
        time.sleep(random.uniform(0.005, 0.1))    # ack → fill
        mon.mark(token, "fill_received")
        rec = mon.finish_trace(token)
        if rec:
            spike_tag = " [SPIKE]" if rec.spike else ""
            print(f"  {rec.symbol:>6} {rec.order_id:<12} total={rec.total_ms:>8.2f}ms "
                  f"bucket={rec.bucket}{spike_tag}")
    print()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    mon = LatencyMonitor(log_path=Path(args.log))

    if args.demo:
        _run_demo(mon)

    if args.tail or args.stats:
        rows = LatencyMonitor.read_log(
            log_path=Path(args.log),
            symbol=args.symbol,
            n=args.n if args.tail else 10_000,
        )

        if not rows:
            print("No records in log (or log is empty).")
            return

        if args.tail:
            print(f"Last {min(args.n, len(rows))} records:")
            print(f"  {'Symbol':<8} {'Order ID':<20} {'Total ms':>10} "
                  f"{'S→Sub':>8} {'Sub→Ack':>8} {'Ack→Fill':>9} {'Bucket':<12} {'Spike'}")
            for r in rows[-args.n:]:
                spike = "YES" if r.get("spike") else "-"
                print(
                    f"  {r.get('symbol','?'):<8} {r.get('order_id','?'):<20} "
                    f"{r.get('total_ms',0):>10.2f} "
                    f"{r.get('signal_to_submit_ms',0):>8.2f} "
                    f"{r.get('submit_to_ack_ms',0):>8.2f} "
                    f"{r.get('ack_to_fill_ms',0):>9.2f} "
                    f"{r.get('bucket','?'):<12} {spike}"
                )

        if args.stats:
            # Load into monitor for stats
            for r in rows:
                sym = r.get("symbol", "UNK")
                mon.record_direct(
                    order_id=r.get("order_id", ""),
                    symbol=sym,
                    side=r.get("side", "buy"),
                    signal_to_submit_ms=float(r.get("signal_to_submit_ms", 0)),
                    submit_to_ack_ms=float(r.get("submit_to_ack_ms", 0)),
                    ack_to_fill_ms=float(r.get("ack_to_fill_ms", 0)),
                    wall_time=r.get("wall_time"),
                )

            print("\nPer-symbol latency stats:")
            print(f"  {'Symbol':<8} {'Class':<8} {'N':>6} {'P50 ms':>8} {'P95 ms':>8} "
                  f"{'P99 ms':>8} {'Mean ms':>9} {'Spikes':>7}")
            for s in sorted(mon.all_stats(), key=lambda x: x.p50_ms, reverse=True):
                print(
                    f"  {s.symbol:<8} {s.asset_class:<8} {s.n_records:>6} "
                    f"{s.p50_ms:>8.2f} {s.p95_ms:>8.2f} {s.p99_ms:>8.2f} "
                    f"{s.mean_ms:>9.2f} {s.spike_count:>7}"
                )

            ac_stats = mon.asset_class_stats()
            if ac_stats:
                print("\nAsset class aggregates:")
                for ac, stats in ac_stats.items():
                    print(f"  {ac:<8}: N={stats['n_records']:>6}  "
                          f"P50={stats['p50_ms']:.2f}ms  "
                          f"P95={stats['p95_ms']:.2f}ms  "
                          f"spikes={stats['spike_count']}")

            bucket_total: Dict[str, int] = {label: 0 for label in _BUCKET_LABELS}
            for r in rows:
                b = r.get("bucket", "")
                if b in bucket_total:
                    bucket_total[b] += 1

            print("\nGlobal bucket distribution:")
            total = max(sum(bucket_total.values()), 1)
            for label in _BUCKET_LABELS:
                count = bucket_total[label]
                pct = count / total * 100
                bar = "|" * int(pct / 2)
                print(f"  {label:<12}: {count:>7,}  ({pct:>5.1f}%)  {bar}")

    mon.close()


if __name__ == "__main__":
    main()
