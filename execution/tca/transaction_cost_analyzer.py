"""
execution/tca/transaction_cost_analyzer.py
============================================
Transaction Cost Analysis (TCA) framework.

For each filled order this module computes:

- **Implementation shortfall**: difference between decision price and fill price.
- **Market impact**: price movement attributable to our order.
- **Timing cost**: price drift from decision to submission time.
- **Spread cost**: half-spread paid on execution.

Rolling statistics are maintained so the system can detect systematic
patterns (e.g. certain UTC hours always having worse fills).

IAE integration
---------------
If average slippage in a given hour X exceeds 2× the overall baseline, the
analyzer logs an IAE hypothesis entry.  This mirrors the hypothesis-driven
improvement process documented in the lab.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("execution.tca")

# ---------------------------------------------------------------------------
# Bucket boundaries for order-size analysis
# ---------------------------------------------------------------------------

SIZE_BUCKETS: list[tuple[str, float, float]] = [
    ("micro",   0.0,     500.0),
    ("small",   500.0,   5_000.0),
    ("medium",  5_000.0, 50_000.0),
    ("large",   50_000.0, math.inf),
]

IAE_MULTIPLIER: float = 2.0   # flag if hourly slippage > 2x baseline


# ---------------------------------------------------------------------------
# ExecutionRecord
# ---------------------------------------------------------------------------

@dataclass
class ExecutionRecord:
    """
    Immutable record of a single fill's cost decomposition.

    All cost components are in basis points (bps) unless noted.
    """
    order_id:                str
    symbol:                  str
    side:                    str
    fill_qty:                float
    fill_price:              float
    decision_price:          float
    pre_trade_mid:           float
    commission_usd:          float
    fill_time:               datetime

    # Derived (computed on construction)
    implementation_shortfall: float = 0.0   # bps
    spread_cost:              float = 0.0   # bps
    timing_cost:              float = 0.0   # bps
    market_impact:            float = 0.0   # bps (residual)
    notional_usd:             float = 0.0

    def __post_init__(self) -> None:
        side_sign = 1.0 if self.side == "BUY" else -1.0
        dp = self.decision_price
        mp = self.pre_trade_mid
        fp = self.fill_price

        if dp > 0:
            self.implementation_shortfall = (
                (fp - dp) / dp * 10_000 * side_sign
            )
        if mp > 0 and dp > 0:
            self.timing_cost  = (mp - dp) / dp * 10_000 * side_sign
            self.spread_cost  = abs(fp - mp) / mp * 10_000
            self.market_impact = self.implementation_shortfall - self.timing_cost - self.spread_cost

        self.notional_usd = abs(self.fill_qty) * self.fill_price

    def to_dict(self) -> dict:
        return {
            "order_id":                self.order_id,
            "symbol":                  self.symbol,
            "side":                    self.side,
            "fill_qty":                self.fill_qty,
            "fill_price":              self.fill_price,
            "decision_price":          self.decision_price,
            "pre_trade_mid":           self.pre_trade_mid,
            "commission_usd":          self.commission_usd,
            "fill_time":               self.fill_time.isoformat(),
            "implementation_shortfall": self.implementation_shortfall,
            "spread_cost":             self.spread_cost,
            "timing_cost":             self.timing_cost,
            "market_impact":           self.market_impact,
            "notional_usd":            self.notional_usd,
        }


# ---------------------------------------------------------------------------
# TransactionCostAnalyzer
# ---------------------------------------------------------------------------

class TransactionCostAnalyzer:
    """
    Records every fill and computes aggregate TCA statistics.

    Internal accumulators
    ---------------------
    - Per-symbol rolling stats (deque of slippage values).
    - Per-UTC-hour rolling stats.
    - Per-order-size-bucket stats.

    The ``detect_iae_hypotheses`` method scans for hours where slippage
    systematically exceeds 2× the overall average.

    Thread safety: RLock on all writes.
    """

    def __init__(self) -> None:
        self._records:        list[ExecutionRecord]                = []
        self._by_symbol:      dict[str, list[ExecutionRecord]]     = defaultdict(list)
        self._by_hour:        dict[int, list[float]]               = defaultdict(list)  # hour -> [is_bps]
        self._by_size_bucket: dict[str, list[float]]               = defaultdict(list)
        self._lock            = threading.RLock()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_execution(
        self,
        order,                      # Order object
        pre_trade_mid: float,
    ) -> Optional[ExecutionRecord]:
        """
        Record a filled order and compute its cost components.

        Parameters
        ----------
        order : Order
            A FILLED OMS Order.
        pre_trade_mid : float
            The mid-price captured immediately before the order was submitted.

        Returns
        -------
        ExecutionRecord | None
            The computed record, or None if the order is not filled.
        """
        from ..oms.order import OrderStatus

        if order.status != OrderStatus.FILLED:
            return None
        if order.fill_price is None or order.fill_price <= 0:
            return None

        rec = ExecutionRecord(
            order_id       = order.order_id,
            symbol         = order.symbol,
            side           = order.side.value,
            fill_qty       = order.fill_qty,
            fill_price     = order.fill_price,
            decision_price = order.decision_price or order.fill_price,
            pre_trade_mid  = pre_trade_mid or order.fill_price,
            commission_usd = order.commission_usd,
            fill_time      = order.filled_at or datetime.now(timezone.utc),
        )

        with self._lock:
            self._records.append(rec)
            self._by_symbol[rec.symbol].append(rec)
            hour = rec.fill_time.hour
            self._by_hour[hour].append(rec.implementation_shortfall)
            bucket = self._get_size_bucket(rec.notional_usd)
            self._by_size_bucket[bucket].append(rec.implementation_shortfall)

        log.info(
            "TCA: %s %s IS=%.1fbps spread=%.1fbps impact=%.1fbps timing=%.1fbps",
            rec.symbol, rec.side,
            rec.implementation_shortfall,
            rec.spread_cost,
            rec.market_impact,
            rec.timing_cost,
        )
        return rec

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def avg_slippage_by_symbol(self) -> dict[str, float]:
        """Return average implementation shortfall (bps) per symbol."""
        with self._lock:
            return {
                sym: sum(r.implementation_shortfall for r in recs) / len(recs)
                for sym, recs in self._by_symbol.items()
                if recs
            }

    def avg_slippage_by_hour(self) -> dict[int, float]:
        """Return average IS per UTC hour of day."""
        with self._lock:
            return {
                hour: sum(vals) / len(vals)
                for hour, vals in self._by_hour.items()
                if vals
            }

    def avg_slippage_by_size_bucket(self) -> dict[str, float]:
        """Return average IS per order-size bucket."""
        with self._lock:
            return {
                bucket: sum(vals) / len(vals)
                for bucket, vals in self._by_size_bucket.items()
                if vals
            }

    def overall_avg_slippage(self) -> float:
        """Overall average implementation shortfall across all fills."""
        with self._lock:
            if not self._records:
                return 0.0
            return sum(r.implementation_shortfall for r in self._records) / len(self._records)

    def total_commissions(self) -> float:
        """Total commission paid across all recorded fills."""
        with self._lock:
            return sum(r.commission_usd for r in self._records)

    # ------------------------------------------------------------------
    # IAE hypothesis generation
    # ------------------------------------------------------------------

    def detect_iae_hypotheses(self) -> list[str]:
        """
        Scan hourly slippage stats and generate hypothesis strings for any
        UTC hour where average IS > 2× overall baseline.

        Returns
        -------
        list[str]
            Human-readable hypothesis descriptions.  Empty if no anomalies.
        """
        with self._lock:
            baseline = self.overall_avg_slippage()
            if abs(baseline) < 0.1:
                return []

            hypotheses: list[str] = []
            for hour, vals in self._by_hour.items():
                if len(vals) < 5:
                    continue   # not enough data
                avg = sum(vals) / len(vals)
                if avg > IAE_MULTIPLIER * baseline:
                    msg = (
                        f"IAE hypothesis: UTC hour {hour:02d} has avg IS "
                        f"{avg:.1f} bps vs baseline {baseline:.1f} bps "
                        f"({avg/baseline:.1f}x) — consider blocking this hour "
                        f"[n={len(vals)} trades]"
                    )
                    hypotheses.append(msg)
                    log.warning(msg)

            return hypotheses

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_records(self) -> list[dict]:
        """Return all execution records as a list of dicts (for CSV/JSON export)."""
        with self._lock:
            return [r.to_dict() for r in self._records]

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_size_bucket(notional: float) -> str:
        for name, lo, hi in SIZE_BUCKETS:
            if lo <= notional < hi:
                return name
        return "large"
