"""
execution/tca/execution_quality_report.py
==========================================
Daily execution quality report generator.

Aggregates the TransactionCostAnalyzer records and produces:
  - Summary statistics (fill rate, avg slippage, commissions)
  - Best/worst execution hours
  - VWAP benchmark comparison
  - Slippage trend analysis (is quality improving?)

Reports are returned as a structured dict and can optionally be written
to a dated file under ``execution/reports/``.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("execution.exec_quality_report")

REPORTS_DIR = Path(__file__).parent.parent / "reports"


# ---------------------------------------------------------------------------
# ExecutionQualityReport
# ---------------------------------------------------------------------------

@dataclass
class DailyReport:
    """Structured daily execution quality report."""
    report_date:          str
    total_orders:         int
    filled_orders:        int
    rejected_orders:      int
    cancelled_orders:     int
    fill_rate:            float           # filled / total
    avg_slippage_bps:     float
    median_slippage_bps:  float
    total_commissions:    float
    total_notional:       float
    best_hour_utc:        Optional[int]
    worst_hour_utc:       Optional[int]
    best_hour_slippage:   float
    worst_hour_slippage:  float
    vwap_benchmark:       float           # system VWAP price (avg fill price weighted by qty)
    beat_vwap:            bool            # True if fill_vwap < vwap_benchmark (BUY side)
    slippage_trend:       str             # "improving" / "degrading" / "stable" / "insufficient_data"
    per_symbol_slippage:  dict[str, float]
    iae_hypotheses:       list[str]
    generated_at:         str


class ExecutionQualityReport:
    """
    Generates daily execution quality summaries from TCA records.

    Parameters
    ----------
    tca : TransactionCostAnalyzer
        The live TCA instance to read records from.
    order_book : OrderBook | None
        If provided, order counts (total, rejected, cancelled) are pulled
        from the book rather than estimated from TCA records.
    reports_dir : Path | None
        Directory to write JSON reports.  Defaults to ``execution/reports/``.
    """

    def __init__(self, tca, order_book=None, reports_dir: Optional[Path] = None) -> None:
        self._tca         = tca
        self._order_book  = order_book
        self._reports_dir = reports_dir or REPORTS_DIR

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate(self, for_date: Optional[date] = None) -> DailyReport:
        """
        Generate a report for *for_date* (default: today UTC).

        Filters TCA records to those whose fill_time falls on *for_date*.
        If no date is specified, all records are included (useful for
        intraday snapshots).

        Returns
        -------
        DailyReport
        """
        target_date = for_date or datetime.now(timezone.utc).date()
        records     = self._get_records_for_date(target_date)

        # ── Fill counts ──────────────────────────────────────────────
        filled    = len(records)
        total, rejected, cancelled = self._get_order_counts(target_date)
        fill_rate = filled / max(total, 1)

        # ── Slippage stats ───────────────────────────────────────────
        slippages = [r.implementation_shortfall for r in records]
        avg_slip  = sum(slippages) / len(slippages) if slippages else 0.0
        med_slip  = self._median(slippages)

        # ── Hourly stats ─────────────────────────────────────────────
        hourly = self._tca.avg_slippage_by_hour()
        best_hour  = min(hourly, key=lambda h: hourly[h], default=None)
        worst_hour = max(hourly, key=lambda h: hourly[h], default=None)

        # ── VWAP benchmark ───────────────────────────────────────────
        total_notional = sum(r.notional_usd for r in records)
        total_qty      = sum(r.fill_qty for r in records)
        fill_vwap      = total_notional / total_qty if total_qty > 0 else 0.0
        # Simple VWAP benchmark: arithmetic mean of decision prices
        avg_decision   = (
            sum(r.decision_price for r in records) / len(records) if records else 0.0
        )
        beat_vwap      = fill_vwap < avg_decision if records else False

        # ── Trend ────────────────────────────────────────────────────
        trend = self._compute_slippage_trend()

        # ── Per-symbol slippage ──────────────────────────────────────
        per_sym: dict[str, list[float]] = {}
        for r in records:
            per_sym.setdefault(r.symbol, []).append(r.implementation_shortfall)
        per_sym_avg = {sym: sum(v)/len(v) for sym, v in per_sym.items()}

        # ── IAE hypotheses ───────────────────────────────────────────
        iae = self._tca.detect_iae_hypotheses()

        report = DailyReport(
            report_date         = str(target_date),
            total_orders        = total,
            filled_orders       = filled,
            rejected_orders     = rejected,
            cancelled_orders    = cancelled,
            fill_rate           = fill_rate,
            avg_slippage_bps    = avg_slip,
            median_slippage_bps = med_slip,
            total_commissions   = self._tca.total_commissions(),
            total_notional      = total_notional,
            best_hour_utc       = best_hour,
            worst_hour_utc      = worst_hour,
            best_hour_slippage  = hourly.get(best_hour, 0.0) if best_hour is not None else 0.0,
            worst_hour_slippage = hourly.get(worst_hour, 0.0) if worst_hour is not None else 0.0,
            vwap_benchmark      = avg_decision,
            beat_vwap           = beat_vwap,
            slippage_trend      = trend,
            per_symbol_slippage = per_sym_avg,
            iae_hypotheses      = iae,
            generated_at        = datetime.now(timezone.utc).isoformat(),
        )

        self._log_summary(report)
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, report: DailyReport) -> Path:
        """
        Write the report to a JSON file under reports_dir.

        Returns the file path.
        """
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        filename = self._reports_dir / f"exec_quality_{report.report_date}.json"
        data = {
            "report_date":          report.report_date,
            "total_orders":         report.total_orders,
            "filled_orders":        report.filled_orders,
            "rejected_orders":      report.rejected_orders,
            "cancelled_orders":     report.cancelled_orders,
            "fill_rate":            report.fill_rate,
            "avg_slippage_bps":     report.avg_slippage_bps,
            "median_slippage_bps":  report.median_slippage_bps,
            "total_commissions":    report.total_commissions,
            "total_notional":       report.total_notional,
            "best_hour_utc":        report.best_hour_utc,
            "worst_hour_utc":       report.worst_hour_utc,
            "best_hour_slippage":   report.best_hour_slippage,
            "worst_hour_slippage":  report.worst_hour_slippage,
            "vwap_benchmark":       report.vwap_benchmark,
            "beat_vwap":            report.beat_vwap,
            "slippage_trend":       report.slippage_trend,
            "per_symbol_slippage":  report.per_symbol_slippage,
            "iae_hypotheses":       report.iae_hypotheses,
            "generated_at":         report.generated_at,
        }
        filename.write_text(json.dumps(data, indent=2))
        log.info("ExecutionQualityReport saved: %s", filename)
        return filename

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_records_for_date(self, target_date: date):
        all_records = self._tca._records
        if target_date == datetime.now(timezone.utc).date():
            return all_records   # intraday: all records
        return [
            r for r in all_records
            if r.fill_time.date() == target_date
        ]

    def _get_order_counts(self, target_date: date) -> tuple[int, int, int]:
        """Return (total, rejected, cancelled) from the order book."""
        if self._order_book is None:
            n = len(self._tca)
            return n, 0, 0
        from ..oms.order import OrderStatus
        orders = [
            o for o in self._order_book.all_orders()
            if o.created_at.date() == target_date
        ]
        rejected  = sum(1 for o in orders if o.status == OrderStatus.REJECTED)
        cancelled = sum(1 for o in orders if o.status == OrderStatus.CANCELLED)
        return len(orders), rejected, cancelled

    def _compute_slippage_trend(self) -> str:
        """
        Compare first-half vs second-half average slippage to detect
        improving/degrading trends.  Returns one of:
        ``"improving"``, ``"degrading"``, ``"stable"``, ``"insufficient_data"``.
        """
        records = self._tca._records
        if len(records) < 10:
            return "insufficient_data"
        mid = len(records) // 2
        first_half_avg  = sum(r.implementation_shortfall for r in records[:mid]) / mid
        second_half_avg = sum(r.implementation_shortfall for r in records[mid:]) / (len(records) - mid)
        delta = second_half_avg - first_half_avg
        if delta < -1.0:
            return "improving"
        elif delta > 1.0:
            return "degrading"
        return "stable"

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        sv = sorted(values)
        n  = len(sv)
        if n % 2 == 1:
            return sv[n // 2]
        return (sv[n // 2 - 1] + sv[n // 2]) / 2.0

    def _log_summary(self, r: DailyReport) -> None:
        log.info(
            "ExecQuality[%s]: fills=%d/%d(%.0f%%) slip=%.1fbps comm=$%.2f "
            "beat_vwap=%s trend=%s",
            r.report_date, r.filled_orders, r.total_orders, r.fill_rate * 100,
            r.avg_slippage_bps, r.total_commissions, r.beat_vwap, r.slippage_trend,
        )
        if r.iae_hypotheses:
            for h in r.iae_hypotheses:
                log.warning("IAE: %s", h)
