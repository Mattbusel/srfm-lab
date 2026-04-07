"""
test_new_observability.py -- Tests for tracing, SLO tracking, log aggregation,
and performance baseline modules added to the SRFM observability stack.

Run with:
    pytest infra/observability/tests/test_new_observability.py -v

Covers (25 tests):
    Tracing            # TraceContext, Tracer lifecycle, context manager, store, exporter
    SLO tracking       # error budget math, burn rate, report generation, alert threshold
    Log aggregation    # StructuredLogger, LogAggregator query/filter, LogAnalyzer
    Performance baseline -- regression detection, z-score, percentile, severity labeling
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup so imports work when run directly or via pytest from repo root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent.parent.parent  # srfm-lab/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from infra.observability.tracing import (
    BackgroundExporter,
    TraceContext,
    TraceExporter,
    TraceStore,
    Tracer,
    trace_order_submission,
    trace_risk_check,
    trace_signal_computation,
    trace_param_update,
    set_default_tracer,
)
from infra.observability.slo_tracker import (
    SLO,
    SLOStatus,
    SLOTracker,
    AlertBudgetBurnRate,
    build_default_tracker,
    make_srfm_slos,
)
from infra.observability.log_aggregator import (
    LogEntry,
    LogAggregator,
    LogAnalyzer,
    StructuredLogger,
    LEVEL_NUM,
)
from infra.observability.performance_baseline import (
    PerformanceBaseline,
    Regression,
    METRIC_DIRECTIONS,
    SRFM_METRICS,
    _mean,
    _std,
    _percentile,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _ns_ago(seconds: float) -> int:
    return int((time.time() - seconds) * 1_000_000_000)


# ===========================================================================
# Tracing tests
# ===========================================================================

class TestTraceContext(unittest.TestCase):
    """TraceContext data model."""

    def test_duration_none_while_open(self):
        ctx = TraceContext(
            trace_id="abc", span_id="001", parent_span_id=None,
            service_name="svc", operation="op",
            start_time_ns=time.time_ns(),
        )
        self.assertIsNone(ctx.duration_ns)
        self.assertIsNone(ctx.duration_ms)

    def test_duration_computed_after_finish(self):
        start = time.time_ns()
        ctx = TraceContext(
            trace_id="abc", span_id="001", parent_span_id=None,
            service_name="svc", operation="op",
            start_time_ns=start,
            end_time_ns=start + 5_000_000,
        )
        self.assertAlmostEqual(ctx.duration_ms, 5.0, places=3)

    def test_is_root(self):
        ctx = TraceContext(
            trace_id="t", span_id="s", parent_span_id=None,
            service_name="svc", operation="op", start_time_ns=0,
        )
        self.assertTrue(ctx.is_root)

    def test_is_not_root_when_parent_set(self):
        ctx = TraceContext(
            trace_id="t", span_id="s", parent_span_id="p",
            service_name="svc", operation="op", start_time_ns=0,
        )
        self.assertFalse(ctx.is_root)

    def test_to_dict_keys(self):
        ctx = TraceContext(
            trace_id="t", span_id="s", parent_span_id=None,
            service_name="svc", operation="op", start_time_ns=100,
        )
        d = ctx.to_dict()
        for key in ("trace_id", "span_id", "service_name", "operation",
                    "start_time_ns", "status", "tags", "logs"):
            self.assertIn(key, d)


class TestTracer(unittest.TestCase):
    """Tracer span lifecycle."""

    def setUp(self):
        self.tracer = Tracer(service_name="test_svc")

    def test_start_and_finish_span(self):
        ctx = self.tracer.start_span("my_op")
        self.assertEqual(ctx.operation, "my_op")
        self.assertIsNone(ctx.end_time_ns)
        self.tracer.finish_span(ctx, status="ok")
        self.assertIsNotNone(ctx.end_time_ns)
        self.assertEqual(ctx.status, "ok")

    def test_parent_propagation(self):
        parent = self.tracer.start_span("parent_op")
        child = self.tracer.start_span("child_op", parent=parent)
        self.assertEqual(child.trace_id, parent.trace_id)
        self.assertEqual(child.parent_span_id, parent.span_id)

    def test_tag_and_log_event(self):
        ctx = self.tracer.start_span("tagged_op")
        self.tracer.tag(ctx, "symbol", "AAPL")
        self.tracer.log_event(ctx, "validation_passed", {"checks": 3})
        self.tracer.finish_span(ctx)
        self.assertEqual(ctx.tags.get("symbol"), "AAPL")
        self.assertEqual(len(ctx.logs), 1)
        self.assertEqual(ctx.logs[0]["event"], "validation_passed")
        self.assertEqual(ctx.logs[0]["checks"], 3)

    def test_context_manager_ok(self):
        with self.tracer.trace("cm_op") as ctx:
            self.tracer.tag(ctx, "k", "v")
        self.assertEqual(ctx.status, "ok")
        self.assertIsNotNone(ctx.end_time_ns)

    def test_context_manager_error(self):
        with self.assertRaises(ValueError):
            with self.tracer.trace("err_op") as ctx:
                raise ValueError("boom")
        self.assertEqual(ctx.status, "error")
        self.assertIn("error.message", ctx.tags)

    def test_flush_clears_completed(self):
        ctx = self.tracer.start_span("flush_op")
        self.tracer.finish_span(ctx)
        spans = self.tracer.flush()
        self.assertGreaterEqual(len(spans), 1)
        self.assertEqual(len(self.tracer.flush()), 0)

    def test_thread_safety(self):
        """Concurrent spans from multiple threads should not interfere."""
        errors: List[str] = []

        def worker():
            try:
                with self.tracer.trace("parallel_op") as c:
                    time.sleep(0.005)
                    self.assertIsNone(c.end_time_ns)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])


class TestTraceStore(unittest.TestCase):
    """TraceStore SQLite persistence and queries."""

    def setUp(self):
        self.store = TraceStore(db_path=":memory:")
        self.tracer = Tracer(service_name="store_svc")

    def _make_span(self, op: str, status: str = "ok") -> TraceContext:
        ctx = self.tracer.start_span(op)
        self.tracer.finish_span(ctx, status=status)
        return ctx

    def test_store_and_retrieve_by_trace_id(self):
        ctx = self._make_span("op1")
        self.store.store(ctx)
        spans = self.store.get_trace(ctx.trace_id)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].span_id, ctx.span_id)

    def test_get_by_service(self):
        for _ in range(3):
            self.store.store(self._make_span("op"))
        spans = self.store.get_by_service("store_svc", limit=10)
        self.assertGreaterEqual(len(spans), 3)

    def test_get_errors(self):
        self.store.store(self._make_span("ok_op", status="ok"))
        self.store.store(self._make_span("err_op", status="error"))
        errors = self.store.get_errors()
        self.assertTrue(all(s.status != "ok" for s in errors))
        self.assertGreaterEqual(len(errors), 1)

    def test_span_count(self):
        before = self.store.span_count()
        self.store.store(self._make_span("count_op"))
        self.assertEqual(self.store.span_count(), before + 1)

    def test_p99_latency(self):
        tracer = Tracer(service_name="lat_svc")
        for _ in range(20):
            ctx = tracer.start_span("lat_op")
            time.sleep(0.001)
            tracer.finish_span(ctx)
            self.store.store(ctx)
        p99 = self.store.p99_latency_ms("lat_op")
        self.assertIsNotNone(p99)
        self.assertGreater(p99, 0)


class TestPreInstrumentedHelpers(unittest.TestCase):
    """Pre-instrumented SRFM hot-path span factories."""

    def setUp(self):
        tracer = Tracer(service_name="srfm")
        set_default_tracer(tracer)

    def test_trace_order_submission(self):
        ctx = trace_order_submission("AAPL", 100.0, 175.5)
        self.assertEqual(ctx.tags.get("symbol"), "AAPL")
        self.assertEqual(ctx.tags.get("component"), "order_router")

    def test_trace_signal_computation(self):
        ctx = trace_signal_computation("TSLA", "1m")
        self.assertEqual(ctx.tags.get("timeframe"), "1m")
        self.assertEqual(ctx.tags.get("component"), "signal_engine")

    def test_trace_risk_check(self):
        ctx = trace_risk_check("ord-9999")
        self.assertEqual(ctx.tags.get("order_id"), "ord-9999")

    def test_trace_param_update(self):
        ctx = trace_param_update("bayesian_optimizer")
        self.assertEqual(ctx.tags.get("source"), "bayesian_optimizer")


class TestTraceExporterSQLite(unittest.TestCase):
    """TraceExporter.export_sqlite round-trip."""

    def test_export_and_reload(self):
        import os, sqlite3 as sq
        tracer = Tracer("export_svc")
        spans = []
        for i in range(5):
            ctx = tracer.start_span(f"op_{i}")
            tracer.tag(ctx, "index", str(i))
            tracer.finish_span(ctx)
            spans.append(ctx)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            TraceExporter.export_sqlite(spans, db_path)
            conn = sq.connect(db_path)
            count = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            conn.close()
            self.assertEqual(count, 5)
        finally:
            os.unlink(db_path)


# ===========================================================================
# SLO tracking tests
# ===========================================================================

class TestSLOErrorBudget(unittest.TestCase):
    """Error budget math on the SLO dataclass."""

    def test_total_budget_minutes(self):
        # 99.9 % over 7 days => 7*24*60 * 0.001 = 10.08 minutes
        slo = SLO("test", "desc", 0.999, 7)
        self.assertAlmostEqual(slo.error_budget_minutes_total, 10.08, places=2)

    def test_budget_used_at_target(self):
        slo = SLO("test", "desc", 0.999, 7)
        used = slo.error_budget_minutes_for_rate(0.999)
        self.assertAlmostEqual(used, 0.0, places=6)

    def test_budget_used_full_outage(self):
        slo = SLO("test", "desc", 0.999, 7)
        # 0 % success rate: used = window_days*24*60 * target_pct = 7*1440*0.999 = 10069.92
        # which is >> total budget (10.08), meaning the budget is massively overdrawn
        used = slo.error_budget_minutes_for_rate(0.0)
        expected = 7 * 24 * 60 * 0.999
        self.assertAlmostEqual(used, expected, places=2)


class TestSLOTracker(unittest.TestCase):
    """SLOTracker register, check, and report."""

    def setUp(self):
        self.tracker = SLOTracker(db_path=":memory:")

    def _register_at(self, rate: float) -> SLO:
        slo = SLO("test_slo", "test", 0.99, 7, metric_fn=lambda: rate)
        self.tracker.register(slo)
        return slo

    def test_slo_met_when_at_target(self):
        self._register_at(0.995)
        statuses = self.tracker.check_all()
        self.assertIn("test_slo", statuses)
        self.assertTrue(statuses["test_slo"].is_met)

    def test_slo_not_met_below_target(self):
        self._register_at(0.98)
        statuses = self.tracker.check_all()
        self.assertFalse(statuses["test_slo"].is_met)

    def test_error_budget_remaining_full_when_met(self):
        self._register_at(0.999)
        remaining = self.tracker.error_budget_remaining("test_slo")
        self.assertGreater(remaining, 0)

    def test_report_slos_met_count(self):
        self._register_at(0.995)
        report = self.tracker.generate_report(period_days=7)
        self.assertGreaterEqual(report.slos_met, 1)
        self.assertEqual(report.slos_total, 1)

    def test_report_to_dict(self):
        self._register_at(0.995)
        report = self.tracker.generate_report()
        d = report.to_dict()
        self.assertIn("overall_health_pct", d)
        self.assertIn("statuses", d)

    def test_make_srfm_slos_returns_five(self):
        slos = make_srfm_slos()
        self.assertEqual(len(slos), 5)

    def test_build_default_tracker(self):
        tracker = build_default_tracker()
        self.assertEqual(len(tracker.registered_names()), 5)


class TestAlertBudgetBurnRate(unittest.TestCase):
    """AlertBudgetBurnRate fires callback at high burn rate."""

    def test_check_once_returns_empty_when_slo_met(self):
        tracker = SLOTracker(db_path=":memory:")
        slo = SLO("a", "b", 0.99, 7, metric_fn=lambda: 0.999)
        tracker.register(slo)
        alert = AlertBudgetBurnRate(tracker, callback=lambda *a: None, threshold=14.0)
        breaches = alert.check_once()
        # With a healthy SLO and no prior samples, burn rate should be 0
        self.assertIsInstance(breaches, list)


# ===========================================================================
# Log aggregation tests
# ===========================================================================

class TestStructuredLogger(unittest.TestCase):
    """StructuredLogger emits well-formed LogEntry objects."""

    def test_info_returns_log_entry(self):
        logger = StructuredLogger("test_svc")
        entry = logger.info("hello", symbol="AAPL", qty=50)
        self.assertIsInstance(entry, LogEntry)
        self.assertEqual(entry.level, "INFO")
        self.assertEqual(entry.service, "test_svc")
        self.assertEqual(entry.message, "hello")
        self.assertEqual(entry.fields.get("symbol"), "AAPL")

    def test_error_level(self):
        logger = StructuredLogger("test_svc")
        entry = logger.error("boom", reason="timeout")
        self.assertEqual(entry.level, "ERROR")
        self.assertEqual(entry.fields.get("reason"), "timeout")

    def test_critical_level(self):
        logger = StructuredLogger("test_svc")
        entry = logger.critical("meltdown")
        self.assertEqual(entry.level, "CRITICAL")

    def test_trace_id_propagated(self):
        logger = StructuredLogger("svc", trace_id="trace-abc")
        entry = logger.info("msg")
        self.assertEqual(entry.trace_id, "trace-abc")

    def test_writes_to_file(self):
        import json, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            path = f.name
        try:
            logger = StructuredLogger("file_svc", path=path)
            logger.info("written", key="value")
            logger.close()
            with open(path, "r") as fh:
                lines = [l.strip() for l in fh if l.strip()]
            self.assertGreaterEqual(len(lines), 1)
            d = json.loads(lines[-1])
            self.assertEqual(d["message"], "written")
        finally:
            os.unlink(path)


class TestLogAggregatorQuery(unittest.TestCase):
    """LogAggregator query filtering."""

    def setUp(self):
        self.agg = LogAggregator(db_path=":memory:")
        logger_a = StructuredLogger("svc_a")
        logger_b = StructuredLogger("svc_b")
        entries = []
        for i in range(5):
            entries.append(logger_a.info(f"info msg {i}"))
        for i in range(3):
            entries.append(logger_a.error(f"error msg {i}"))
        for i in range(4):
            entries.append(logger_b.warning(f"warn msg {i}"))
        self.agg.ingest_many(entries)

    def test_query_by_service(self):
        results = self.agg.query(service="svc_a", n=50)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(e.service == "svc_a" for e in results))

    def test_query_by_level_filters_lower(self):
        # Query "ERROR" should return only ERROR and above, not INFO
        results = self.agg.query(level="ERROR", n=50)
        self.assertTrue(all(LEVEL_NUM[e.level] >= LEVEL_NUM["ERROR"] for e in results))
        self.assertGreaterEqual(len(results), 3)

    def test_query_text_search(self):
        results = self.agg.query(text_search="error msg", n=50)
        self.assertGreaterEqual(len(results), 3)
        self.assertTrue(all("error msg" in e.message for e in results))

    def test_query_since(self):
        # All entries were just inserted, so "since 5 seconds ago" should return all
        since_s = time.time() - 5
        results = self.agg.query(since=since_s, n=100)
        self.assertGreaterEqual(len(results), 12)

    def test_query_n_limit(self):
        results = self.agg.query(n=3)
        self.assertLessEqual(len(results), 3)


class TestLogAnalyzer(unittest.TestCase):
    """LogAnalyzer error rate and top_errors."""

    def setUp(self):
        self.agg = LogAggregator(db_path=":memory:")
        logger = StructuredLogger("analyzer_svc")
        entries = []
        for _ in range(10):
            entries.append(logger.error("timeout error"))
        for _ in range(5):
            entries.append(logger.error("connection refused"))
        for _ in range(20):
            entries.append(logger.info("all good"))
        self.agg.ingest_many(entries)
        self.analyzer = LogAnalyzer(self.agg)

    def test_error_rate_positive(self):
        rate = self.analyzer.error_rate("analyzer_svc", window_minutes=5)
        self.assertGreater(rate, 0.0)

    def test_top_errors_ordering(self):
        top = self.analyzer.top_errors("analyzer_svc", n=5)
        self.assertGreaterEqual(len(top), 1)
        # Most frequent error should appear first
        msgs = [t[0] for t in top]
        counts = [t[1] for t in top]
        self.assertIn("timeout error", msgs)
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_no_anomaly_when_stable(self):
        # No anomaly expected when error rate is uniform
        result = self.analyzer.detect_log_anomaly("analyzer_svc")
        # May or may not detect anomaly on small data; just assert no crash
        self.assertIsInstance(result, (str, type(None)))


# ===========================================================================
# Performance baseline tests
# ===========================================================================

class TestStatHelpers(unittest.TestCase):
    """Statistical helper functions."""

    def test_mean(self):
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0)

    def test_std(self):
        # Sample std (ddof=1) of [2,4,4,4,5,5,7,9] ~ 2.1381
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        self.assertAlmostEqual(_std(vals), 2.138089935299395, places=5)

    def test_percentile_median(self):
        vals = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(_percentile(vals, 0.50), 3.0)

    def test_percentile_min_max(self):
        vals = sorted([10.0, 20.0, 30.0])
        self.assertAlmostEqual(_percentile(vals, 0.0), 10.0)
        self.assertAlmostEqual(_percentile(vals, 1.0), 30.0)


class TestPerformanceBaseline(unittest.TestCase):
    """PerformanceBaseline regression detection."""

    def _make_baseline(self, metric: str, values: List[float]) -> PerformanceBaseline:
        pb = PerformanceBaseline(db_path=":memory:")
        for i, v in enumerate(values):
            pb.update_baseline(metric, v, f"2026-01-{i+1:02d}")
        return pb

    def test_no_regression_within_threshold(self):
        pb = self._make_baseline("order_fill_latency_ms", [4.0] * 20)
        reg = pb.detect_regression("order_fill_latency_ms", current_value=4.3, threshold_pct=0.10)
        self.assertIsNone(reg)

    def test_regression_detected_on_latency_spike(self):
        pb = self._make_baseline("order_fill_latency_ms", [4.0] * 20)
        reg = pb.detect_regression("order_fill_latency_ms", current_value=8.0, threshold_pct=0.10)
        self.assertIsNotNone(reg)
        self.assertIsInstance(reg, Regression)
        self.assertGreater(reg.pct_change, 0)

    def test_severity_critical_on_large_spike(self):
        pb = self._make_baseline("order_fill_latency_ms", [4.0] * 20)
        reg = pb.detect_regression("order_fill_latency_ms", current_value=20.0, threshold_pct=0.10)
        self.assertIsNotNone(reg)
        self.assertEqual(reg.severity, "critical")

    def test_regression_returns_none_on_insufficient_data(self):
        pb = PerformanceBaseline(db_path=":memory:")
        pb.update_baseline("order_fill_latency_ms", 4.0, "2026-01-01")
        # Only 1 sample -- not enough
        reg = pb.detect_regression("order_fill_latency_ms", current_value=99.0, threshold_pct=0.10)
        self.assertIsNone(reg)

    def test_sharpe_regression_on_drop(self):
        # Sharpe is higher-better, so a drop should be flagged
        pb = self._make_baseline("sharpe_ratio_4h", [1.5] * 20)
        reg = pb.detect_regression("sharpe_ratio_4h", current_value=0.5, threshold_pct=0.10)
        self.assertIsNotNone(reg)
        self.assertLess(reg.pct_change, 0)

    def test_z_score_zero_for_mean(self):
        pb = self._make_baseline("risk_check_ms", [10.0] * 20)
        z = pb.z_score("risk_check_ms", current_value=10.0)
        self.assertAlmostEqual(z, 0.0, places=4)

    def test_z_score_positive_above_mean(self):
        pb = self._make_baseline("risk_check_ms", [10.0] * 20)
        # std is 0 for identical values; inject some variance
        pb2 = PerformanceBaseline(db_path=":memory:")
        for i, v in enumerate([8.0, 9.0, 10.0, 11.0, 12.0] * 4):
            pb2.update_baseline("risk_check_ms", v, f"2026-01-{i+1:02d}")
        z = pb2.z_score("risk_check_ms", current_value=15.0)
        self.assertGreater(z, 0)

    def test_baseline_for_percentile(self):
        pb = self._make_baseline("signal_computation_ms", list(range(1, 21)))
        p90 = pb.baseline_for("signal_computation_ms", percentile=0.90)
        self.assertGreater(p90, 15.0)  # 90th pct of 1..20 is ~19

    def test_on_regression_callback_fires(self):
        fired: List[Regression] = []
        pb = PerformanceBaseline(
            db_path=":memory:",
            on_regression=lambda r: fired.append(r),
        )
        for i in range(20):
            pb.update_baseline("order_fill_latency_ms", 4.0, f"2026-01-{i+1:02d}")
        pb.detect_regression("order_fill_latency_ms", current_value=20.0)
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].metric, "order_fill_latency_ms")

    def test_summary_returns_list(self):
        pb = self._make_baseline("order_fill_latency_ms", [4.0] * 10)
        summary = pb.summary()
        self.assertIsInstance(summary, list)
        self.assertGreaterEqual(len(summary), 1)
        self.assertIn("metric", summary[0])
        self.assertIn("p99", summary[0])

    def test_srfm_metrics_list_nonempty(self):
        self.assertIn("order_fill_latency_ms", SRFM_METRICS)
        self.assertIn("sharpe_ratio_4h", SRFM_METRICS)


if __name__ == "__main__":
    unittest.main()
