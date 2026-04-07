"""
test_reporting.py -- Test suite for LARSA v18 trade journal and reporting system.

Covers:
- JournalEntry round-trip serialization
- TradeJournal CRUD operations
- WeeklyReport markdown formatting
- PerformanceReport Sharpe/Sortino/Calmar
- EquityCurveAnalyzer drawdown detection and bootstrap Sharpe
- AttributionReport layer attribution and BHB decomposition
- BenchmarkComparison beta/alpha/IR
- 40+ test cases
"""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

# Make tools/ importable
_tools_dir = Path(__file__).resolve().parent.parent
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

from trade_journal import (
    JournalEntry,
    TradeJournal,
    WeeklyReport,
    WeeklyReportData,
    JournalStats,
    _make_sample_entry,
)
from performance_report import (
    PerformanceReport,
    BenchmarkComparison,
    _sharpe,
    _sortino,
    _max_drawdown,
    _calmar,
    _build_equity_curve,
    _find_all_drawdowns,
    _compute_streaks,
)
from equity_curve_analyzer import (
    EquityCurveAnalyzer,
    EquityDecomposition,
    RegimeBreakpoint,
    RecoveryMetrics,
    _hp_filter,
    _baxter_king_bandpass,
    _cusum_detect,
    _bootstrap_sharpe,
    _predict_drawdown_probability,
)
from attribution_report import (
    AttributionReport,
    AttributionResult,
    LayerAttribution,
    _pearson_corr,
    _spearman_corr,
    _rank,
    _simulate_bh_only,
    _simulate_bh_plus_cf,
    _simulate_full,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = random.Random(99)


def _random_entry(
    symbol: str = "BTC",
    net_pnl: float | None = None,
    hold_bars: int | None = None,
    ts_offset_hours: int = 0,
) -> JournalEntry:
    if net_pnl is None:
        net_pnl = RNG.gauss(50, 300)
    if hold_bars is None:
        hold_bars = RNG.randint(2, 48)
    e = _make_sample_entry(symbol, net_pnl, hold_bars)
    base = datetime(2024, 3, 1) + timedelta(hours=ts_offset_hours)
    e.entry_ts = base.isoformat()
    e.exit_ts = (base + timedelta(hours=hold_bars)).isoformat()
    e.portfolio_nav_at_entry = 100_000.0
    e.bh_active = RNG.random() > 0.2
    e.cf_direction = RNG.choice([-1, 0, 1])
    e.hurst_regime = RNG.choice(["trending", "neutral", "mean-reverting"])
    e.vol_regime = RNG.choice(["low", "med", "high"])
    e.was_cf_filtered = RNG.random() > 0.65
    e.was_hurst_damped = RNG.random() > 0.75
    e.was_nav_gated = RNG.random() > 0.85
    e.was_ml_filtered = RNG.random() > 0.9
    e.was_event_calendar_filtered = RNG.random() > 0.93
    e.was_rl_exit = RNG.random() > 0.85
    return e


def _make_trades(n: int = 50) -> List[JournalEntry]:
    symbols = ["BTC", "ETH", "SOL", "AAPL", "NVDA"]
    trades = []
    for i in range(n):
        sym = RNG.choice(symbols)
        pnl = RNG.gauss(60, 400)
        trades.append(_random_entry(sym, pnl, ts_offset_hours=i * 6))
    return trades


def _temp_journal(trades: List[JournalEntry] | None = None) -> TradeJournal:
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    j = TradeJournal(f.name)
    if trades:
        for t in trades:
            j.add_entry(t)
    return j


def _linear_equity(n: int = 100, start: float = 100_000, slope: float = 50) -> List[float]:
    return [start + i * slope for i in range(n)]


def _sinusoidal_equity(n: int = 200, start: float = 100_000, amp: float = 5000) -> List[float]:
    return [start + amp * math.sin(2 * math.pi * i / 40) for i in range(n)]


def _drawdown_equity() -> List[float]:
    """Equity curve with a clear drawdown then recovery."""
    curve = [100_000.0]
    for i in range(49):
        curve.append(curve[-1] + 500)   # up 49 bars
    for i in range(30):
        curve.append(curve[-1] - 800)   # drawdown
    for i in range(40):
        curve.append(curve[-1] + 600)   # recovery
    return curve


# ---------------------------------------------------------------------------
# JournalEntry Tests
# ---------------------------------------------------------------------------


class TestJournalEntry(unittest.TestCase):

    def test_default_fields_are_set(self):
        e = JournalEntry()
        self.assertIsInstance(e.trade_id, str)
        self.assertTrue(len(e.trade_id) > 0)
        self.assertEqual(e.symbol, "")
        self.assertEqual(e.strategy_version, "LARSA_v18")

    def test_roundtrip_to_dict_and_back(self):
        """test_journal_entry_roundtrip -- full field round-trip via dict."""
        e = _random_entry("ETH", 250.0, 10)
        d = e.to_dict()
        e2 = JournalEntry.from_dict(d)
        self.assertEqual(e.trade_id, e2.trade_id)
        self.assertEqual(e.symbol, e2.symbol)
        self.assertAlmostEqual(e.net_pnl, e2.net_pnl, places=6)
        self.assertAlmostEqual(e.bh_mass_15m, e2.bh_mass_15m, places=6)
        self.assertEqual(e.hurst_regime, e2.hurst_regime)
        self.assertEqual(e.was_rl_exit, e2.was_rl_exit)
        self.assertEqual(e.exit_reason, e2.exit_reason)

    def test_from_dict_ignores_extra_keys(self):
        """Extra keys in dict should be silently ignored."""
        e = _random_entry("SOL")
        d = e.to_dict()
        d["__extra_field__"] = "should_be_ignored"
        e2 = JournalEntry.from_dict(d)
        self.assertEqual(e.trade_id, e2.trade_id)

    def test_is_winner(self):
        e = _random_entry(net_pnl=100.0)
        self.assertTrue(e.is_winner())
        e2 = _random_entry(net_pnl=-50.0)
        self.assertFalse(e2.is_winner())

    def test_edge_ratio_finite(self):
        e = _random_entry()
        e.mfe_pct = 0.04
        e.mae_pct = 0.01
        self.assertAlmostEqual(e.edge_ratio(), 4.0, places=6)

    def test_edge_ratio_zero_mae(self):
        e = _random_entry()
        e.mfe_pct = 0.03
        e.mae_pct = 0.0
        self.assertEqual(e.edge_ratio(), float("inf"))

    def test_hold_duration_hours(self):
        e = JournalEntry()
        e.entry_ts = "2024-03-01T10:00:00"
        e.exit_ts = "2024-03-01T22:00:00"
        self.assertAlmostEqual(e.hold_duration_hours(), 12.0, places=4)

    def test_hold_duration_invalid_ts(self):
        e = JournalEntry()
        e.entry_ts = "bad"
        e.exit_ts = "bad"
        self.assertEqual(e.hold_duration_hours(), 0.0)

    def test_unique_trade_ids(self):
        ids = {JournalEntry().trade_id for _ in range(100)}
        self.assertEqual(len(ids), 100)

    def test_nav_quaternion_json(self):
        e = _random_entry("BTC")
        q = [0.92, 0.21, 0.31, 0.12]
        e.nav_quaternion = json.dumps(q)
        d = e.to_dict()
        e2 = JournalEntry.from_dict(d)
        self.assertEqual(json.loads(e2.nav_quaternion), q)

    def test_all_fifty_plus_fields_present(self):
        fields = list(JournalEntry.__dataclass_fields__.keys())
        self.assertGreaterEqual(len(fields), 50)


# ---------------------------------------------------------------------------
# TradeJournal Tests
# ---------------------------------------------------------------------------


class TestTradeJournal(unittest.TestCase):

    def setUp(self):
        self.trades = _make_trades(30)
        self.journal = _temp_journal(self.trades)

    def tearDown(self):
        self.journal.close()

    def test_add_and_count(self):
        self.assertEqual(self.journal.count(), 30)

    def test_get_by_id(self):
        e = self.trades[0]
        retrieved = self.journal.get_by_id(e.trade_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.trade_id, e.trade_id)

    def test_get_recent_default(self):
        recent = self.journal.get_recent()
        self.assertLessEqual(len(recent), 20)

    def test_get_recent_n(self):
        recent = self.journal.get_recent(n=5)
        self.assertEqual(len(recent), 5)

    def test_get_by_symbol(self):
        btc_trades = [t for t in self.trades if t.symbol == "BTC"]
        result = self.journal.get_by_symbol("BTC")
        self.assertEqual(len(result), len(btc_trades))

    def test_get_by_symbol_since(self):
        since = "2024-03-02T00:00:00"
        result = self.journal.get_by_symbol("BTC", since=since)
        for e in result:
            self.assertGreaterEqual(e.entry_ts, since)

    def test_search_filter(self):
        winners = self.journal.search(lambda e: e.net_pnl > 0)
        for e in winners:
            self.assertGreater(e.net_pnl, 0)

    def test_search_symbol_and_pnl(self):
        result = self.journal.search(
            lambda e: e.symbol in ("BTC", "ETH") and e.net_pnl > 100
        )
        for e in result:
            self.assertIn(e.symbol, ("BTC", "ETH"))
            self.assertGreater(e.net_pnl, 100)

    def test_update_entry(self):
        e = self.trades[0]
        self.journal.update_entry(e.trade_id, notes="updated note")
        e2 = self.journal.get_by_id(e.trade_id)
        self.assertEqual(e2.notes, "updated note")

    def test_delete_entry(self):
        e = self.trades[0]
        self.journal.delete_entry(e.trade_id)
        self.assertIsNone(self.journal.get_by_id(e.trade_id))
        self.assertEqual(self.journal.count(), 29)

    def test_get_all_order(self):
        all_entries = self.journal.get_all()
        ts_list = [e.entry_ts for e in all_entries]
        self.assertEqual(ts_list, sorted(ts_list))

    def test_get_between(self):
        start = "2024-03-01T00:00:00"
        end = "2024-03-02T23:59:59"
        result = self.journal.get_between(start, end)
        for e in result:
            self.assertGreaterEqual(e.entry_ts, start)
            self.assertLessEqual(e.entry_ts, end)

    def test_stats_total_trades(self):
        stats = self.journal.get_stats()
        self.assertEqual(stats.total_trades, 30)

    def test_stats_win_rate_range(self):
        stats = self.journal.get_stats()
        self.assertGreaterEqual(stats.win_rate, 0.0)
        self.assertLessEqual(stats.win_rate, 1.0)

    def test_stats_profit_factor_positive(self):
        stats = self.journal.get_stats()
        self.assertGreaterEqual(stats.profit_factor, 0.0)

    def test_stats_most_traded_symbol(self):
        stats = self.journal.get_stats()
        self.assertIn(stats.most_traded_symbol, ["BTC", "ETH", "SOL", "AAPL", "NVDA"])

    def test_stats_best_worst_trade(self):
        stats = self.journal.get_stats()
        self.assertGreaterEqual(stats.best_trade_pnl, stats.worst_trade_pnl)


# ---------------------------------------------------------------------------
# WeeklyReport Tests
# ---------------------------------------------------------------------------


class TestWeeklyReport(unittest.TestCase):

    def setUp(self):
        # Create trades in week of 2024-03-04 (Monday)
        self.week_start = date(2024, 3, 4)
        trades = []
        base = datetime(2024, 3, 4, 9, 0, 0)
        for i in range(20):
            sym = ["BTC", "ETH", "SOL"][i % 3]
            pnl = (i % 2 * 2 - 1) * (100 + i * 20)   # alternating win/loss
            e = _make_sample_entry(sym, float(pnl), 4)
            ts = base + timedelta(hours=i * 8)
            e.entry_ts = ts.isoformat()
            e.exit_ts = (ts + timedelta(hours=4)).isoformat()
            trades.append(e)
        self.journal = _temp_journal(trades)
        self.reporter = WeeklyReport(self.journal)

    def tearDown(self):
        self.journal.close()

    def test_generate_returns_data(self):
        data = self.reporter.generate(self.week_start)
        self.assertIsInstance(data, WeeklyReportData)

    def test_generate_total_trades(self):
        data = self.reporter.generate(self.week_start)
        self.assertEqual(data.total_trades, 20)

    def test_generate_empty_week(self):
        empty_week = date(2020, 1, 6)
        data = self.reporter.generate(empty_week)
        self.assertEqual(data.total_trades, 0)
        self.assertEqual(data.net_pnl, 0.0)

    def test_generate_best_worst_trades(self):
        data = self.reporter.generate(self.week_start)
        self.assertLessEqual(len(data.best_trades), 5)
        self.assertLessEqual(len(data.worst_trades), 5)
        if data.best_trades and data.worst_trades:
            self.assertGreaterEqual(
                data.best_trades[0].net_pnl, data.worst_trades[-1].net_pnl
            )

    def test_generate_symbol_pnl_keys(self):
        data = self.reporter.generate(self.week_start)
        for sym in data.symbol_pnl:
            self.assertIn(sym, ["BTC", "ETH", "SOL"])

    def test_to_markdown_format(self):
        """test_weekly_report_markdown_format -- verify structure of markdown output."""
        data = self.reporter.generate(self.week_start)
        md = self.reporter.to_markdown(data)
        self.assertIn("# Weekly Trading Report", md)
        self.assertIn("## Summary", md)
        self.assertIn("## Daily P&L", md)
        self.assertIn("## Symbol Breakdown", md)
        self.assertIn("## Top 5 Trades", md)
        self.assertIn("## Bottom 5 Trades", md)
        self.assertIn("## Signal Attribution", md)
        self.assertIn("## Regime Breakdown", md)

    def test_to_markdown_contains_pnl(self):
        data = self.reporter.generate(self.week_start)
        md = self.reporter.to_markdown(data)
        self.assertIn("Net P&L", md)
        self.assertIn("Win rate", md)

    def test_to_markdown_tables_have_pipes(self):
        data = self.reporter.generate(self.week_start)
        md = self.reporter.to_markdown(data)
        table_lines = [l for l in md.splitlines() if "|" in l]
        self.assertGreater(len(table_lines), 5)

    def test_to_html_produces_html(self):
        data = self.reporter.generate(self.week_start)
        html = self.reporter.to_html(data)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Weekly Trading Report", html)
        self.assertIn("<table>", html)

    def test_signal_attribution_keys(self):
        data = self.reporter.generate(self.week_start)
        expected_keys = {
            "bh_only", "cf_filtered", "hurst_damped",
            "nav_gated", "ml_filtered", "rl_exit", "event_filtered",
        }
        self.assertEqual(set(data.signal_attribution.keys()), expected_keys)

    def test_win_rate_valid(self):
        data = self.reporter.generate(self.week_start)
        self.assertGreaterEqual(data.win_rate, 0.0)
        self.assertLessEqual(data.win_rate, 1.0)


# ---------------------------------------------------------------------------
# PerformanceReport Tests
# ---------------------------------------------------------------------------


class TestPerformanceReport(unittest.TestCase):

    def setUp(self):
        self.trades = _make_trades(80)
        self.journal = _temp_journal(self.trades)

    def tearDown(self):
        self.journal.close()

    def test_generate_full_report_keys(self):
        rpt = PerformanceReport(self.journal.db_path)
        report = rpt.generate_full_report()
        rpt.close()
        expected_keys = {
            "overview", "monthly_breakdown", "symbol_breakdown",
            "signal_attribution", "regime_analysis", "drawdown_analysis",
            "trade_statistics",
        }
        self.assertEqual(set(report.keys()), expected_keys)

    def test_overview_contains_sharpe(self):
        rpt = PerformanceReport(self.journal.db_path)
        report = rpt.generate_full_report()
        rpt.close()
        self.assertIn("sharpe", report["overview"])
        self.assertIsInstance(report["overview"]["sharpe"], float)

    def test_sharpe_function(self):
        """test_performance_report_sharpe -- validate Sharpe calculation."""
        # All returns equal to rf -> Sharpe = 0
        returns = [0.01] * 100
        s = _sharpe(returns, rf_per_period=0.01)
        self.assertAlmostEqual(s, 0.0, places=6)

    def test_sharpe_positive_for_good_returns(self):
        rng = random.Random(1)
        returns = [rng.gauss(0.005, 0.01) for _ in range(200)]
        s = _sharpe(returns, rf_per_period=0.0)
        self.assertGreater(s, 0)

    def test_sharpe_empty(self):
        self.assertEqual(_sharpe([]), 0.0)

    def test_sortino_positive(self):
        rng = random.Random(2)
        returns = [rng.gauss(0.005, 0.01) for _ in range(200)]
        s = _sortino(returns)
        self.assertGreater(s, 0)

    def test_sortino_no_losses(self):
        returns = [0.01] * 50
        s = _sortino(returns)
        self.assertEqual(s, float("inf"))

    def test_max_drawdown_flat_curve(self):
        curve = [100.0] * 50
        dd, _, _ = _max_drawdown(curve)
        self.assertAlmostEqual(dd, 0.0, places=6)

    def test_max_drawdown_known_curve(self):
        curve = [100, 110, 90, 95, 80, 100]
        dd, pk, tr = _max_drawdown(curve)
        # Peak at 110, trough at 80: DD = 30/110 ~ 0.272
        self.assertAlmostEqual(dd, 30 / 110, places=4)
        self.assertEqual(pk, 1)  # index of 110
        self.assertEqual(tr, 4)  # index of 80

    def test_max_drawdown_monotone_rise(self):
        curve = list(range(1, 101))
        dd, _, _ = _max_drawdown(curve)
        self.assertAlmostEqual(dd, 0.0, places=6)

    def test_calmar_zero_dd(self):
        c = _calmar(0.5, 0.0)
        self.assertEqual(c, 0.0)

    def test_calmar_positive(self):
        c = _calmar(0.3, 0.1, years=1.0)
        self.assertGreater(c, 0)

    def test_build_equity_curve_length(self):
        trades = _make_trades(20)
        curve = _build_equity_curve(trades, 100_000)
        self.assertEqual(len(curve), 21)  # start + 1 per trade

    def test_find_all_drawdowns(self):
        curve = _drawdown_equity()
        periods = _find_all_drawdowns(curve)
        self.assertGreaterEqual(len(periods), 1)
        # All depths should be positive
        for p in periods:
            self.assertGreater(p.depth, 0)

    def test_compute_streaks(self):
        outcomes = [True, True, True, False, False, True]
        result = _compute_streaks(outcomes)
        self.assertEqual(result["max_win"], 3)
        self.assertEqual(result["max_loss"], 2)
        self.assertEqual(result["current"], 1)  # ends on win

    def test_export_csv(self):
        import csv as _csv
        rpt = PerformanceReport(self.journal.db_path)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            csv_path = f.name
        rpt.export_csv(csv_path)
        rpt.close()
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 80)
        self.assertIn("trade_id", rows[0])
        self.assertIn("net_pnl", rows[0])

    def test_monthly_breakdown_keys_are_iso(self):
        rpt = PerformanceReport(self.journal.db_path)
        report = rpt.generate_full_report()
        rpt.close()
        for key in report["monthly_breakdown"]:
            self.assertRegex(key, r"^\d{4}-\d{2}$")

    def test_symbol_pct_of_total_sums_to_one(self):
        rpt = PerformanceReport(self.journal.db_path)
        report = rpt.generate_full_report()
        rpt.close()
        total = sum(
            v["pct_of_total_pnl"] for v in report["symbol_breakdown"].values()
        )
        self.assertAlmostEqual(total, 1.0, places=4)


# ---------------------------------------------------------------------------
# BenchmarkComparison Tests
# ---------------------------------------------------------------------------


class TestBenchmarkComparison(unittest.TestCase):

    def setUp(self):
        self.rng = random.Random(7)
        self.n = 100
        self.strat = [self.rng.gauss(0.005, 0.02) for _ in range(self.n)]
        self.bench = [self.rng.gauss(0.003, 0.025) for _ in range(self.n)]

    def test_beta_positive_correlated(self):
        """test_benchmark_comparison_beta -- beta > 0 for correlated returns."""
        # Make strategy strongly correlated to benchmark
        strat = [b + self.rng.gauss(0, 0.005) for b in self.bench]
        beta = BenchmarkComparison.compute_beta(strat, self.bench)
        self.assertGreater(beta, 0.5)

    def test_beta_negatively_correlated(self):
        strat = [-b + self.rng.gauss(0, 0.002) for b in self.bench]
        beta = BenchmarkComparison.compute_beta(strat, self.bench)
        self.assertLess(beta, 0)

    def test_beta_uncorrelated_near_zero(self):
        rng2 = random.Random(13)
        strat = [rng2.gauss(0, 0.02) for _ in range(500)]
        bench = [rng2.gauss(0, 0.02) for _ in range(500)]
        beta = BenchmarkComparison.compute_beta(strat, bench)
        self.assertLess(abs(beta), 0.3)

    def test_alpha_zero_when_equal(self):
        """If strategy == benchmark, alpha should be ~0."""
        alpha = BenchmarkComparison.compute_alpha(self.bench, self.bench)
        self.assertAlmostEqual(alpha, 0.0, places=6)

    def test_alpha_positive_for_outperformer(self):
        better = [b + 0.005 for b in self.bench]
        alpha = BenchmarkComparison.compute_alpha(better, self.bench)
        self.assertGreater(alpha, 0)

    def test_information_ratio_perfect_tracking(self):
        """IR should be very large (or inf) if tracking error is zero."""
        ir = BenchmarkComparison.compute_information_ratio(self.bench, self.bench)
        # active return = 0, IR = 0/0 -> we return 0
        self.assertEqual(ir, 0.0)

    def test_information_ratio_outperformer(self):
        better = [b + 0.003 for b in self.bench]
        ir = BenchmarkComparison.compute_information_ratio(better, self.bench)
        self.assertGreater(ir, 0)

    def test_information_ratio_mismatched_lengths(self):
        short_bench = self.bench[:50]
        ir = BenchmarkComparison.compute_information_ratio(self.strat, short_bench)
        self.assertIsInstance(ir, float)

    def test_btc_buyhold_returns_length(self):
        trades = _make_trades(10)
        returns = BenchmarkComparison.btc_buyhold_returns(trades)
        self.assertEqual(len(returns), 10)

    def test_sixty_forty_returns_length(self):
        returns = BenchmarkComparison.sixty_forty_returns(50)
        self.assertEqual(len(returns), 50)


# ---------------------------------------------------------------------------
# EquityCurveAnalyzer Tests
# ---------------------------------------------------------------------------


class TestEquityCurveAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = EquityCurveAnalyzer(hp_lambda=100.0, bk_K=6)
        rng = random.Random(5)
        nav = 100_000.0
        self.curve = [nav]
        for _ in range(199):
            nav *= 1.0 + rng.gauss(0.0005, 0.015)
            self.curve.append(nav)

    def test_decompose_lengths(self):
        decomp = self.analyzer.decompose(self.curve)
        n = len(self.curve)
        self.assertEqual(len(decomp.trend), n)
        self.assertEqual(len(decomp.cyclical), n)
        self.assertEqual(len(decomp.noise), n)

    def test_decompose_trend_positive(self):
        decomp = self.analyzer.decompose(_linear_equity(100))
        for t in decomp.trend:
            self.assertGreater(t, 0)

    def test_decompose_short_curve(self):
        decomp = self.analyzer.decompose([100.0, 102.0])
        self.assertEqual(len(decomp.trend), 2)

    def test_hp_filter_output_lengths(self):
        series = [float(i) for i in range(50)]
        trend, cycle = _hp_filter(series, lam=1600)
        self.assertEqual(len(trend), 50)
        self.assertEqual(len(cycle), 50)

    def test_bk_bandpass_length(self):
        series = [float(i) + 0.1 * i ** 0.5 for i in range(80)]
        filtered = _baxter_king_bandpass(series, K=6)
        self.assertEqual(len(filtered), 80)

    def test_bk_edges_are_zero(self):
        series = [100.0 + i for i in range(80)]
        filtered = _baxter_king_bandpass(series, K=6)
        self.assertEqual(filtered[0], 0.0)
        self.assertEqual(filtered[-1], 0.0)

    def test_detect_regime_changes_returns_list(self):
        breakpoints = self.analyzer.detect_regime_changes(self.curve)
        self.assertIsInstance(breakpoints, list)

    def test_detect_regime_changes_on_step_function(self):
        """test_equity_curve_drawdown_detection -- CUSUM should catch a structural break."""
        # Create a curve with a clear mean shift at bar 100
        rng = random.Random(10)
        curve = [100_000.0]
        for i in range(199):
            if i < 100:
                r = rng.gauss(0.003, 0.01)
            else:
                r = rng.gauss(-0.003, 0.01)   # regime flip
            curve.append(curve[-1] * (1.0 + r))
        bps = self.analyzer.detect_regime_changes(curve)
        self.assertGreater(len(bps), 0)

    def test_regime_breakpoint_index_in_range(self):
        bps = self.analyzer.detect_regime_changes(self.curve)
        for bp in bps:
            self.assertGreater(bp.index, 0)
            self.assertLess(bp.index, len(self.curve))

    def test_regime_breakpoint_confidence_range(self):
        bps = self.analyzer.detect_regime_changes(self.curve)
        for bp in bps:
            self.assertGreaterEqual(bp.confidence, 0.0)
            self.assertLessEqual(bp.confidence, 1.0)

    def test_recovery_metrics_on_drawdown_curve(self):
        curve = _drawdown_equity()
        metrics = self.analyzer.compute_recovery_metrics(curve)
        self.assertGreater(metrics.num_drawdown_periods, 0)
        self.assertGreater(metrics.max_drawdown_depth, 0)
        self.assertGreaterEqual(metrics.underwater_fraction, 0)
        self.assertLessEqual(metrics.underwater_fraction, 1)

    def test_recovery_factor_positive(self):
        curve = _drawdown_equity()
        metrics = self.analyzer.compute_recovery_metrics(curve)
        # Total return should be positive (ends above start)
        if curve[-1] > curve[0] and metrics.max_drawdown_depth > 0:
            self.assertGreater(metrics.recovery_factor, 0)

    def test_bootstrap_sharpe_returns_tuple(self):
        sharpe, ci = self.analyzer.bootstrap_sharpe(self.curve, n_boot=100)
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(ci, float)
        self.assertGreaterEqual(ci, 0.0)

    def test_bootstrap_sharpe_ci_sensible(self):
        rng = random.Random(6)
        curve = [100_000.0]
        for _ in range(99):
            curve.append(curve[-1] * (1 + rng.gauss(0.001, 0.02)))
        sharpe, ci = self.analyzer.bootstrap_sharpe(curve, n_boot=200)
        # CI half-width should be a positive fraction of |sharpe|
        self.assertGreater(ci, 0)

    def test_bootstrap_sharpe_standalone(self):
        returns = [0.005 + random.gauss(0, 0.01) for _ in range(100)]
        pt, hw = _bootstrap_sharpe(returns, n_boot=200)
        self.assertIsInstance(pt, float)
        self.assertGreater(hw, 0)

    def test_predict_drawdown_probability_range(self):
        prob = self.analyzer.predict_next_drawdown(self.curve)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_drawdown_short_curve(self):
        prob = self.analyzer.predict_next_drawdown([100.0, 105.0, 103.0])
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_full_analysis_keys(self):
        result = self.analyzer.full_analysis(self.curve, n_boot=50)
        self.assertIn("decomposition", result)
        self.assertIn("regime_breakpoints", result)
        self.assertIn("recovery", result)
        self.assertIn("bootstrap_sharpe", result)
        self.assertIn("drawdown_prediction", result)

    def test_noise_std_non_negative(self):
        decomp = self.analyzer.decompose(self.curve)
        self.assertGreaterEqual(decomp.noise_std(), 0.0)

    def test_cyclical_amplitude_non_negative(self):
        decomp = self.analyzer.decompose(self.curve)
        self.assertGreaterEqual(decomp.cyclical_amplitude(), 0.0)


# ---------------------------------------------------------------------------
# AttributionReport Tests
# ---------------------------------------------------------------------------


class TestAttributionReport(unittest.TestCase):

    def setUp(self):
        self.trades = _make_trades(60)
        self.rpt = AttributionReport(self.trades)

    def test_compute_layer_attribution_returns_result(self):
        result = self.rpt.compute_layer_attribution()
        self.assertIsInstance(result, AttributionResult)

    def test_layers_count(self):
        result = self.rpt.compute_layer_attribution()
        self.assertEqual(len(result.layers), len(self.rpt.simulate_fns))

    def test_layers_sum_close_to_total(self):
        """test_attribution_report_layers_sum -- sum of deltas == final - baseline."""
        result = self.rpt.compute_layer_attribution()
        delta_sum = sum(la.delta_pnl for la in result.layers[1:])
        expected = result.final_pnl - result.baseline_pnl
        self.assertAlmostEqual(delta_sum + result.unexplained, expected, places=4)

    def test_baseline_is_bh_mass_only(self):
        result = self.rpt.compute_layer_attribution()
        bh_trades = _simulate_bh_only(self.trades)
        expected_pnl = sum(t.net_pnl for t in bh_trades)
        self.assertAlmostEqual(result.baseline_pnl, expected_pnl, places=4)

    def test_final_pnl_is_full_stack(self):
        result = self.rpt.compute_layer_attribution()
        full_trades = _simulate_full(self.trades)
        expected = sum(t.net_pnl for t in full_trades)
        self.assertAlmostEqual(result.final_pnl, expected, places=4)

    def test_win_rates_in_range(self):
        result = self.rpt.compute_layer_attribution()
        for la in result.layers:
            self.assertGreaterEqual(la.win_rate, 0.0)
            self.assertLessEqual(la.win_rate, 1.0)

    def test_trades_monotone_decrease_or_equal(self):
        """Each layer should have <= trades than previous (filters only remove)."""
        result = self.rpt.compute_layer_attribution()
        for i in range(1, len(result.layers)):
            self.assertLessEqual(
                result.layers[i].trades_included,
                result.layers[i - 1].trades_included,
            )

    def test_attribution_table_length(self):
        result = self.rpt.compute_layer_attribution()
        table = result.attribution_table()
        self.assertEqual(len(table), len(result.layers))

    def test_attribution_table_keys(self):
        result = self.rpt.compute_layer_attribution()
        row = result.attribution_table()[0]
        for key in ("layer", "delta_pnl", "cumulative_pnl", "win_rate", "trades"):
            self.assertIn(key, row)

    def test_to_markdown_contains_layers(self):
        result = self.rpt.compute_layer_attribution()
        md = self.rpt.to_markdown(result)
        self.assertIn("BH mass (baseline)", md)
        self.assertIn("CF cross-filter", md)
        self.assertIn("Hurst", md)

    def test_to_markdown_has_bhb_section(self):
        result = self.rpt.compute_layer_attribution()
        md = self.rpt.to_markdown(result)
        self.assertIn("BHB Decomposition", md)

    def test_pearson_corr_perfect(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(_pearson_corr(x, x), 1.0, places=6)

    def test_pearson_corr_anti(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(_pearson_corr(x, y), -1.0, places=6)

    def test_pearson_corr_zero_variance(self):
        x = [1.0] * 10
        y = list(range(10))
        self.assertEqual(_pearson_corr(x, y), 0.0)

    def test_spearman_corr_monotone(self):
        x = [1.0, 3.0, 5.0, 7.0, 9.0]
        self.assertAlmostEqual(_spearman_corr(x, x), 1.0, places=4)

    def test_rank_basic(self):
        vals = [3.0, 1.0, 2.0]
        ranks = _rank(vals)
        self.assertEqual(ranks[0], 3.0)   # 3 is highest
        self.assertEqual(ranks[1], 1.0)   # 1 is lowest
        self.assertEqual(ranks[2], 2.0)

    def test_rank_ties(self):
        vals = [1.0, 1.0, 3.0]
        ranks = _rank(vals)
        # Tied at 1.0 -> avg rank 1.5
        self.assertAlmostEqual(ranks[0], 1.5, places=4)
        self.assertAlmostEqual(ranks[1], 1.5, places=4)
        self.assertAlmostEqual(ranks[2], 3.0, places=4)

    def test_signal_correlations_keys(self):
        corrs = self.rpt.compute_signal_correlations()
        self.assertIn("bh_mass_15m", corrs)
        self.assertIn("ml_signal", corrs)
        for v in corrs.values():
            self.assertGreaterEqual(v, -1.0)
            self.assertLessEqual(v, 1.0)

    def test_compute_all_ics(self):
        ics = self.rpt.compute_all_ics()
        expected_fields = [
            "bh_mass_15m", "bh_mass_1h", "bh_mass_4h",
            "cf_alignment", "hurst_h", "nav_omega", "ml_signal",
        ]
        for f in expected_fields:
            self.assertIn(f, ics)
            self.assertGreaterEqual(ics[f], -1.0)
            self.assertLessEqual(ics[f], 1.0)

    def test_top_contributing_trades(self):
        top = self.rpt.top_contributing_trades(n=5)
        self.assertLessEqual(len(top), 5)
        if len(top) > 1:
            self.assertGreaterEqual(abs(top[0].net_pnl), abs(top[-1].net_pnl))

    def test_empty_trades(self):
        rpt = AttributionReport([])
        result = rpt.compute_layer_attribution()
        self.assertEqual(result.baseline_pnl, 0.0)
        self.assertEqual(result.final_pnl, 0.0)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        """
        End-to-end: generate trades -> journal -> performance report
        -> equity analyzer -> attribution.
        """
        trades = _make_trades(40)
        with _temp_journal(trades) as journal:
            stats = journal.get_stats()
            self.assertEqual(stats.total_trades, 40)

            rpt = PerformanceReport(journal.db_path)
            report = rpt.generate_full_report()
            rpt.close()
            self.assertIn("overview", report)
            self.assertNotIn("error", report)

        entries = trades
        curve = _build_equity_curve(entries, 100_000)
        analyzer = EquityCurveAnalyzer(hp_lambda=100.0, bk_K=4)
        full = analyzer.full_analysis(curve, n_boot=50)
        self.assertIn("bootstrap_sharpe", full)

        attr = AttributionReport(entries)
        attr_result = attr.compute_layer_attribution()
        self.assertGreater(len(attr_result.layers), 0)

    def test_journal_context_manager(self):
        with _temp_journal() as j:
            j.add_entry(_random_entry("BTC"))
            self.assertEqual(j.count(), 1)
        # After exit, connection closed -- should raise
        with self.assertRaises(Exception):
            j.count()

    def test_weekly_report_empty_then_populated(self):
        with _temp_journal() as j:
            reporter = WeeklyReport(j)
            week = date(2024, 6, 3)
            empty = reporter.generate(week)
            self.assertEqual(empty.total_trades, 0)

            for i in range(5):
                e = _random_entry("ETH", ts_offset_hours=i * 8)
                e.entry_ts = f"2024-06-0{3+i}T10:00:00"
                e.exit_ts = f"2024-06-0{3+i}T14:00:00"
                j.add_entry(e)

            populated = reporter.generate(week)
            self.assertGreater(populated.total_trades, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
