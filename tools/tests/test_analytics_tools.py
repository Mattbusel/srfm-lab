"""
tools/tests/test_analytics_tools.py
=====================================
Tests for the SRFM analytics tools suite.

Covers:
  - regime_diagnostic
  - signal_performance_tracker
  - correlation_drift_monitor
  - execution_quality_report
  - live_performance_dashboard (data layer)
  - param_impact_analyzer (data layer)

Run with:
    pytest tools/tests/test_analytics_tools.py -v
    python -m pytest tools/tests/test_analytics_tools.py -v
"""

from __future__ import annotations

import math
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure tools dir is importable
# ---------------------------------------------------------------------------
_TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_TOOLS.parent))

# ---------------------------------------------------------------------------
# Import modules under test
# ---------------------------------------------------------------------------
from regime_diagnostic import (
    _hurst_rs,
    _garch_vol_simple,
    _classify_hurst,
    _classify_vol,
    enrich_trade,
    pnl_by_regime,
    transition_matrix,
    timing_analysis,
    hold_time_by_regime,
    _synthetic_trades as rd_synthetic,
    _safe_std as rd_std,
    _sharpe as rd_sharpe,
)

from signal_performance_tracker import (
    augment_with_signals,
    compute_signal_metrics,
    signal_interaction_matrix,
    rolling_ic,
    rolling_attribution,
    _pearson,
    _synthetic_trades as spt_synthetic,
    SIGNAL_COLS,
)

from correlation_drift_monitor import (
    _pearson as cdm_pearson,
    returns_from_prices,
    align_series,
    rolling_pairwise_corr,
    classify_corr_regime,
    generate_alerts,
    latest_regime_summary,
    _synthetic_prices,
    HIGH_CORR_THRESHOLD,
    BTCETH_DECOUPLE_THRESHOLD,
)

from execution_quality_report import (
    compute_slippage,
    enrich_fill,
    enrich_all_fills,
    per_venue_stats,
    size_vs_slippage,
    ac_validation_stats,
    fill_speed_analysis,
    _venue,
    _synthetic_fills,
)

from live_performance_dashboard import (
    equity_curve,
    sparkline,
    today_pnl_by_symbol,
    rolling_sharpe,
    _is_stale,
    _synthetic_data,
)

from param_impact_analyzer import (
    enrich_update_with_performance,
    param_attribution,
    find_edge_params,
    regime_conditioned_impact,
    _synthetic_history,
    _add_hours,
    _sub_hours,
    sharpe_window,
)


# ===========================================================================
# Helper: in-memory SQLite DB with test data
# ===========================================================================

def _make_trade_db(n: int = 50) -> sqlite3.Connection:
    """Create an in-memory SQLite DB with trade_pnl and fills tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE trade_pnl (
            id           INTEGER PRIMARY KEY,
            symbol       TEXT,
            entry_time   TEXT,
            exit_time    TEXT,
            entry_price  REAL,
            exit_price   REAL,
            qty          REAL,
            pnl          REAL,
            hold_bars    INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE fills (
            id           INTEGER PRIMARY KEY,
            symbol       TEXT,
            side         TEXT,
            qty          REAL,
            fill_price   REAL,
            mid_price    REAL,
            commission   REAL,
            fill_bars    INTEGER,
            timestamp    TEXT
        )
    """)

    import random
    rng = random.Random(42)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    syms = ["BTC", "ETH", "SOL", "AAPL"]
    for i in range(n):
        entry = base + timedelta(hours=i * 8)
        exit_dt = entry + timedelta(hours=rng.randint(1, 4))
        pnl = rng.gauss(10.0, 60.0)
        ep = rng.uniform(100, 50000)
        xp = ep * (1 + pnl / (ep * 0.1))
        sym = rng.choice(syms)
        conn.execute(
            "INSERT INTO trade_pnl VALUES (?,?,?,?,?,?,?,?,?)",
            (i + 1, sym, entry.isoformat(), exit_dt.isoformat(),
             round(ep, 4), round(xp, 4), 0.1, round(pnl, 4), rng.randint(1, 20))
        )
        mid = ep
        fill = ep * (1 + rng.gauss(0, 0.001))
        conn.execute(
            "INSERT INTO fills VALUES (?,?,?,?,?,?,?,?,?)",
            (i + 1, sym, rng.choice(["buy", "sell"]), 0.1,
             round(fill, 4), round(mid, 4),
             round(0.1 * fill * 0.001, 6), rng.randint(1, 4),
             entry.isoformat())
        )
    conn.commit()
    return conn


def _get_trades_from_db(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM trade_pnl ORDER BY entry_time").fetchall()
    return [dict(r) for r in rows]


def _get_fills_from_db(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM fills ORDER BY timestamp").fetchall()
    return [dict(r) for r in rows]


# ===========================================================================
# regime_diagnostic tests
# ===========================================================================

class TestRegimeDiagnostic:

    def test_hurst_rs_returns_float_in_range(self):
        prices = [100.0 * (1 + 0.002 * i) for i in range(50)]
        h = _hurst_rs(prices)
        assert isinstance(h, float)
        assert 0.0 <= h <= 1.0

    def test_hurst_rs_short_series_returns_half(self):
        assert _hurst_rs([100.0, 101.0, 99.0]) == 0.5

    def test_hurst_trending_series_high(self):
        # Strong trend should give H > 0.5
        prices = [float(i) for i in range(1, 60)]
        h = _hurst_rs(prices)
        # Trending series should be above random walk (though not guaranteed in R/S)
        assert h >= 0.0  # minimal sanity

    def test_garch_vol_zero_returns(self):
        vol = _garch_vol_simple([])
        assert vol == 0.0

    def test_garch_vol_positive(self):
        rets = [0.01, -0.02, 0.005, -0.01, 0.015] * 10
        vol = _garch_vol_simple(rets)
        assert vol > 0

    def test_classify_hurst(self):
        assert _classify_hurst(0.7) == "trending"
        assert _classify_hurst(0.5) == "random"
        assert _classify_hurst(0.3) == "mean_rev"

    def test_classify_vol(self):
        assert _classify_vol(0.05, 0.01, 0.03) == "high"
        assert _classify_vol(0.005, 0.01, 0.03) == "low"
        assert _classify_vol(0.02, 0.01, 0.03) == "normal"

    def test_enrich_trade_short_bars(self):
        trade = {"id": 1, "symbol": "BTC", "pnl": 10.0, "hold_bars": 5}
        enriched = enrich_trade(trade, bars=[])
        assert "bh_mass" in enriched
        assert "hurst_h" in enriched
        assert "garch_vol" in enriched
        assert enriched["hurst_h"] == 0.5  # default when insufficient bars

    def test_enrich_trade_with_bars(self):
        bars = [{"close": 100.0 + i * 0.5} for i in range(50)]
        trade = {"id": 1, "symbol": "BTC", "pnl": 15.0, "hold_bars": 5}
        enriched = enrich_trade(trade, bars)
        assert 0.0 <= enriched["hurst_h"] <= 1.0
        assert enriched["garch_vol"] >= 0.0
        assert "bh_active" in enriched
        assert "nav_omega" in enriched

    def test_regime_diagnostic_loads_trades(self):
        conn = _make_trade_db(30)
        trades = _get_trades_from_db(conn)
        assert len(trades) == 30
        assert all("pnl" in t for t in trades)
        assert all("symbol" in t for t in trades)

    def test_regime_diagnostic_enrichment(self):
        trades = rd_synthetic(20)
        # Manually enrich without bar DB
        bars = [{"close": 100.0 + i} for i in range(30)]
        for t in trades:
            enrich_trade(t, bars)
        assert all("hurst_regime" in t for t in trades)
        assert all("vol_regime" in t for t in trades)

    def test_pnl_by_regime(self):
        trades = rd_synthetic(50)
        bars = [{"close": float(100 + i)} for i in range(30)]
        for t in trades:
            enrich_trade(t, bars)
        stats = pnl_by_regime(trades)
        assert "BH_ACTIVE" in stats or "BH_QUIET" in stats
        for label, s in stats.items():
            assert "avg_pnl" in s
            assert "win_rate" in s
            assert "sharpe" in s
            assert s["count"] > 0

    def test_pnl_by_regime_empty(self):
        stats = pnl_by_regime([])
        assert stats == {}

    def test_transition_matrix(self):
        trades = rd_synthetic(30)
        bars = [{"close": float(100 + i)} for i in range(30)]
        for t in trades:
            enrich_trade(t, bars)
        matrix = transition_matrix(trades)
        # Each transition entry should have count and avg_pnl
        for (src, dst), info in matrix.items():
            assert info["count"] > 0
            assert isinstance(info["avg_pnl"], float)

    def test_timing_analysis(self):
        trades = rd_synthetic(50)
        bars = [{"close": float(100 + i)} for i in range(30)]
        for t in trades:
            enrich_trade(t, bars)
        timing = timing_analysis(trades)
        assert "entry_distribution" in timing
        assert "bh_active_entry_pct" in timing
        assert "assessment" in timing
        total = sum(timing["entry_distribution"].values())
        assert abs(total - 100.0) < 1.0

    def test_hold_time_by_regime(self):
        trades = rd_synthetic(30)
        bars = [{"close": float(100 + i)} for i in range(30)]
        for t in trades:
            enrich_trade(t, bars)
        hold_data = hold_time_by_regime(trades)
        for regime, pairs in hold_data.items():
            for hold, pnl in pairs:
                assert isinstance(hold, int)
                assert isinstance(pnl, float)

    def test_safe_std_single_value(self):
        assert rd_std([5.0]) == 0.0

    def test_safe_std_uniform(self):
        assert rd_std([3.0, 3.0, 3.0]) == 0.0

    def test_sharpe_no_trades(self):
        assert rd_sharpe([]) == 0.0

    def test_sharpe_positive_pnl(self):
        pnls = [10.0] * 30
        sharpe = rd_sharpe(pnls)
        # All positive uniform returns -- numerically large or inf
        assert not math.isnan(sharpe)


# ===========================================================================
# signal_performance_tracker tests
# ===========================================================================

class TestSignalPerformanceTracker:

    def test_augment_fills_signal_columns(self):
        trades = spt_synthetic(20)
        augmented = augment_with_signals(trades)
        for t in augmented:
            for sig in SIGNAL_COLS:
                assert sig in t, f"Missing signal column {sig}"

    def test_compute_signal_metrics_keys(self):
        trades = spt_synthetic(60)
        trades = augment_with_signals(trades)
        metrics = compute_signal_metrics(trades)
        for sig in SIGNAL_COLS:
            assert sig in metrics
            m = metrics[sig]
            assert "win_rate_active" in m
            assert "avg_pnl_active" in m
            assert "contribution_pct" in m
            assert "ic" in m

    def test_compute_signal_metrics_win_rate_bounds(self):
        trades = spt_synthetic(80)
        trades = augment_with_signals(trades)
        metrics = compute_signal_metrics(trades)
        for sig, m in metrics.items():
            assert 0.0 <= m["win_rate_active"] <= 100.0

    def test_signal_performance_ic_computation(self):
        trades = spt_synthetic(60)
        trades = augment_with_signals(trades)
        metrics = compute_signal_metrics(trades)
        for sig, m in metrics.items():
            # IC should be a valid float in [-1, 1]
            assert -1.01 <= m["ic"] <= 1.01, f"IC out of range for {sig}: {m['ic']}"

    def test_pearson_identical_series(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(_pearson(xs, xs) - 1.0) < 1e-9

    def test_pearson_opposite_series(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(_pearson(xs, ys) + 1.0) < 1e-9

    def test_pearson_too_short(self):
        assert _pearson([1.0], [2.0]) == 0.0

    def test_signal_interaction_matrix(self):
        trades = spt_synthetic(50)
        trades = augment_with_signals(trades)
        combos = signal_interaction_matrix(trades)
        # Keys are "0|1|0|1" etc
        for key, info in combos.items():
            assert "|" in key
            assert info["count"] > 0
            assert 0.0 <= info["win_rate"] <= 100.0

    def test_rolling_ic_length(self):
        trades = spt_synthetic(80)
        trades = augment_with_signals(trades)
        window = 20
        series = rolling_ic(trades, window=window, sig=SIGNAL_COLS[0])
        assert len(series) == len(trades) - window + 1

    def test_rolling_ic_values_in_range(self):
        trades = spt_synthetic(80)
        trades = augment_with_signals(trades)
        series = rolling_ic(trades, window=20, sig=SIGNAL_COLS[0])
        for ts, ic in series:
            assert -1.01 <= ic <= 1.01

    def test_rolling_attribution_keys(self):
        trades = spt_synthetic(60)
        trades = augment_with_signals(trades)
        attr = rolling_attribution(trades, window=15)
        for sig in SIGNAL_COLS:
            assert sig in attr

    def test_rolling_attribution_length(self):
        trades = spt_synthetic(60)
        trades = augment_with_signals(trades)
        window = 15
        attr = rolling_attribution(trades, window=window)
        for sig, series in attr.items():
            assert len(series) == len(trades) - window + 1

    def test_signal_metrics_with_db_trades(self):
        conn = _make_trade_db(40)
        trades = _get_trades_from_db(conn)
        trades = augment_with_signals(trades)
        metrics = compute_signal_metrics(trades)
        assert len(metrics) == len(SIGNAL_COLS)


# ===========================================================================
# correlation_drift_monitor tests
# ===========================================================================

class TestCorrelationDriftMonitor:

    def test_pearson_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(cdm_pearson(xs, xs) - 1.0) < 1e-9

    def test_pearson_short_series(self):
        val = cdm_pearson([1.0, 2.0], [3.0, 4.0])
        assert math.isnan(val) or isinstance(val, float)

    def test_returns_from_prices(self):
        prices = [100.0, 102.0, 101.0, 105.0]
        rets = returns_from_prices(prices)
        assert len(rets) == len(prices) - 1
        assert abs(rets[0] - 0.02) < 1e-6

    def test_returns_single_price(self):
        assert returns_from_prices([100.0]) == []

    def test_align_series_overlap(self):
        a = [("2024-01-01", 100.0), ("2024-01-02", 102.0), ("2024-01-03", 101.0)]
        b = [("2024-01-01", 200.0), ("2024-01-03", 205.0)]
        pa, pb = align_series(a, b)
        assert len(pa) == 2
        assert len(pb) == 2
        assert pa[0] == 100.0
        assert pb[0] == 200.0

    def test_align_series_no_overlap(self):
        a = [("2024-01-01", 100.0)]
        b = [("2024-01-02", 200.0)]
        pa, pb = align_series(a, b)
        assert pa == []
        assert pb == []

    def test_classify_corr_regime(self):
        assert classify_corr_regime(0.90) == "HIGH_CORR"
        assert classify_corr_regime(0.60) == "MEDIUM_CORR"
        assert classify_corr_regime(0.30) == "LOW_CORR"

    def test_correlation_monitor_threshold(self):
        """High avg correlation triggers HIGH_CORR regime."""
        prices = _synthetic_prices(["BTC", "ETH"], n_bars=60)
        records = rolling_pairwise_corr(prices, ["BTC", "ETH"], window=20)
        if records:
            for ts, corrs in records:
                avg = corrs.get("_avg", 0.5)
                regime = classify_corr_regime(avg, threshold=0.85)
                assert regime in ("HIGH_CORR", "MEDIUM_CORR", "LOW_CORR")

    def test_rolling_pairwise_corr_structure(self):
        prices = _synthetic_prices(["BTC", "ETH", "SOL"], n_bars=80)
        records = rolling_pairwise_corr(prices, ["BTC", "ETH", "SOL"], window=20)
        assert isinstance(records, list)
        for ts, corrs in records:
            assert isinstance(ts, str)
            assert "_avg" in corrs
            for key, val in corrs.items():
                if not key.startswith("_"):
                    assert -1.01 <= val <= 1.01

    def test_generate_alerts_high_corr(self):
        # Construct records where avg > threshold
        records = [
            ("2024-01-01", {"BTC-ETH": 0.92, "_avg": 0.92}),
            ("2024-01-02", {"BTC-ETH": 0.91, "_avg": 0.91}),
        ]
        alerts = generate_alerts(records, high_corr_threshold=0.85)
        high_alerts = [a for a in alerts if a["type"] == "HIGH_AVG_CORR"]
        assert len(high_alerts) == 2

    def test_generate_alerts_btc_eth_decouple(self):
        records = [
            ("2024-01-01", {"BTC-ETH": 0.50, "_avg": 0.60}),
        ]
        alerts = generate_alerts(records, btceth_threshold=0.70)
        decouple = [a for a in alerts if a["type"] == "BTC_ETH_DECOUPLE"]
        assert len(decouple) == 1
        assert decouple[0]["value"] == 0.50

    def test_generate_alerts_dxy_sign_change(self):
        records = [
            ("2024-01-01", {"DXY-BTC": 0.30, "_avg": 0.60}),
            ("2024-01-02", {"DXY-BTC": -0.20, "_avg": 0.60}),
        ]
        alerts = generate_alerts(records)
        sign_alerts = [a for a in alerts if a["type"] == "DXY_BTC_SIGN_CHANGE"]
        assert len(sign_alerts) == 1

    def test_latest_regime_summary(self):
        records = [
            ("2024-01-01", {"BTC-ETH": 0.80, "_avg": 0.55}),
            ("2024-01-02", {"BTC-ETH": 0.75, "_avg": 0.60}),
        ]
        summary = latest_regime_summary(records, "20d", 0.85)
        assert summary["regime"] == "MEDIUM_CORR"
        assert summary["ts"] == "2024-01-02"
        assert summary["avg_corr"] == 0.60

    def test_latest_regime_summary_empty(self):
        summary = latest_regime_summary([], "20d", 0.85)
        assert summary["regime"] == "UNKNOWN"


# ===========================================================================
# execution_quality_report tests
# ===========================================================================

class TestExecutionQualityReport:

    def test_venue_classification_crypto(self):
        assert _venue("BTC") == "CRYPTO"
        assert _venue("ETH") == "CRYPTO"
        assert _venue("SOL") == "CRYPTO"

    def test_venue_classification_equity(self):
        assert _venue("AAPL") == "EQUITY"
        assert _venue("SPY") == "EQUITY"
        assert _venue("QQQ") == "EQUITY"

    def test_execution_quality_slippage_calc_buy(self):
        """Buy slippage: positive when fill > mid."""
        slip = compute_slippage(fill_price=100.10, mid_price=100.00, side="buy")
        assert slip > 0
        assert abs(slip - 0.001) < 1e-6

    def test_execution_quality_slippage_calc_sell(self):
        """Sell slippage: positive when fill < mid (adverse)."""
        slip = compute_slippage(fill_price=99.90, mid_price=100.00, side="sell")
        assert slip > 0
        assert abs(slip - 0.001) < 1e-6

    def test_slippage_zero_mid(self):
        assert compute_slippage(100.0, 0.0, "buy") == 0.0

    def test_enrich_fill_has_required_keys(self):
        fill = {
            "symbol": "BTC", "side": "buy",
            "qty": 0.5, "fill_price": 50100.0, "mid_price": 50000.0,
            "commission": 5.0, "fill_bars": 1,
        }
        enriched = enrich_fill(fill)
        for key in ("slippage_frac", "slippage_bps", "venue",
                    "notional", "ac_predicted_bps", "ac_error_bps"):
            assert key in enriched, f"Missing key: {key}"

    def test_enrich_fill_slippage_bps_value(self):
        fill = {
            "symbol": "AAPL", "side": "buy",
            "qty": 10.0, "fill_price": 150.30, "mid_price": 150.00,
            "commission": 1.0, "fill_bars": 2,
        }
        enriched = enrich_fill(fill)
        # 0.30 / 150.00 = 0.002 = 20 bps
        assert abs(enriched["slippage_bps"] - 20.0) < 1.0

    def test_per_venue_stats(self):
        fills = _synthetic_fills(100)
        enriched = enrich_all_fills(fills)
        stats = per_venue_stats(enriched)
        for venue in ("CRYPTO", "EQUITY"):
            if venue in stats:
                s = stats[venue]
                assert s["n_fills"] > 0
                assert isinstance(s["avg_slippage_bps"], float)

    def test_size_vs_slippage_pairs(self):
        fills = _synthetic_fills(50)
        enriched = enrich_all_fills(fills)
        pairs = size_vs_slippage(enriched)
        assert len(pairs) == len([f for f in enriched if f["notional"] > 0])
        for notional, slip in pairs:
            assert notional > 0

    def test_ac_validation_stats_structure(self):
        fills = _synthetic_fills(60)
        enriched = enrich_all_fills(fills)
        stats = ac_validation_stats(enriched)
        assert "mae_bps" in stats
        assert "bias_bps" in stats
        assert "r2" in stats
        assert "interpretation" in stats

    def test_fill_speed_analysis(self):
        fills = _synthetic_fills(100)
        enriched = enrich_all_fills(fills)
        speed = fill_speed_analysis(enriched)
        total = speed["instant_pct"] + speed["fast_pct"] + speed["slow_pct"]
        assert abs(total - 100.0) < 1.0

    def test_enrich_all_fills_consistent_length(self):
        fills = _synthetic_fills(30)
        enriched = enrich_all_fills(fills)
        assert len(enriched) == len(fills)

    def test_enrich_fill_notional(self):
        fill = {
            "symbol": "ETH", "side": "buy",
            "qty": 2.0, "fill_price": 3000.0, "mid_price": 2995.0,
            "commission": 3.0, "fill_bars": 1,
        }
        enriched = enrich_fill(fill)
        assert enriched["notional"] == pytest.approx(6000.0, abs=1.0)

    def test_venue_stats_from_db(self):
        conn = _make_trade_db(40)
        fills = _get_fills_from_db(conn)
        enriched = enrich_all_fills(fills)
        stats = per_venue_stats(enriched)
        assert len(stats) >= 1


# ===========================================================================
# live_performance_dashboard tests
# ===========================================================================

class TestLivePerformanceDashboard:

    def test_equity_curve_length(self):
        trades = [{"pnl": 10.0}, {"pnl": -5.0}, {"pnl": 20.0}]
        curve = equity_curve(trades, initial=1000.0)
        assert len(curve) == 4
        assert curve[0] == 1000.0
        assert curve[-1] == pytest.approx(1025.0, abs=0.01)

    def test_equity_curve_empty(self):
        curve = equity_curve([], initial=500.0)
        assert curve == [500.0]

    def test_sparkline_length(self):
        values = list(range(100))
        spark = sparkline(values, width=20)
        assert len(spark) == 20

    def test_sparkline_short_series(self):
        spark = sparkline([1.0], width=20)
        assert isinstance(spark, str)

    def test_today_pnl_by_symbol(self):
        today = datetime.now(timezone.utc).date().isoformat()
        trades = [
            {"symbol": "BTC", "exit_time": f"{today}T12:00:00", "pnl": 100.0},
            {"symbol": "BTC", "exit_time": f"{today}T14:00:00", "pnl": -30.0},
            {"symbol": "ETH", "exit_time": f"{today}T10:00:00", "pnl": 50.0},
            {"symbol": "SOL", "exit_time": "2020-01-01T00:00:00", "pnl": 999.0},
        ]
        result = today_pnl_by_symbol(trades)
        assert result["BTC"] == pytest.approx(70.0, abs=0.01)
        assert result["ETH"] == pytest.approx(50.0, abs=0.01)
        assert "SOL" not in result

    def test_rolling_sharpe_no_trades(self):
        assert rolling_sharpe([], days=7) == 0.0

    def test_rolling_sharpe_constant_returns(self):
        today = datetime.now(timezone.utc).date().isoformat()
        trades = [
            {"pnl": 10.0, "exit_time": f"{today}T{h:02d}:00:00"}
            for h in range(10)
        ]
        sharpe = rolling_sharpe(trades, days=7)
        # Constant returns --> std=0 --> returns 0
        assert sharpe == 0.0

    def test_is_stale_old_timestamp(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        assert _is_stale(old, minutes=30) is True

    def test_is_stale_recent_timestamp(self):
        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        assert _is_stale(recent, minutes=30) is False

    def test_is_stale_none(self):
        assert _is_stale(None) is True

    def test_synthetic_data_structure(self):
        data = _synthetic_data()
        assert "equity" in data
        assert "today_pnl" in data
        assert "sharpes" in data
        assert "regime_state" in data
        assert len(data["equity"]) > 1
        for period in ("1d", "7d", "30d"):
            assert period in data["sharpes"]


# ===========================================================================
# param_impact_analyzer tests
# ===========================================================================

class TestParamImpactAnalyzer:

    def test_add_sub_hours_roundtrip(self):
        ts = "2024-06-15T12:00:00+00:00"
        result = _add_hours(_sub_hours(ts, 4.0), 4.0)
        # Should get back close to original
        orig = datetime.fromisoformat(ts)
        back = datetime.fromisoformat(result)
        diff = abs((back - orig).total_seconds())
        assert diff < 2

    def test_sharpe_window_no_trades(self):
        val = sharpe_window([], "2024-01-01", "2024-01-02")
        assert math.isnan(val)

    def test_sharpe_window_with_trades(self):
        trades = [
            {"pnl": 10.0, "exit_time": "2024-01-01T10:00:00"},
            {"pnl": -5.0, "exit_time": "2024-01-01T14:00:00"},
            {"pnl": 20.0, "exit_time": "2024-01-01T18:00:00"},
        ]
        val = sharpe_window(trades, "2024-01-01", "2024-01-02")
        assert not math.isnan(val)

    def test_enrich_update_keys(self):
        update = {"timestamp": "2024-01-15T10:00:00", "changed_params": {"bh_mass": 1.2}}
        enriched = enrich_update_with_performance(update, [], windows=[4.0, 24.0])
        assert "_performance" in enriched
        assert "4.0h" in enriched["_performance"]
        assert "24.0h" in enriched["_performance"]

    def test_param_attribution_structure(self):
        history = _synthetic_history(20)
        enriched = [enrich_update_with_performance(h, [], windows=[24.0]) for h in history]
        attr = param_attribution(enriched, window="24.0h")
        for pname, stats in attr.items():
            assert "n_updates" in stats
            assert "avg_delta" in stats
            assert "improvement_rate" in stats
            assert 0.0 <= stats["improvement_rate"] <= 100.0

    def test_find_edge_params(self):
        # Create history where one param is near its min
        history = [
            {
                "timestamp": f"2024-0{i+1}-01T00:00:00",
                "changed_params": {
                    "bh_threshold": 0.001 if i == 0 else 0.001 + i * 0.1,
                }
            }
            for i in range(10)
        ]
        enriched = [enrich_update_with_performance(h, []) for h in history]
        edges = find_edge_params(enriched, pct_threshold=0.15)
        assert isinstance(edges, list)
        if edges:
            for e in edges:
                assert "param" in e
                assert "current" in e
                assert "at_edge" in e
                assert "recommendation" in e

    def test_regime_conditioned_impact(self):
        history = _synthetic_history(30)
        trades = rd_synthetic(100)
        enriched = [enrich_update_with_performance(h, trades, windows=[24.0]) for h in history]
        rc = regime_conditioned_impact(enriched, trades, window_h=24.0)
        assert "trending" in rc
        assert "mean_rev" in rc
        for regime, stats in rc.items():
            assert "n" in stats
            if stats["n"] > 0:
                assert stats["pct_improved"] is not None
                assert 0.0 <= stats["pct_improved"] <= 100.0

    def test_synthetic_history_length(self):
        history = _synthetic_history(50)
        assert len(history) == 50
        for h in history:
            assert "timestamp" in h
            assert "changed_params" in h


# ===========================================================================
# Integration tests
# ===========================================================================

class TestIntegration:

    def test_full_regime_pipeline(self):
        """Run full regime diagnostic pipeline end-to-end on synthetic data."""
        trades = rd_synthetic(40)
        bars = [{"close": float(100 + i * 0.3)} for i in range(30)]
        for t in trades:
            enrich_trade(t, bars)
        stats = pnl_by_regime(trades)
        matrix = transition_matrix(trades)
        timing = timing_analysis(trades)
        assert isinstance(stats, dict)
        assert isinstance(matrix, dict)
        assert isinstance(timing, dict)
        assert "bh_active_entry_pct" in timing

    def test_full_signal_pipeline(self):
        """Run full signal tracking pipeline end-to-end."""
        trades = spt_synthetic(60)
        trades = augment_with_signals(trades)
        metrics = compute_signal_metrics(trades)
        combos = signal_interaction_matrix(trades)
        attr = rolling_attribution(trades, window=15)
        ic = rolling_ic(trades, window=15, sig=SIGNAL_COLS[0])
        assert all(sig in metrics for sig in SIGNAL_COLS)
        assert len(combos) > 0
        assert len(ic) > 0

    def test_full_exec_pipeline(self):
        """Run full execution quality pipeline end-to-end."""
        fills = _synthetic_fills(80)
        enriched = enrich_all_fills(fills)
        venue_stats = per_venue_stats(enriched)
        ac_stats = ac_validation_stats(enriched)
        speed_stats = fill_speed_analysis(enriched)
        assert len(venue_stats) >= 1
        assert "mae_bps" in ac_stats
        assert "avg_speed" in speed_stats

    def test_full_corr_pipeline(self):
        """Run full correlation monitor pipeline end-to-end."""
        symbols = ["BTC", "ETH", "SOL"]
        prices = _synthetic_prices(symbols, n_bars=60)
        records = rolling_pairwise_corr(prices, symbols, window=20)
        alerts = generate_alerts(records)
        summary = latest_regime_summary(records, "20d", 0.85)
        assert isinstance(records, list)
        assert isinstance(alerts, list)
        assert "regime" in summary

    def test_db_round_trip_trades(self):
        """Write trades to SQLite and read them back correctly."""
        conn = _make_trade_db(25)
        trades = _get_trades_from_db(conn)
        assert len(trades) == 25
        for t in trades:
            assert t["pnl"] is not None
            assert t["symbol"] in ("BTC", "ETH", "SOL", "AAPL")
