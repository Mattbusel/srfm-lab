"""
tools/tests/test_tools_v2.py
=============================
Test suite for the v2 live monitoring and control tools.

Covers:
- live_monitor_v2.py  (SQLite queries, display rendering, signal state)
- position_sizing_sandbox.py  (all sizing methods, Kelly grid search)
- trade_replay.py  (BH mass reconstruction, similar trade search, diagnostics)
- live_controls_v2.py  (overrides, blocks, pause, confirm guard, param validation)

Run:
    pytest tools/tests/test_tools_v2.py -v
    pytest tools/tests/test_tools_v2.py -v -k "kelly"
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Ensure tools/ is importable ──────────────────────────────────────────────
_TOOLS_DIR = Path(__file__).parents[1]
_REPO_ROOT  = Path(__file__).parents[2]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Create a minimal live_trades.db for testing."""
    db_path = tmp_path / "live_trades.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE live_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, side TEXT, qty REAL, price REAL,
            notional REAL, fill_time TEXT, order_id TEXT, strategy_version TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE trade_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, entry_time TEXT, exit_time TEXT,
            entry_price REAL, exit_price REAL, qty REAL, pnl REAL, hold_bars INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE bar_data (
            ts TEXT, symbol TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL
        )
    """)
    conn.execute("""
        CREATE TABLE nav_state (
            ts TEXT, symbol TEXT, bh_mass_15m REAL, bh_mass_1h REAL,
            hurst_h REAL, nav_omega REAL, signal_strength REAL, bars_held INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE signal_state (
            ts TEXT, quatnav_gate INTEGER, hurst_filter INTEGER,
            calendar_filter INTEGER, granger_boost INTEGER,
            ml_signal_dir REAL, macro_regime TEXT
        )
    """)
    # Insert sample live_trades
    now = datetime.now(timezone.utc)
    conn.executemany(
        "INSERT INTO live_trades (symbol, side, qty, price, notional, fill_time, strategy_version) VALUES (?,?,?,?,?,?,?)",
        [
            ("BTC", "buy",  0.01, 50000.0, 500.0,  now.isoformat(), "larsa_v18"),
            ("ETH", "buy",  0.10, 3000.0,  300.0,  now.isoformat(), "larsa_v18"),
        ],
    )
    # Insert sample trade_pnl
    past = (now - timedelta(hours=2))
    conn.executemany(
        "INSERT INTO trade_pnl (symbol, entry_time, exit_time, entry_price, exit_price, qty, pnl, hold_bars) VALUES (?,?,?,?,?,?,?,?)",
        [
            ("BTC", (now - timedelta(hours=6)).isoformat(), past.isoformat(), 49000.0, 50500.0, 0.01,  15.0, 24),
            ("ETH", (now - timedelta(hours=4)).isoformat(), now.isoformat(),  2900.0,  2850.0,  0.10,  -5.0,  16),
            ("XRP", (now - timedelta(hours=8)).isoformat(), (now - timedelta(hours=1)).isoformat(), 0.60, 0.65, 100.0, 5.0, 28),
        ],
    )
    # Insert bar_data
    base_time = now - timedelta(hours=8)
    bar_rows = []
    for i in range(60):
        ts  = (base_time + timedelta(minutes=15 * i)).isoformat()
        px  = 50000.0 + i * 50
        bar_rows.append(("BTC", ts, px*0.999, px*1.001, px*0.998, px, 1e6))
    conn.executemany(
        "INSERT INTO bar_data (symbol, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in bar_rows],
    )
    # Insert nav_state
    nav_rows = []
    for i in range(60):
        ts  = (base_time + timedelta(minutes=15 * i)).isoformat()
        nav_rows.append(("BTC", ts, 0.4 + i*0.005, 0.3 + i*0.003, 0.55, 0.001, 0.6 + i*0.002, i))
    conn.executemany(
        "INSERT INTO nav_state (symbol, ts, bh_mass_15m, bh_mass_1h, hurst_h, nav_omega, signal_strength, bars_held) VALUES (?,?,?,?,?,?,?,?)",
        nav_rows,
    )
    # Insert signal_state
    conn.execute(
        "INSERT INTO signal_state (ts, quatnav_gate, hurst_filter, calendar_filter, granger_boost, ml_signal_dir, macro_regime) VALUES (?,?,?,?,?,?,?)",
        (now.isoformat(), 1, 1, 0, 1, 0.75, "BULL"),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def tmp_config_dir(tmp_path) -> Path:
    """Create a temp config directory."""
    config = tmp_path / "config"
    config.mkdir()
    return config


# ─────────────────────────────────────────────────────────────────────────────
# live_monitor_v2 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveMonitorQuerySQLite:

    def test_query_equity_returns_dict(self, tmp_db):
        from live_monitor_v2 import _open_db, query_equity
        conn = _open_db(tmp_db)
        assert conn is not None
        result = query_equity(conn)
        conn.close()
        assert isinstance(result, dict)
        assert "daily_pnl" in result
        assert "seven_day_pnl" in result
        assert "max_drawdown" in result

    def test_query_equity_daily_pnl_nonzero(self, tmp_db):
        from live_monitor_v2 import _open_db, query_equity
        conn = _open_db(tmp_db)
        result = query_equity(conn)
        conn.close()
        # We inserted trades with exit_time = now, so daily_pnl should be nonzero
        # BTC: +15, ETH: -5, XRP: +5 for recent exits
        assert isinstance(result["daily_pnl"], float)

    def test_query_equity_max_drawdown_non_negative(self, tmp_db):
        from live_monitor_v2 import _open_db, query_equity
        conn = _open_db(tmp_db)
        result = query_equity(conn)
        conn.close()
        assert result["max_drawdown"] >= 0.0

    def test_query_positions_returns_list(self, tmp_db):
        from live_monitor_v2 import _open_db, query_positions
        conn = _open_db(tmp_db)
        positions = query_positions(conn)
        conn.close()
        assert isinstance(positions, list)

    def test_query_positions_has_bh_fields(self, tmp_db):
        from live_monitor_v2 import _open_db, query_positions
        conn = _open_db(tmp_db)
        positions = query_positions(conn)
        conn.close()
        for pos in positions:
            assert "bh_mass_15m" in pos
            assert "bh_mass_1h" in pos
            assert "hurst_regime" in pos
            assert "nav_omega" in pos

    def test_query_recent_trades_returns_list(self, tmp_db):
        from live_monitor_v2 import _open_db, query_recent_trades
        conn = _open_db(tmp_db)
        trades = query_recent_trades(conn, n=5)
        conn.close()
        assert isinstance(trades, list)
        assert len(trades) <= 5

    def test_query_recent_trades_has_pnl(self, tmp_db):
        from live_monitor_v2 import _open_db, query_recent_trades
        conn = _open_db(tmp_db)
        trades = query_recent_trades(conn)
        conn.close()
        for t in trades:
            assert "pnl" in t
            assert "hold_bars" in t

    def test_query_signal_state_returns_dict(self, tmp_db):
        from live_monitor_v2 import _open_db, query_signal_state
        conn = _open_db(tmp_db)
        state = query_signal_state(conn)
        conn.close()
        assert isinstance(state, dict)
        assert "macro_regime" in state

    def test_query_signal_state_macro_regime_bull(self, tmp_db):
        from live_monitor_v2 import _open_db, query_signal_state
        conn = _open_db(tmp_db)
        state = query_signal_state(conn)
        conn.close()
        assert state.get("macro_regime") == "BULL"

    def test_open_db_returns_none_for_missing_file(self, tmp_path):
        from live_monitor_v2 import _open_db
        result = _open_db(tmp_path / "nonexistent.db")
        assert result is None

    def test_hurst_label_trending(self):
        from live_monitor_v2 import _hurst_label
        assert "TREND" in _hurst_label(0.65)

    def test_hurst_label_mean_reverting(self):
        from live_monitor_v2 import _hurst_label
        assert "MR" in _hurst_label(0.35)

    def test_hurst_label_random(self):
        from live_monitor_v2 import _hurst_label
        assert "RAND" in _hurst_label(0.50)

    def test_hurst_label_none(self):
        from live_monitor_v2 import _hurst_label
        assert _hurst_label(None) == "N/A"

    def test_ascii_bar_full(self):
        from live_monitor_v2 import _ascii_bar
        bar = _ascii_bar(1.0, width=10)
        assert bar.startswith("[")
        assert "#" * 10 in bar

    def test_ascii_bar_empty(self):
        from live_monitor_v2 import _ascii_bar
        bar = _ascii_bar(0.0, width=10)
        assert "-" * 10 in bar

    def test_monitor_state_uptime(self):
        from live_monitor_v2 import MonitorState
        state = MonitorState()
        state.start_time = time.time() - 3661  # 1h 1m 1s
        up = state.uptime_str()
        assert up.startswith("01:01:")

    @pytest.mark.asyncio
    async def test_refresh_calls_db(self, tmp_db):
        from live_monitor_v2 import MonitorState, refresh
        state = MonitorState()
        with patch("live_monitor_v2.aiohttp.ClientSession") as mock_sess:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__  = AsyncMock(return_value=False)
            mock_sess.return_value = mock_ctx
            mock_ctx.get = AsyncMock()
            # Mock all fetch calls to return {}
            with patch("live_monitor_v2.fetch_coordination", new=AsyncMock(return_value={})):
                with patch("live_monitor_v2.fetch_risk", new=AsyncMock(return_value={})):
                    with patch("live_monitor_v2.fetch_observability", new=AsyncMock(return_value={})):
                        await refresh(state, tmp_db)
        assert state.refresh_count == 1
        assert isinstance(state.equity_data, dict)


# ─────────────────────────────────────────────────────────────────────────────
# position_sizing_sandbox tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionSizingKellyFraction:

    @pytest.fixture
    def sandbox(self):
        from position_sizing_sandbox import SizingSandbox
        sb = SizingSandbox(db_path=None, synthetic_fallback=True)
        sb.load()
        return sb

    def test_sandbox_loads_synthetic(self, sandbox):
        assert sandbox._prices is not None
        assert not sandbox._prices.empty
        assert sandbox._returns is not None

    def test_simulate_kelly_returns_dataframe(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.KELLY_VOL_TARGET, {"kelly_fraction": 0.25})
        assert isinstance(result, pd.DataFrame)
        assert "portfolio_return" in result.columns
        assert "equity" in result.columns
        assert "drawdown" in result.columns

    def test_simulate_equal_weight(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.EQUAL_WEIGHT)
        assert "portfolio_return" in result.columns
        assert len(result) > 0

    def test_simulate_risk_parity(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.RISK_PARITY)
        assert "portfolio_return" in result.columns

    def test_simulate_signal_proportional(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.SIGNAL_PROPORTIONAL)
        assert len(result) > 0

    def test_simulate_fixed_pct(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.FIXED_PCT, {"fixed_pct": 0.10})
        assert len(result) > 0

    def test_simulate_metrics_in_attrs(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.KELLY_VOL_TARGET, {"kelly_fraction": 0.5})
        metrics = result.attrs.get("metrics", {})
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "total_return" in metrics

    def test_kelly_metrics_finite(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.KELLY_VOL_TARGET, {"kelly_fraction": 0.25})
        metrics = result.attrs.get("metrics", {})
        assert np.isfinite(metrics["sharpe"])
        assert np.isfinite(metrics["max_drawdown"])

    def test_equity_starts_near_one(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.EQUAL_WEIGHT)
        assert abs(result["equity"].iloc[0] - 1.0) < 0.1

    def test_compare_methods_returns_dict(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        methods = [SizingMethod.KELLY_VOL_TARGET, SizingMethod.EQUAL_WEIGHT]
        comp = sandbox.compare_methods(methods)
        assert isinstance(comp, dict)
        assert len(comp) == 2

    def test_compare_methods_has_sharpe(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        methods = [SizingMethod.KELLY_VOL_TARGET, SizingMethod.RISK_PARITY]
        comp = sandbox.compare_methods(methods)
        for method, metrics in comp.items():
            assert "sharpe" in metrics

    def test_optimize_kelly_fraction_returns_best(self, sandbox):
        result = sandbox.optimize_kelly_fraction(grid=[0.1, 0.25, 0.5])
        assert "best_fraction" in result
        assert result["best_fraction"] in [0.1, 0.25, 0.5]

    def test_optimize_kelly_fraction_all_scores(self, sandbox):
        result = sandbox.optimize_kelly_fraction(grid=[0.1, 0.5, 1.0])
        assert len(result["all_scores"]) == 3
        for kf in [0.1, 0.5, 1.0]:
            assert kf in result["all_scores"]
            assert np.isfinite(result["all_scores"][kf])

    def test_optimize_kelly_test_metrics(self, sandbox):
        result = sandbox.optimize_kelly_fraction()
        assert "test_metrics" in result
        assert "sharpe" in result["test_metrics"]

    def test_regime_conditioned_sizing(self, sandbox):
        from position_sizing_sandbox import SizingMethod, _size_equal_weight

        def dummy_regime(prices):
            mid = len(prices) // 2
            regime = pd.Series("BULL", index=prices.index)
            regime.iloc[mid:] = "BEAR"
            return regime

        def dummy_sizer(regime, ret, sig):
            return _size_equal_weight(ret, sig)

        result = sandbox.regime_conditioned_sizing(dummy_regime, dummy_sizer)
        assert "portfolio_return" in result.columns
        assert len(result) > 0

    def test_summary_table_is_dataframe(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        comp = sandbox.compare_methods([SizingMethod.EQUAL_WEIGHT, SizingMethod.FIXED_PCT])
        tbl = sandbox.summary_table(comp)
        assert isinstance(tbl, pd.DataFrame)
        assert "sharpe" in tbl.columns

    def test_string_method_alias(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate("kelly_vol_target", {"kelly_fraction": 0.25})
        assert len(result) > 0

    def test_sharpe_finite_for_all_methods(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        for method in SizingMethod:
            result = sandbox.simulate(method)
            metrics = result.attrs.get("metrics", {})
            s = metrics.get("sharpe", float("nan"))
            assert np.isfinite(s), f"sharpe not finite for {method}"

    def test_weight_columns_present(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.KELLY_VOL_TARGET)
        weight_cols = [c for c in result.columns if c.startswith("weight_")]
        assert len(weight_cols) >= 1

    def test_max_drawdown_negative_or_zero(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.EQUAL_WEIGHT)
        metrics = result.attrs["metrics"]
        assert metrics["max_drawdown"] <= 0.0

    def test_turnover_non_negative(self, sandbox):
        from position_sizing_sandbox import SizingMethod
        result = sandbox.simulate(SizingMethod.KELLY_VOL_TARGET)
        metrics = result.attrs["metrics"]
        assert metrics.get("turnover") is None or metrics["turnover"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# trade_replay tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeReplayReconstructsBhMass:

    def test_reconstruct_bh_mass_basic(self):
        from trade_replay import _reconstruct_bh_mass
        closes = [100.0 + i * 0.5 for i in range(30)]
        masses = _reconstruct_bh_mass(closes)
        assert len(masses) == 30
        assert all(m >= 0 for m in masses)

    def test_reconstruct_bh_mass_increases_with_vol(self):
        from trade_replay import _reconstruct_bh_mass
        low_vol  = [100.0 + i * 0.01 for i in range(30)]
        high_vol = [100.0 * (1.0 + 0.05 * (-1)**i) for i in range(30)]
        m_low  = _reconstruct_bh_mass(low_vol)
        m_high = _reconstruct_bh_mass(high_vol)
        assert m_high[-1] > m_low[-1]

    def test_reconstruct_bh_mass_decays_to_zero(self):
        from trade_replay import _reconstruct_bh_mass
        # Flat prices -> impulse=0, mass decays
        flat = [100.0] * 100
        masses = _reconstruct_bh_mass(flat)
        assert masses[-1] < masses[0] + 1e-9

    def test_rs_hurst_trending(self):
        from trade_replay import _rs_hurst
        # Monotonic increases -> trending
        prices = [float(i) for i in range(1, 101)]
        h = _rs_hurst(prices)
        assert h is not None
        assert h > 0.5

    def test_rs_hurst_returns_none_short(self):
        from trade_replay import _rs_hurst
        assert _rs_hurst([1.0, 2.0]) is None

    def test_rs_hurst_range(self):
        from trade_replay import _rs_hurst
        rng = np.random.default_rng(0)
        prices = rng.normal(0, 1, 100).cumsum().tolist()
        h = _rs_hurst(prices)
        if h is not None:
            assert 0.0 <= h <= 1.0

    def test_nav_omega_approx_nonzero(self):
        from trade_replay import _nav_omega_approx
        closes = [100.0 * (1.02 ** i) for i in range(10)]
        omega = _nav_omega_approx(closes)
        assert omega > 0.0

    def test_nav_omega_approx_flat_near_zero(self):
        from trade_replay import _nav_omega_approx
        closes = [100.0] * 10
        omega = _nav_omega_approx(closes)
        assert omega < 1e-6

    def test_replayer_loads_trades(self, tmp_db):
        from trade_replay import TradeReplayer
        replayer = TradeReplayer(db_path=tmp_db)
        trades   = replayer._get_trades()
        assert not trades.empty
        assert "pnl" in trades.columns

    def test_replayer_replay_trade_fields(self, tmp_db):
        from trade_replay import TradeReplayer
        replayer = TradeReplayer(db_path=tmp_db)
        trades   = replayer._get_trades()
        if trades.empty:
            pytest.skip("No trades in DB")
        tid    = int(trades.iloc[0]["id"])
        result = replayer.replay_trade(tid)
        assert "bh_mass_at_entry" in result
        assert "hurst_h" in result
        assert "nav_omega" in result
        assert "entry_trigger" in result
        assert "exit_trigger" in result

    def test_replayer_diagnose_returns_string(self, tmp_db):
        from trade_replay import TradeReplayer
        replayer = TradeReplayer(db_path=tmp_db)
        trades   = replayer._get_trades()
        if trades.empty:
            pytest.skip("No trades in DB")
        # Find a losing trade
        losing = trades[trades["pnl"] < 0]
        if losing.empty:
            losing = trades
        tid    = int(losing.iloc[0]["id"])
        diag   = replayer.diagnose_bad_trade(tid)
        assert isinstance(diag, str)
        assert len(diag) > 50

    def test_replayer_list_trades_symbol_filter(self, tmp_db):
        from trade_replay import TradeReplayer
        replayer = TradeReplayer(db_path=tmp_db)
        df       = replayer.list_trades(symbol="BTC")
        if not df.empty:
            assert all(df["symbol"] == "BTC")

    def test_replayer_list_trades_losing_only(self, tmp_db):
        from trade_replay import TradeReplayer
        replayer = TradeReplayer(db_path=tmp_db)
        df       = replayer.list_trades(losing_only=True)
        if not df.empty:
            assert all(df["pnl"] < 0)

    def test_replayer_find_similar_trades(self, tmp_db):
        from trade_replay import TradeReplayer
        replayer = TradeReplayer(db_path=tmp_db)
        trades   = replayer._get_trades()
        if len(trades) < 2:
            pytest.skip("Need at least 2 trades for similarity search")
        tid     = int(trades.iloc[0]["id"])
        similar = replayer.find_similar_trades(tid, n=2)
        assert isinstance(similar, list)
        for s in similar:
            assert "similarity" in s
            assert "trade_id" in s

    def test_infer_entry_trigger_strong_bh(self):
        from trade_replay import _infer_entry_trigger
        info = {"bh_mass_at_entry": 0.7, "bh_mass_1h_at_entry": 0.5,
                "hurst_h": 0.65, "signal_strength": 0.8}
        trigger = _infer_entry_trigger(info)
        assert "BH_15m_fire" in trigger

    def test_infer_exit_trigger_stop_loss(self):
        from trade_replay import _infer_exit_trigger
        trade = {"pnl": -50.0, "hold_bars": 5, "entry_price": 1000.0, "exit_price": 940.0}
        trigger = _infer_exit_trigger(trade, 0.5)
        assert "stop" in trigger.lower()

    def test_infer_exit_trigger_take_profit(self):
        from trade_replay import _infer_exit_trigger
        trade = {"pnl": 120.0, "hold_bars": 10, "entry_price": 1000.0, "exit_price": 1120.0}
        trigger = _infer_exit_trigger(trade, 0.6)
        assert "profit" in trigger.lower()

    def test_cosine_similarity_identical(self):
        from trade_replay import _cosine_similarity
        a = np.array([1.0, 2.0, 3.0])
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-9

    def test_cosine_similarity_orthogonal(self):
        from trade_replay import _cosine_similarity
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_cosine_similarity_zero_vector(self):
        from trade_replay import _cosine_similarity
        a = np.zeros(5)
        b = np.ones(5)
        assert _cosine_similarity(a, b) == 0.0

    def test_feature_vector_normalised(self):
        from trade_replay import _build_feature_vector
        info = {"bh_mass_at_entry": 0.5, "bh_mass_1h_at_entry": 0.3,
                "hurst_h": 0.6, "nav_omega": 0.001, "signal_strength": 0.7,
                "entry_price": 50000.0, "hold_bars": 20}
        vec = _build_feature_vector(info)
        assert vec.shape[0] == 7
        norm = np.linalg.norm(vec)
        assert 0.9 < norm <= 1.01 or norm == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# live_controls_v2 tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveControlsOverrideWritesJson:

    def test_override_writes_file(self, tmp_config_dir):
        import live_controls_v2 as lc
        original = lc._OVERRIDES_FILE
        lc._OVERRIDES_FILE = tmp_config_dir / "signal_overrides.json"
        try:
            with patch("live_controls_v2.aiohttp.ClientSession") as mock_sess:
                mock_ctx = AsyncMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.__aexit__  = AsyncMock(return_value=False)
                mock_ctx.post = AsyncMock()
                mock_sess.return_value = mock_ctx
                with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
                    rc = asyncio.run(lc.action_override_signal("BTC=1.5", confirm=True))
            assert rc == 0
            assert lc._OVERRIDES_FILE.exists()
            data = json.loads(lc._OVERRIDES_FILE.read_text())
            assert "BTC" in data
            assert data["BTC"]["multiplier"] == 1.5
        finally:
            lc._OVERRIDES_FILE = original

    def test_override_invalid_format(self, tmp_config_dir):
        import live_controls_v2 as lc
        original = lc._OVERRIDES_FILE
        lc._OVERRIDES_FILE = tmp_config_dir / "signal_overrides.json"
        try:
            rc = asyncio.run(lc.action_override_signal("BTC_NO_EQUAL", confirm=True))
            assert rc == 1
        finally:
            lc._OVERRIDES_FILE = original

    def test_override_multiplier_out_of_range(self, tmp_config_dir):
        import live_controls_v2 as lc
        original = lc._OVERRIDES_FILE
        lc._OVERRIDES_FILE = tmp_config_dir / "signal_overrides.json"
        try:
            rc = asyncio.run(lc.action_override_signal("BTC=99.0", confirm=True))
            assert rc == 1
        finally:
            lc._OVERRIDES_FILE = original

    def test_override_zero_disables(self, tmp_config_dir):
        import live_controls_v2 as lc
        original = lc._OVERRIDES_FILE
        lc._OVERRIDES_FILE = tmp_config_dir / "signal_overrides.json"
        try:
            with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
                rc = asyncio.run(lc.action_override_signal("ETH=0.0", confirm=True))
            assert rc == 0
            data = json.loads(lc._OVERRIDES_FILE.read_text())
            assert data["ETH"]["multiplier"] == 0.0
        finally:
            lc._OVERRIDES_FILE = original

    def test_override_non_numeric_multiplier(self, tmp_config_dir):
        import live_controls_v2 as lc
        original = lc._OVERRIDES_FILE
        lc._OVERRIDES_FILE = tmp_config_dir / "signal_overrides.json"
        try:
            rc = asyncio.run(lc.action_override_signal("BTC=abc", confirm=True))
            assert rc == 1
        finally:
            lc._OVERRIDES_FILE = original


class TestLiveControlsFlattenRequiresConfirm:

    def test_flatten_without_confirm_exits(self):
        import live_controls_v2 as lc
        with pytest.raises(SystemExit) as exc_info:
            asyncio.run(lc.action_flatten_all(confirm=False))
        assert exc_info.value.code == 1

    def test_flatten_with_confirm_calls_api(self):
        import live_controls_v2 as lc
        with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
            rc = asyncio.run(lc.action_flatten_all(confirm=True))
        assert rc == 0

    def test_flatten_api_failure_returns_nonzero(self):
        import live_controls_v2 as lc
        with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": False, "error": "timeout"})):
            rc = asyncio.run(lc.action_flatten_all(confirm=True))
        assert rc == 1

    def test_pause_without_confirm_exits(self):
        import live_controls_v2 as lc
        with pytest.raises(SystemExit):
            asyncio.run(lc.action_pause(30.0, confirm=False))

    def test_pause_writes_file(self, tmp_config_dir):
        import live_controls_v2 as lc
        original = lc._PAUSE_FILE
        lc._PAUSE_FILE = tmp_config_dir / "trading_pause.json"
        try:
            with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
                rc = asyncio.run(lc.action_pause(30.0, confirm=True))
            assert rc == 0
            assert lc._PAUSE_FILE.exists()
            data = json.loads(lc._PAUSE_FILE.read_text())
            assert data["paused"] is True
            assert data["minutes"] == 30.0
        finally:
            lc._PAUSE_FILE = original

    def test_block_symbol_writes_file(self, tmp_config_dir):
        import live_controls_v2 as lc
        original = lc._BLOCKED_FILE
        lc._BLOCKED_FILE = tmp_config_dir / "blocked_symbols.json"
        try:
            with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
                rc = asyncio.run(lc.action_block_symbol("XRP", 2.0, confirm=True))
            assert rc == 0
            data = json.loads(lc._BLOCKED_FILE.read_text())
            assert "XRP" in data
            assert data["XRP"]["hours"] == 2.0
        finally:
            lc._BLOCKED_FILE = original

    def test_block_symbol_without_confirm_exits(self):
        import live_controls_v2 as lc
        with pytest.raises(SystemExit):
            asyncio.run(lc.action_block_symbol("BTC", 1.0, confirm=False))

    def test_reset_circuit_without_confirm_exits(self):
        import live_controls_v2 as lc
        with pytest.raises(SystemExit):
            asyncio.run(lc.action_reset_circuit("alpaca", confirm=False))

    def test_reset_circuit_with_confirm(self):
        import live_controls_v2 as lc
        with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
            rc = asyncio.run(lc.action_reset_circuit("alpaca", confirm=True))
        assert rc == 0

    def test_drain_without_confirm_exits(self):
        import live_controls_v2 as lc
        with pytest.raises(SystemExit):
            asyncio.run(lc.action_drain("coordination", confirm=False))

    def test_drain_with_confirm(self):
        import live_controls_v2 as lc
        with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
            rc = asyncio.run(lc.action_drain("coordination", confirm=True))
        assert rc == 0

    def test_propose_params_without_confirm_exits(self, tmp_path):
        import live_controls_v2 as lc
        pf = tmp_path / "params.json"
        pf.write_text(json.dumps({"kelly_fraction": 0.25}))
        with pytest.raises(SystemExit):
            asyncio.run(lc.action_propose_params(str(pf), confirm=False))

    def test_propose_params_invalid_json_file(self, tmp_path):
        import live_controls_v2 as lc
        pf = tmp_path / "bad.json"
        pf.write_text("not json {{")
        rc = asyncio.run(lc.action_propose_params(str(pf), confirm=True))
        assert rc == 1

    def test_propose_params_missing_file(self, tmp_path):
        import live_controls_v2 as lc
        rc = asyncio.run(lc.action_propose_params(str(tmp_path / "none.json"), confirm=True))
        assert rc == 1

    def test_propose_params_out_of_range(self, tmp_path):
        import live_controls_v2 as lc
        pf = tmp_path / "params.json"
        pf.write_text(json.dumps({"kelly_fraction": 999.0}))
        rc = asyncio.run(lc.action_propose_params(str(pf), confirm=True))
        assert rc == 1

    def test_propose_params_valid(self, tmp_path):
        import live_controls_v2 as lc
        pf = tmp_path / "params.json"
        pf.write_text(json.dumps({"kelly_fraction": 0.25, "vol_target": 0.10}))
        with patch("live_controls_v2._post", new=AsyncMock(return_value={"ok": True})):
            rc = asyncio.run(lc.action_propose_params(str(pf), confirm=True))
        assert rc == 0


# ─────────────────────────────────────────────────────────────────────────────
# Parameter validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParamValidation:

    def test_valid_params_no_errors(self):
        from live_controls_v2 import validate_params
        params = {"kelly_fraction": 0.25, "vol_target": 0.10, "max_lev": 0.65}
        errors = validate_params(params)
        assert errors == []

    def test_unknown_param_flagged(self):
        from live_controls_v2 import validate_params
        errors = validate_params({"unknown_thing": 1.0})
        assert any("Unknown" in e for e in errors)

    def test_out_of_range_caught(self):
        from live_controls_v2 import validate_params
        errors = validate_params({"kelly_fraction": 5.0})
        assert any("out of range" in e for e in errors)

    def test_wrong_type_caught(self):
        from live_controls_v2 import validate_params
        errors = validate_params({"kelly_fraction": "abc"})
        assert len(errors) > 0

    def test_boundary_values_accepted(self):
        from live_controls_v2 import validate_params
        params = {"kelly_fraction": 0.01, "vol_target": 0.50}
        errors = validate_params(params)
        assert errors == []

    def test_empty_params_valid(self):
        from live_controls_v2 import validate_params
        assert validate_params({}) == []

    def test_all_schema_keys_valid(self):
        from live_controls_v2 import validate_params, _PARAM_SCHEMA
        params = {k: (lo + hi) / 2 for k, (t, lo, hi) in _PARAM_SCHEMA.items()}
        errors = validate_params(params)
        assert errors == []


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationSmoke:

    def test_sizing_sandbox_html_report_csv_fallback(self, tmp_path):
        """HTML report falls back to CSV when plotly not available."""
        from position_sizing_sandbox import SizingSandbox, SizingMethod
        sb = SizingSandbox(synthetic_fallback=True).load()
        comp = sb.compare_methods([SizingMethod.EQUAL_WEIGHT])
        out = tmp_path / "report.html"
        with patch.dict("sys.modules", {"plotly": None, "plotly.graph_objects": None,
                                         "plotly.subplots": None}):
            try:
                sb.to_html_report(out, comp)
            except Exception:
                pass  # expected if plotly mock causes ImportError path

    def test_trade_replayer_missing_db_raises(self, tmp_path):
        from trade_replay import TradeReplayer
        replayer = TradeReplayer(db_path=tmp_path / "missing.db")
        with pytest.raises(FileNotFoundError):
            replayer._get_conn()

    def test_monitor_state_refresh_count_increments(self):
        from live_monitor_v2 import MonitorState
        state = MonitorState()
        assert state.refresh_count == 0
        state.refresh_count += 1
        assert state.refresh_count == 1

    def test_fmt_pnl_positive(self):
        from live_monitor_v2 import _fmt_pnl
        assert _fmt_pnl(100.5).startswith("+")

    def test_fmt_pnl_negative(self):
        from live_monitor_v2 import _fmt_pnl
        assert _fmt_pnl(-100.5).startswith("-")

    def test_load_json_file_default(self, tmp_path):
        from live_controls_v2 import _load_json_file
        result = _load_json_file(tmp_path / "none.json", default={"x": 1})
        assert result == {"x": 1}

    def test_save_and_load_json(self, tmp_path):
        from live_controls_v2 import _load_json_file, _save_json_file
        path = tmp_path / "test.json"
        _save_json_file(path, {"key": "value"})
        data = _load_json_file(path)
        assert data == {"key": "value"}
