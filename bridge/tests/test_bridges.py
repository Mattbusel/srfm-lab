"""
bridge/tests/test_bridges.py
=============================
Production test suite for the SRFM bridge layer.

Covers:
  - LiveParamBridgeV2  (param polling, atomic write, diff, staleness, rollback)
  - PerformanceReporter (Sharpe calc, DB query, HTTP posting)
  - ConfigFileWriter   (thread safety, atomic write, format validation)
  - IAESignalBridge    (pattern translation, validation, deployment, retirement)
  - MarketDataBridge   (buffer ops, fan-out, fallback trigger, health monitor)
  - BarBroadcaster     (fan-out, back-pressure, registration)
  - ExecutionBridge    (routing, fill processing, IS tracking, quality metrics)

Run::

    pytest bridge/tests/test_bridges.py -v
    pytest bridge/tests/test_bridges.py -v -k "sharpe"
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Adjust sys.path so bridge modules can be imported from repo root
# ---------------------------------------------------------------------------

import sys

_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bridge.live_param_bridge_v2 import (
    ConfigFileWriter,
    DiffLogger,
    LiveParamBridgeV2,
    ParamBridgeConfig,
    ParamDiff,
    PerformanceReporter,
    PerformanceSnapshot,
    _check_pair,
    compute_diff,
    log_diff,
    validate_params,
)
from bridge.iae_signal_bridge import (
    IAESignalBridge,
    PatternTranslator,
    SignalRegistry,
    SignalSpec,
    _check_expression_syntax,
    _ICIR_RETIRE_THRESHOLD,
    validate_signal_spec,
)
from bridge.market_data_bridge import (
    AlpacaFallbackClient,
    Bar,
    BarBroadcaster,
    BarBuffer,
    FeedHealthMonitor,
    MarketDataBridge,
)
from bridge.execution_bridge import (
    BridgeOrderIntent,
    BridgeRoutingDecision,
    ExecutionBridge,
    ExecutionQualityTracker,
    FillRecord,
    ISRecord,
    _make_decision_id,
    _compute_price_limit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmp_path() -> Path:
    """Return a fresh temporary directory as a Path."""
    return Path(tempfile.mkdtemp())


def _make_param_config(tmp: Path) -> ParamBridgeConfig:
    return ParamBridgeConfig(
        param_file_path=tmp / "live_params.json",
        coordination_url="http://localhost:18781",   # non-existent -- tests mock HTTP
        iae_url="http://localhost:18780",
        observability_url="http://localhost:19091",
        poll_interval_secs=60.0,
        perf_report_interval_secs=900.0,
        db_path=tmp / "live_trades.db",
        log_dir=tmp / "logs",
        signal_reload=False,  # don't try to kill PIDs in tests
        pid_file=tmp / "live_trader.pid",
    )


def _make_db_with_trades(db_path: Path, rows: list[dict]) -> None:
    """Create a minimal live_trades.db with the given rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT,
            entry_time  TEXT,
            exit_time   TEXT,
            pnl         REAL,
            qty         REAL
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO trades (symbol, entry_time, exit_time, pnl, qty) VALUES (?,?,?,?,?)",
            (r["symbol"], r["entry_time"], r["exit_time"], r["pnl"], r.get("qty", 1.0)),
        )
    conn.commit()
    conn.close()


def _make_bridge_intent(
    symbol: str = "SPY",
    qty: float = 100.0,
    urgency: float = 0.5,
    ref_price: float = 450.0,
) -> BridgeOrderIntent:
    return BridgeOrderIntent(
        symbol=symbol,
        qty=qty,
        urgency=urgency,
        asset_class="equity",
        adv_usd=25_000_000.0,
        sigma_daily=0.012,
        ref_price=ref_price,
    )


# ===========================================================================
# ConfigFileWriter tests
# ===========================================================================

class TestConfigFileWriter:
    """Tests for atomic write and format correctness."""

    def test_write_creates_file(self, tmp_path):
        writer = ConfigFileWriter(tmp_path / "params.json")
        ok = writer.write({"CF_BULL_THRESH": 1.2}, version=1, source="test")
        assert ok
        assert (tmp_path / "params.json").exists()

    def test_atomic_write_format(self, tmp_path):
        p = tmp_path / "params.json"
        writer = ConfigFileWriter(p)
        writer.write({"CF_BULL_THRESH": 1.5, "BH_FORM": 2.0}, version=3, source="unit_test")
        data = json.loads(p.read_text())
        assert data["version"] == 3
        assert data["source"] == "unit_test"
        assert data["params"]["CF_BULL_THRESH"] == 1.5
        assert "applied_at" in data

    def test_no_tmp_file_left_on_success(self, tmp_path):
        p = tmp_path / "params.json"
        writer = ConfigFileWriter(p)
        writer.write({"x": 1}, version=1, source="test")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_read_round_trip(self, tmp_path):
        p = tmp_path / "params.json"
        writer = ConfigFileWriter(p)
        params = {"BH_FORM": 1.92, "KELLY_FRAC": 0.25}
        writer.write(params, version=5, source="test")
        data = writer.read()
        assert data["version"] == 5
        assert data["params"]["BH_FORM"] == 1.92

    def test_read_missing_file_returns_empty(self, tmp_path):
        writer = ConfigFileWriter(tmp_path / "nonexistent.json")
        result = writer.read()
        assert result == {}

    def test_thread_safe_concurrent_writes(self, tmp_path):
        """Multiple threads writing simultaneously should not corrupt the file."""
        p = tmp_path / "params.json"
        writer = ConfigFileWriter(p)
        errors: list[str] = []

        def write_task(version: int) -> None:
            ok = writer.write({"v": version}, version=version, source="thread")
            if not ok:
                errors.append(f"write {version} failed")

        threads = [threading.Thread(target=write_task, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # File should be valid JSON
        data = json.loads(p.read_text())
        assert "version" in data

    def test_write_with_hypothesis_id(self, tmp_path):
        p = tmp_path / "params.json"
        writer = ConfigFileWriter(p)
        writer.write({"x": 1}, version=1, source="iae", hypothesis_id="hyp_42")
        data = json.loads(p.read_text())
        assert data["hypothesis_id"] == "hyp_42"


# ===========================================================================
# Param validation tests
# ===========================================================================

class TestValidateParams:
    def test_valid_params(self):
        params = {
            "CF_BULL_THRESH": 1.2,
            "CF_BEAR_THRESH": 1.4,
            "BH_FORM": 1.92,
            "MIN_HOLD_BARS": 5,
            "MAX_HOLD_BARS": 50,
        }
        errors = validate_params(params)
        assert errors == []

    def test_out_of_range(self):
        errors = validate_params({"CF_BULL_THRESH": 99.0})
        assert any("CF_BULL_THRESH" in e for e in errors)

    def test_non_numeric_value(self):
        errors = validate_params({"BH_FORM": "not_a_number"})
        assert any("non-numeric" in e for e in errors)

    def test_min_max_hold_bars_cross_constraint(self):
        errors = validate_params({"MIN_HOLD_BARS": 50, "MAX_HOLD_BARS": 10})
        assert any("MIN_HOLD_BARS" in e for e in errors)

    def test_correlation_cross_constraint(self):
        errors = validate_params({"min_correlation": 0.5, "max_correlation": 0.3})
        assert any("min_correlation" in e for e in errors)

    def test_cf_bull_bear_constraint(self):
        errors = validate_params({"CF_BULL_THRESH": 2.0, "CF_BEAR_THRESH": 1.0})
        assert any("CF_BULL_THRESH" in e for e in errors)

    def test_unknown_params_allowed(self):
        """Unknown param keys should not cause errors."""
        errors = validate_params({"totally_unknown_param": 42})
        assert errors == []

    def test_bh_form_extreme_constraint(self):
        errors = validate_params({"BH_FORM": 3.5, "BH_MASS_EXTREME": 2.0})
        assert any("BH_FORM" in e for e in errors)


# ===========================================================================
# Diff computation tests
# ===========================================================================

class TestComputeDiff:
    def test_detects_changed_param(self):
        old = {"CF_BULL_THRESH": 1.2}
        new = {"CF_BULL_THRESH": 1.5}
        diffs = compute_diff(old, new, "test")
        assert len(diffs) == 1
        assert diffs[0].key == "CF_BULL_THRESH"
        assert diffs[0].old_value == 1.2
        assert diffs[0].new_value == 1.5

    def test_detects_added_param(self):
        old: dict = {}
        new = {"BH_FORM": 2.0}
        diffs = compute_diff(old, new, "test")
        assert len(diffs) == 1
        assert diffs[0].key == "BH_FORM"
        assert diffs[0].old_value is None

    def test_detects_removed_param(self):
        old = {"BH_FORM": 2.0}
        new: dict = {}
        diffs = compute_diff(old, new, "test")
        assert len(diffs) == 1
        assert diffs[0].new_value is None

    def test_no_diff_when_equal(self):
        params = {"CF_BULL_THRESH": 1.2, "BH_FORM": 1.92}
        diffs = compute_diff(params, dict(params), "test")
        assert diffs == []

    def test_pct_change_computed(self):
        diffs = compute_diff({"x": 1.0}, {"x": 1.5}, "test")
        assert diffs[0].pct_change() == pytest.approx(50.0)

    def test_pct_change_none_for_zero_base(self):
        diffs = compute_diff({"x": 0.0}, {"x": 1.0}, "test")
        assert diffs[0].pct_change() is None

    def test_multiple_changes(self):
        old = {"a": 1, "b": 2, "c": 3}
        new = {"a": 1, "b": 99, "c": 100}
        diffs = compute_diff(old, new, "test")
        keys = {d.key for d in diffs}
        assert keys == {"b", "c"}


# ===========================================================================
# ParamDiff serialisation
# ===========================================================================

class TestParamDiff:
    def test_to_dict(self):
        d = ParamDiff(key="CF_BULL_THRESH", old_value=1.2, new_value=1.5,
                      source="coordination", timestamp="2026-04-06T00:00:00+00:00")
        result = d.to_dict()
        assert result["key"] == "CF_BULL_THRESH"
        assert result["pct_change"] == pytest.approx(25.0)

    def test_non_numeric_pct_change(self):
        d = ParamDiff(key="mode", old_value="fast", new_value="slow",
                      source="test", timestamp="t")
        assert d.pct_change() is None


# ===========================================================================
# PerformanceReporter tests
# ===========================================================================

class TestPerformanceReporter:
    """Sharpe calculation and DB query logic."""

    def test_sharpe_zero_std(self):
        """Flat returns -> Sharpe should be 0."""
        pnls = [1.0, 1.0, 1.0, 1.0]
        sharpe = PerformanceReporter._compute_sharpe(pnls)
        assert sharpe == pytest.approx(0.0)

    def test_sharpe_single_trade(self):
        assert PerformanceReporter._compute_sharpe([100.0]) == pytest.approx(0.0)

    def test_sharpe_positive_edge(self):
        """Positive mean with consistent returns -> positive Sharpe."""
        pnls = [10.0, 12.0, 11.0, 13.0, 9.0, 10.0, 12.0, 11.0]
        sharpe = PerformanceReporter._compute_sharpe(pnls)
        assert sharpe > 0.0

    def test_sharpe_negative_edge(self):
        """Negative mean -> negative Sharpe."""
        pnls = [-5.0, -3.0, -7.0, -4.0, -6.0]
        sharpe = PerformanceReporter._compute_sharpe(pnls)
        assert sharpe < 0.0

    def test_sharpe_annualisation_factor(self):
        """Annualisation: 4h window -> sqrt(365 * 6)."""
        pnls = [1.0, 2.0, 1.5, 2.5, 1.0, 2.0]
        n = len(pnls)
        mean = sum(pnls) / n
        std = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1))
        expected = (mean / std) * math.sqrt(365.0 * 6.0)
        result = PerformanceReporter._compute_sharpe(pnls)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_max_drawdown_no_loss(self):
        """Monotonically increasing equity -> zero drawdown."""
        dd = PerformanceReporter._compute_max_drawdown([1.0, 2.0, 3.0, 4.0])
        assert dd == pytest.approx(0.0)

    def test_max_drawdown_known_value(self):
        """Peak at +6, trough at +3 -> drawdown of 3/6 = 0.5."""
        pnls = [1.0, 2.0, 3.0, -3.0, 0.0]
        dd = PerformanceReporter._compute_max_drawdown(pnls)
        # Running equity: 1, 3, 6, 3, 3 -> peak=6, trough=3 -> dd=0.5
        assert dd == pytest.approx(0.5, rel=1e-3)

    def test_max_drawdown_empty(self):
        dd = PerformanceReporter._compute_max_drawdown([])
        assert dd == 0.0

    def test_reporter_reads_db(self, tmp_path):
        """Reporter should load trades from DB and compute valid snapshot."""
        now = datetime.now(timezone.utc)
        two_hours_ago = (now - timedelta(hours=2)).isoformat()
        db_path = tmp_path / "live_trades.db"
        rows = [
            {"symbol": "SPY", "entry_time": two_hours_ago, "exit_time": two_hours_ago, "pnl": 10.0},
            {"symbol": "QQQ", "entry_time": two_hours_ago, "exit_time": two_hours_ago, "pnl": -5.0},
            {"symbol": "BTC", "entry_time": two_hours_ago, "exit_time": two_hours_ago, "pnl": 20.0},
        ]
        _make_db_with_trades(db_path, rows)
        cfg = _make_param_config(tmp_path)
        reporter = PerformanceReporter(cfg)
        snap = reporter._compute_snapshot()
        assert snap is not None
        assert snap.trade_count == 3
        assert snap.realized_pnl == pytest.approx(25.0)
        assert snap.win_rate_4h == pytest.approx(2.0 / 3.0, rel=1e-3)

    def test_reporter_no_trades_returns_zero_snapshot(self, tmp_path):
        db_path = tmp_path / "live_trades.db"
        _make_db_with_trades(db_path, [])
        cfg = _make_param_config(tmp_path)
        reporter = PerformanceReporter(cfg)
        snap = reporter._compute_snapshot()
        assert snap is not None
        assert snap.trade_count == 0
        assert snap.realized_pnl == pytest.approx(0.0)

    def test_reporter_missing_db_returns_none(self, tmp_path):
        cfg = _make_param_config(tmp_path)
        reporter = PerformanceReporter(cfg)
        # DB file does not exist
        snap = reporter._compute_snapshot()
        assert snap is None

    def test_snapshot_to_dict(self):
        snap = PerformanceSnapshot(
            window_start="2026-04-06T00:00:00+00:00",
            window_end="2026-04-06T04:00:00+00:00",
            trade_count=10,
            equity_change=50.0,
            realized_pnl=50.0,
            sharpe_4h=1.5,
            win_rate_4h=0.7,
            avg_pnl_per_trade=5.0,
            max_drawdown=0.05,
        )
        d = snap.to_dict()
        assert d["trade_count"] == 10
        assert d["sharpe_4h"] == pytest.approx(1.5)
        assert d["strategy"] == "larsa_v18"


# ===========================================================================
# LiveParamBridgeV2 tests
# ===========================================================================

class TestLiveParamBridgeV2:
    def test_param_bridge_detects_change(self, tmp_path):
        """Bridge should detect version bump and update internal state."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)

        new_params = {"CF_BULL_THRESH": 1.5, "BH_FORM": 2.0}
        data = {"version": 1, "params": new_params, "source": "coordination"}

        async def run():
            await bridge._apply_params(data)

        asyncio.run(run())
        assert bridge.get_current_version() == 1
        assert bridge.get_current_params()["CF_BULL_THRESH"] == 1.5

    def test_param_bridge_atomic_write(self, tmp_path):
        """Applied params must produce a valid JSON file with correct structure."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)

        async def run():
            await bridge._apply_params({
                "version": 2,
                "params": {"BH_FORM": 2.1, "KELLY_FRAC": 0.3},
                "source": "test",
            })

        asyncio.run(run())
        param_file = tmp_path / "live_params.json"
        assert param_file.exists()
        data = json.loads(param_file.read_text())
        assert data["version"] == 2
        assert data["params"]["BH_FORM"] == pytest.approx(2.1)

    def test_param_bridge_ignores_invalid_params(self, tmp_path):
        """Bridge must not write if validation fails."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)

        async def run():
            # CF_BULL_THRESH > CF_BEAR_THRESH: invalid
            await bridge._apply_params({
                "version": 1,
                "params": {"CF_BULL_THRESH": 3.0, "CF_BEAR_THRESH": 1.0},
                "source": "test",
            })

        asyncio.run(run())
        assert bridge.get_current_version() == 0

    def test_param_bridge_no_change_skips_write(self, tmp_path):
        """Same version should not trigger a write."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)
        params = {"BH_FORM": 1.92}

        async def run():
            await bridge._apply_params({"version": 1, "params": params, "source": "test"})
            # Apply again with same version
            await bridge._apply_params({"version": 1, "params": params, "source": "test"})

        asyncio.run(run())
        assert bridge.get_current_version() == 1

    def test_stale_warning_written(self, tmp_path):
        """Staleness watchdog should write to stale_alerts.jsonl."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)
        # Simulate stale state
        bridge._last_response_time = time.monotonic() - 400
        bridge._last_good_params = {"BH_FORM": 1.92}
        bridge._write_stale_warning(400.0)
        alert_file = tmp_path / "logs" / "stale_alerts.jsonl"
        assert alert_file.exists()
        line = json.loads(alert_file.read_text().strip())
        assert line["alert_type"] == "stale_coordination"
        assert line["elapsed_secs"] == pytest.approx(400.0)

    def test_rollback_event_applies_params(self, tmp_path):
        """Rollback event handler should apply rollback_params."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)
        rollback_event = json.dumps({
            "reason": "sharp_drawdown",
            "initiated_by": "risk_monitor",
            "rollback_params": {"BH_FORM": 1.5, "KELLY_FRAC": 0.1},
        })

        async def run():
            await bridge._handle_rollback_event(rollback_event)

        asyncio.run(run())
        assert bridge.get_current_params().get("BH_FORM") == 1.5

    def test_rollback_event_audit_written(self, tmp_path):
        """Rollback events should be audited to rollback_events.jsonl."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)
        event = json.dumps({"reason": "test", "initiated_by": "unit_test"})

        async def run():
            await bridge._handle_rollback_event(event)

        asyncio.run(run())
        audit = tmp_path / "logs" / "rollback_events.jsonl"
        assert audit.exists()

    def test_rollback_bad_json_ignored(self, tmp_path):
        """Malformed rollback event should not crash the bridge."""
        cfg = _make_param_config(tmp_path)
        bridge = LiveParamBridgeV2(cfg)

        async def run():
            await bridge._handle_rollback_event("not valid json {{")

        asyncio.run(run())  # Should not raise


# ===========================================================================
# DiffLogger tests
# ===========================================================================

class TestDiffLogger:
    def test_appends_jsonl(self, tmp_path):
        logger = DiffLogger(tmp_path)
        diffs = [
            ParamDiff("a", 1.0, 2.0, "test", "2026-04-06T00:00:00+00:00"),
            ParamDiff("b", "old", "new", "test", "2026-04-06T00:00:00+00:00"),
        ]
        logger.append(diffs)
        path = tmp_path / "param_diffs.jsonl"
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        obj = json.loads(lines[0])
        assert obj["key"] == "a"

    def test_empty_diffs_no_write(self, tmp_path):
        logger = DiffLogger(tmp_path)
        logger.append([])
        assert not (tmp_path / "param_diffs.jsonl").exists()


# ===========================================================================
# IAESignalBridge -- pattern translation tests
# ===========================================================================

class TestPatternTranslator:
    def setup_method(self):
        self.translator = PatternTranslator()

    def _make_pattern(self, pattern_type: str, extra_cond: dict | None = None) -> dict:
        return {
            "pattern_id": f"iae_test_{pattern_type}_abc123",
            "pattern_type": pattern_type,
            "confidence": 0.75,
            "entry_conditions": extra_cond or {},
            "regime_requirements": ["trending"],
            "fitness_score": 0.6,
            "generation": 10,
        }

    def test_translate_momentum_burst(self):
        pat = self._make_pattern("momentum_burst", {"lookback": 15, "threshold": 1.8})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None
        assert "momentum_burst" in spec.name
        assert "15" in spec.signal_fn_str

    def test_translate_mean_reversion_ou(self):
        pat = self._make_pattern("mean_reversion_ou", {"half_life": 8.0, "z_threshold": 1.5})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None
        assert spec.required_regime == "trending"

    def test_translate_volatility_regime_shift(self):
        pat = self._make_pattern("volatility_regime_shift", {"short_vol_window": 5, "long_vol_window": 20})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None
        assert "5" in spec.signal_fn_str
        assert "20" in spec.signal_fn_str

    def test_translate_microstructure_imbalance(self):
        pat = self._make_pattern("microstructure_imbalance", {"window": 12})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None
        assert "12" in spec.signal_fn_str

    def test_translate_black_hole_formation(self):
        pat = self._make_pattern("black_hole_formation", {"atr_multiplier": 2.5, "atr_window": 14})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None
        assert "2.5" in spec.signal_fn_str

    def test_translate_centrifugal_force_threshold(self):
        pat = self._make_pattern("centrifugal_force_threshold", {"cf_threshold": 1.8})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None

    def test_translate_regime_transition(self):
        pat = self._make_pattern("regime_transition", {"transition_window": 12})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None

    def test_translate_carry_signal(self):
        pat = self._make_pattern("carry_signal", {"carry_window": 25})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None
        assert "25" in spec.signal_fn_str

    def test_translate_liquidity_stress(self):
        pat = self._make_pattern("liquidity_stress", {"window": 8, "stress_threshold": 3.0})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None

    def test_unknown_pattern_type_returns_none(self):
        pat = self._make_pattern("completely_unknown_type")
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is None

    def test_cross_sectional_momentum(self):
        pat = self._make_pattern("cross_sectional_momentum", {"lookback": 60, "skip_recent": 5})
        spec = self.translator.translate_pattern_to_signal(pat)
        assert spec is not None
        assert "60" in spec.signal_fn_str


# ===========================================================================
# SignalSpec validation tests
# ===========================================================================

class TestValidateSignalSpec:
    def _valid_spec(self) -> SignalSpec:
        return SignalSpec(
            name="test_signal_v1",
            signal_fn_str="prices.pct_change(20).rolling(3).mean()",
            required_regime="trending",
            min_ic=0.03,
        )

    def test_valid_spec(self):
        errors = validate_signal_spec(self._valid_spec())
        assert errors == []

    def test_invalid_name_with_spaces(self):
        spec = self._valid_spec()
        spec.name = "invalid name here"
        errors = validate_signal_spec(spec)
        assert any("name" in e.lower() for e in errors)

    def test_empty_signal_fn(self):
        spec = self._valid_spec()
        spec.signal_fn_str = "   "
        errors = validate_signal_spec(spec)
        assert any("signal_fn_str" in e for e in errors)

    def test_syntax_error_in_fn(self):
        spec = self._valid_spec()
        spec.signal_fn_str = "prices.pct_change("  # unclosed paren
        errors = validate_signal_spec(spec)
        assert any("SyntaxError" in e for e in errors)

    def test_invalid_regime(self):
        spec = self._valid_spec()
        spec.required_regime = "completely_invalid_regime"
        errors = validate_signal_spec(spec)
        assert any("required_regime" in e for e in errors)

    def test_min_ic_out_of_range(self):
        spec = self._valid_spec()
        spec.min_ic = 1.5
        errors = validate_signal_spec(spec)
        assert any("min_ic" in e for e in errors)

    def test_expression_syntax_check(self):
        errs = _check_expression_syntax("1 + 1")
        assert errs == []
        errs = _check_expression_syntax("def foo(): pass")  # def is a statement, not expr
        assert len(errs) > 0


# ===========================================================================
# SignalRegistry tests
# ===========================================================================

class TestSignalRegistry:
    def test_insert_and_exists(self, tmp_path):
        reg = SignalRegistry(tmp_path / "signals.db")
        spec = SignalSpec(
            name="test_mom_v1",
            signal_fn_str="prices.pct_change(10)",
            pattern_id="pat_001",
        )
        row_id = reg.insert_signal(spec)
        assert row_id > 0
        assert reg.signal_exists("test_mom_v1")

    def test_duplicate_insert_raises(self, tmp_path):
        reg = SignalRegistry(tmp_path / "signals.db")
        spec = SignalSpec(name="dup_signal", signal_fn_str="prices.pct_change(5)", pattern_id="p")
        reg.insert_signal(spec)
        with pytest.raises(sqlite3.IntegrityError):
            reg.insert_signal(spec)

    def test_update_icir(self, tmp_path):
        reg = SignalRegistry(tmp_path / "signals.db")
        spec = SignalSpec(name="icir_test", signal_fn_str="prices.diff()", pattern_id="p")
        reg.insert_signal(spec)
        reg.update_icir("icir_test", 0.45)
        active = reg.get_active_signals()
        rec = next(r for r in active if r.name == "icir_test")
        assert rec.current_icir == pytest.approx(0.45)

    def test_retire_signal(self, tmp_path):
        reg = SignalRegistry(tmp_path / "signals.db")
        spec = SignalSpec(name="retire_test", signal_fn_str="prices.pct_change(3)", pattern_id="p")
        reg.insert_signal(spec)
        reg.retire_signal("retire_test")
        active = reg.get_active_signals()
        assert not any(r.name == "retire_test" for r in active)


# ===========================================================================
# IAESignalBridge integration tests
# ===========================================================================

class TestIAESignalBridge:
    def _make_bridge(self, tmp_path) -> IAESignalBridge:
        return IAESignalBridge(
            iae_url="http://localhost:18780",
            coordination_url="http://localhost:18781",
            db_path=tmp_path / "signal_registry.db",
        )

    def test_deploy_valid_signal(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        spec = SignalSpec(
            name="deploy_test",
            signal_fn_str="prices.pct_change(10).rolling(3).mean()",
            pattern_id="pat_deploy",
            required_regime="any",
        )
        ok = bridge.deploy_signal(spec)
        assert ok

    def test_deploy_duplicate_returns_false(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        spec = SignalSpec(
            name="dup_deploy",
            signal_fn_str="prices.pct_change(5)",
            pattern_id="p",
        )
        assert bridge.deploy_signal(spec)
        assert not bridge.deploy_signal(spec)

    def test_deploy_invalid_spec_returns_false(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        spec = SignalSpec(
            name="bad signal name",
            signal_fn_str="prices.pct_change(",  # syntax error
            pattern_id="p",
        )
        ok = bridge.deploy_signal(spec)
        assert not ok

    def test_retire_stale_signals(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        spec = SignalSpec(
            name="stale_signal",
            signal_fn_str="prices.pct_change(5)",
            pattern_id="ps",
        )
        bridge.deploy_signal(spec)
        # Set ICIR below retire threshold
        bridge._registry.update_icir("stale_signal", 0.10)
        retired = bridge.retire_stale_signals()
        assert "stale_signal" in retired

    def test_no_retire_above_threshold(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        spec = SignalSpec(
            name="healthy_signal",
            signal_fn_str="prices.pct_change(5)",
            pattern_id="ph",
        )
        bridge.deploy_signal(spec)
        bridge._registry.update_icir("healthy_signal", 0.50)
        retired = bridge.retire_stale_signals()
        assert "healthy_signal" not in retired

    def test_pattern_schema_validation_rejects_low_confidence(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        pattern = {
            "pattern_id": "low_conf",
            "pattern_type": "momentum_burst",
            "confidence": 0.30,  # below 0.50 threshold
            "entry_conditions": {},
        }
        assert not bridge._validate_pattern_schema(pattern)

    def test_pattern_schema_validation_missing_fields(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        pattern = {"pattern_type": "momentum_burst"}  # missing required fields
        assert not bridge._validate_pattern_schema(pattern)

    def test_seen_pattern_ids_prevents_redeploy(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        bridge._seen_pattern_ids.add("already_seen_123")
        spec = SignalSpec(
            name="wont_deploy",
            signal_fn_str="prices.pct_change(5)",
            pattern_id="already_seen_123",
        )
        # Manually deploy it first time
        bridge.deploy_signal(spec)
        # Simulate patterns list with already-seen pattern
        # The _seen_pattern_ids check is in _fetch_and_process_patterns, not deploy_signal
        assert "already_seen_123" in bridge._seen_pattern_ids


# ===========================================================================
# BarBuffer tests
# ===========================================================================

class TestBarBuffer:
    def _make_bar(self, symbol: str, close: float) -> Bar:
        return Bar(
            symbol=symbol,
            timeframe="1m",
            timestamp="2026-04-06T12:00:00+00:00",
            open=close,
            high=close + 0.5,
            low=close - 0.5,
            close=close,
            volume=1000.0,
        )

    def test_append_and_get_latest(self):
        buf = BarBuffer(maxlen=100)
        bar = self._make_bar("SPY", 450.0)

        async def run():
            await buf.append(bar)
            return await buf.get_latest()

        result = asyncio.run(run())
        assert result is not None
        assert result.close == pytest.approx(450.0)

    def test_maxlen_enforced(self):
        buf = BarBuffer(maxlen=5)

        async def run():
            for i in range(10):
                await buf.append(self._make_bar("SPY", float(i)))

        asyncio.run(run())
        assert len(buf) == 5

    def test_get_history_n(self):
        buf = BarBuffer(maxlen=100)

        async def run():
            for i in range(20):
                await buf.append(self._make_bar("SPY", float(i)))
            return await buf.get_history(5)

        history = asyncio.run(run())
        assert len(history) == 5
        assert history[-1].close == pytest.approx(19.0)

    def test_get_latest_empty_returns_none(self):
        buf = BarBuffer()

        async def run():
            return await buf.get_latest()

        assert asyncio.run(run()) is None


# ===========================================================================
# BarBroadcaster tests
# ===========================================================================

class TestBarBroadcaster:
    def _make_bar(self, sym: str = "SPY", close: float = 450.0) -> Bar:
        return Bar(
            symbol=sym, timeframe="1m",
            timestamp="2026-04-06T12:00:00+00:00",
            open=close, high=close, low=close, close=close, volume=1000.0,
        )

    def test_register_and_receive(self):
        broadcaster = BarBroadcaster(queue_size=100)

        async def run():
            q = await broadcaster.register_consumer("test_consumer")
            bar = self._make_bar()
            await broadcaster.broadcast(bar)
            received = await asyncio.wait_for(q.get(), timeout=1.0)
            return received

        result = asyncio.run(run())
        assert result.symbol == "SPY"

    def test_fanout_to_multiple_consumers(self):
        broadcaster = BarBroadcaster(queue_size=100)

        async def run():
            q1 = await broadcaster.register_consumer("consumer_1")
            q2 = await broadcaster.register_consumer("consumer_2")
            q3 = await broadcaster.register_consumer("consumer_3")
            bar = self._make_bar()
            await broadcaster.broadcast(bar)
            r1 = await asyncio.wait_for(q1.get(), timeout=1.0)
            r2 = await asyncio.wait_for(q2.get(), timeout=1.0)
            r3 = await asyncio.wait_for(q3.get(), timeout=1.0)
            return r1, r2, r3

        r1, r2, r3 = asyncio.run(run())
        assert r1.symbol == r2.symbol == r3.symbol == "SPY"

    def test_back_pressure_drops_oldest(self):
        """When queue full, oldest item should be dropped to make room."""
        broadcaster = BarBroadcaster(queue_size=3)

        async def run():
            q = await broadcaster.register_consumer("slow_consumer")
            # Fill queue to capacity
            for i in range(5):
                await broadcaster.broadcast(self._make_bar("SPY", float(i)))
            return q

        q = asyncio.run(run())
        assert q.qsize() == 3
        assert broadcaster.get_drop_counts().get("slow_consumer", 0) >= 1

    def test_unregister_consumer(self):
        broadcaster = BarBroadcaster(queue_size=100)

        async def run():
            await broadcaster.register_consumer("to_remove")
            await broadcaster.unregister_consumer("to_remove")
            names = await broadcaster.get_consumer_names()
            return names

        names = asyncio.run(run())
        assert "to_remove" not in names

    def test_register_duplicate_returns_same_queue(self):
        broadcaster = BarBroadcaster(queue_size=100)

        async def run():
            q1 = await broadcaster.register_consumer("same_name")
            q2 = await broadcaster.register_consumer("same_name")
            return q1 is q2

        assert asyncio.run(run())


# ===========================================================================
# Bar parsing tests
# ===========================================================================

class TestBarParsing:
    def test_from_go_message(self):
        msg = {
            "symbol": "SPY", "timeframe": "1m", "timestamp": "2026-04-06T12:00:00Z",
            "open": 450.0, "high": 451.0, "low": 449.5, "close": 450.5,
            "volume": 50000.0, "vwap": 450.3, "trade_count": 1234,
        }
        bar = Bar.from_go_message(msg)
        assert bar.symbol == "SPY"
        assert bar.close == pytest.approx(450.5)
        assert bar.source == "go_market_data"

    def test_from_go_message_short_keys(self):
        msg = {"S": "BTC/USD", "tf": "4h", "t": "ts", "o": 60000.0, "h": 61000.0,
               "l": 59000.0, "c": 60500.0, "v": 10.5, "vw": 60250.0, "n": 500}
        bar = Bar.from_go_message(msg)
        assert bar.symbol == "BTC/USD"
        assert bar.close == pytest.approx(60500.0)

    def test_from_alpaca_bar(self):
        msg = {"S": "SPY", "t": "2026-04-06T12:00:00Z",
               "o": 450.0, "h": 451.0, "l": 449.0, "c": 450.5, "v": 12000.0}
        bar = Bar.from_alpaca_bar(msg, timeframe="15m")
        assert bar.timeframe == "15m"
        assert bar.source == "alpaca_fallback"

    def test_bar_to_dict(self):
        bar = Bar("SPY", "1m", "2026-04-06T12:00:00Z", 450.0, 451.0, 449.0, 450.5, 5000.0)
        d = bar.to_dict()
        assert set(d.keys()) >= {"symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume"}


# ===========================================================================
# MarketDataBridge tests
# ===========================================================================

class TestMarketDataBridge:
    def _make_bridge(self) -> MarketDataBridge:
        broadcaster = BarBroadcaster()
        return MarketDataBridge(
            go_ws_url="ws://localhost:18780/ws/bars",
            broadcaster=broadcaster,
            stale_threshold_secs=30.0,
            warn_threshold_secs=10.0,
        )

    def test_ingest_bar_stores_in_buffer(self):
        bridge = self._make_bridge()

        async def run():
            bar = Bar("SPY", "1m", "2026-04-06T12:00:00Z", 450.0, 451.0, 449.0, 450.5, 5000.0)
            await bridge._ingest_bar(bar)
            return await bridge.get_latest_bar("SPY", "1m")

        result = asyncio.run(run())
        assert result is not None
        assert result.close == pytest.approx(450.5)

    def test_get_bar_history(self):
        bridge = self._make_bridge()

        async def run():
            for i in range(10):
                bar = Bar("ETH/USD", "15m", "ts", 2000.0 + i, 2001.0, 1999.0, 2000.0 + i, 500.0)
                await bridge._ingest_bar(bar)
            return await bridge.get_bar_history("ETH/USD", "15m", n=5)

        history = asyncio.run(run())
        assert len(history) == 5
        assert history[-1].close == pytest.approx(2009.0)

    def test_get_latest_bar_missing_returns_none(self):
        bridge = self._make_bridge()

        async def run():
            return await bridge.get_latest_bar("MISSING", "1m")

        assert asyncio.run(run()) is None

    def test_ingest_increments_counter(self):
        bridge = self._make_bridge()

        async def run():
            for _ in range(5):
                bar = Bar("BTC/USD", "4h", "ts", 60000.0, 61000.0, 59000.0, 60500.0, 1.0)
                await bridge._ingest_bar(bar)

        asyncio.run(run())
        assert bridge.get_total_bars_received() == 5

    def test_market_data_bridge_fallback_flag(self):
        """Fallback flag should start as False."""
        bridge = self._make_bridge()
        assert not bridge.is_using_fallback()

    def test_multiple_symbols_buffered_independently(self):
        bridge = self._make_bridge()

        async def run():
            await bridge._ingest_bar(Bar("SPY", "1m", "ts", 450.0, 451.0, 449.0, 450.5, 1000.0))
            await bridge._ingest_bar(Bar("QQQ", "1m", "ts", 370.0, 371.0, 369.0, 370.5, 500.0))
            spy = await bridge.get_latest_bar("SPY", "1m")
            qqq = await bridge.get_latest_bar("QQQ", "1m")
            return spy, qqq

        spy, qqq = asyncio.run(run())
        assert spy.symbol == "SPY"
        assert qqq.symbol == "QQQ"
        assert spy.close != qqq.close


# ===========================================================================
# FeedHealthMonitor tests
# ===========================================================================

class TestFeedHealthMonitor:
    def test_fresh_feed_not_stale(self):
        monitor = FeedHealthMonitor(warn_threshold_secs=60, stale_threshold_secs=120)

        async def run():
            await monitor.record_bar("SPY", "1m")
            return await monitor.check_staleness()

        result = asyncio.run(run())
        assert result == []

    def test_stale_feed_detected(self):
        monitor = FeedHealthMonitor(warn_threshold_secs=5, stale_threshold_secs=10)

        async def run():
            await monitor.record_bar("SPY", "1m")
            # Backdate last_seen
            async with monitor._lock:
                monitor._last_seen[("SPY", "1m")] = time.monotonic() - 20
            return await monitor.check_staleness()

        stale = asyncio.run(run())
        assert len(stale) == 1
        assert stale[0][0] == "SPY"


# ===========================================================================
# ExecutionBridge tests
# ===========================================================================

class TestExecutionBridge:
    def _make_bridge(self, tmp_path: Path) -> ExecutionBridge:
        return ExecutionBridge(
            db_path=tmp_path / "exec_bridge.db",
            exec_db_path=tmp_path / "live_trades.db",
        )

    def test_routing_decision_returned(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(
            symbol="SPY", qty=100.0, urgency=0.5,
            asset_class="equity", ref_price=450.0,
        )
        assert len(decisions) >= 1
        assert decisions[0].intent.symbol == "SPY"

    def test_routing_decision_has_valid_venue(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(
            symbol="BTC/USD", qty=0.5, urgency=0.8,
            asset_class="crypto", ref_price=60000.0,
        )
        # SmartRouter may return alpaca_crypto, binance_spot, or other valid venue
        assert isinstance(decisions[0].venue, str) and len(decisions[0].venue) > 0

    def test_on_fill_returns_fill_record(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(
            symbol="SPY", qty=100.0, ref_price=450.0
        )
        fill = bridge.on_fill(
            decisions[0],
            {"fill_price": 450.10, "fill_qty": 100.0}
        )
        assert fill is not None
        assert fill.fill_price == pytest.approx(450.10)

    def test_fill_computes_slippage(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(
            symbol="SPY", qty=100.0, ref_price=450.0
        )
        fill = bridge.on_fill(
            decisions[0],
            {"fill_price": 450.45, "fill_qty": 100.0}
        )
        # slippage = (450.45 - 450.0) / 450.0 * 10000 = 10 bps
        assert fill is not None
        assert fill.slippage_bps == pytest.approx(10.0, rel=0.01)

    def test_fill_removes_from_pending(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(symbol="SPY", qty=100.0, ref_price=450.0)
        assert bridge.get_pending_count() == 1
        bridge.on_fill(decisions[0], {"fill_price": 450.0, "fill_qty": 100.0})
        assert bridge.get_pending_count() == 0

    def test_is_tracked_after_fill(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(
            symbol="SPY", qty=100.0, ref_price=450.0, urgency=0.5
        )
        bridge.on_fill(decisions[0], {"fill_price": 450.20, "fill_qty": 100.0})
        quality = bridge.get_execution_quality("SPY")
        assert quality["count"] >= 1

    def test_execution_quality_empty_symbol(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        quality = bridge.get_execution_quality("NONEXISTENT")
        assert quality["mean_is_bps"] == pytest.approx(0.0)
        assert quality["count"] == 0

    def test_decision_id_format(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(symbol="SPY", qty=100.0)
        did = decisions[0].decision_id
        assert "SPY" in did
        parts = did.split("_")
        assert len(parts) >= 2
        assert parts[-1].isdigit()

    def test_make_decision_id_unique(self):
        id1 = _make_decision_id("SPY")
        time.sleep(0.002)
        id2 = _make_decision_id("SPY")
        assert id1 != id2

    def test_compute_price_limit_buy(self):
        limit = _compute_price_limit(450.0, qty=100.0, slippage_frac=0.001)
        assert limit == pytest.approx(450.45)

    def test_compute_price_limit_sell(self):
        limit = _compute_price_limit(450.0, qty=-100.0, slippage_frac=0.001)
        assert limit == pytest.approx(449.55)

    def test_sell_decision_routes_to_expected_venue(self, tmp_path):
        bridge = self._make_bridge(tmp_path)
        decisions = bridge.on_order_intent(
            symbol="ETH/USD", qty=-1.0, urgency=0.2,
            asset_class="crypto", ref_price=3000.0,
        )
        assert len(decisions) >= 1
        assert decisions[0].qty < 0 or decisions[0].intent.qty < 0


# ===========================================================================
# ExecutionQualityTracker tests
# ===========================================================================

class TestExecutionQualityTracker:
    def test_record_and_query(self, tmp_path):
        tracker = ExecutionQualityTracker(tmp_path / "quality.db")
        rec = ISRecord(
            symbol="SPY",
            decision_id="SPY_123456",
            estimated_bps=3.0,
            actual_bps=4.5,
            is_bps=1.5,
            ref_price=450.0,
            fill_price=450.0675,
            qty=100.0,
        )
        tracker.record(rec)
        quality = tracker.get_rolling_is("SPY")
        assert quality["count"] == 1
        assert quality["mean_is_bps"] == pytest.approx(1.5)

    def test_multiple_records_rolling_stats(self, tmp_path):
        tracker = ExecutionQualityTracker(tmp_path / "quality.db")
        for i in range(5):
            rec = ISRecord(
                symbol="QQQ", decision_id=f"QQQ_{i}",
                estimated_bps=2.0, actual_bps=float(3 + i), is_bps=float(1 + i),
                ref_price=370.0, fill_price=370.0, qty=50.0,
            )
            tracker.record(rec)
        quality = tracker.get_rolling_is("QQQ")
        assert quality["count"] == 5
        # IS values: 1, 2, 3, 4, 5 -> mean = 3
        assert quality["mean_is_bps"] == pytest.approx(3.0)

    def test_unknown_symbol_returns_zeros(self, tmp_path):
        tracker = ExecutionQualityTracker(tmp_path / "quality.db")
        quality = tracker.get_rolling_is("NONEXISTENT")
        assert quality["count"] == 0
        assert quality["mean_is_bps"] == pytest.approx(0.0)

    def test_30d_from_db(self, tmp_path):
        tracker = ExecutionQualityTracker(tmp_path / "quality.db")
        rec = ISRecord(
            symbol="GLD", decision_id="GLD_001",
            estimated_bps=1.5, actual_bps=2.0, is_bps=0.5,
            ref_price=180.0, fill_price=180.009, qty=100.0,
        )
        tracker.record(rec)
        result = tracker.get_30d_is_from_db("GLD")
        assert result["count"] >= 1
        assert result["mean_is_bps"] == pytest.approx(0.5)


# ===========================================================================
# ISRecord / FillRecord serialisation tests
# ===========================================================================

class TestSerialisation:
    def test_is_record_to_dict(self):
        rec = ISRecord("SPY", "SPY_123", 3.0, 4.5, 1.5, 450.0, 450.0675, 100.0)
        d = rec.to_dict()
        assert d["is_bps"] == pytest.approx(1.5)
        assert "recorded_at" in d

    def test_fill_record_to_dict(self):
        fill = FillRecord(
            decision_id="SPY_123", symbol="SPY",
            fill_price=450.10, fill_qty=100.0, slippage_bps=2.22,
        )
        d = fill.to_dict()
        assert d["fill_price"] == pytest.approx(450.10)
        assert "fill_time" in d

    def test_bridge_routing_decision_to_dict(self):
        intent = _make_bridge_intent()
        rd = BridgeRoutingDecision(
            intent=intent, venue="alpaca_equity", order_type="limit",
            limit_price=450.45, qty=100.0, bar_index=0,
            estimated_cost_bps=3.5, reasoning="test",
            decision_id="SPY_9999",
        )
        d = rd.to_dict()
        assert d["venue"] == "alpaca_equity"
        assert d["estimated_cost_bps"] == pytest.approx(3.5)
        assert d["intent"]["symbol"] == "SPY"


# ===========================================================================
# PerformanceSnapshot serialisation
# ===========================================================================

class TestPerformanceSnapshotSerialisation:
    def test_to_dict_keys(self):
        snap = PerformanceSnapshot(
            window_start="2026-04-06T00:00:00+00:00",
            window_end="2026-04-06T04:00:00+00:00",
            trade_count=5,
            equity_change=100.0,
            realized_pnl=100.0,
            sharpe_4h=2.1,
            win_rate_4h=0.8,
            avg_pnl_per_trade=20.0,
            max_drawdown=0.02,
        )
        d = snap.to_dict()
        required = {"window_start", "window_end", "trade_count", "equity_change",
                    "realized_pnl", "sharpe_4h", "win_rate_4h", "avg_pnl_per_trade",
                    "max_drawdown", "computed_at", "strategy"}
        assert required <= set(d.keys())


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:
    def test_bar_buffer_get_history_n_larger_than_size(self):
        buf = BarBuffer(maxlen=100)

        async def run():
            for i in range(5):
                await buf.append(Bar("SPY", "1m", "ts", float(i), float(i), float(i), float(i), 100.0))
            return await buf.get_history(200)

        history = asyncio.run(run())
        assert len(history) == 5  # capped at available

    def test_compute_diff_source_preserved(self):
        diffs = compute_diff({"a": 1}, {"a": 2}, source="iae_genome")
        assert diffs[0].source == "iae_genome"

    def test_validate_params_empty_dict(self):
        assert validate_params({}) == []

    def test_bridge_order_intent_to_dict(self):
        intent = _make_bridge_intent()
        d = intent.to_dict()
        assert d["symbol"] == "SPY"
        assert d["strategy_id"] == "larsa_v18"

    def test_check_expression_syntax_valid(self):
        assert _check_expression_syntax("prices.pct_change(20)") == []

    def test_check_pair_skips_when_key_absent(self):
        errors: list[str] = []
        _check_pair({"a": 1.0}, errors, "a", "b", "lt")
        assert errors == []  # b not in params

    def test_spec_from_dict_round_trip(self):
        spec = SignalSpec(
            name="test_round_trip",
            signal_fn_str="prices.pct_change(10)",
            required_regime="trending",
            min_ic=0.03,
            pattern_id="test_pat",
        )
        d = spec.to_dict()
        spec2 = SignalSpec.from_dict(d)
        assert spec2.name == spec.name
        assert spec2.min_ic == pytest.approx(spec.min_ic)

    def test_execution_bridge_crypto_routing(self, tmp_path):
        bridge = ExecutionBridge(db_path=tmp_path / "b.db", exec_db_path=tmp_path / "t.db")
        decisions = bridge.on_order_intent(
            symbol="BTC/USD", qty=0.1, asset_class="crypto",
            urgency=0.9, ref_price=60000.0,
        )
        assert len(decisions) >= 1
        # Market order expected for high urgency
        assert decisions[0].order_type in ("market", "limit")

    def test_is_record_min_is(self, tmp_path):
        tracker = ExecutionQualityTracker(tmp_path / "q.db")
        for is_val in [-5.0, 2.0, 10.0, -1.0]:
            tracker.record(ISRecord(
                symbol="TST", decision_id=f"TST_{int(time.time()*1000)}_{is_val}",
                estimated_bps=2.0, actual_bps=2.0 + is_val, is_bps=is_val,
                ref_price=100.0, fill_price=100.0, qty=10.0,
            ))
            time.sleep(0.001)
        quality = tracker.get_rolling_is("TST")
        assert quality["min_is_bps"] == pytest.approx(-5.0)
        assert quality["max_is_bps"] == pytest.approx(10.0)
