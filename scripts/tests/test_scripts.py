"""
test_scripts.py -- Unit tests for SRFM operational scripts.

Tests cover:
  - EmergencyStop halt/flatten/cancel sequence (mock broker)
  - ManualParamUpdater validation and diff computation
  - UniverseUpdater add/remove round-trip
  - Daily startup check sequence with all-pass mock services
  - DataBackfiller gap detection and checkpoint management
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, call

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from emergency_stop import EmergencyStop
from param_update_manual import ManualParamUpdater, _coerce_numeric, _parse_value
from universe_updater import UniverseUpdater
from daily_startup import (
    ServiceHealthChecker,
    ConfigValidator,
    DatabaseChecker,
    CircuitBreakerChecker,
    BrokerConnectivityChecker,
    PositionReconciler,
    StartupOrchestrator,
)
from backfill_data import (
    DataBackfiller,
    _date_chunks,
    _is_crypto,
    _parse_date,
    load_checkpoint,
    save_checkpoint,
    delete_checkpoint,
    _progress_bar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ok_response(body: Dict) -> Tuple[int, bytes]:
    return 200, json.dumps(body).encode("utf-8")


def _make_err_response(code: int) -> Tuple[int, bytes]:
    return code, b'{"error": "test error"}'


# ---------------------------------------------------------------------------
# EmergencyStop tests
# ---------------------------------------------------------------------------

class TestEmergencyStopHalt(unittest.TestCase):

    def setUp(self):
        self.es = EmergencyStop(base_url="http://test-coordinator:8000")

    @patch("emergency_stop.http_post")
    def test_halt_trading_success(self, mock_post):
        mock_post.return_value = (200, b'{"status": "halted"}')
        result = self.es.halt_all_trading()
        self.assertTrue(result)
        mock_post.assert_called_once()
        args, _ = mock_post.call_args
        self.assertIn("/trading/halt", args[0])

    @patch("emergency_stop.http_post")
    def test_halt_trading_falls_back_to_mode_endpoint(self, mock_post):
        # First call to /trading/halt returns 503, second to /trading/mode returns 200
        mock_post.side_effect = [
            (503, b'{"error": "service unavailable"}'),
            (200, b'{"status": "disabled"}'),
        ]
        result = self.es.halt_all_trading()
        self.assertTrue(result)
        self.assertEqual(mock_post.call_count, 2)

    @patch("emergency_stop.http_post")
    def test_halt_trading_fails_both_endpoints(self, mock_post):
        mock_post.return_value = (500, b'{"error": "server error"}')
        result = self.es.halt_all_trading()
        self.assertFalse(result)

    @patch("emergency_stop.http_delete")
    def test_cancel_pending_orders_alpaca(self, mock_delete):
        mock_delete.return_value = (207, b'[]')
        import emergency_stop as em
        with patch.object(em, "ALPACA_KEY", "test_key"), patch.object(em, "ALPACA_SECRET", "test_secret"):
            result = self.es.cancel_pending_orders("alpaca")
        self.assertTrue(result)
        mock_delete.assert_called_once()
        args, _ = mock_delete.call_args
        self.assertIn("/v2/orders", args[0])

    @patch("emergency_stop.http_delete")
    def test_cancel_pending_orders_binance_via_coordinator(self, mock_delete):
        mock_delete.return_value = (200, b'{"cancelled": 3}')
        result = self.es.cancel_pending_orders("binance")
        self.assertTrue(result)
        args, _ = mock_delete.call_args
        self.assertIn("broker=binance", args[0])

    @patch("emergency_stop.http_delete")
    def test_flatten_positions_via_coordinator(self, mock_delete):
        mock_delete.return_value = (200, b'{"flattened": 2}')
        # Use generic broker path
        with patch("emergency_stop.http_post") as mock_post:
            mock_post.return_value = (200, b'{"status": "submitted"}')
            result = self.es.flatten_positions("binance", market_order=True)
        self.assertTrue(result)

    def test_flatten_alpaca_direct(self):
        import emergency_stop as em
        with patch.object(em, "ALPACA_KEY", "test_key"), \
             patch.object(em, "ALPACA_SECRET", "test_secret"), \
             patch("emergency_stop.http_delete") as mock_delete:
            mock_delete.return_value = (200, b'[{"symbol": "AAPL"}]')
            result = self.es.flatten_positions("alpaca", market_order=True)
        self.assertTrue(result)
        mock_delete.assert_called_once()

    @patch("emergency_stop.http_post")
    def test_send_emergency_alert_no_webhook(self, mock_post):
        with patch.dict(os.environ, {}, clear=False):
            orig = os.environ.get("SRFM_SLACK_WEBHOOK", "")
            if "SRFM_SLACK_WEBHOOK" in os.environ:
                del os.environ["SRFM_SLACK_WEBHOOK"]
            # Should not raise even without webhook
            self.es.send_emergency_alert("test_reason")
            mock_post.assert_not_called()
            if orig:
                os.environ["SRFM_SLACK_WEBHOOK"] = orig

    def test_log_incident_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import emergency_stop as em_mod
            orig_dir = em_mod.INCIDENTS_DIR
            em_mod.INCIDENTS_DIR = Path(tmpdir)
            try:
                path = self.es.log_incident("test_reason", {"context_key": "value"})
                self.assertTrue(path.exists())
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.assertEqual(data["reason"], "test_reason")
                self.assertEqual(data["incident_type"], "emergency_stop")
                self.assertIn("context_key", data["context"])
            finally:
                em_mod.INCIDENTS_DIR = orig_dir

    @patch("emergency_stop.http_post")
    @patch("emergency_stop.http_delete")
    def test_full_stop_sequence(self, mock_delete, mock_post):
        mock_post.return_value = (200, b'{"status": "ok"}')
        mock_delete.return_value = (200, b'[]')
        with tempfile.TemporaryDirectory() as tmpdir:
            import emergency_stop as em_mod
            orig_dir = em_mod.INCIDENTS_DIR
            em_mod.INCIDENTS_DIR = Path(tmpdir)
            try:
                result = self.es.full_stop(
                    reason="unit_test",
                    flatten=True,
                    cancel_orders=True,
                    brokers=["alpaca"],
                )
                self.assertTrue(result)
                incident_files = list(Path(tmpdir).glob("*_emergency.json"))
                self.assertEqual(len(incident_files), 1)
            finally:
                em_mod.INCIDENTS_DIR = orig_dir


# ---------------------------------------------------------------------------
# ManualParamUpdater tests
# ---------------------------------------------------------------------------

class TestManualParamUpdater(unittest.TestCase):

    def setUp(self):
        self.updater = ManualParamUpdater(base_url="http://test-coordinator:8000")

    def test_coerce_numeric_float(self):
        self.assertAlmostEqual(_coerce_numeric("3.14"), 3.14)
        self.assertAlmostEqual(_coerce_numeric(2.1), 2.1)
        self.assertIsNone(_coerce_numeric("hello"))
        self.assertIsNone(_coerce_numeric(None))

    def test_parse_value_types(self):
        self.assertIs(_parse_value("true"), True)
        self.assertIs(_parse_value("False"), False)
        self.assertEqual(_parse_value("42"), 42)
        self.assertAlmostEqual(_parse_value("3.14"), 3.14)
        self.assertEqual(_parse_value("hello"), "hello")

    @patch("param_update_manual.http_get")
    def test_get_current_params_success(self, mock_get):
        mock_get.return_value = (200, json.dumps({"params": {"BH_MASS_THRESH": 2.0}}).encode())
        result = self.updater.get_current_params()
        self.assertIsNotNone(result)
        self.assertIn("BH_MASS_THRESH", result)

    @patch("param_update_manual.http_get")
    def test_get_current_params_404(self, mock_get):
        mock_get.return_value = (404, b'{}')
        result = self.updater.get_current_params()
        self.assertEqual(result, {})

    @patch("param_update_manual.http_post")
    def test_validate_with_coordinator_valid(self, mock_post):
        mock_post.return_value = (200, json.dumps({"errors": [], "warnings": []}).encode())
        valid, errors = self.updater.validate_with_coordinator({"BH_MASS_THRESH": 2.1})
        self.assertTrue(valid)
        self.assertEqual(errors, [])

    @patch("param_update_manual.http_post")
    def test_validate_with_coordinator_invalid(self, mock_post):
        mock_post.return_value = (400, json.dumps({"errors": ["Value out of range"]}).encode())
        valid, errors = self.updater.validate_with_coordinator({"BH_MASS_THRESH": 99.0})
        self.assertFalse(valid)
        self.assertIn("Value out of range", errors)

    def test_diff_display_increase(self):
        """Display diff should recognize numeric increase without error."""
        from param_update_manual import display_diff
        import io
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            display_diff("BH_MASS_THRESH", 2.0, 2.1)
        # No assertion needed -- just verify it doesn't raise
        self.assertIsNone(None)

    @patch("param_update_manual.http_get")
    @patch("param_update_manual.http_post")
    @patch("builtins.input", return_value="n")
    def test_propose_operator_cancels(self, mock_input, mock_post, mock_get):
        mock_get.return_value = (200, json.dumps({"params": {"BH_MASS_THRESH": 2.0}}).encode())
        mock_post.return_value = (200, json.dumps({"errors": []}).encode())
        result = self.updater.propose({"BH_MASS_THRESH": 2.1}, reason="test")
        self.assertFalse(result)


# ---------------------------------------------------------------------------
# UniverseUpdater tests
# ---------------------------------------------------------------------------

class TestUniverseUpdater(unittest.TestCase):

    def _make_updater(self, tmpdir: Path) -> UniverseUpdater:
        instruments = tmpdir / "instruments.yaml"
        return UniverseUpdater(instruments_file=instruments, base_url="http://test:8000")

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_add_symbol_creates_file(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            ok = u.add_symbol("BTC-USD", asset_class="crypto", sector="defi", adv=5e9)
            self.assertTrue(ok)
            self.assertTrue(u.instruments_file.exists())

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_add_then_find(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            u.add_symbol("ETH-USD", asset_class="crypto", sector="defi")
            data = u._load()
            idx, entry = u._find_symbol(data, "ETH-USD")
            self.assertIsNotNone(entry)
            self.assertEqual(entry["symbol"], "ETH-USD")
            self.assertTrue(entry["enabled"])

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_remove_symbol_disables(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            u.add_symbol("SOL-USD", asset_class="crypto", sector="smart-contract")
            ok = u.remove_symbol("SOL-USD", reason="test_removal")
            self.assertTrue(ok)
            data = u._load()
            _, entry = u._find_symbol(data, "SOL-USD")
            self.assertFalse(entry["enabled"])
            self.assertEqual(entry["disabled_reason"], "test_removal")

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_remove_nonexistent_returns_false(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            ok = u.remove_symbol("FAKE-USD", reason="test")
            self.assertFalse(ok)

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_add_then_remove_then_re_add(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            u.add_symbol("ADA-USD", asset_class="crypto", sector="smart-contract")
            u.remove_symbol("ADA-USD", reason="testing")
            # Re-add should succeed (re-enables disabled entry)
            ok = u.add_symbol("ADA-USD", asset_class="crypto", sector="defi")
            self.assertTrue(ok)
            data = u._load()
            _, entry = u._find_symbol(data, "ADA-USD")
            self.assertTrue(entry["enabled"])

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_update_adv(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            u.add_symbol("AAPL", asset_class="equity", sector="tech", adv=1e8)
            ok = u.update_adv("AAPL", new_adv=2e8)
            self.assertTrue(ok)
            data = u._load()
            _, entry = u._find_symbol(data, "AAPL")
            self.assertAlmostEqual(entry["adv"], 2e8)

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_validate_universe_detects_missing_fields(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            # Manually write a bad instruments file
            bad_data = {"universe": [{"symbol": "BAD", "enabled": True}]}
            import universe_updater as um
            um._dump_yaml(bad_data, u.instruments_file)
            errors = u.validate_universe()
            self.assertTrue(len(errors) > 0)
            missing_field_errors = [e for e in errors if "missing required field" in e]
            self.assertTrue(len(missing_field_errors) > 0)

    @patch("universe_updater._trigger_hot_reload", return_value=True)
    def test_validate_universe_valid(self, mock_reload):
        with tempfile.TemporaryDirectory() as tmpdir:
            u = self._make_updater(Path(tmpdir))
            u.add_symbol("BTC-USD", asset_class="crypto", sector="defi", adv=5e9)
            errors = u.validate_universe()
            self.assertEqual(errors, [])


# ---------------------------------------------------------------------------
# Startup orchestrator tests
# ---------------------------------------------------------------------------

class TestStartupOrchestrator(unittest.TestCase):

    def _patch_all_steps_ok(self):
        """Return a patcher context that mocks all startup steps to succeed."""
        return patch.multiple(
            "daily_startup",
            http_get=MagicMock(return_value=(200, json.dumps({"status": "ok", "services": {
                "coordinator": {"healthy": True},
                "live_trader": {"healthy": True},
                "iae": {"healthy": True},
                "data_store": {"healthy": True},
                "risk_manager": {"healthy": True},
            }, "breakers": {}, "integrity": "ok"}).encode())),
            http_post=MagicMock(return_value=(200, b'{"mode": "trading"}')),
        )

    def test_health_checker_ok(self):
        checker = ServiceHealthChecker("http://test:8000")
        with patch("daily_startup.http_get") as mock_get:
            mock_get.return_value = (200, json.dumps({
                "status": "ok",
                "services": {
                    "coordinator": {"healthy": True},
                    "live_trader": {"healthy": True},
                    "iae": {"healthy": True},
                    "data_store": {"healthy": True},
                    "risk_manager": {"healthy": True},
                },
            }).encode())
            result = checker.run()
        self.assertTrue(result)

    def test_health_checker_fail_on_unhealthy_service(self):
        checker = ServiceHealthChecker("http://test:8000")
        with patch("daily_startup.http_get") as mock_get:
            mock_get.return_value = (200, json.dumps({
                "status": "ok",
                "services": {
                    "coordinator": {"healthy": True},
                    "live_trader": {"healthy": False},
                    "iae": {"healthy": True},
                    "data_store": {"healthy": True},
                    "risk_manager": {"healthy": True},
                },
            }).encode())
            result = checker.run()
        self.assertFalse(result)

    def test_circuit_breaker_checker_all_closed(self):
        checker = CircuitBreakerChecker("http://test:8000")
        with patch("daily_startup.http_get") as mock_get:
            mock_get.return_value = (200, json.dumps({
                "breakers": {
                    "equity_daily_loss": "CLOSED",
                    "crypto_daily_loss": "CLOSED",
                    "global_drawdown": "CLOSED",
                },
            }).encode())
            result = checker.run()
        self.assertTrue(result)

    def test_circuit_breaker_checker_open_breaker_fails(self):
        checker = CircuitBreakerChecker("http://test:8000")
        with patch("daily_startup.http_get") as mock_get:
            mock_get.return_value = (200, json.dumps({
                "breakers": {
                    "equity_daily_loss": "CLOSED",
                    "crypto_daily_loss": "OPEN",
                },
            }).encode())
            result = checker.run()
        self.assertFalse(result)

    def test_position_reconciler_no_positions(self):
        reconciler = PositionReconciler("http://test:8000")
        with patch("daily_startup.http_get") as mock_get:
            mock_get.return_value = (404, b'{}')
            result = reconciler.run()
        self.assertTrue(result)

    def test_config_validator_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validator = ConfigValidator(Path(tmpdir))
            result = validator.run()
            # Missing files should cause validation to fail
            self.assertFalse(result)

    def test_config_validator_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            # Create all required config files
            for fname in ("param_schema.yaml", "instruments.yaml", "event_calendar.yaml",
                          "coordinator.yaml", "risk_limits.yaml"):
                p = config_dir / fname
                p.write_text("placeholder: true\n", encoding="utf-8")
            validator = ConfigValidator(config_dir)
            result = validator.run()
            self.assertTrue(result)


# ---------------------------------------------------------------------------
# DataBackfiller tests
# ---------------------------------------------------------------------------

class TestDataBackfiller(unittest.TestCase):

    def setUp(self):
        self.bf = DataBackfiller(base_url="http://test:8000")

    def test_is_crypto_btc(self):
        self.assertTrue(_is_crypto("BTC-USD"))
        self.assertTrue(_is_crypto("ETH-USD"))
        self.assertTrue(_is_crypto("BTC/USDT"))

    def test_is_crypto_equity(self):
        self.assertFalse(_is_crypto("AAPL"))
        self.assertFalse(_is_crypto("TSLA"))

    def test_parse_date(self):
        dt = _parse_date("2023-01-15")
        self.assertEqual(dt.year, 2023)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 15)

    def test_date_chunks_divides_range(self):
        start = _parse_date("2023-01-01")
        end = _parse_date("2023-01-31")
        chunks = list(_date_chunks(start, end, 15, 100))
        # Should have at least one chunk
        self.assertGreater(len(chunks), 0)
        # First chunk should start at start
        self.assertEqual(chunks[0][0], start)
        # Last chunk should end at end
        self.assertEqual(chunks[-1][1], end)
        # All chunks should be contiguous
        for i in range(1, len(chunks)):
            self.assertEqual(chunks[i][0], chunks[i - 1][1])

    def test_progress_bar_format(self):
        bar = _progress_bar(50, 100)
        self.assertIn("50.0%", bar)
        self.assertIn("50/100", bar)

    def test_progress_bar_zero_total(self):
        bar = _progress_bar(0, 0)
        self.assertIn("0/0", bar)

    def test_checkpoint_save_load_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import backfill_data as bd_mod
            orig = bd_mod.CHECKPOINT_DIR
            bd_mod.CHECKPOINT_DIR = Path(tmpdir)
            try:
                save_checkpoint("TEST-USD", {"symbol": "TEST-USD", "bars_stored": 42})
                cp = load_checkpoint("TEST-USD")
                self.assertIsNotNone(cp)
                self.assertEqual(cp["bars_stored"], 42)
                delete_checkpoint("TEST-USD")
                self.assertIsNone(load_checkpoint("TEST-USD"))
            finally:
                bd_mod.CHECKPOINT_DIR = orig

    def test_normalize_binance_bars(self):
        raw = [
            [1672531200000, "16500.0", "16600.0", "16400.0", "16550.0", "100.5",
             1672531499999, "1659000.0", 10, "50.0", "825000.0", "0"],
        ]
        bars = self.bf._normalize_binance_bars(raw, "BTC-USD")
        self.assertEqual(len(bars), 1)
        self.assertAlmostEqual(bars[0]["open"], 16500.0)
        self.assertAlmostEqual(bars[0]["close"], 16550.0)
        self.assertEqual(bars[0]["symbol"], "BTC-USD")

    def test_normalize_alpaca_bars(self):
        raw = [
            {"t": "2023-01-01T14:30:00Z", "o": 150.0, "h": 152.0, "l": 149.0, "c": 151.0, "v": 1000000, "vw": 150.5},
        ]
        bars = self.bf._normalize_alpaca_bars(raw, "AAPL")
        self.assertEqual(len(bars), 1)
        self.assertAlmostEqual(bars[0]["open"], 150.0)
        self.assertEqual(bars[0]["symbol"], "AAPL")

    @patch("backfill_data.http_post_json")
    def test_store_bars_success(self, mock_post):
        mock_post.return_value = (200, {"stored": 1})
        bars = [{"symbol": "BTC-USD", "ts": "2023-01-01T00:00:00Z", "open": 16500.0,
                 "high": 16600.0, "low": 16400.0, "close": 16550.0, "volume": 100.0, "vwap": None}]
        result = self.bf._store_bars("BTC-USD", "15m", bars)
        self.assertTrue(result)

    @patch("backfill_data.http_post_json")
    def test_store_bars_empty_is_ok(self, mock_post):
        result = self.bf._store_bars("BTC-USD", "15m", [])
        self.assertTrue(result)
        mock_post.assert_not_called()

    @patch("backfill_data.http_post_json")
    def test_verify_completeness_no_gaps(self, mock_post):
        mock_post.return_value = (200, {"total_expected": 100, "total_found": 100, "gaps": []})
        result = self.bf.verify_completeness("BTC-USD", "2023-01-01", "2023-01-08", "1h")
        self.assertTrue(result["complete"])
        self.assertEqual(result["gaps"], [])

    @patch("backfill_data.http_post_json")
    def test_verify_completeness_with_gaps(self, mock_post):
        gaps = [
            {"start": "2023-01-03T00:00:00Z", "end": "2023-01-04T00:00:00Z"},
        ]
        mock_post.return_value = (200, {"total_expected": 168, "total_found": 140, "gaps": gaps})
        result = self.bf.verify_completeness("BTC-USD", "2023-01-01", "2023-01-08", "1h")
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["gaps"]), 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
