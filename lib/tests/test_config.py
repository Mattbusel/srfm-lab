"""
lib/tests/test_config.py
========================
Production test suite for the LARSA strategy configuration and hot-reload
infrastructure.

Covers:
  - LiveConfig hot-reload (valid and invalid)
  - ConfigDiff significant change detection
  - ConfigSchema validation and cross-constraints
  - InstrumentRegistry CRUD, disable/enable, CF calibration, order validation
  - SessionState pause, block, emergency mode, composite tradeable check
  - SignalRegistry IC update, ICIR computation, auto status transitions
  - TradeLog entry/exit round-trip, P&L computation, open/closed queries

Run with:
    pytest lib/tests/test_config.py -v
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Imports under test -- adjust sys.path so tests work from repo root
# ---------------------------------------------------------------------------
import sys

_REPO_ROOT = Path(__file__).parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.config_loader import (
    ConfigSchema,
    ConfigDiff,
    ParamChange,
    LiveConfig,
    ConfigValidationError,
    SCHEMA,
    _SCHEMA_LIST,
    write_default_params,
)
from lib.instrument_registry import (
    Instrument,
    CfCalibration,
    InstrumentRegistry,
    _parse_instrument,
)
from lib.session_state import (
    SessionState,
    SymbolBlock,
    reset_singleton_for_testing as ss_reset,
)
from lib.signal_registry import (
    SignalState,
    SignalRegistry,
    STATUS_ACTIVE,
    STATUS_PROBATION,
    STATUS_RETIRED,
    IC_WINDOW,
    reset_singleton_for_testing as sr_reset,
)
from lib.trade_log import (
    TradeEntry,
    TradeExit,
    TradeLog,
    _opt_float,
    _opt_int,
    reset_singleton_for_testing as tl_reset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_json(tmp_path):
    """Return a factory that writes a JSON file to a temp dir."""
    def _write(data: dict, filename: str = "live_params.json") -> Path:
        p = tmp_path / filename
        with open(p, "w") as fh:
            json.dump(data, fh)
        return p
    return _write


@pytest.fixture
def tmp_yaml(tmp_path):
    """Return a factory that writes a YAML file to a temp dir."""
    def _write(content: str, filename: str = "instruments.yaml") -> Path:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return p
    return _write


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path for a temporary SQLite database."""
    return tmp_path / "test_live.db"


@pytest.fixture
def session_state(tmp_db):
    """Fresh SessionState using a temp DB."""
    ss_reset(db_path=None)   # ensure global singleton not used
    return SessionState(db_path=tmp_db)


@pytest.fixture
def signal_registry(tmp_db):
    """Fresh SignalRegistry using a temp DB."""
    sr_reset(db_path=None)
    return SignalRegistry(db_path=tmp_db)


@pytest.fixture
def trade_log(tmp_db):
    """Fresh TradeLog using a temp DB."""
    tl_reset(db_path=None)
    return TradeLog(db_path=tmp_db)


MINIMAL_INSTRUMENTS_YAML = """\
---
instruments:
  BTC:
    name: "Bitcoin"
    asset_class: crypto
    base_currency: BTC
    quote_currency: USD
    alpaca_ticker: "BTC/USD"
    type: crypto
    tick_size: 1.0
    cf_15m: 0.005
    cf_1h:  0.015
    cf_1d:  0.050
    bh_form: 2.0
    bh_collapse: 0.992
    bh_decay: 0.924
    is_active: true
  ETH:
    name: "Ethereum"
    asset_class: crypto
    base_currency: ETH
    quote_currency: USD
    alpaca_ticker: "ETH/USD"
    type: crypto
    tick_size: 0.01
    cf_15m: 0.007
    cf_1h:  0.020
    cf_1d:  0.070
    bh_form: 2.0
    bh_collapse: 0.992
    bh_decay: 0.924
    is_active: true
  SPY:
    name: "SPDR S&P 500 ETF"
    asset_class: equity_index
    base_currency: USD
    quote_currency: USD
    alpaca_ticker: SPY
    type: stock
    tick_size: 0.01
    cf_15m: 0.00030
    cf_1h:  0.00100
    cf_1d:  0.00500
    bh_form: 1.5
    bh_collapse: 0.992
    bh_decay: 0.924
    options_overlay: true
    is_active: true
  DISABLED_COIN:
    name: "Test Disabled"
    asset_class: crypto
    base_currency: TEST
    quote_currency: USD
    alpaca_ticker: null
    type: crypto
    tick_size: 0.001
    cf_15m: 0.010
    cf_1h:  0.030
    cf_1d:  0.100
    bh_form: 2.0
    bh_collapse: 0.992
    bh_decay: 0.924
    is_active: false
"""


@pytest.fixture
def instrument_registry(tmp_yaml):
    yaml_path = tmp_yaml(MINIMAL_INSTRUMENTS_YAML)
    return InstrumentRegistry(yaml_path=yaml_path, watch=False)


# ===========================================================================
# ConfigSchema tests
# ===========================================================================

class TestConfigSchema:
    def test_schema_is_populated(self):
        assert len(SCHEMA) > 20, "schema should have many entries"

    def test_schema_hot_reloadable_params(self):
        hot = [s for s in _SCHEMA_LIST if s.hot_reloadable]
        non = [s for s in _SCHEMA_LIST if not s.hot_reloadable]
        assert len(hot) > 0
        assert len(non) > 0

    def test_schema_bh_mass_thresh_range(self):
        s = SCHEMA["BH_MASS_THRESH"]
        assert s.min_val == 1.0
        assert s.max_val == 4.0
        assert s.default == 1.92
        assert s.hot_reloadable is True

    def test_schema_db_path_not_hot_reloadable(self):
        s = SCHEMA["DB_PATH"]
        assert s.hot_reloadable is False

    def test_schema_instruments_list_not_in_schema(self):
        # INSTRUMENTS dict is not in schema -- it belongs to instrument_registry
        assert "INSTRUMENTS" not in SCHEMA

    def test_schema_all_types_valid(self):
        valid_types = {"float", "int", "bool", "list_int", "str"}
        for s in _SCHEMA_LIST:
            assert s.type_str in valid_types, f"{s.param_name} has invalid type {s.type_str}"

    def test_schema_float_params_have_ranges(self):
        for s in _SCHEMA_LIST:
            if s.type_str == "float":
                assert s.min_val is not None, f"{s.param_name} float missing min_val"
                assert s.max_val is not None, f"{s.param_name} float missing max_val"
                assert s.min_val < s.max_val

    def test_schema_credentials_not_hot_reloadable(self):
        for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER"):
            assert SCHEMA[name].hot_reloadable is False


# ===========================================================================
# ConfigDiff tests
# ===========================================================================

class TestConfigDiff:
    def test_no_changes(self):
        cfg = {"BH_MASS_THRESH": 1.92, "CF_BULL_THRESH": 1.2}
        changes = ConfigDiff.compute(cfg, cfg)
        assert changes == []

    def test_single_change(self):
        old = {"BH_MASS_THRESH": 1.92}
        new = {"BH_MASS_THRESH": 2.10}
        changes = ConfigDiff.compute(old, new)
        assert len(changes) == 1
        assert changes[0].name == "BH_MASS_THRESH"
        assert changes[0].old_value == 1.92
        assert changes[0].new_value == 2.10

    def test_config_diff_significant_change(self):
        """A > 10% change should be flagged as significant."""
        old = {"BH_MASS_THRESH": 1.0}
        new = {"BH_MASS_THRESH": 1.15}   # 15% increase
        changes = ConfigDiff.compute(old, new)
        assert len(changes) == 1
        c = changes[0]
        assert c.is_significant is True
        assert c.change_pct is not None
        assert abs(c.change_pct - 15.0) < 0.1

    def test_config_diff_non_significant_change(self):
        """A <= 10% change should NOT be flagged."""
        old = {"CF_BULL_THRESH": 1.2}
        new = {"CF_BULL_THRESH": 1.25}   # ~4% increase
        changes = ConfigDiff.compute(old, new)
        assert len(changes) == 1
        assert changes[0].is_significant is False

    def test_config_diff_bool_change(self):
        old = {"RL_EXIT_ACTIVE": True}
        new = {"RL_EXIT_ACTIVE": False}
        changes = ConfigDiff.compute(old, new)
        assert len(changes) == 1
        assert changes[0].change_pct is None   # bool has no pct

    def test_config_diff_added_key(self):
        old = {}
        new = {"ML_SIGNAL_BOOST": 1.2}
        changes = ConfigDiff.compute(old, new)
        assert len(changes) == 1
        assert changes[0].old_value is None
        assert changes[0].new_value == 1.2

    def test_config_diff_removed_key(self):
        old = {"ML_SIGNAL_BOOST": 1.2}
        new = {}
        changes = ConfigDiff.compute(old, new)
        assert len(changes) == 1
        assert changes[0].new_value is None

    def test_config_diff_multiple_changes(self):
        old = {"A": 1.0, "B": 2.0, "C": True}
        new = {"A": 1.0, "B": 2.5, "C": False}
        changes = ConfigDiff.compute(old, new)
        assert len(changes) == 2
        names = {c.name for c in changes}
        assert names == {"B", "C"}

    def test_param_change_str(self):
        c = ParamChange("X", 1.0, 2.0, 100.0, True)
        s = str(c)
        assert "X" in s
        assert "SIGNIFICANT" in s

    def test_change_pct_zero_denominator(self):
        old = {"X": 0.0}
        new = {"X": 1.0}
        changes = ConfigDiff.compute(old, new)
        assert changes[0].change_pct is None   # division by zero -> None


# ===========================================================================
# LiveConfig tests
# ===========================================================================

class TestLiveConfig:
    def test_live_config_defaults_on_missing_file(self, tmp_path):
        cfg = LiveConfig(config_path=tmp_path / "nonexistent.json", watch=False)
        assert cfg.get("BH_MASS_THRESH") == 1.92
        assert cfg.get("CF_BULL_THRESH") == 1.2
        cfg.stop()

    def test_live_config_loads_from_file(self, tmp_json):
        # Provide both CF params to satisfy CF_BEAR_THRESH >= CF_BULL_THRESH constraint
        params = {"BH_MASS_THRESH": 2.5, "CF_BULL_THRESH": 1.3, "CF_BEAR_THRESH": 1.5}
        p = tmp_json(params)
        cfg = LiveConfig(config_path=p, watch=False)
        assert cfg.get("BH_MASS_THRESH") == 2.5
        assert cfg.get("CF_BULL_THRESH") == 1.3
        cfg.stop()

    def test_live_config_hot_reload_valid(self, tmp_json):
        """Hot reload with valid data updates the config."""
        params = {"BH_MASS_THRESH": 1.92}
        p = tmp_json(params)
        cfg = LiveConfig(config_path=p, watch=False)
        assert cfg.get("BH_MASS_THRESH") == 1.92

        # Write new value
        with open(p, "w") as fh:
            json.dump({"BH_MASS_THRESH": 2.50}, fh)

        changed = cfg.reload()
        assert changed is True
        assert cfg.get("BH_MASS_THRESH") == 2.50
        cfg.stop()

    def test_live_config_rejects_invalid_range(self, tmp_json):
        """Reload with out-of-range value keeps the current config."""
        params = {"BH_MASS_THRESH": 1.92}
        p = tmp_json(params)
        cfg = LiveConfig(config_path=p, watch=False)

        # Write invalid value (min is 1.0, max is 4.0 -- 99.0 is out of range)
        with open(p, "w") as fh:
            json.dump({"BH_MASS_THRESH": 99.0}, fh)

        changed = cfg.reload()
        assert changed is False
        assert cfg.get("BH_MASS_THRESH") == 1.92   # unchanged
        cfg.stop()

    def test_live_config_rejects_invalid_type(self, tmp_json):
        """Reload with wrong type (string for float) is rejected."""
        p = tmp_json({})
        cfg = LiveConfig(config_path=p, watch=False)
        with open(p, "w") as fh:
            json.dump({"BH_MASS_THRESH": "not_a_number"}, fh)
        changed = cfg.reload()
        assert changed is False
        cfg.stop()

    def test_live_config_cross_constraint_violation(self, tmp_json):
        """Reload that violates cross-constraint is rejected."""
        p = tmp_json({})
        cfg = LiveConfig(config_path=p, watch=False)
        # CF_BEAR_THRESH must be >= CF_BULL_THRESH
        with open(p, "w") as fh:
            json.dump({"CF_BULL_THRESH": 3.0, "CF_BEAR_THRESH": 1.0}, fh)
        changed = cfg.reload()
        assert changed is False
        cfg.stop()

    def test_live_config_garch_stationarity_constraint(self, tmp_json):
        """GARCH_ALPHA + GARCH_BETA >= 1 is rejected."""
        p = tmp_json({})
        cfg = LiveConfig(config_path=p, watch=False)
        with open(p, "w") as fh:
            json.dump({"GARCH_ALPHA": 0.20, "GARCH_BETA": 0.90}, fh)
        changed = cfg.reload()
        assert changed is False
        cfg.stop()

    def test_live_config_callback_fired_on_reload(self, tmp_json):
        """Registered callback is called with change list on successful reload."""
        # Start with valid constraint-satisfying values
        p = tmp_json({"CF_BULL_THRESH": 1.2, "CF_BEAR_THRESH": 1.4})
        cfg = LiveConfig(config_path=p, watch=False)

        received: list = []
        cfg.register_callback(received.append)

        # New value: CF_BULL_THRESH=1.3 still < CF_BEAR_THRESH default (1.4) -> valid
        with open(p, "w") as fh:
            json.dump({"CF_BULL_THRESH": 1.3, "CF_BEAR_THRESH": 1.5}, fh)
        cfg.reload()

        assert len(received) == 1
        changes = received[0]
        names = {c.name for c in changes}
        assert "CF_BULL_THRESH" in names or "CF_BEAR_THRESH" in names
        cfg.stop()

    def test_live_config_callback_not_fired_when_no_change(self, tmp_json):
        p = tmp_json({"CF_BULL_THRESH": 1.2})
        cfg = LiveConfig(config_path=p, watch=False)
        received: list = []
        cfg.register_callback(received.append)
        # Same value -- no change
        with open(p, "w") as fh:
            json.dump({"CF_BULL_THRESH": 1.2}, fh)
        cfg.reload()
        assert len(received) == 0
        cfg.stop()

    def test_live_config_get_snapshot(self, tmp_json):
        p = tmp_json({"BH_MASS_THRESH": 2.0})
        cfg = LiveConfig(config_path=p, watch=False)
        snap = cfg.get_snapshot()
        assert isinstance(snap, dict)
        assert snap["BH_MASS_THRESH"] == 2.0
        # Snapshot is a copy -- mutating it doesn't affect config
        snap["BH_MASS_THRESH"] = 99.0
        assert cfg.get("BH_MASS_THRESH") == 2.0
        cfg.stop()

    def test_live_config_unknown_key_ignored(self, tmp_json):
        p = tmp_json({"UNKNOWN_FUTURE_PARAM": 42.0, "BH_MASS_THRESH": 2.1})
        cfg = LiveConfig(config_path=p, watch=False)
        assert cfg.get("BH_MASS_THRESH") == 2.1
        assert cfg.get("UNKNOWN_FUTURE_PARAM") is None
        cfg.stop()

    def test_live_config_bool_coercion(self, tmp_json):
        p = tmp_json({"RL_EXIT_ACTIVE": "true"})
        cfg = LiveConfig(config_path=p, watch=False)
        assert cfg.get("RL_EXIT_ACTIVE") is True
        cfg.stop()

    def test_live_config_list_int_coercion(self, tmp_json):
        p = tmp_json({"BLOCKED_HOURS": [1, 13, 14]})
        cfg = LiveConfig(config_path=p, watch=False)
        val = cfg.get("BLOCKED_HOURS")
        assert val == [1, 13, 14]
        cfg.stop()

    def test_live_config_hot_reloadable_param_list(self, tmp_json):
        p = tmp_json({})
        cfg = LiveConfig(config_path=p, watch=False)
        hot = cfg.get_hot_reloadable_params()
        assert "BH_MASS_THRESH" in hot
        assert "CF_BULL_THRESH" in hot
        assert "DB_PATH" not in hot
        assert "ALPACA_API_KEY" not in hot
        cfg.stop()

    def test_live_config_non_hot_reloadable_param_list(self, tmp_json):
        p = tmp_json({})
        cfg = LiveConfig(config_path=p, watch=False)
        non = cfg.get_non_hot_reloadable_params()
        assert "DB_PATH" in non
        assert "ALPACA_API_KEY" in non
        assert "BH_MASS_THRESH" not in non
        cfg.stop()

    def test_live_config_load_count_increments(self, tmp_json):
        p = tmp_json({"BH_MASS_THRESH": 1.92})
        cfg = LiveConfig(config_path=p, watch=False)
        assert cfg.load_count() == 0
        with open(p, "w") as fh:
            json.dump({"BH_MASS_THRESH": 2.0}, fh)
        cfg.reload()
        assert cfg.load_count() == 1
        cfg.stop()

    def test_write_default_params(self, tmp_path):
        p = tmp_path / "defaults.json"
        write_default_params(path=p)
        assert p.exists()
        with open(p) as fh:
            data = json.load(fh)
        assert "BH_MASS_THRESH" in data
        assert "_written_at" in data

    def test_live_config_thread_safety(self, tmp_json):
        """Concurrent reads from multiple threads should not corrupt state."""
        p = tmp_json({"BH_MASS_THRESH": 1.92})
        cfg = LiveConfig(config_path=p, watch=False)
        errors: list[Exception] = []

        def reader():
            for _ in range(200):
                try:
                    v = cfg.get("BH_MASS_THRESH")
                    assert isinstance(v, float)
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        cfg.stop()


# ===========================================================================
# InstrumentRegistry tests
# ===========================================================================

class TestInstrumentRegistry:
    def test_load_instruments(self, instrument_registry):
        assert instrument_registry.get("BTC") is not None
        assert instrument_registry.get("ETH") is not None
        assert instrument_registry.get("SPY") is not None

    def test_instrument_registry_get_crypto(self, instrument_registry):
        """get_crypto() returns active crypto instruments only."""
        cryptos = instrument_registry.get_crypto()
        symbols = {i.symbol for i in cryptos}
        assert "BTC" in symbols
        assert "ETH" in symbols
        assert "SPY" not in symbols      # SPY is equity_index
        assert "DISABLED_COIN" not in symbols  # inactive

    def test_instrument_registry_get_active(self, instrument_registry):
        active = instrument_registry.get_active()
        symbols = {i.symbol for i in active}
        assert "BTC" in symbols
        assert "DISABLED_COIN" not in symbols

    def test_instrument_registry_get_all(self, instrument_registry):
        all_inst = instrument_registry.get_all()
        symbols = {i.symbol for i in all_inst}
        assert "DISABLED_COIN" in symbols   # get_all includes inactive

    def test_instrument_registry_get_equity(self, instrument_registry):
        equities = instrument_registry.get_equity()
        symbols = {i.symbol for i in equities}
        assert "SPY" in symbols
        assert "BTC" not in symbols

    def test_instrument_unknown_raises(self, instrument_registry):
        with pytest.raises(KeyError):
            instrument_registry.get("UNKNOWN_XYZ")

    def test_get_or_none_returns_none(self, instrument_registry):
        assert instrument_registry.get_or_none("UNKNOWN_XYZ") is None

    def test_instrument_registry_disable_reenable(self, instrument_registry):
        """Disabling then re-enabling a symbol works correctly."""
        reg = instrument_registry
        reg.disable_instrument("BTC", reason="test disable")
        btc = reg.get("BTC")
        assert btc.is_active is False
        assert btc.disabled_reason == "test disable"

        active_symbols = reg.active_symbols()
        assert "BTC" not in active_symbols

        reg.enable_instrument("BTC")
        btc = reg.get("BTC")
        assert btc.is_active is True
        assert btc.disabled_reason is None

        active_symbols = reg.active_symbols()
        assert "BTC" in active_symbols

    def test_disable_unknown_raises(self, instrument_registry):
        with pytest.raises(KeyError):
            instrument_registry.disable_instrument("NONEXISTENT")

    def test_enable_unknown_raises(self, instrument_registry):
        with pytest.raises(KeyError):
            instrument_registry.enable_instrument("NONEXISTENT")

    def test_get_cf_calibration(self, instrument_registry):
        cf = instrument_registry.get_cf_calibration("BTC")
        assert "cf_15m" in cf
        assert "cf_1h" in cf
        assert "bh_form" in cf
        assert cf["bh_form"] == 2.0
        assert cf["bh_collapse"] == 0.992

    def test_validate_order_valid(self, instrument_registry):
        ok, msg = instrument_registry.validate_order("BTC", qty=0.1, price=65000.0)
        assert ok is True
        assert msg == ""

    def test_validate_order_notional_too_large(self, instrument_registry):
        # BTC default max_position_usd is 500_000
        ok, msg = instrument_registry.validate_order("BTC", qty=100.0, price=65000.0)
        assert ok is False
        assert "max_position_usd" in msg

    def test_validate_order_zero_price(self, instrument_registry):
        ok, msg = instrument_registry.validate_order("BTC", qty=0.1, price=0.0)
        assert ok is False

    def test_validate_order_zero_qty(self, instrument_registry):
        ok, msg = instrument_registry.validate_order("BTC", qty=0.0, price=65000.0)
        assert ok is False

    def test_validate_order_disabled_symbol(self, instrument_registry):
        ok, msg = instrument_registry.validate_order("DISABLED_COIN", qty=1.0, price=1.0)
        assert ok is False
        assert "disabled" in msg.lower()

    def test_validate_order_unknown_symbol(self, instrument_registry):
        ok, msg = instrument_registry.validate_order("UNKNOWN", qty=1.0, price=100.0)
        assert ok is False

    def test_add_instrument(self, instrument_registry):
        new_inst = Instrument(
            symbol          = "SOL",
            name            = "Solana",
            asset_class     = "crypto",
            base_currency   = "SOL",
            quote_currency  = "USD",
            alpaca_ticker   = "SOL/USD",
            instrument_type = "crypto",
            tick_size       = 0.01,
            is_active       = True,
        )
        instrument_registry.add_instrument(new_inst)
        assert instrument_registry.get("SOL") is not None
        assert instrument_registry.get("SOL").name == "Solana"

    def test_symbols_list_sorted(self, instrument_registry):
        syms = instrument_registry.symbols()
        assert syms == sorted(syms)

    def test_get_by_class(self, instrument_registry):
        equity_index = instrument_registry.get_by_class("equity_index")
        assert any(i.symbol == "SPY" for i in equity_index)

    def test_options_overlay_flag(self, instrument_registry):
        spy = instrument_registry.get("SPY")
        assert spy.options_overlay is True
        btc = instrument_registry.get("BTC")
        assert btc.options_overlay is False

    def test_disabled_log_tracks_events(self, instrument_registry):
        instrument_registry.disable_instrument("ETH", reason="volatility halt")
        log = instrument_registry.get_disabled_log()
        assert any(e["symbol"] == "ETH" and e["action"] == "disable" for e in log)

    def test_hot_reload_preserves_disabled_state(self, tmp_yaml):
        """Hot reload from YAML preserves runtime-disabled symbols."""
        yaml_path = tmp_yaml(MINIMAL_INSTRUMENTS_YAML)
        reg = InstrumentRegistry(yaml_path=yaml_path, watch=False)
        reg.disable_instrument("BTC", reason="halted")
        reg.hot_reload()
        btc = reg.get("BTC")
        assert btc.is_active is False
        assert btc.disabled_reason == "halted"

    def test_parse_instrument_cf_4h_derived(self):
        """cf_4h is derived from cf_1h * 1.5 when not in YAML."""
        data = {
            "name": "Test",
            "asset_class": "crypto",
            "base_currency": "T",
            "quote_currency": "USD",
            "alpaca_ticker": "T/USD",
            "type": "crypto",
            "tick_size": 0.01,
            "cf_15m": 0.005,
            "cf_1h": 0.015,
            "cf_1d": 0.050,
            "bh_form": 2.0,
            "is_active": True,
        }
        inst = _parse_instrument("TEST", data)
        assert abs(inst.cf_calibration.cf_4h - 0.015 * 1.5) < 1e-9


# ===========================================================================
# SessionState tests
# ===========================================================================

class TestSessionState:
    def test_trading_enabled_by_default(self, session_state):
        assert session_state.is_trading_enabled() is True

    def test_disable_trading(self, session_state):
        session_state.disable_trading(reason="test")
        assert session_state.is_trading_enabled() is False

    def test_enable_trading(self, session_state):
        session_state.disable_trading()
        session_state.enable_trading()
        assert session_state.is_trading_enabled() is True

    def test_session_state_pause_blocks_entry(self, session_state):
        """After pause(), is_tradeable() returns False."""
        assert session_state.is_tradeable("BTC") is True
        session_state.pause(minutes=10)
        assert session_state.is_tradeable("BTC") is False

    def test_session_state_pause_auto_expires(self, session_state):
        """A pause with a past expiry auto-clears."""
        from unittest.mock import patch
        future_time = datetime.now(timezone.utc) + timedelta(minutes=100)
        with patch("lib.session_state.datetime") as mock_dt:
            mock_dt.now.return_value = future_time
            mock_dt.fromisoformat = datetime.fromisoformat
            # Expire the pause
            result = session_state.is_paused()
        # After the patched "now" is past expiry the pause should have cleared
        # We use a direct approach: set a very short pause and wait
        session_state.resume()   # clean state
        session_state._paused_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert session_state.is_paused() is False   # auto-expired

    def test_session_state_resume_clears_pause(self, session_state):
        session_state.pause(minutes=60)
        assert session_state.is_paused() is True
        session_state.resume()
        assert session_state.is_paused() is False

    def test_pause_duration(self, session_state):
        before = datetime.now(timezone.utc)
        session_state.pause(minutes=30)
        expiry = session_state.paused_until()
        assert expiry is not None
        delta = expiry - before
        assert 29 * 60 < delta.total_seconds() < 31 * 60

    def test_block_symbol(self, session_state):
        session_state.block_symbol("TSLA", reason="flash crash")
        assert session_state.is_symbol_blocked("TSLA") is True

    def test_unblock_symbol(self, session_state):
        session_state.block_symbol("TSLA")
        session_state.unblock_symbol("TSLA")
        assert session_state.is_symbol_blocked("TSLA") is False

    def test_block_symbol_with_duration(self, session_state):
        session_state.block_symbol("NVDA", duration_hours=2)
        assert session_state.is_symbol_blocked("NVDA") is True

    def test_block_expires_automatically(self, session_state):
        """Blocks with past expiry are auto-removed on check."""
        now = datetime.now(timezone.utc)
        session_state._symbol_blocks["AAPL"] = SymbolBlock(
            symbol     = "AAPL",
            blocked_at = now - timedelta(hours=3),
            expires_at = now - timedelta(hours=1),
            reason     = "expired",
        )
        assert session_state.is_symbol_blocked("AAPL") is False
        assert "AAPL" not in session_state._symbol_blocks

    def test_is_tradeable_blocked_symbol(self, session_state):
        session_state.block_symbol("BTC")
        assert session_state.is_tradeable("BTC") is False

    def test_is_tradeable_when_trading_disabled(self, session_state):
        session_state.disable_trading()
        assert session_state.is_tradeable("BTC") is False

    def test_session_state_emergency_mode(self, session_state):
        """Emergency mode disables all trading and is_tradeable returns False."""
        session_state.set_emergency_mode(reason="liquidation breach")
        assert session_state.is_emergency() is True
        assert session_state.is_tradeable("BTC") is False
        assert session_state.is_tradeable("SPY") is False
        assert session_state.is_trading_enabled() is False

    def test_emergency_mode_cleared_by_enable_trading(self, session_state):
        session_state.set_emergency_mode()
        assert session_state.is_emergency() is True
        session_state.enable_trading()
        assert session_state.is_emergency() is False
        assert session_state.is_trading_enabled() is True

    def test_tradeable_reason_ok(self, session_state):
        assert session_state.tradeable_reason("BTC") == "OK"

    def test_tradeable_reason_emergency(self, session_state):
        session_state.set_emergency_mode()
        reason = session_state.tradeable_reason("BTC")
        assert "emergency" in reason.lower()

    def test_tradeable_reason_paused(self, session_state):
        session_state.pause(minutes=10)
        reason = session_state.tradeable_reason("BTC")
        assert "paused" in reason.lower()

    def test_tradeable_reason_blocked(self, session_state):
        session_state.block_symbol("BTC", reason="test")
        reason = session_state.tradeable_reason("BTC")
        assert "block" in reason.lower()

    def test_snapshot_structure(self, session_state):
        snap = session_state.snapshot()
        assert "trading_enabled" in snap
        assert "emergency_mode" in snap
        assert "paused_until" in snap
        assert "blocked_symbols" in snap
        assert "snapshot_at" in snap

    def test_get_recent_events_returns_list(self, session_state):
        session_state.disable_trading(reason="test event")
        events = session_state.get_recent_events(limit=10)
        assert isinstance(events, list)
        assert len(events) >= 1

    def test_thread_safety_concurrent_blocks(self, session_state):
        """Multiple threads blocking/unblocking different symbols is safe."""
        errors: list[Exception] = []
        symbols = [f"SYM{i}" for i in range(20)]

        def blocker(sym):
            try:
                session_state.block_symbol(sym)
                time.sleep(0.001)
                session_state.unblock_symbol(sym)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=blocker, args=(s,)) for s in symbols]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_get_blocked_symbols_snapshot(self, session_state):
        session_state.block_symbol("ETH", reason="stress")
        session_state.block_symbol("BTC", reason="circuit")
        blocked = session_state.get_blocked_symbols()
        assert "ETH" in blocked
        assert "BTC" in blocked


# ===========================================================================
# SignalRegistry tests
# ===========================================================================

class TestSignalRegistry:
    def test_all_signals_initialized(self, signal_registry):
        for name in ["BH_MASS_15m", "BH_MASS_1h", "CF_CROSS", "ML_SIGNAL", "RL_EXIT"]:
            sig = signal_registry.get(name)
            assert sig is not None
            assert sig.status == STATUS_ACTIVE

    def test_unknown_signal_raises(self, signal_registry):
        with pytest.raises(KeyError):
            signal_registry.get("NONEXISTENT_SIG")

    def test_signal_registry_ic_update(self, signal_registry):
        """update_ic stores the IC and updates rolling mean."""
        signal_registry.update_ic("BH_MASS_15m", 0.045)
        sig = signal_registry.get("BH_MASS_15m")
        assert sig.last_ic == 0.045
        assert sig.ic_rolling_30d == 0.045   # single obs mean

    def test_signal_ic_rolling_mean(self, signal_registry):
        """Rolling mean is correctly computed over multiple updates."""
        values = [0.02, 0.04, 0.06, 0.08, 0.10]
        for v in values:
            signal_registry.update_ic("CF_CROSS", v)
        sig = signal_registry.get("CF_CROSS")
        expected_mean = sum(values) / len(values)
        assert abs(sig.ic_rolling_30d - expected_mean) < 1e-9

    def test_signal_icir_computation(self, signal_registry):
        """ICIR = mean / std of IC history."""
        import math
        values = [0.03, 0.04, 0.05, 0.03, 0.04, 0.05]
        for v in values:
            signal_registry.update_ic("GARCH_VOL", v)
        sig = signal_registry.get("GARCH_VOL")
        assert sig.icir_30d is not None
        n     = len(values)
        mean  = sum(values) / n
        var   = sum((v - mean) ** 2 for v in values) / (n - 1)
        std   = math.sqrt(var)
        expected_icir = mean / std
        assert abs(sig.icir_30d - expected_icir) < 1e-6

    def test_signal_probation_on_low_icir(self, signal_registry):
        """Signal enters PROBATION when ICIR drops below threshold."""
        from lib.signal_registry import ICIR_WARN_THRESHOLD
        # Feed negative IC values -- low enough to trigger probation
        for _ in range(10):
            signal_registry.update_ic("ML_SIGNAL", -0.10)
        sig = signal_registry.get("ML_SIGNAL")
        assert sig.status in (STATUS_PROBATION, STATUS_RETIRED)

    def test_signal_retirement_on_negative_rolling_ic(self, signal_registry):
        """Signal retires when rolling IC mean <= 0."""
        # First push to probation, then continue with negative IC
        for _ in range(5):
            signal_registry.update_ic("GRANGER_BTC", -0.05)
        # Force to probation
        signal_registry.set_signal_status("GRANGER_BTC", STATUS_PROBATION, "test")
        signal_registry.clear_manual_override("GRANGER_BTC")
        for _ in range(10):
            signal_registry.update_ic("GRANGER_BTC", -0.02)
        sig = signal_registry.get("GRANGER_BTC")
        assert sig.status == STATUS_RETIRED
        assert sig.is_active is False

    def test_set_signal_status_manual_override(self, signal_registry):
        signal_registry.set_signal_status("RL_EXIT", STATUS_PROBATION, "manual test")
        sig = signal_registry.get("RL_EXIT")
        assert sig.status == STATUS_PROBATION
        assert sig.manual_override is True

    def test_clear_manual_override(self, signal_registry):
        signal_registry.set_signal_status("NAV_OMEGA", STATUS_PROBATION)
        signal_registry.clear_manual_override("NAV_OMEGA")
        sig = signal_registry.get("NAV_OMEGA")
        assert sig.manual_override is False

    def test_get_active_signals_excludes_retired(self, signal_registry):
        signal_registry.set_signal_status("EVENT_CALENDAR", STATUS_RETIRED)
        active = signal_registry.get_active_signals()
        assert "EVENT_CALENDAR" not in active

    def test_get_fired_signals_excludes_probation(self, signal_registry):
        signal_registry.set_signal_status("HURST_REGIME", STATUS_PROBATION)
        fired = signal_registry.get_fired_signals()
        assert "HURST_REGIME" not in fired

    def test_get_report_structure(self, signal_registry):
        report = signal_registry.get_report()
        assert "signals" in report
        assert "total_signals" in report
        assert "active_count" in report
        assert report["total_signals"] == len(report["signals"])

    def test_ic_window_caps_at_30(self, signal_registry):
        """IC deque has max length IC_WINDOW."""
        for i in range(IC_WINDOW + 20):
            signal_registry.update_ic("BH_MASS_4h", float(i) * 0.001)
        history = signal_registry.get_ic_history("BH_MASS_4h")
        assert len(history) == IC_WINDOW

    def test_get_icir_by_signal(self, signal_registry):
        signal_registry.update_ic("NAV_GEODESIC", 0.05)
        signal_registry.update_ic("NAV_GEODESIC", 0.06)
        icirs = signal_registry.get_icir_by_signal()
        assert "NAV_GEODESIC" in icirs
        assert "BH_MASS_1h" in icirs

    def test_thread_safe_ic_updates(self, signal_registry):
        """Concurrent IC updates to multiple signals don't corrupt state."""
        errors: list[Exception] = []

        def updater(sig_name, n):
            try:
                for i in range(n):
                    signal_registry.update_ic(sig_name, float(i) * 0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=updater, args=(name, 50))
            for name in ["BH_MASS_15m", "BH_MASS_1h", "CF_CROSS", "ML_SIGNAL"]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"


# ===========================================================================
# TradeLog tests
# ===========================================================================

def _make_context(**kwargs) -> dict[str, Any]:
    """Build a minimal trade context dict."""
    defaults = {
        "bh_mass_15m":  1.95,
        "bh_mass_1h":   2.10,
        "bh_mass_4h":   2.30,
        "bh_dir_15m":   1,
        "tf_score":     5,
        "garch_vol":    0.85,
        "garch_vol_scale": 1.05,
        "ml_signal":    0.35,
        "regime":       "trending",
        "corr_mode":    "normal",
        "portfolio_nav": 100_000.0,
        "param_snapshot": {"BH_MASS_THRESH": 1.92, "CF_BULL_THRESH": 1.2},
    }
    defaults.update(kwargs)
    return defaults


class TestTradeLog:
    def test_trade_log_entry_exit_roundtrip(self, trade_log):
        """log_entry then log_exit produces a complete P&L record."""
        tid = trade_log.log_entry("BTC", price=65000.0, qty=0.1, side="long", context=_make_context())
        assert tid is not None

        open_pos = trade_log.get_open_positions()
        assert any(e.trade_id == tid for e in open_pos)

        exit_rec = trade_log.log_exit(tid, price=66000.0, reason="BH_COLLAPSE", bars_held=12)
        assert exit_rec is not None
        assert exit_rec.trade_id == tid
        assert exit_rec.pnl_abs is not None
        # long 0.1 BTC, entry 65000, exit 66000 -> pnl = (66000-65000)*0.1 = 100
        assert abs(exit_rec.pnl_abs - 100.0) < 0.01

    def test_trade_log_pnl_pct_computed(self, trade_log):
        tid = trade_log.log_entry("ETH", price=3000.0, qty=1.0, side="long", context=_make_context())
        exit_rec = trade_log.log_exit(tid, price=3150.0, reason="RL_EXIT", bars_held=5)
        assert exit_rec.pnl_pct is not None
        assert abs(exit_rec.pnl_pct - 5.0) < 0.01  # 150/3000 * 100 = 5%

    def test_trade_log_short_pnl(self, trade_log):
        """Short trade: profit when exit < entry."""
        tid      = trade_log.log_entry("SPY", price=500.0, qty=10.0, side="short", context=_make_context())
        exit_rec = trade_log.log_exit(tid, price=480.0, reason="BH_COLLAPSE", bars_held=8)
        # short 10 SPY, entry 500, exit 480 -> pnl = -(480-500)*10 = +200
        assert exit_rec.pnl_abs is not None
        assert exit_rec.pnl_abs > 0

    def test_position_removed_from_open_after_exit(self, trade_log):
        tid = trade_log.log_entry("BTC", price=65000.0, qty=0.01, side="long", context=_make_context())
        trade_log.log_exit(tid, price=65500.0, reason="STOP_LOSS")
        open_pos = trade_log.get_open_positions()
        assert not any(e.trade_id == tid for e in open_pos)

    def test_get_open_by_symbol(self, trade_log):
        trade_log.log_entry("BTC", price=65000.0, qty=0.01, side="long", context=_make_context())
        trade_log.log_entry("BTC", price=65500.0, qty=0.02, side="long", context=_make_context())
        trade_log.log_entry("ETH", price=3000.0,  qty=1.0,  side="long", context=_make_context())
        btc_open = trade_log.get_open_by_symbol("BTC")
        assert len(btc_open) == 2
        eth_open = trade_log.get_open_by_symbol("ETH")
        assert len(eth_open) == 1

    def test_get_closed_trades_returns_pairs(self, trade_log):
        tid = trade_log.log_entry("BTC", price=65000.0, qty=0.05, side="long", context=_make_context())
        trade_log.log_exit(tid, price=66000.0, reason="TIMEOUT")
        closed = trade_log.get_closed_trades()
        assert len(closed) >= 1
        entry, exit_rec = closed[0]
        assert isinstance(entry, TradeEntry)
        assert isinstance(exit_rec, TradeExit)

    def test_get_entry_context_in_memory(self, trade_log):
        ctx = _make_context(ml_signal=0.42, regime="ranging")
        tid = trade_log.log_entry("BTC", price=65000.0, qty=0.01, side="long", context=ctx)
        result = trade_log.get_entry_context(tid)
        assert result["trade_id"] == tid
        assert result["symbol"] == "BTC"

    def test_get_entry_context_from_db(self, trade_log):
        """After moving to closed, context is retrievable via DB fallback."""
        ctx = _make_context(ml_signal=0.55)
        tid = trade_log.log_entry("ETH", price=3000.0, qty=1.0, side="long", context=ctx)
        trade_log.log_exit(tid, price=3100.0, reason="MANUAL")
        # Force DB lookup by clearing in-memory closed dict
        with trade_log._lock:
            trade_log._closed.clear()
        result = trade_log.get_entry_context(tid)
        assert result.get("trade_id") == tid

    def test_log_exit_unknown_trade_id(self, trade_log):
        result = trade_log.log_exit("NONEXISTENT_ID", price=100.0, reason="TEST")
        assert result is None

    def test_open_trade_ids(self, trade_log):
        trade_log.log_entry("BTC", price=65000.0, qty=0.01, side="long", context=_make_context())
        trade_log.log_entry("ETH", price=3000.0,  qty=1.0,  side="long", context=_make_context())
        ids = trade_log.open_trade_ids()
        assert len(ids) == 2
        assert ids == sorted(ids)

    def test_pnl_summary(self, trade_log):
        for i in range(5):
            tid = trade_log.log_entry("BTC", price=65000.0, qty=0.01, side="long", context=_make_context())
            pnl = (i - 2) * 50.0   # mix of wins and losses
            trade_log.log_exit(tid, price=65000.0 + (i - 2) * 5000.0, reason="TEST",
                                context={"notes": f"trade {i}"})
        summary = trade_log.get_pnl_summary()
        assert "n_trades" in summary
        assert summary["n_trades"] == 5

    def test_context_param_snapshot_persisted(self, trade_log):
        snap = {"BH_MASS_THRESH": 2.0, "CF_BULL_THRESH": 1.5}
        ctx  = _make_context(param_snapshot=snap)
        tid  = trade_log.log_entry("BTC", price=65000.0, qty=0.01, side="long", context=ctx)
        entry = trade_log.get_open_position(tid)
        assert entry is not None
        assert entry.param_snapshot["BH_MASS_THRESH"] == 2.0

    def test_trade_id_uniqueness(self, trade_log):
        ids = set()
        for _ in range(20):
            tid = trade_log.log_entry("BTC", price=65000.0, qty=0.001, side="long",
                                       context=_make_context())
            ids.add(tid)
        assert len(ids) == 20   # all unique

    def test_thread_safety_concurrent_entries(self, trade_log):
        """Concurrent log_entry calls from multiple threads don't corrupt state."""
        errors: list[Exception] = []

        def logger(symbol, n):
            try:
                for _ in range(n):
                    trade_log.log_entry(symbol, price=100.0, qty=1.0, side="long",
                                         context=_make_context())
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=logger, args=(f"SYM{i}", 10))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert len(trade_log.get_open_positions()) == 50


# ===========================================================================
# _opt_float / _opt_int helpers
# ===========================================================================

class TestOptHelpers:
    def test_opt_float_none(self):
        assert _opt_float(None) is None

    def test_opt_float_numeric(self):
        assert _opt_float(1.5) == 1.5

    def test_opt_float_string(self):
        assert _opt_float("3.14") == pytest.approx(3.14)

    def test_opt_float_invalid(self):
        assert _opt_float("nope") is None

    def test_opt_int_none(self):
        assert _opt_int(None) is None

    def test_opt_int_float(self):
        assert _opt_int(3.9) == 3

    def test_opt_int_string(self):
        assert _opt_int("7") == 7

    def test_opt_int_invalid(self):
        assert _opt_int("abc") is None


# ===========================================================================
# Integration test: all four modules together
# ===========================================================================

class TestIntegration:
    def test_full_pipeline_with_config(self, tmp_json, tmp_db):
        """Config change -> trade context captures param snapshot."""
        p   = tmp_json({"BH_MASS_THRESH": 1.92, "CF_BULL_THRESH": 1.2})
        cfg = LiveConfig(config_path=p, watch=False)
        tl  = TradeLog(db_path=tmp_db)
        ss  = SessionState(db_path=tmp_db)

        assert ss.is_tradeable("BTC")

        snap = cfg.get_snapshot()
        tid  = tl.log_entry("BTC", price=65000.0, qty=0.01, side="long",
                              context=_make_context(param_snapshot=snap))
        entry = tl.get_open_position(tid)
        assert entry.param_snapshot["BH_MASS_THRESH"] == 1.92

        # Update config
        with open(p, "w") as fh:
            json.dump({"BH_MASS_THRESH": 2.50}, fh)
        cfg.reload()
        new_snap = cfg.get_snapshot()
        assert new_snap["BH_MASS_THRESH"] == 2.50

        tl.log_exit(tid, price=65500.0, reason="BH_COLLAPSE", bars_held=4)

        ss.close()
        tl.close()
        cfg.stop()

    def test_session_blocks_entry_during_emergency(self, tmp_db):
        """Emergency mode prevents trade log entries from being semantically valid."""
        ss = SessionState(db_path=tmp_db)
        ss.set_emergency_mode("test emergency")

        tradeable = ss.is_tradeable("BTC")
        assert tradeable is False

        ss.enable_trading()
        assert ss.is_tradeable("BTC") is True
        ss.close()

    def test_signal_registry_report_after_updates(self, tmp_db):
        sr = SignalRegistry(db_path=tmp_db)
        for v in [0.04, 0.05, 0.06, 0.04, 0.05]:
            sr.update_ic("BH_MASS_15m", v)
        report = sr.get_report()
        bh_entry = next(
            (s for s in report["signals"] if s["name"] == "BH_MASS_15m"), None
        )
        assert bh_entry is not None
        assert bh_entry["ic_rolling_30d"] is not None
        sr.close()
