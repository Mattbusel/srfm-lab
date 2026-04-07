"""
execution/tests/test_options_live_risk.py -- Tests for OptionsRiskMonitor.

Covers:
  - OptionsPosition dataclass and Greeks computation
  - LiveGreeks aggregation
  - VolSurfaceCache (SVI lookup, fallback, cache refresh)
  - OptionsRiskMonitor (add/remove, portfolio Greeks, hedge signal, scenario matrix)
  - Scenario P&L correctness checks
  - Vega P&L estimation
  - Gamma scalp signal

Run with: pytest execution/tests/test_options_live_risk.py -v
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from execution.options_live_risk import (
    LiveGreeks,
    OptionsPosition,
    OptionsRiskMonitor,
    VolSurfaceCache,
    create_options_risk_monitor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos(
    symbol: str = "AAPL",
    expiry: str = "2027-01-15",
    strike: float = 150.0,
    right: str = "call",
    qty: float = 10.0,
    entry_price: float = 5.0,
    spot: float = 150.0,
    sigma: float = 0.25,
    r: float = 0.05,
    q: float = 0.0,
    multiplier: float = 100.0,
) -> OptionsPosition:
    return OptionsPosition(
        symbol=symbol,
        expiry=expiry,
        strike=strike,
        right=right,
        qty=qty,
        entry_price=entry_price,
        spot=spot,
        sigma=sigma,
        r=r,
        q=q,
        multiplier=multiplier,
    )


def run(coro):
    """Run an async coroutine in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# OptionsPosition
# ---------------------------------------------------------------------------

class TestOptionsPosition:
    def test_position_id_format(self):
        p = _pos(symbol="AAPL", expiry="2027-01-15", strike=150.0, right="call")
        assert p.position_id == "AAPL_2027-01-15_150.0_call"

    def test_time_to_expiry_positive(self):
        p = _pos(expiry="2027-01-15")
        T = p.time_to_expiry()
        assert T > 0.0

    def test_time_to_expiry_past_date(self):
        p = _pos(expiry="2020-01-01")
        T = p.time_to_expiry()
        assert T >= 0.0  # clamped to min

    def test_current_greeks_returns_greeks_result(self):
        p = _pos(spot=150.0, strike=150.0, sigma=0.25)
        g = p.current_greeks()
        assert hasattr(g, "delta")
        assert hasattr(g, "gamma")
        assert hasattr(g, "vega")

    def test_call_delta_positive(self):
        p = _pos(right="call", spot=150.0, strike=150.0, sigma=0.25)
        g = p.current_greeks()
        assert g.delta > 0

    def test_put_delta_negative(self):
        p = _pos(right="put", spot=150.0, strike=150.0, sigma=0.25)
        g = p.current_greeks()
        assert g.delta < 0

    def test_gamma_positive(self):
        p = _pos(spot=150.0, strike=150.0)
        g = p.current_greeks()
        assert g.gamma > 0

    def test_current_price_positive(self):
        p = _pos(spot=150.0, strike=150.0, sigma=0.25)
        price = p.current_price()
        assert price > 0

    def test_otm_call_cheaper_than_atm(self):
        atm = _pos(spot=150.0, strike=150.0, sigma=0.25)
        otm = _pos(spot=150.0, strike=180.0, sigma=0.25)
        assert atm.current_price() > otm.current_price()

    def test_sigma_attribute_set(self):
        p = _pos(sigma=0.30)
        assert p.sigma == 0.30


# ---------------------------------------------------------------------------
# LiveGreeks
# ---------------------------------------------------------------------------

class TestLiveGreeks:
    def test_as_dict_keys(self):
        g = LiveGreeks(delta=0.5, gamma=0.02, vega=0.1)
        d = g.as_dict()
        for key in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga"):
            assert key in d

    def test_addition(self):
        g1 = LiveGreeks(delta=0.3, gamma=0.01, vega=0.05, n_positions=2)
        g2 = LiveGreeks(delta=-0.1, gamma=0.02, vega=0.03, n_positions=1)
        g3 = g1 + g2
        assert g3.delta == pytest.approx(0.2)
        assert g3.gamma == pytest.approx(0.03)
        assert g3.n_positions == 3

    def test_default_values_are_zero(self):
        g = LiveGreeks()
        assert g.delta == 0.0
        assert g.gamma == 0.0
        assert g.n_positions == 0

    def test_timestamp_set_on_creation(self):
        before = time.time()
        g = LiveGreeks()
        after = time.time()
        assert before <= g.timestamp <= after

    def test_as_dict_rounded(self):
        g = LiveGreeks(delta=0.123456789)
        d = g.as_dict()
        assert d["delta"] == round(0.123456789, 6)


# ---------------------------------------------------------------------------
# VolSurfaceCache
# ---------------------------------------------------------------------------

class TestVolSurfaceCache:
    def test_returns_fallback_when_no_data(self):
        cache = VolSurfaceCache(fallback_vol=0.25)
        vol = run(cache.get_vol("AAPL", 150.0, 0.25, 150.0))
        assert vol == 0.25

    def test_fallback_on_surface_fetch_failure(self):
        async def bad_fetch(sym, spot):
            raise RuntimeError("fetch failed")

        cache = VolSurfaceCache(fallback_vol=0.20, fetch_fn=bad_fetch)
        vol = run(cache.get_vol("AAPL", 150.0, 0.5, 150.0))
        assert vol == 0.20

    def test_uses_set_fallback_vol(self):
        cache = VolSurfaceCache(fallback_vol=0.20)
        cache.set_fallback_vol("MSFT", 0.30)
        vol = run(cache.get_vol("MSFT", 200.0, 0.5, 200.0))
        assert vol == pytest.approx(0.30, abs=0.01)

    def test_svi_vol_returned_when_params_set(self):
        cache = VolSurfaceCache(fallback_vol=0.20)
        svi_params = {"a": 0.04, "b": 0.10, "rho": -0.3, "m": 0.0, "sigma": 0.10}
        cache.set_svi_params("AAPL", svi_params)
        vol = run(cache.get_vol("AAPL", 150.0, 0.5, 150.0))
        # SVI should give a reasonable vol
        assert 0.01 <= vol <= 2.0

    def test_realized_vol_update(self):
        cache = VolSurfaceCache()
        rets = [0.01, -0.02, 0.005, 0.015, -0.008, 0.003, 0.011, -0.005]
        for r in rets:
            cache.update_realized_vol("AAPL", r)
        assert cache.is_cached("AAPL")

    def test_cache_age_infinite_before_first_refresh(self):
        cache = VolSurfaceCache()
        assert cache.cache_age("UNKNOWN") == float("inf")

    def test_cache_age_after_set(self):
        cache = VolSurfaceCache()
        cache.set_svi_params("AAPL", {"a": 0.04, "b": 0.1, "rho": -0.3, "m": 0.0, "sigma": 0.1})
        assert cache.cache_age("AAPL") < 5.0  # set just now

    def test_clear_specific_symbol(self):
        cache = VolSurfaceCache()
        cache.set_fallback_vol("AAPL", 0.25)
        cache.set_fallback_vol("MSFT", 0.30)
        cache.clear("AAPL")
        assert not cache.is_cached("AAPL")
        assert cache.is_cached("MSFT")

    def test_clear_all(self):
        cache = VolSurfaceCache()
        for sym in ("AAPL", "MSFT", "GOOG"):
            cache.set_fallback_vol(sym, 0.25)
        cache.clear()
        for sym in ("AAPL", "MSFT", "GOOG"):
            assert not cache.is_cached(sym)

    def test_svi_vol_otm_skew(self):
        """OTM put vol should be higher than ATM with typical negative rho."""
        cache = VolSurfaceCache()
        svi_params = {"a": 0.04, "b": 0.15, "rho": -0.5, "m": 0.0, "sigma": 0.10}
        cache.set_svi_params("SPY", svi_params)
        atm_vol = VolSurfaceCache._svi_vol(svi_params, 400.0, 400.0, 0.25)
        otm_put_vol = VolSurfaceCache._svi_vol(svi_params, 400.0, 360.0, 0.25)
        assert otm_put_vol > atm_vol  # typical negative skew

    def test_successful_fetch_stores_result(self):
        async def good_fetch(sym, spot):
            return {"svi_params": {"a": 0.04, "b": 0.10, "rho": -0.3, "m": 0.0, "sigma": 0.10},
                    "realized_vol": 0.22}

        cache = VolSurfaceCache(refresh_interval=0.0, fetch_fn=good_fetch)
        vol = run(cache.get_vol("AAPL", 150.0, 0.5, 150.0))
        assert vol > 0.0


# ---------------------------------------------------------------------------
# OptionsRiskMonitor -- position management
# ---------------------------------------------------------------------------

class TestPositionAddRemove:
    """test_position_add_remove"""

    def test_add_position(self):
        mon = create_options_risk_monitor()
        p = _pos()
        run(mon.add_position(p))
        assert mon.n_positions == 1

    def test_add_multiple_positions(self):
        mon = create_options_risk_monitor()
        for i in range(5):
            p = _pos(strike=140.0 + i * 10, right="call")
            run(mon.add_position(p))
        assert mon.n_positions == 5

    def test_remove_existing_position(self):
        mon = create_options_risk_monitor()
        p = _pos(symbol="AAPL", expiry="2027-01-15", strike=150.0, right="call")
        run(mon.add_position(p))
        removed = run(mon.remove_position("AAPL", "2027-01-15", 150.0, "call"))
        assert removed
        assert mon.n_positions == 0

    def test_remove_nonexistent_returns_false(self):
        mon = create_options_risk_monitor()
        removed = run(mon.remove_position("AAPL", "2027-01-15", 999.0))
        assert not removed

    def test_remove_by_symbol_only(self):
        mon = create_options_risk_monitor()
        p1 = _pos(symbol="AAPL", expiry="2027-01-15", strike=150.0, right="call")
        p2 = _pos(symbol="AAPL", expiry="2027-01-15", strike=150.0, right="put")
        run(mon.add_position(p1))
        run(mon.add_position(p2))
        removed = run(mon.remove_position("AAPL", "2027-01-15", 150.0))  # both
        assert removed
        assert mon.n_positions == 0

    def test_position_id_unique_for_call_put(self):
        p_call = _pos(right="call")
        p_put = _pos(right="put")
        assert p_call.position_id != p_put.position_id

    def test_update_spot_propagates(self):
        mon = create_options_risk_monitor()
        p = _pos(symbol="AAPL", spot=150.0)
        run(mon.add_position(p))
        run(mon.update_spot("AAPL", 160.0))
        assert mon.positions[p.position_id].spot == 160.0

    def test_add_duplicate_position_overwrites(self):
        mon = create_options_risk_monitor()
        p1 = _pos(qty=10.0)
        p2 = _pos(qty=20.0)  # same position_id
        run(mon.add_position(p1))
        run(mon.add_position(p2))
        # Still 1 unique position
        assert mon.n_positions == 1
        assert list(mon.positions.values())[0].qty == 20.0


# ---------------------------------------------------------------------------
# OptionsRiskMonitor -- Greeks aggregation
# ---------------------------------------------------------------------------

class TestGreeksAggregation:
    """test_greeks_aggregation"""

    def test_portfolio_delta_single_long_call(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, sigma=0.25, multiplier=1.0)
        run(mon.add_position(p))
        greeks = run(mon.compute_portfolio_greeks({"AAPL": 150.0}))
        # ATM call delta ~ 0.5-0.6, no multiplier scaling
        assert 0.3 < greeks.delta < 0.7

    def test_portfolio_delta_long_short_cancel(self):
        mon = create_options_risk_monitor()
        p_long = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, multiplier=1.0)
        p_short = _pos(right="call", qty=-1.0, spot=150.0, strike=150.0, multiplier=1.0)
        run(mon.add_position(p_long))
        run(mon.add_position(p_short))
        # Need different position_ids
        p_short2 = OptionsPosition(
            symbol="AAPL", expiry="2027-06-15", strike=150.0, right="call",
            qty=-1.0, entry_price=5.0, spot=150.0, sigma=0.25, r=0.05, q=0.0, multiplier=1.0
        )
        mon2 = create_options_risk_monitor()
        run(mon2.add_position(p_long))
        run(mon2.add_position(p_short2))
        greeks = run(mon2.compute_portfolio_greeks({"AAPL": 150.0}))
        assert abs(greeks.delta) < 0.1  # nearly zero

    def test_portfolio_greeks_n_positions(self):
        mon = create_options_risk_monitor()
        for i in range(3):
            p = _pos(strike=140.0 + i * 10, expiry=f"2027-0{i+1}-15")
            run(mon.add_position(p))
        greeks = run(mon.compute_portfolio_greeks())
        assert greeks.n_positions == 3

    def test_portfolio_gamma_positive_for_long(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=5.0, spot=150.0, strike=150.0, multiplier=100.0)
        run(mon.add_position(p))
        greeks = run(mon.compute_portfolio_greeks())
        assert greeks.gamma > 0

    def test_portfolio_gamma_negative_for_short(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=-5.0, spot=150.0, strike=150.0, multiplier=100.0)
        run(mon.add_position(p))
        greeks = run(mon.compute_portfolio_greeks())
        assert greeks.gamma < 0

    def test_portfolio_theta_negative_for_long_option(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, multiplier=1.0)
        run(mon.add_position(p))
        greeks = run(mon.compute_portfolio_greeks())
        assert greeks.theta < 0  # long option loses time value

    def test_put_call_delta_sum_near_one(self):
        """Call delta + put delta ~ 1 for same ATM strike."""
        p_call = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, sigma=0.25, multiplier=1.0)
        p_put = _pos(right="put", qty=1.0, spot=150.0, strike=150.0, sigma=0.25, multiplier=1.0)
        g_call = p_call.current_greeks(spot=150.0, sigma=0.25)
        g_put = p_put.current_greeks(spot=150.0, sigma=0.25)
        # For European options: delta_call - delta_put = exp(-qT) ~ 1
        assert abs(g_call.delta + abs(g_put.delta) - 1.0) < 0.05

    def test_greeks_cached_in_last_greeks(self):
        mon = create_options_risk_monitor()
        p = _pos()
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        assert mon._last_greeks is not None

    def test_empty_portfolio_zero_greeks(self):
        mon = create_options_risk_monitor()
        greeks = run(mon.compute_portfolio_greeks())
        assert greeks.delta == 0.0
        assert greeks.gamma == 0.0
        assert greeks.n_positions == 0


# ---------------------------------------------------------------------------
# OptionsRiskMonitor -- delta hedge signal
# ---------------------------------------------------------------------------

class TestDeltaHedgeSignal:
    """test_delta_hedge_signal"""

    def test_no_signal_below_threshold(self):
        mon = create_options_risk_monitor(delta_limit=0.50, notional=1_000_000.0)
        p = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, multiplier=1.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        # Delta ~ 0.5, norm_delta = 0.5 / 1_000_000 << threshold
        signals = mon.get_pending_hedge_signals()
        assert len(signals) == 0

    def test_signal_emitted_above_threshold(self):
        # Small notional so even tiny delta triggers
        mon = create_options_risk_monitor(delta_limit=0.01, notional=1.0)
        p = _pos(right="call", qty=100.0, spot=150.0, strike=150.0, sigma=0.25, multiplier=100.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks({"AAPL": 150.0}))
        signals = mon.get_pending_hedge_signals()
        assert len(signals) > 0

    def test_hedge_signal_has_correct_type(self):
        mon = create_options_risk_monitor(delta_limit=0.01, notional=1.0)
        p = _pos(qty=100.0, multiplier=100.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        signals = mon.get_pending_hedge_signals()
        if signals:
            assert signals[0]["type"] == "delta_hedge"

    def test_hedge_signal_contains_portfolio_delta(self):
        mon = create_options_risk_monitor(delta_limit=0.01, notional=1.0)
        p = _pos(qty=100.0, multiplier=100.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        signals = mon.get_pending_hedge_signals()
        if signals:
            assert "portfolio_delta" in signals[0]

    def test_clear_hedge_signals(self):
        mon = create_options_risk_monitor(delta_limit=0.01, notional=1.0)
        p = _pos(qty=100.0, multiplier=100.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        mon.clear_hedge_signals()
        assert len(mon.get_pending_hedge_signals()) == 0

    def test_hedge_qty_opposite_sign_to_delta(self):
        mon = create_options_risk_monitor(delta_limit=0.01, notional=1.0)
        p = _pos(right="call", qty=100.0, multiplier=100.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        signals = mon.get_pending_hedge_signals()
        if signals:
            # Positive delta should produce negative hedge qty
            if signals[0]["portfolio_delta"] > 0:
                assert signals[0]["hedge_qty"] < 0


# ---------------------------------------------------------------------------
# OptionsRiskMonitor -- scenario matrix
# ---------------------------------------------------------------------------

class TestScenarioMatrix:
    """test_scenario_matrix"""

    def test_scenario_matrix_shape(self):
        mon = create_options_risk_monitor()
        p = _pos()
        run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix())
        n_scenarios = len(OptionsRiskMonitor.SCENARIO_SPOTS) * len(OptionsRiskMonitor.SCENARIO_VOLS)
        assert len(result["scenarios"]) == n_scenarios

    def test_zero_shift_scenario_near_zero_pnl(self):
        mon = create_options_risk_monitor()
        p = _pos(spot=150.0, sigma=0.25)
        run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix({"AAPL": 150.0}))
        zero_scenario = next(
            s for s in result["scenarios"]
            if s["spot_shift_pct"] == 0.0 and s["vol_shift"] == 0.0
        )
        assert abs(zero_scenario["portfolio_pnl"]) < 1.0  # should be ~0 P&L

    def test_spot_up_call_pnl_positive(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, sigma=0.25, multiplier=1.0)
        run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix({"AAPL": 150.0}))
        spot_up = next(
            s for s in result["scenarios"]
            if s["spot_shift_pct"] == 0.10 and s["vol_shift"] == 0.0
        )
        assert spot_up["portfolio_pnl"] > 0

    def test_spot_down_call_pnl_negative(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, sigma=0.25, multiplier=1.0)
        run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix({"AAPL": 150.0}))
        spot_down = next(
            s for s in result["scenarios"]
            if s["spot_shift_pct"] == -0.10 and s["vol_shift"] == 0.0
        )
        assert spot_down["portfolio_pnl"] < 0

    def test_vol_up_call_pnl_positive(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, sigma=0.20, multiplier=1.0)
        run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix({"AAPL": 150.0}))
        vol_up = next(
            s for s in result["scenarios"]
            if s["spot_shift_pct"] == 0.0 and s["vol_shift"] == 0.02
        )
        assert vol_up["portfolio_pnl"] > 0

    def test_scenario_pnl_for_short_call_reversed(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=-1.0, spot=150.0, strike=150.0, sigma=0.20, multiplier=1.0)
        run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix({"AAPL": 150.0}))
        spot_up = next(
            s for s in result["scenarios"]
            if s["spot_shift_pct"] == 0.10 and s["vol_shift"] == 0.0
        )
        assert spot_up["portfolio_pnl"] < 0  # short call loses when spot rises

    def test_scenario_n_positions_correct(self):
        mon = create_options_risk_monitor()
        for i in range(3):
            p = _pos(strike=140 + i * 10, expiry=f"2027-0{i+1}-15")
            run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix())
        assert result["n_positions"] == 3

    def test_scenario_has_timestamp(self):
        mon = create_options_risk_monitor()
        result = run(mon.compute_scenario_matrix())
        assert "timestamp" in result

    def test_scenario_pnl_monotone_in_spot_for_long_call(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, spot=150.0, strike=150.0, sigma=0.20, multiplier=1.0)
        run(mon.add_position(p))
        result = run(mon.compute_scenario_matrix({"AAPL": 150.0}))
        # Get scenarios with zero vol shift, sorted by spot
        zero_vol = sorted(
            [s for s in result["scenarios"] if s["vol_shift"] == 0.0],
            key=lambda x: x["spot_shift_pct"]
        )
        pnls = [s["portfolio_pnl"] for s in zero_vol]
        # Should be monotonically increasing for long call
        assert all(pnls[i] <= pnls[i + 1] for i in range(len(pnls) - 1))


# ---------------------------------------------------------------------------
# OptionsRiskMonitor -- vega P&L and gamma scalp
# ---------------------------------------------------------------------------

class TestVegaPnLAndGammaScalp:
    def test_vega_pnl_positive_for_long_call_vol_up(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, spot=150.0, sigma=0.20, multiplier=1.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        pnl = mon.estimate_vega_pnl(vol_change=0.02)
        assert pnl > 0

    def test_vega_pnl_negative_for_long_call_vol_down(self):
        mon = create_options_risk_monitor()
        p = _pos(right="call", qty=1.0, spot=150.0, sigma=0.20, multiplier=1.0)
        run(mon.add_position(p))
        run(mon.compute_portfolio_greeks())
        pnl = mon.estimate_vega_pnl(vol_change=-0.02)
        assert pnl < 0

    def test_vega_pnl_zero_with_no_greeks(self):
        mon = create_options_risk_monitor()
        pnl = mon.estimate_vega_pnl(0.02)
        assert pnl == 0.0

    def test_gamma_scalp_signal_when_large_gamma(self):
        mon = create_options_risk_monitor(gamma_threshold=0.001)
        big_greeks = LiveGreeks(gamma=1.0, n_positions=1)
        signal = mon.check_gamma_scalp(spot=150.0, greeks=big_greeks)
        assert signal is not None
        assert signal["type"] == "gamma_scalp"

    def test_no_gamma_scalp_signal_when_small_gamma(self):
        mon = create_options_risk_monitor(gamma_threshold=1.0)
        small_greeks = LiveGreeks(gamma=0.0001, n_positions=1)
        signal = mon.check_gamma_scalp(spot=150.0, greeks=small_greeks)
        assert signal is None

    def test_gamma_scalp_contains_threshold(self):
        mon = create_options_risk_monitor(gamma_threshold=0.001)
        greeks = LiveGreeks(gamma=10.0)
        signal = mon.check_gamma_scalp(150.0, greeks)
        if signal:
            assert "threshold" in signal
            assert "dollar_gamma_risk" in signal

    def test_risk_snapshot_structure(self):
        mon = create_options_risk_monitor()
        p = _pos()
        run(mon.add_position(p))
        snapshot = run(mon.get_risk_snapshot())
        assert "greeks" in snapshot
        assert "scenarios" in snapshot
        assert "hedge_signals" in snapshot
        assert "timestamp" in snapshot

    def test_risk_snapshot_greeks_has_delta(self):
        mon = create_options_risk_monitor()
        p = _pos()
        run(mon.add_position(p))
        snapshot = run(mon.get_risk_snapshot())
        assert "delta" in snapshot["greeks"]


# ---------------------------------------------------------------------------
# VolSurfaceCache fallback test (explicit)
# ---------------------------------------------------------------------------

class TestVolSurfaceCacheFallback:
    """test_vol_surface_cache_fallback"""

    def test_fallback_used_when_fetch_raises(self):
        async def broken_fetch(sym, spot):
            raise ConnectionError("broker offline")

        cache = VolSurfaceCache(
            refresh_interval=0.0,
            fallback_vol=0.33,
            fetch_fn=broken_fetch,
        )
        vol = run(cache.get_vol("SPY", 400.0, 0.5, 400.0))
        assert vol == pytest.approx(0.33, abs=0.01)

    def test_fallback_used_when_fetch_returns_none(self):
        async def null_fetch(sym, spot):
            return None

        cache = VolSurfaceCache(
            refresh_interval=0.0,
            fallback_vol=0.18,
            fetch_fn=null_fetch,
        )
        vol = run(cache.get_vol("QQQ", 350.0, 0.5, 350.0))
        assert vol == pytest.approx(0.18, abs=0.01)

    def test_fallback_after_stale_svi(self):
        """If SVI gives out-of-range vol, fallback is used."""
        cache = VolSurfaceCache(fallback_vol=0.22)
        # Set extreme SVI params that produce garbage
        cache.set_svi_params("AAPL", {"a": -100.0, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1})
        vol = run(cache.get_vol("AAPL", 150.0, 0.5, 150.0))
        # Should fall back since SVI gives negative/bad variance
        assert vol == pytest.approx(0.22, abs=0.01)

    def test_realized_vol_fallback_after_enough_returns(self):
        cache = VolSurfaceCache(fallback_vol=0.50)
        rets = [0.01, -0.005, 0.008, 0.003, -0.006, 0.004, 0.002, -0.003]
        for r in rets:
            cache.update_realized_vol("GLD", r)
        vol = run(cache.get_vol("GLD", 180.0, 0.5, 180.0))
        # Should use realized vol, not default fallback of 0.50
        realized = float(np.std(rets) * math.sqrt(252))
        assert vol == pytest.approx(realized, abs=0.05)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_monitor_defaults(self):
        mon = create_options_risk_monitor()
        assert mon._delta_limit == OptionsRiskMonitor.DEFAULT_DELTA_LIMIT
        assert mon._gamma_threshold == OptionsRiskMonitor.DEFAULT_GAMMA_THRESHOLD
        assert mon._notional == OptionsRiskMonitor.DEFAULT_NOTIONAL

    def test_create_monitor_custom_params(self):
        mon = create_options_risk_monitor(delta_limit=0.05, notional=500_000.0)
        assert mon._delta_limit == 0.05
        assert mon._notional == 500_000.0

    def test_vol_cache_attached(self):
        mon = create_options_risk_monitor()
        assert mon.vol_cache is not None
        assert isinstance(mon.vol_cache, VolSurfaceCache)
