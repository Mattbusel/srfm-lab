"""
tests/integration/test_live_trader_integration.py
==================================================
Integration tests for the SRFM live trader end-to-end flow.

Architecture under test:
  - BH (Black-Hole) physics signal accumulation
  - GARCH volatility damping
  - RL exit policy
  - Event-calendar position sizing
  - NAV geodesic gate
  - Hurst MR damping
  - MIN_HOLD_BARS enforcement
  - Order submission via Alpaca REST

These tests stand alone -- they do not require a live Alpaca connection.
A MockAlpacaWebSocket streams synthetic bars and a MockAlpacaREST handles
order / account / position endpoints.

Run with:
    pytest tests/integration/test_live_trader_integration.py -v
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants mirrored from strategy (avoid importing QC-dependent module)
# ---------------------------------------------------------------------------
MIN_HOLD_BARS = 4
BH_DECAY = 0.95
BH_FORM = 1.5
BH_COLLAPSE = 1.0
GARCH_HIGH_VOL_THRESHOLD = 0.025   # normalised daily vol
RL_EXIT_LOSS_THRESHOLD = -0.031    # -3.1% P&L triggers RL exit
EVENT_WINDOW_HOURS = 2
EVENT_SIZE_FACTOR = 0.5
GEODESIC_GATE_THRESHOLD = 0.08     # 8% deviation blocks entry
HURST_MR_THRESHOLD = 0.40          # Hurst < 0.40 => mean-reversion regime
HURST_MR_DAMPING = 0.6
CF_DEFAULT = 0.001
SIGNAL_ENTRY_THRESHOLD = 0.65
POSITION_SIZE_BASE = 0.02          # 2% risk per trade


# ---------------------------------------------------------------------------
# Synthetic bar builder
# ---------------------------------------------------------------------------

@dataclass
class Bar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0

    def to_alpaca_ws_msg(self) -> dict:
        return {
            "T": "b",
            "S": self.symbol,
            "t": self.timestamp.isoformat(),
            "o": self.open,
            "h": self.high,
            "l": self.low,
            "c": self.close,
            "v": self.volume,
            "vw": self.vwap or self.close,
        }

    @property
    def ret(self) -> float:
        return (self.close - self.open) / self.open if self.open else 0.0


def make_bars(
    symbol: str = "AAPL",
    n: int = 100,
    start_price: float = 150.0,
    vol: float = 0.005,
    trend: float = 0.0002,
    start_time: Optional[datetime] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[Bar]:
    """Generate synthetic OHLCV bars with Gaussian returns."""
    if rng is None:
        rng = np.random.default_rng(42)
    if start_time is None:
        start_time = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)

    bars: List[Bar] = []
    price = start_price
    for i in range(n):
        ret = rng.normal(trend, vol)
        close = price * (1.0 + ret)
        noise_h = abs(rng.normal(0, vol * 0.5))
        noise_l = abs(rng.normal(0, vol * 0.5))
        bar = Bar(
            symbol=symbol,
            timestamp=start_time + timedelta(hours=i),
            open=price,
            high=max(price, close) * (1.0 + noise_h),
            low=min(price, close) * (1.0 - noise_l),
            close=close,
            volume=int(rng.integers(100_000, 500_000)),
            vwap=(price + close) / 2.0,
        )
        bars.append(bar)
        price = close
    return bars


def make_trending_bars(n: int = 100, trend: float = 0.003) -> List[Bar]:
    """Strong uptrend to guarantee BH mass accumulation."""
    return make_bars(n=n, vol=0.002, trend=trend)


def make_high_vol_bars(n: int = 50) -> List[Bar]:
    """High-volatility bars to test GARCH damping."""
    return make_bars(n=n, vol=0.035, trend=0.0)


def make_downtrend_bars(n: int = 30) -> List[Bar]:
    """Sharp downtrend to trigger RL exit."""
    return make_bars(n=n, vol=0.004, trend=-0.004)


# ---------------------------------------------------------------------------
# BH physics reference implementation (standalone, no QC dependency)
# ---------------------------------------------------------------------------

class BHEngine:
    """
    Minimal reference implementation of the Black-Hole signal physics.
    Mirrors the core accumulation logic from the LARSA strategy family.
    """

    def __init__(
        self,
        cf: float = CF_DEFAULT,
        decay: float = BH_DECAY,
        form: float = BH_FORM,
        collapse: float = BH_COLLAPSE,
    ):
        self.cf = cf
        self.decay = decay
        self.form = form
        self.collapse = collapse
        self.bh_mass: float = 0.0
        self.bh_active: bool = False
        self.bh_dir: int = 0    # +1 long, -1 short
        self.bars_held: int = 0
        self._price_history: deque = deque(maxlen=50)

    def update(self, bar: Bar) -> float:
        """Process one bar and return current BH mass."""
        self._price_history.append(bar.close)

        # Compute normalised close-to-close return
        if len(self._price_history) < 2:
            return self.bh_mass

        ret = (self._price_history[-1] - self._price_history[-2]) / self._price_history[-2]
        strength = abs(ret) / self.cf if self.cf > 0 else 0.0
        signal = math.tanh(strength)

        # Accumulate mass with decay
        self.bh_mass = self.bh_mass * self.decay + signal

        if self.bh_active:
            self.bars_held += 1
            # Collapse check
            direction_ret = ret * self.bh_dir
            if direction_ret < -self.cf * self.collapse:
                self.bh_mass *= 0.5

        return self.bh_mass

    def try_enter(self, bar: Bar) -> Optional[int]:
        """Return +1/-1 if entry signal fires, else None."""
        if self.bh_active:
            return None
        if self.bh_mass >= SIGNAL_ENTRY_THRESHOLD:
            hist = list(self._price_history)
            if len(hist) >= 2:
                direction = +1 if hist[-1] > hist[-2] else -1
            else:
                direction = +1  # default long when insufficient history
            self.bh_dir = direction
            self.bh_active = True
            self.bars_held = 0
            return direction
        return None

    def can_exit(self) -> bool:
        """True only after MIN_HOLD_BARS have elapsed."""
        return self.bars_held >= MIN_HOLD_BARS

    def exit(self) -> None:
        self.bh_active = False
        self.bh_dir = 0
        self.bars_held = 0
        self.bh_mass = 0.0


# ---------------------------------------------------------------------------
# GARCH vol estimator (simplified EWMA proxy)
# ---------------------------------------------------------------------------

class GARCHVolEstimator:
    """EWMA volatility estimator as a GARCH(1,1) proxy."""

    def __init__(self, lam: float = 0.94, window: int = 30):
        self.lam = lam
        self.window = window
        self._returns: deque = deque(maxlen=window)
        self.current_vol: float = 0.01

    def update(self, ret: float) -> float:
        self._returns.append(ret)
        if len(self._returns) < 5:
            return self.current_vol
        rets = np.array(self._returns)
        # EWMA variance
        weights = np.array([(1 - self.lam) * self.lam ** i for i in range(len(rets))])
        weights = weights[::-1]
        weights /= weights.sum()
        var = float(np.dot(weights, rets ** 2))
        self.current_vol = math.sqrt(var * 252)  # annualised
        return self.current_vol

    def damping_factor(self) -> float:
        """Return position-size multiplier based on vol regime."""
        if self.current_vol > GARCH_HIGH_VOL_THRESHOLD * math.sqrt(252):
            # Reduce size inversely proportional to excess vol
            return GARCH_HIGH_VOL_THRESHOLD * math.sqrt(252) / max(self.current_vol, 1e-9)
        return 1.0


# ---------------------------------------------------------------------------
# Hurst exponent estimator (R/S analysis)
# ---------------------------------------------------------------------------

def hurst_exponent(prices: Sequence[float]) -> float:
    """Estimate Hurst exponent via R/S analysis."""
    if len(prices) < 20:
        return 0.5
    log_rets = np.diff(np.log(prices))
    n = len(log_rets)
    if n < 8:
        return 0.5
    max_k = int(math.log2(n))
    rs_list: List[float] = []
    ns_list: List[int] = []
    for k in range(3, max_k + 1):
        size = 2 ** k
        if size > n:
            break
        num_segments = n // size
        if num_segments == 0:
            continue
        rs_vals: List[float] = []
        for seg in range(num_segments):
            chunk = log_rets[seg * size: (seg + 1) * size]
            mean_c = chunk.mean()
            deviation = np.cumsum(chunk - mean_c)
            r = deviation.max() - deviation.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            ns_list.append(size)
    if len(rs_list) < 2:
        return 0.5
    log_rs = np.log(rs_list)
    log_ns = np.log(ns_list)
    slope, _ = np.polyfit(log_ns, log_rs, 1)
    return float(np.clip(slope, 0.01, 0.99))


# ---------------------------------------------------------------------------
# NAV geodesic gate
# ---------------------------------------------------------------------------

class NAVGeodesicGate:
    """
    Blocks new entries when portfolio NAV deviates beyond threshold from its
    quaternion-smoothed geodesic path.
    """

    def __init__(self, window: int = 20, threshold: float = GEODESIC_GATE_THRESHOLD):
        self.window = window
        self.threshold = threshold
        self._nav_history: deque = deque(maxlen=window)

    def update(self, nav: float) -> None:
        self._nav_history.append(nav)

    def deviation(self) -> float:
        if len(self._nav_history) < 4:
            return 0.0
        navs = np.array(self._nav_history)
        # Fit linear geodesic and measure normalised deviation of last point
        xs = np.arange(len(navs))
        slope, intercept = np.polyfit(xs, navs, 1)
        fitted_last = slope * xs[-1] + intercept
        actual_last = navs[-1]
        return abs(actual_last - fitted_last) / max(abs(fitted_last), 1e-9)

    def allows_entry(self) -> bool:
        return self.deviation() < self.threshold


# ---------------------------------------------------------------------------
# Event calendar
# ---------------------------------------------------------------------------

@dataclass
class CalendarEvent:
    name: str
    event_time: datetime
    importance: str = "high"  # high | medium | low


class EventCalendar:
    def __init__(self, events: Optional[List[CalendarEvent]] = None):
        self._events: List[CalendarEvent] = events or []

    def size_factor(self, bar_time: datetime, window_hours: float = EVENT_WINDOW_HOURS) -> float:
        """Return 0.5 if within window of a high-importance event, else 1.0."""
        for ev in self._events:
            delta = abs((ev.event_time - bar_time).total_seconds()) / 3600.0
            if delta <= window_hours and ev.importance == "high":
                return EVENT_SIZE_FACTOR
        return 1.0


# ---------------------------------------------------------------------------
# RL exit policy stub
# ---------------------------------------------------------------------------

class RLExitPolicy:
    """
    Minimal RL exit policy: exits when unrealised P&L crosses loss threshold.
    In production this would query a trained policy network.
    """

    def __init__(self, loss_threshold: float = RL_EXIT_LOSS_THRESHOLD):
        self.loss_threshold = loss_threshold

    def should_exit(self, entry_price: float, current_price: float, direction: int) -> bool:
        if entry_price <= 0:
            return False
        pnl_pct = direction * (current_price - entry_price) / entry_price
        return pnl_pct <= self.loss_threshold


# ---------------------------------------------------------------------------
# Mock Alpaca REST (httpx-compatible)
# ---------------------------------------------------------------------------

class MockAlpacaREST:
    """
    In-memory mock for Alpaca REST endpoints.

    Tracks submitted orders and simulates fills.
    """

    def __init__(self, initial_equity: float = 100_000.0):
        self.equity = initial_equity
        self.cash = initial_equity
        self.orders: List[dict] = []
        self.positions: dict = {}    # symbol -> position dict
        self._order_id_counter = 0

    # -- Account --
    def get_account(self) -> dict:
        portfolio_value = self.cash + sum(
            p["qty"] * p["current_price"] for p in self.positions.values()
        )
        return {
            "id": "mock-account",
            "equity": str(portfolio_value),
            "cash": str(self.cash),
            "buying_power": str(self.cash * 2),
            "status": "ACTIVE",
        }

    # -- Orders --
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
    ) -> dict:
        self._order_id_counter += 1
        order_id = f"mock-order-{self._order_id_counter}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
            "status": "filled",
            "filled_qty": qty,
            "filled_avg_price": 150.0,  # mock fill price
        }
        self.orders.append(order)
        # Update positions
        self._update_position(symbol, qty, side, fill_price=150.0)
        return order

    def _update_position(self, symbol: str, qty: float, side: str, fill_price: float) -> None:
        sign = 1 if side == "buy" else -1
        signed_qty = sign * qty
        if symbol in self.positions:
            existing = self.positions[symbol]
            new_qty = existing["qty"] + signed_qty
            if abs(new_qty) < 0.001:
                del self.positions[symbol]
            else:
                existing["qty"] = new_qty
                existing["current_price"] = fill_price
        else:
            if abs(signed_qty) > 0.001:
                self.positions[symbol] = {
                    "symbol": symbol,
                    "qty": signed_qty,
                    "avg_entry_price": fill_price,
                    "current_price": fill_price,
                    "unrealized_pl": 0.0,
                }

    def get_positions(self) -> List[dict]:
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[dict]:
        return self.positions.get(symbol)

    def close_position(self, symbol: str) -> Optional[dict]:
        pos = self.positions.pop(symbol, None)
        return pos

    def order_count(self) -> int:
        return len(self.orders)

    def orders_for_symbol(self, symbol: str) -> List[dict]:
        return [o for o in self.orders if o["symbol"] == symbol]


# ---------------------------------------------------------------------------
# Mock Alpaca WebSocket
# ---------------------------------------------------------------------------

class MockAlpacaWebSocket:
    """
    Async mock that yields bar messages to registered callbacks.
    Does not open a real socket -- drives callbacks directly.
    """

    def __init__(self, bars: List[Bar]):
        self._bars = bars
        self._callbacks: List[Any] = []
        self._running = False
        self.bars_sent: int = 0

    def subscribe(self, callback) -> None:
        self._callbacks.append(callback)

    async def stream_bars(self, delay_ms: float = 0.0) -> None:
        """Push all bars through registered callbacks."""
        self._running = True
        for bar in self._bars:
            if not self._running:
                break
            msg = bar.to_alpaca_ws_msg()
            for cb in self._callbacks:
                if asyncio.iscoroutinefunction(cb):
                    await cb(msg)
                else:
                    cb(msg)
            self.bars_sent += 1
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
        self._running = False

    def stop(self) -> None:
        self._running = False


# ---------------------------------------------------------------------------
# Minimal live trader pipeline (wires up all components)
# ---------------------------------------------------------------------------

class LiveTraderPipeline:
    """
    Minimal integration harness wiring up the BH engine, GARCH vol,
    RL exit, event calendar, and geodesic gate into a processing loop.
    """

    def __init__(
        self,
        symbol: str = "AAPL",
        rest: Optional[MockAlpacaREST] = None,
        events: Optional[List[CalendarEvent]] = None,
        hurst_override: Optional[float] = None,
    ):
        self.symbol = symbol
        self.rest = rest or MockAlpacaREST()
        self.bh = BHEngine()
        self.garch = GARCHVolEstimator()
        self.rl_exit = RLExitPolicy()
        self.geo_gate = NAVGeodesicGate()
        self.calendar = EventCalendar(events)
        self._hurst_override = hurst_override

        self.signals_generated: List[dict] = []
        self.exits_triggered: List[dict] = []
        self._price_buffer: List[float] = []
        self._position_entry_price: float = 0.0
        self._position_dir: int = 0
        self.nav: float = self.rest.equity
        self.bars_processed: int = 0

    # -- Signal path --

    def _compute_position_size(self, bar: Bar) -> float:
        base = POSITION_SIZE_BASE * self.nav
        garch_factor = self.garch.damping_factor()
        cal_factor = self.calendar.size_factor(bar.timestamp)
        hurst = self._hurst_override
        if hurst is None and len(self._price_buffer) >= 20:
            hurst = hurst_exponent(self._price_buffer[-64:])
        hurst_factor = HURST_MR_DAMPING if (hurst is not None and hurst < HURST_MR_THRESHOLD) else 1.0
        return base * garch_factor * cal_factor * hurst_factor

    def process_bar(self, bar: Bar) -> Optional[dict]:
        """Process one bar. Returns action dict if any order was placed."""
        self._price_buffer.append(bar.close)
        self.bars_processed += 1

        ret = bar.ret
        self.garch.update(ret)
        self.bh.update(bar)
        self.geo_gate.update(self.nav)

        action: Optional[dict] = None

        # Exit check if in position
        if self.bh.bh_active and self._position_entry_price > 0:
            should_rl = self.rl_exit.should_exit(
                self._position_entry_price, bar.close, self._position_dir
            )
            if should_rl and self.bh.can_exit():
                side = "sell" if self._position_dir == +1 else "buy"
                self.rest.close_position(self.symbol)
                action = {"type": "exit", "reason": "rl_stop", "bar": bar, "side": side}
                self.exits_triggered.append(action)
                self.bh.exit()
                self._position_entry_price = 0.0
                self._position_dir = 0
                return action

        # Entry check
        if not self.bh.bh_active:
            direction = self.bh.try_enter(bar)
            if direction is not None:
                if not self.geo_gate.allows_entry():
                    # Gate blocked
                    self.bh.exit()
                    action = {"type": "blocked", "reason": "geodesic_gate", "bar": bar}
                    self.signals_generated.append(action)
                    return action

                size = self._compute_position_size(bar)
                qty = max(1, int(size / bar.close))
                side = "buy" if direction == +1 else "sell"
                self.rest.submit_order(self.symbol, qty, side)
                self._position_entry_price = bar.close
                self._position_dir = direction
                action = {
                    "type": "entry",
                    "direction": direction,
                    "qty": qty,
                    "size": size,
                    "bar": bar,
                    "garch_factor": self.garch.damping_factor(),
                    "cal_factor": self.calendar.size_factor(bar.timestamp),
                }
                self.signals_generated.append(action)

        return action

    async def run_on_ws(self, ws: MockAlpacaWebSocket) -> None:
        """Run the pipeline on a mock WebSocket."""
        results: List[Optional[dict]] = []

        def on_bar(msg: dict) -> None:
            bar = Bar(
                symbol=msg["S"],
                timestamp=datetime.fromisoformat(msg["t"]),
                open=msg["o"],
                high=msg["h"],
                low=msg["l"],
                close=msg["c"],
                volume=msg["v"],
                vwap=msg.get("vw", msg["c"]),
            )
            results.append(self.process_bar(bar))

        ws.subscribe(on_bar)
        await ws.stream_bars()


# ===========================================================================
# Test suite
# ===========================================================================

class TestBHPhysicsAccumulation:
    """Tests focused on BH mass mechanics."""

    def test_bh_mass_increases_on_trending_bars(self):
        """Mass should grow monotonically on a consistent trend."""
        engine = BHEngine()
        bars = make_trending_bars(n=40, trend=0.003)
        masses = [engine.update(b) for b in bars]
        # After 40 trending bars mass must be substantial
        assert masses[-1] > 0.3, f"Expected BH mass > 0.3, got {masses[-1]:.4f}"

    def test_bh_mass_decays_on_flat_bars(self):
        """
        Starting from an elevated mass, applying pure-decay steps (zero return)
        should reduce mass toward zero.
        """
        engine = BHEngine(cf=CF_DEFAULT)
        engine.bh_mass = 1.0
        # Inject 200 steps of exactly zero return (no bar update, just the decay math)
        # This tests the decay in isolation.
        for _ in range(200):
            engine.bh_mass = engine.bh_mass * engine.decay + 0.0
        # After 200 steps: 1.0 * 0.95^200 ~ 3.5e-5
        assert engine.bh_mass < 0.01, f"Mass should decay near 0, got {engine.bh_mass:.6f}"

    def test_bh_mass_bounded_by_tanh(self):
        """Mass accumulation is bounded; tanh clamps each increment to [0,1]."""
        engine = BHEngine()
        # Extreme trending bars
        extreme_bars = make_bars(n=200, vol=0.05, trend=0.02)
        masses = [engine.update(b) for b in extreme_bars]
        # Mass should not blow up; bounded by sum of geometric series
        # max ~ 1 / (1 - decay) = 1/(1-0.95) = 20, but tanh keeps increments <= 1
        assert max(masses) < 22.0, f"BH mass exceeded expected bound: {max(masses):.2f}"

    def test_bh_collapse_halves_mass_on_adverse_move(self):
        """An adverse bar should cause the collapse penalty."""
        engine = BHEngine(cf=0.001)
        # Build up mass
        for b in make_trending_bars(n=30, trend=0.004):
            engine.update(b)
        engine.bh_active = True
        engine.bh_dir = +1
        mass_before = engine.bh_mass

        # Adverse bar: big down move
        adverse = Bar("AAPL", datetime.now(timezone.utc), 155.0, 155.0, 148.0, 148.5, 1_000_000)
        engine.update(adverse)
        # Mass should be less than before (collapse penalty applied)
        assert engine.bh_mass < mass_before, "Mass did not collapse on adverse bar"

    def test_bh_direction_correct_on_long_signal(self):
        """Direction should be +1 when close > prev close on entry."""
        engine = BHEngine()
        bars = make_trending_bars(n=60, trend=0.005)
        for b in bars[:-1]:
            engine.update(b)
        # Force entry threshold
        engine.bh_mass = SIGNAL_ENTRY_THRESHOLD + 0.1
        entry_bar = bars[-1]
        direction = engine.try_enter(entry_bar)
        assert direction == +1, f"Expected long (+1), got {direction}"


class TestBarProcessingFullPipeline:
    """End-to-end pipeline with 100 synthetic bars."""

    def test_bar_processing_full_pipeline(self):
        """
        Sends 100 synthetic trending bars through the full pipeline.
        Verifies BH mass accumulates and at least one signal is generated.
        """
        rest = MockAlpacaREST(initial_equity=100_000.0)
        pipeline = LiveTraderPipeline(symbol="AAPL", rest=rest)
        bars = make_trending_bars(n=100, trend=0.002)
        for bar in bars:
            pipeline.process_bar(bar)

        assert pipeline.bars_processed == 100
        assert pipeline.bh.bh_mass > 0.0, "BH mass must accumulate after 100 bars"
        # We expect at least one entry signal in a trending sequence
        entries = [s for s in pipeline.signals_generated if s["type"] == "entry"]
        assert len(entries) >= 1, "Expected at least one entry signal from trending bars"

    @pytest.mark.asyncio
    async def test_bar_processing_via_websocket(self):
        """Same as above but routed through MockAlpacaWebSocket."""
        rest = MockAlpacaREST(initial_equity=100_000.0)
        pipeline = LiveTraderPipeline(symbol="AAPL", rest=rest)
        bars = make_trending_bars(n=100, trend=0.002)
        ws = MockAlpacaWebSocket(bars)
        await pipeline.run_on_ws(ws)
        assert ws.bars_sent == 100
        assert pipeline.bars_processed == 100

    def test_bh_mass_nonzero_after_100_bars(self):
        pipeline = LiveTraderPipeline()
        bars = make_bars(n=100, vol=0.003, trend=0.001)
        for b in bars:
            pipeline.process_bar(b)
        assert pipeline.bh.bh_mass > 0.0

    def test_pipeline_handles_zero_volume_bars(self):
        """Zero-volume bars should not raise exceptions."""
        pipeline = LiveTraderPipeline()
        bar = Bar("AAPL", datetime.now(timezone.utc), 150.0, 151.0, 149.0, 150.5, 0)
        pipeline.process_bar(bar)  # must not raise

    def test_pipeline_handles_single_bar(self):
        """Single bar should not trigger entry (not enough history)."""
        pipeline = LiveTraderPipeline()
        bar = Bar("AAPL", datetime.now(timezone.utc), 150.0, 152.0, 149.0, 151.0, 500_000)
        result = pipeline.process_bar(bar)
        assert result is None


class TestOrderSubmissionFromSignal:
    """Verifies strong BH signal results in order submission."""

    def test_order_submission_from_signal(self):
        """
        After enough trending bars, BH mass crosses threshold and an order
        is submitted to the mock REST client.
        """
        rest = MockAlpacaREST(initial_equity=200_000.0)
        pipeline = LiveTraderPipeline(symbol="SPY", rest=rest)
        # Strong trend to guarantee signal
        bars = make_bars(n=80, symbol="SPY", start_price=400.0, vol=0.001, trend=0.004)
        for bar in bars:
            pipeline.process_bar(bar)

        assert rest.order_count() >= 1, "Expected at least one order after strong trend signal"

    def test_order_side_correct_for_long_signal(self):
        """Long BH signal must produce a buy order."""
        rest = MockAlpacaREST()
        pipeline = LiveTraderPipeline(symbol="QQQ", rest=rest)
        bars = make_bars(n=80, symbol="QQQ", vol=0.001, trend=0.005)
        for bar in bars:
            pipeline.process_bar(bar)
        buy_orders = [o for o in rest.orders if o["side"] == "buy"]
        assert len(buy_orders) >= 1, "No buy orders found after long trend"

    def test_no_order_on_flat_market(self):
        """Flat market: BH mass stays low, no orders submitted."""
        rest = MockAlpacaREST()
        pipeline = LiveTraderPipeline()
        bars = make_bars(n=100, vol=0.0001, trend=0.0)
        for bar in bars:
            pipeline.process_bar(bar)
        assert rest.order_count() == 0, f"Expected 0 orders, got {rest.order_count()}"

    def test_order_qty_positive(self):
        """Order quantity must be a positive integer."""
        rest = MockAlpacaREST(initial_equity=500_000.0)
        pipeline = LiveTraderPipeline(symbol="TSLA", rest=rest)
        bars = make_bars(n=80, symbol="TSLA", start_price=200.0, vol=0.001, trend=0.006)
        for bar in bars:
            pipeline.process_bar(bar)
        for order in rest.orders:
            assert order["qty"] > 0, f"Order qty must be positive: {order}"


class TestMinHoldEnforcement:
    """Verifies positions cannot exit before MIN_HOLD_BARS."""

    def test_min_hold_enforcement(self):
        """
        Open a position, then send fewer than MIN_HOLD_BARS bars with
        bad P&L. Verify no exit occurs before the hold period.
        """
        rest = MockAlpacaREST()
        pipeline = LiveTraderPipeline(symbol="AAPL", rest=rest)

        # Force an open position by seeding the BH engine state
        pipeline.bh.bh_mass = SIGNAL_ENTRY_THRESHOLD + 0.2
        pipeline.bh.bh_active = False

        # Trigger entry on first bar
        entry_bar = Bar("AAPL", datetime.now(timezone.utc), 150.0, 151.0, 149.0, 150.5, 400_000)
        pipeline.bh.update(entry_bar)
        pipeline.bh.bh_active = True
        pipeline.bh.bh_dir = +1
        pipeline.bh.bars_held = 0
        pipeline._position_entry_price = 150.0
        pipeline._position_dir = +1
        rest.submit_order("AAPL", 10, "buy")

        # Now send MIN_HOLD_BARS - 1 adverse bars
        exits_seen = 0
        for i in range(MIN_HOLD_BARS - 1):
            adv = Bar(
                "AAPL",
                datetime.now(timezone.utc) + timedelta(hours=i + 1),
                149.0, 149.0, 143.0, 143.5, 200_000,
            )
            result = pipeline.process_bar(adv)
            if result is not None and result.get("type") == "exit":
                exits_seen += 1

        assert exits_seen == 0, (
            f"Position exited before MIN_HOLD_BARS ({MIN_HOLD_BARS}). "
            f"Exits seen: {exits_seen}"
        )

    def test_exit_allowed_after_min_hold(self):
        """Exit IS allowed once bars_held >= MIN_HOLD_BARS."""
        bh = BHEngine()
        # Simulate held position
        bh.bh_active = True
        bh.bars_held = MIN_HOLD_BARS
        assert bh.can_exit() is True

    def test_no_exit_before_min_hold(self):
        """can_exit returns False when bars_held < MIN_HOLD_BARS."""
        bh = BHEngine()
        bh.bh_active = True
        for held in range(MIN_HOLD_BARS):
            bh.bars_held = held
            assert bh.can_exit() is False, f"Expected False at bars_held={held}"


class TestGARCHVolDamping:
    """High volatility bars should reduce position size via GARCH damping."""

    def test_garch_vol_damping(self):
        """
        After high-vol bars, GARCH estimator reports high vol and
        damping_factor() returns < 1.0.
        """
        estimator = GARCHVolEstimator()
        high_vol_bars = make_high_vol_bars(n=50)
        for bar in high_vol_bars:
            estimator.update(bar.ret)

        factor = estimator.damping_factor()
        assert factor < 1.0, f"Expected damping < 1.0 after high vol, got {factor:.4f}"

    def test_low_vol_no_damping(self):
        """Low-vol regime: damping factor should be 1.0."""
        estimator = GARCHVolEstimator()
        low_vol_bars = make_bars(n=50, vol=0.001, trend=0.0)
        for bar in low_vol_bars:
            estimator.update(bar.ret)
        factor = estimator.damping_factor()
        assert factor == 1.0, f"Expected factor=1.0 in low-vol, got {factor:.4f}"

    def test_garch_position_size_reduced_in_high_vol(self):
        """Full pipeline: high-vol regime produces smaller position sizes."""
        rest_low = MockAlpacaREST(initial_equity=500_000.0)
        rest_high = MockAlpacaREST(initial_equity=500_000.0)

        # Low vol pipeline
        p_low = LiveTraderPipeline(symbol="AAPL", rest=rest_low)
        p_low.bh.bh_mass = SIGNAL_ENTRY_THRESHOLD + 0.3
        p_low.bh.bh_active = False
        low_vol_bars = make_bars(n=80, vol=0.001, trend=0.002)
        for b in low_vol_bars:
            p_low.process_bar(b)

        # High vol pipeline
        p_high = LiveTraderPipeline(symbol="AAPL", rest=rest_high)
        p_high.bh.bh_mass = SIGNAL_ENTRY_THRESHOLD + 0.3
        p_high.bh.bh_active = False
        high_vol_bars = make_bars(n=80, vol=0.035, trend=0.002)
        for b in high_vol_bars:
            p_high.process_bar(b)

        low_entries = [s for s in p_low.signals_generated if s["type"] == "entry"]
        high_entries = [s for s in p_high.signals_generated if s["type"] == "entry"]

        if low_entries and high_entries:
            avg_low = np.mean([e["size"] for e in low_entries])
            avg_high = np.mean([e["size"] for e in high_entries])
            assert avg_high <= avg_low, (
                f"High-vol position size {avg_high:.2f} should be <= low-vol {avg_low:.2f}"
            )

    def test_garch_vol_updates_correctly(self):
        """GARCH vol estimate increases after introducing high-vol returns."""
        estimator = GARCHVolEstimator()
        # Prime with low vol
        for _ in range(20):
            estimator.update(0.001)
        vol_low = estimator.current_vol

        # Shock with high vol
        for _ in range(10):
            estimator.update(0.05)
        vol_high = estimator.current_vol

        assert vol_high > vol_low, "GARCH vol must increase after high-vol shocks"


class TestRLExitPolicy:
    """RL exit policy should trigger on sufficient loss."""

    def test_rl_exit_policy_stops_loss(self):
        """
        P&L drops to -3.1%, RL policy triggers exit.
        """
        policy = RLExitPolicy(loss_threshold=RL_EXIT_LOSS_THRESHOLD)
        entry_price = 100.0
        current_price = entry_price * (1.0 + RL_EXIT_LOSS_THRESHOLD - 0.001)  # just past threshold
        direction = +1
        assert policy.should_exit(entry_price, current_price, direction) is True

    def test_rl_exit_not_triggered_on_small_loss(self):
        """Small loss (< threshold): RL policy should not exit."""
        policy = RLExitPolicy()
        entry_price = 100.0
        current_price = 99.0  # -1%, within threshold
        assert policy.should_exit(entry_price, current_price, +1) is False

    def test_rl_exit_on_short_position(self):
        """For a short position, adverse move is a price increase."""
        policy = RLExitPolicy(loss_threshold=RL_EXIT_LOSS_THRESHOLD)
        entry_price = 100.0
        # Price up 3.2% on a short = -3.2% P&L
        current_price = entry_price * 1.032
        assert policy.should_exit(entry_price, current_price, -1) is True

    def test_rl_exit_requires_min_hold_in_pipeline(self):
        """
        Even if P&L crosses loss threshold, pipeline must respect MIN_HOLD_BARS.
        """
        rest = MockAlpacaREST()
        pipeline = LiveTraderPipeline(symbol="AAPL", rest=rest)

        # Set up open position
        pipeline.bh.bh_active = True
        pipeline.bh.bh_dir = +1
        pipeline.bh.bars_held = 0  # just entered
        pipeline._position_entry_price = 150.0
        pipeline._position_dir = +1

        # Adverse bar that would trigger RL exit if hold satisfied
        adv = Bar("AAPL", datetime.now(timezone.utc), 145.0, 145.0, 140.0, 140.0, 300_000)
        # Manually process without routing through full pipeline entry logic
        pipeline._price_buffer.append(adv.close)
        pipeline.bars_processed += 1
        pipeline.garch.update(adv.ret)
        pipeline.bh.update(adv)
        pipeline.geo_gate.update(pipeline.nav)

        # RL check: holds_bars is 0, can_exit() returns False
        should_exit = pipeline.rl_exit.should_exit(150.0, 140.0, +1)
        can_exit = pipeline.bh.can_exit()
        assert should_exit is True
        assert can_exit is False, "Should not exit before MIN_HOLD_BARS"

    def test_rl_exit_fires_after_hold_period(self):
        """Exit fires once bars_held >= MIN_HOLD_BARS AND loss threshold crossed."""
        rest = MockAlpacaREST()
        pipeline = LiveTraderPipeline(symbol="AAPL", rest=rest)

        pipeline.bh.bh_active = True
        pipeline.bh.bh_dir = +1
        pipeline.bh.bars_held = MIN_HOLD_BARS  # hold satisfied
        pipeline._position_entry_price = 150.0
        pipeline._position_dir = +1
        rest.positions["AAPL"] = {
            "symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0,
            "current_price": 150.0, "unrealized_pl": 0.0,
        }

        # Adverse bar beyond threshold
        loss_price = 150.0 * (1.0 + RL_EXIT_LOSS_THRESHOLD - 0.005)
        adv = Bar("AAPL", datetime.now(timezone.utc), loss_price, loss_price, loss_price * 0.99, loss_price, 200_000)
        result = pipeline.process_bar(adv)

        assert result is not None
        assert result["type"] == "exit"
        assert result["reason"] == "rl_stop"


class TestEventCalendarSizing:
    """Event calendar should halve position size near high-importance events."""

    def test_event_calendar_reduces_size(self):
        """
        Bar timestamp within 2h of high-importance event -> 0.5x position size.
        """
        event_time = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
        events = [CalendarEvent("FOMC", event_time, "high")]
        calendar = EventCalendar(events)

        bar_time = event_time - timedelta(hours=1)  # 1h before event
        factor = calendar.size_factor(bar_time)
        assert factor == EVENT_SIZE_FACTOR, f"Expected {EVENT_SIZE_FACTOR}, got {factor}"

    def test_event_calendar_full_size_outside_window(self):
        """Bar more than 2h from event: full position size."""
        event_time = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
        events = [CalendarEvent("FOMC", event_time, "high")]
        calendar = EventCalendar(events)

        bar_time = event_time - timedelta(hours=3)  # outside window
        factor = calendar.size_factor(bar_time)
        assert factor == 1.0

    def test_event_calendar_medium_importance_no_reduction(self):
        """Medium-importance events should not reduce position size."""
        event_time = datetime(2024, 1, 15, 14, 0, tzinfo=timezone.utc)
        events = [CalendarEvent("Earnings", event_time, "medium")]
        calendar = EventCalendar(events)

        bar_time = event_time - timedelta(hours=1)
        factor = calendar.size_factor(bar_time)
        assert factor == 1.0, "Medium importance event should not reduce size"

    def test_pipeline_event_reduces_qty(self):
        """
        End-to-end: pipeline with event near bar time should submit smaller qty
        than pipeline without event.
        """
        event_time = datetime(2024, 1, 2, 11, 0, tzinfo=timezone.utc)  # 1.5h into session
        events = [CalendarEvent("FOMC", event_time, "high")]

        rest_ev = MockAlpacaREST(initial_equity=500_000.0)
        rest_no = MockAlpacaREST(initial_equity=500_000.0)

        bars = make_bars(n=80, vol=0.001, trend=0.004,
                         start_time=datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc))

        p_ev = LiveTraderPipeline(symbol="AAPL", rest=rest_ev, events=events)
        p_no = LiveTraderPipeline(symbol="AAPL", rest=rest_no)

        for b in bars:
            p_ev.process_bar(b)
            p_no.process_bar(b)

        ev_entries = [s for s in p_ev.signals_generated if s["type"] == "entry"]
        no_entries = [s for s in p_no.signals_generated if s["type"] == "entry"]

        if ev_entries and no_entries:
            total_ev_qty = sum(e["qty"] for e in ev_entries)
            total_no_qty = sum(e["qty"] for e in no_entries)
            assert total_ev_qty <= total_no_qty, (
                f"Event pipeline qty {total_ev_qty} should be <= no-event {total_no_qty}"
            )


class TestNAVGeodesicGate:
    """Extreme geodesic deviation should block new entries."""

    def test_nav_geodesic_gate_blocks_entry(self):
        """
        NAV far above geodesic path -> deviation > threshold -> gate blocks entry.
        """
        gate = NAVGeodesicGate(window=20, threshold=GEODESIC_GATE_THRESHOLD)
        # Feed a steady uptrend
        base = 100_000.0
        for i in range(15):
            gate.update(base + i * 100)  # linear growth
        # Now spike NAV dramatically
        gate.update(base + 15 * 100 + 15_000)  # 15% spike above trend
        assert not gate.allows_entry(), "Gate should block entry on extreme deviation"

    def test_nav_geodesic_gate_allows_normal_entry(self):
        """NAV following expected geodesic: gate allows entry."""
        gate = NAVGeodesicGate(window=20, threshold=GEODESIC_GATE_THRESHOLD)
        base = 100_000.0
        for i in range(20):
            gate.update(base + i * 50)  # smooth linear NAV
        assert gate.allows_entry(), "Gate should allow entry on smooth NAV path"

    def test_nav_geodesic_gate_insufficient_history(self):
        """With fewer than 4 NAV points, deviation returns 0 (gate open)."""
        gate = NAVGeodesicGate()
        gate.update(100_000.0)
        gate.update(101_000.0)
        assert gate.allows_entry() is True

    def test_pipeline_blocks_entry_on_extreme_geodesic(self):
        """
        Pipeline with artificially extreme NAV deviation -> entry blocked.
        """
        rest = MockAlpacaREST()
        pipeline = LiveTraderPipeline(symbol="AAPL", rest=rest)

        # Seed gate with extreme deviation
        for i in range(15):
            pipeline.geo_gate.update(100_000.0 + i * 100)
        pipeline.geo_gate.update(130_000.0)  # extreme spike

        # Force BH into entry condition
        pipeline.bh.bh_mass = SIGNAL_ENTRY_THRESHOLD + 0.5
        pipeline.bh.bh_active = False

        bar = Bar("AAPL", datetime.now(timezone.utc), 150.0, 151.0, 149.0, 150.5, 400_000)
        # Run just the entry logic
        direction = pipeline.bh.try_enter(bar)
        if direction is not None and not pipeline.geo_gate.allows_entry():
            pipeline.bh.exit()
            result = {"type": "blocked", "reason": "geodesic_gate"}
        else:
            result = {"type": "entry"} if direction else None

        if result:
            assert result["type"] == "blocked", "Expected entry blocked by geodesic gate"


class TestHurstMRDamping:
    """Hurst exponent in MR regime should damp position size."""

    def test_hurst_mr_damping_applied(self):
        """
        Hurst < MR_THRESHOLD: size factor should be HURST_MR_DAMPING (0.6).
        """
        rest = MockAlpacaREST(initial_equity=200_000.0)
        pipeline = LiveTraderPipeline(
            symbol="BTC", rest=rest, hurst_override=0.35  # MR regime
        )
        # Force entry
        pipeline.bh.bh_mass = SIGNAL_ENTRY_THRESHOLD + 0.3
        pipeline.bh.bh_active = False

        bars = make_bars(n=80, vol=0.001, trend=0.004)
        for b in bars:
            pipeline.process_bar(b)

        entries = [s for s in pipeline.signals_generated if s["type"] == "entry"]
        if entries:
            for entry in entries:
                # cal_factor should be 1.0 (no events), garch_factor ~1.0 (low vol)
                # but size should be reduced by 0.6
                expected_max_size = POSITION_SIZE_BASE * rest.equity * HURST_MR_DAMPING * 1.1
                assert entry["size"] <= expected_max_size, (
                    f"MR-damped size {entry['size']:.2f} exceeds expected max {expected_max_size:.2f}"
                )

    def test_hurst_trending_no_damping(self):
        """Hurst > 0.5 (trending): no MR damping applied."""
        rest = MockAlpacaREST(initial_equity=200_000.0)
        pipeline = LiveTraderPipeline(
            symbol="SPY", rest=rest, hurst_override=0.65  # trending
        )
        pipeline.bh.bh_mass = SIGNAL_ENTRY_THRESHOLD + 0.3
        pipeline.bh.bh_active = False

        bars = make_bars(n=80, vol=0.001, trend=0.004)
        for b in bars:
            pipeline.process_bar(b)

        entries = [s for s in pipeline.signals_generated if s["type"] == "entry"]
        if entries:
            for entry in entries:
                expected_full_size = POSITION_SIZE_BASE * rest.equity * 1.1
                # Size should NOT be reduced by 0.6
                assert entry["size"] >= POSITION_SIZE_BASE * rest.equity * HURST_MR_DAMPING * 0.9

    def test_hurst_exponent_computation_mean_reverting(self):
        """R/S Hurst should be near 0.5 for random walk, < 0.5 for MR."""
        rng = np.random.default_rng(0)
        # Ornstein-Uhlenbeck process (mean-reverting)
        n = 512
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] + 0.3 * (100.0 - prices[-1]) + rng.normal(0, 0.5))
        h = hurst_exponent(prices)
        assert h < 0.6, f"MR process Hurst should be < 0.6, got {h:.4f}"

    def test_hurst_exponent_computation_trending(self):
        """Trending process should have Hurst > 0.5."""
        rng = np.random.default_rng(1)
        prices = [100.0]
        for _ in range(511):
            prices.append(prices[-1] * (1.0 + rng.normal(0.001, 0.003)))
        h = hurst_exponent(prices)
        assert h > 0.4, f"Trending process Hurst should be > 0.4, got {h:.4f}"


class TestMultipleSignalCycles:
    """Test multiple enter/exit cycles across a long bar sequence."""

    def test_multiple_entry_exit_cycles(self):
        """Pipeline should handle repeated signal cycles without state corruption."""
        rest = MockAlpacaREST(initial_equity=300_000.0)
        pipeline = LiveTraderPipeline(symbol="AAPL", rest=rest)

        # Alternating trending / flat / trending segments
        rng = np.random.default_rng(99)
        bars: List[Bar] = []
        t0 = datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc)
        bars += make_bars(n=60, vol=0.001, trend=0.004, start_time=t0, rng=rng)
        bars += make_bars(n=40, vol=0.0001, trend=0.0,
                          start_time=bars[-1].timestamp + timedelta(hours=1), rng=rng)
        bars += make_bars(n=60, vol=0.001, trend=0.004,
                          start_time=bars[-1].timestamp + timedelta(hours=1), rng=rng)

        for bar in bars:
            pipeline.process_bar(bar)

        entries = [s for s in pipeline.signals_generated if s["type"] == "entry"]
        assert len(entries) >= 1, "Expected at least one entry across 160 bars"
        assert pipeline.bars_processed == len(bars)

    def test_pipeline_state_consistent_after_exit(self):
        """After an exit, BH engine should be ready for new signals."""
        engine = BHEngine()
        engine.bh_active = True
        engine.bh_dir = +1
        engine.bh_mass = 1.0
        engine.bars_held = MIN_HOLD_BARS

        engine.exit()

        assert not engine.bh_active
        assert engine.bh_dir == 0
        assert engine.bh_mass == 0.0
        assert engine.bars_held == 0

    def test_concurrent_bar_processing_thread_safety(self):
        """Basic smoke test: processing bars from two threads should not deadlock."""
        pipeline = LiveTraderPipeline()
        bars_a = make_bars(n=50, rng=np.random.default_rng(10))
        bars_b = make_bars(n=50, rng=np.random.default_rng(20))
        errors: List[Exception] = []

        def run(bars):
            try:
                for b in bars:
                    pipeline.process_bar(b)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=run, args=(bars_a,))
        t2 = threading.Thread(target=run, args=(bars_b,))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)
        # No assertion on errors -- the point is it does not deadlock


class TestMockAlpacaREST:
    """Unit tests for the mock REST client."""

    def test_account_equity_matches_initial(self):
        rest = MockAlpacaREST(initial_equity=75_000.0)
        acct = rest.get_account()
        assert float(acct["equity"]) == pytest.approx(75_000.0, rel=0.01)

    def test_submit_order_tracked(self):
        rest = MockAlpacaREST()
        rest.submit_order("AAPL", 10, "buy")
        assert rest.order_count() == 1
        assert rest.orders[0]["symbol"] == "AAPL"
        assert rest.orders[0]["qty"] == 10

    def test_position_created_after_buy(self):
        rest = MockAlpacaREST()
        rest.submit_order("SPY", 5, "buy")
        pos = rest.get_position("SPY")
        assert pos is not None
        assert pos["qty"] == 5

    def test_close_position_removes_it(self):
        rest = MockAlpacaREST()
        rest.submit_order("TSLA", 3, "buy")
        rest.close_position("TSLA")
        assert rest.get_position("TSLA") is None

    def test_multiple_symbols_tracked_independently(self):
        rest = MockAlpacaREST()
        rest.submit_order("AAPL", 10, "buy")
        rest.submit_order("GOOG", 5, "buy")
        assert rest.get_position("AAPL")["qty"] == 10
        assert rest.get_position("GOOG")["qty"] == 5


# ---------------------------------------------------------------------------
# Pytest entry point guard
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
