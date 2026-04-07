"""
lib/microstructure_signals.py
==============================
LARSA v18 -- Real-time microstructure signals for the live trader.

Provides four classes:
  VPINSignal            -- Volume-synchronized probability of informed trading
  OrderFlowSignal       -- Cumulative order-flow delta with price divergence detection
  AmihudSignal          -- Rolling Amihud illiquidity ratio (22-bar window)
  MicrostructureComposite -- Weighted composite of the three signals above

All mutable state is protected by asyncio.Lock for use in async contexts.
All signal() methods return a float in [-1.0, 1.0] unless noted otherwise.

Usage in live trader:
    composite = MicrostructureComposite("BTC")
    # on each tick:
    composite.vpin.update(price, volume, side)
    composite.order_flow.update(bar_buy_vol, bar_sell_vol)
    composite.amihud.update(abs_return, dollar_volume)
    sig = composite.composite_signal()
    if composite.should_skip_entry():
        return   # adverse microstructure -- skip this bar
"""

from __future__ import annotations

import asyncio
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation: map x in [x0, x1] to [y0, y1], clamped."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


# ---------------------------------------------------------------------------
# VPINSignal
# ---------------------------------------------------------------------------

class VPINSignal:
    """
    Volume-synchronized Probability of Informed Trading (VPIN).

    Classic VPIN partitions total volume into equal-size buckets and estimates
    the fraction of informed volume (order-flow imbalance) in each bucket.
    VPIN is the rolling average of per-bucket imbalances.

    High VPIN -> adverse selection risk -> reduce / skip entry.
    Low VPIN  -> low toxicity            -> full size.

    Signal mapping:
      VPIN >= 0.40  ->  -1.0  (high toxicity, reduce position)
      VPIN <= 0.15  ->  +1.0  (low toxicity, full size)
      linear interpolation between those thresholds

    Thread-safety: all mutation is protected by asyncio.Lock.
    """

    # Threshold constants
    VPIN_HIGH: float = 0.40   # above this -> max adverse signal
    VPIN_LOW:  float = 0.15   # below this -> max favorable signal

    def __init__(self, symbol: str, bucket_count: int = 50) -> None:
        """
        Parameters
        ----------
        symbol:
            Instrument symbol (used for logging/identification only).
        bucket_count:
            Number of equal-volume buckets to maintain. VPIN is the average
            imbalance across all completed buckets. Larger values produce a
            smoother, slower-reacting estimate.
        """
        if bucket_count < 1:
            raise ValueError(f"bucket_count must be >= 1, got {bucket_count}")

        self.symbol: str = symbol
        self.bucket_count: int = bucket_count
        self._lock: asyncio.Lock = asyncio.Lock()

        # Running bucket state
        self._bucket_volume: float = 0.0    # total volume accumulated in current bucket
        self._bucket_buy_vol: float = 0.0   # buy volume in current bucket
        self._bucket_sell_vol: float = 0.0  # sell volume in current bucket

        # We need to know the target bucket size. We estimate it lazily once we
        # have seen enough ticks. Until then we use a calibration window.
        self._calibration_ticks: List[float] = []  # raw volumes seen so far
        self._calibration_done: bool = False
        self._bucket_size: float = 0.0  # target volume per bucket (set after calibration)
        self._calibration_window: int = 200  # ticks before we fix bucket size

        # Completed bucket imbalances: |buy_vol - sell_vol| / total_vol
        self._bucket_imbalances: Deque[float] = deque(maxlen=bucket_count)

        # Most recent VPIN value (cached)
        self._vpin: float = 0.0

        # Total ticks processed
        self._tick_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, price: float, volume: float, side: str) -> None:
        """
        Record a new tick.

        Parameters
        ----------
        price:
            Last trade price. Not used directly in basic VPIN but stored for
            potential price-impact extensions.
        volume:
            Trade size (in base currency units, e.g. BTC).
        side:
            'buy' or 'sell' (case-insensitive). Any unrecognised string is
            treated as neutral (volume split 50/50).
        """
        if volume <= 0.0:
            return

        side_lower = side.lower().strip()

        async with self._lock:
            self._tick_count += 1

            # Calibration phase -- collect raw volumes to estimate bucket size
            if not self._calibration_done:
                self._calibration_ticks.append(volume)
                if len(self._calibration_ticks) >= self._calibration_window:
                    total = sum(self._calibration_ticks)
                    # Each bucket should represent 1/bucket_count of total vol
                    # seen in the calibration window -- but scaled to a
                    # per-tick average so we get natural bucket fills.
                    avg_vol_per_tick = total / len(self._calibration_ticks)
                    # Aim for buckets to fill in ~4 ticks on average
                    self._bucket_size = avg_vol_per_tick * 4.0
                    self._calibration_done = True
                    self._calibration_ticks.clear()
                # During calibration still accumulate into current bucket
                # using a provisional size (we will just not emit a bucket yet)

            # Classify volume
            buy_vol: float = 0.0
            sell_vol: float = 0.0
            if side_lower == "buy":
                buy_vol = volume
            elif side_lower == "sell":
                sell_vol = volume
            else:
                # Unknown side -- split 50/50
                buy_vol = volume * 0.5
                sell_vol = volume * 0.5

            # Accumulate into current bucket, potentially completing it
            remaining = volume
            while remaining > 0.0 and self._calibration_done:
                space_in_bucket = self._bucket_size - self._bucket_volume
                if space_in_bucket <= 0.0:
                    # Should not happen, but guard
                    self._flush_bucket()
                    space_in_bucket = self._bucket_size

                fill = min(remaining, space_in_bucket)
                frac = fill / volume if volume > 0 else 0.5
                self._bucket_volume += fill
                self._bucket_buy_vol += buy_vol * frac
                self._bucket_sell_vol += sell_vol * frac
                remaining -= fill

                if self._bucket_volume >= self._bucket_size:
                    self._flush_bucket()

            if not self._calibration_done:
                # Simple accumulation during calibration
                self._bucket_volume += volume
                self._bucket_buy_vol += buy_vol
                self._bucket_sell_vol += sell_vol

            # Recompute VPIN
            if self._bucket_imbalances:
                self._vpin = sum(self._bucket_imbalances) / len(self._bucket_imbalances)

    def signal(self) -> float:
        """
        Return a float in [-1.0, 1.0] based on current VPIN.

          VPIN >= VPIN_HIGH  ->  -1.0 (toxic flow, reduce)
          VPIN <= VPIN_LOW   ->  +1.0 (clean flow, full size)
          linear interpolation between

        If fewer than bucket_count buckets have been completed the signal
        returns 0.0 (neutral) to avoid acting on an unreliable estimate.
        """
        if len(self._bucket_imbalances) < self.bucket_count:
            return 0.0
        vpin = self._vpin
        # Map [VPIN_HIGH, VPIN_LOW] -> [-1, +1] (note: high VPIN = bad)
        raw = _lerp(vpin, self.VPIN_LOW, self.VPIN_HIGH, 1.0, -1.0)
        return _clamp(raw)

    @property
    def vpin(self) -> float:
        """Current VPIN estimate (0.0 if not yet calibrated)."""
        return self._vpin

    @property
    def tick_count(self) -> int:
        """Total ticks processed."""
        return self._tick_count

    @property
    def bucket_fill_count(self) -> int:
        """Number of completed buckets in history."""
        return len(self._bucket_imbalances)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _flush_bucket(self) -> None:
        """Complete the current bucket and record its imbalance."""
        total = self._bucket_volume
        if total > 0.0:
            imbalance = abs(self._bucket_buy_vol - self._bucket_sell_vol) / total
        else:
            imbalance = 0.0
        self._bucket_imbalances.append(imbalance)
        self._bucket_volume = 0.0
        self._bucket_buy_vol = 0.0
        self._bucket_sell_vol = 0.0

    def __repr__(self) -> str:
        return (
            f"VPINSignal(symbol={self.symbol!r}, "
            f"vpin={self._vpin:.4f}, "
            f"buckets={len(self._bucket_imbalances)}/{self.bucket_count}, "
            f"ticks={self._tick_count})"
        )


# ---------------------------------------------------------------------------
# OrderFlowSignal
# ---------------------------------------------------------------------------

class OrderFlowSignal:
    """
    Cumulative order-flow delta tracker with price-divergence detection.

    Tracks cumulative delta (buy_volume - sell_volume) across bars.
    Compares the direction of the delta trend with the direction of price
    to detect momentum confirmation or divergence.

    Signal logic:
      Momentum confirmation (price rising, delta rising)  -> +1.0
      Momentum confirmation (price falling, delta falling) -> -0.5
        (short-side signals are dampened: we favour long entries)
      Divergence (price rising, delta falling)             -> -0.5
      Divergence (price falling, delta rising)             ->  0.0
      Flat / insufficient data                             ->  0.0

    Thread-safety: all mutation protected by asyncio.Lock.
    """

    # Minimum number of bars before the signal is considered reliable
    MIN_BARS: int = 3
    # Number of bars used to assess recent delta trend
    TREND_WINDOW: int = 5

    def __init__(self, symbol: str) -> None:
        self.symbol: str = symbol
        self._lock: asyncio.Lock = asyncio.Lock()

        # Per-bar records (cumulative delta up to that bar, closing price)
        self._cum_delta: float = 0.0
        self._delta_history: Deque[float] = deque(maxlen=self.TREND_WINDOW + 1)
        self._price_history: Deque[float] = deque(maxlen=self.TREND_WINDOW + 1)

        self._bar_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, bar_buy_vol: float, bar_sell_vol: float, close_price: Optional[float] = None) -> None:
        """
        Record a completed 15-minute bar.

        Parameters
        ----------
        bar_buy_vol:
            Total buy volume for this bar (taker buys).
        bar_sell_vol:
            Total sell volume for this bar (taker sells).
        close_price:
            Closing price of the bar. Optional -- if provided, enables
            price-divergence detection. Omit when only delta tracking is needed.
        """
        async with self._lock:
            bar_delta = bar_buy_vol - bar_sell_vol
            self._cum_delta += bar_delta
            self._delta_history.append(self._cum_delta)
            if close_price is not None:
                self._price_history.append(close_price)
            self._bar_count += 1

    def signal(self) -> float:
        """
        Return a float in [-1.0, 1.0] based on cumulative-delta vs price trend.

        Requires at least MIN_BARS of history and a non-empty price history
        to detect divergence. Returns 0.0 (neutral) if data is insufficient.
        """
        if self._bar_count < self.MIN_BARS:
            return 0.0

        delta_trend = self._slope(self._delta_history)
        price_trend = self._slope(self._price_history) if len(self._price_history) >= 2 else None

        # If no price data, fall back to pure delta direction
        if price_trend is None:
            if delta_trend > 0:
                return 0.5
            elif delta_trend < 0:
                return -0.5
            return 0.0

        price_rising = price_trend > 0.0
        delta_rising = delta_trend > 0.0

        if price_rising and delta_rising:
            # Momentum confirmation -- bullish
            return _clamp(1.0)
        elif not price_rising and not delta_rising:
            # Momentum confirmation -- bearish (damped short signal)
            return _clamp(-0.5)
        elif price_rising and not delta_rising:
            # Bullish price, falling delta -- divergence warning
            return _clamp(-0.5)
        else:
            # Falling price, rising delta -- possible absorption / reversal
            return _clamp(0.0)

    async def reset_session(self) -> None:
        """
        Reset cumulative delta at market open.

        Call this at the start of each trading session to avoid stale
        carry-over delta from a prior session polluting the signal.
        """
        async with self._lock:
            self._cum_delta = 0.0
            self._delta_history.clear()
            self._price_history.clear()
            self._bar_count = 0

    @property
    def cumulative_delta(self) -> float:
        """Current cumulative delta (buy_vol - sell_vol) since last reset."""
        return self._cum_delta

    @property
    def bar_count(self) -> int:
        """Number of bars processed since last reset."""
        return self._bar_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slope(series: Deque[float]) -> float:
        """
        Compute the sign of the simple linear trend across the series.
        Returns positive, negative, or zero.
        """
        pts = list(series)
        n = len(pts)
        if n < 2:
            return 0.0
        # Simple: compare first half average vs second half average
        mid = n // 2
        first_avg = sum(pts[:mid]) / mid
        second_avg = sum(pts[mid:]) / (n - mid)
        return second_avg - first_avg

    def __repr__(self) -> str:
        return (
            f"OrderFlowSignal(symbol={self.symbol!r}, "
            f"cum_delta={self._cum_delta:.2f}, "
            f"bars={self._bar_count})"
        )


# ---------------------------------------------------------------------------
# AmihudSignal
# ---------------------------------------------------------------------------

class AmihudSignal:
    """
    Rolling Amihud (2002) illiquidity ratio.

    Illiquidity_t = |R_t| / DollarVolume_t

    Rolling estimate: average of last 22 bars.

    High illiquidity -> spread wider, fills worse -> reduce position size.
    Low illiquidity  -> liquid market              -> full size allowed.

    Signal mapping:
      illiquidity >= high_threshold  ->  -0.5  (illiquid, reduce)
      illiquidity <= low_threshold   ->  +1.0  (liquid, full size)
      linear interpolation between

    The thresholds are set relative to the historical median of the asset,
    so they self-calibrate as more data accumulates. Until enough bars are
    seen the signal returns 0.0 (neutral).

    Thread-safety: all mutation protected by asyncio.Lock.
    """

    WINDOW: int = 22        # rolling window length
    WARM_UP: int = 22       # bars needed before signal is active
    # Signal output bounds
    SIGNAL_LOW:  float = -0.5
    SIGNAL_HIGH: float = 1.0

    def __init__(self, symbol: str, window: int = 22) -> None:
        self.symbol: str = symbol
        self._window = window
        self._lock: asyncio.Lock = asyncio.Lock()

        self._ratios: Deque[float] = deque(maxlen=window)
        self._bar_count: int = 0

        # Dynamic thresholds (updated when we have enough data)
        self._high_threshold: float = float("inf")
        self._low_threshold:  float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, abs_return: float, dollar_volume: float) -> None:
        """
        Add a new bar's illiquidity observation.

        Parameters
        ----------
        abs_return:
            Absolute bar return (|close/open - 1| or similar). Must be >= 0.
        dollar_volume:
            Bar dollar volume (price * volume). Must be > 0 for a valid ratio.
        """
        async with self._lock:
            self._bar_count += 1
            if dollar_volume > 0.0 and abs_return >= 0.0:
                ratio = abs_return / dollar_volume
            else:
                # Zero volume or negative return (shouldn't happen) -- use zero
                ratio = 0.0
            self._ratios.append(ratio)
            self._recompute_thresholds()

    def signal(self) -> float:
        """
        Return a float in [SIGNAL_LOW, SIGNAL_HIGH] based on current
        Amihud illiquidity ratio relative to dynamic thresholds.

        Returns 0.0 if insufficient history.
        """
        if len(self._ratios) < self.WARM_UP:
            return 0.0
        current = self.illiquidity_ratio()
        raw = _lerp(
            current,
            self._low_threshold,
            self._high_threshold,
            self.SIGNAL_HIGH,
            self.SIGNAL_LOW,
        )
        return _clamp(raw, self.SIGNAL_LOW, self.SIGNAL_HIGH)

    def illiquidity_ratio(self) -> float:
        """Current rolling Amihud illiquidity ratio (mean of window)."""
        if not self._ratios:
            return 0.0
        return sum(self._ratios) / len(self._ratios)

    @property
    def bar_count(self) -> int:
        """Total bars observed."""
        return self._bar_count

    @property
    def window_size(self) -> int:
        """Length of rolling window."""
        return self._window

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recompute_thresholds(self) -> None:
        """
        Recompute dynamic low/high illiquidity thresholds from the current
        rolling window. Uses 25th and 75th percentile of observed ratios.
        """
        if len(self._ratios) < self.WARM_UP:
            return
        sorted_ratios = sorted(self._ratios)
        n = len(sorted_ratios)
        lo_idx = max(0, int(n * 0.25))
        hi_idx = min(n - 1, int(n * 0.75))
        self._low_threshold  = sorted_ratios[lo_idx]
        self._high_threshold = sorted_ratios[hi_idx]
        # Guard: if thresholds are equal or degenerate, keep a small spread
        if self._high_threshold <= self._low_threshold:
            self._high_threshold = self._low_threshold * 1.5 + 1e-12

    def __repr__(self) -> str:
        return (
            f"AmihudSignal(symbol={self.symbol!r}, "
            f"illiquidity={self.illiquidity_ratio():.6e}, "
            f"bars={self._bar_count})"
        )


# ---------------------------------------------------------------------------
# MicrostructureComposite
# ---------------------------------------------------------------------------

@dataclass
class _ComponentSignals:
    """Container for individual component signal values."""
    vpin: float = 0.0
    order_flow: float = 0.0
    amihud: float = 0.0


class MicrostructureComposite:
    """
    Weighted composite of VPIN, OrderFlow, and Amihud signals.

    Weights:
      VPIN       : 0.40
      OrderFlow  : 0.35
      Amihud     : 0.25

    composite_signal() -> float in [-1.0, 1.0]
    should_skip_entry() -> bool: True when composite < SKIP_THRESHOLD (-0.5)

    Each sub-signal component is publicly accessible for inspection and
    individual updating.

    Thread-safety: composite_signal() and should_skip_entry() are pure reads
    of sub-signal state (each sub-signal is individually thread-safe).
    """

    WEIGHT_VPIN:       float = 0.40
    WEIGHT_ORDER_FLOW: float = 0.35
    WEIGHT_AMIHUD:     float = 0.25

    # Threshold below which entry should be skipped
    SKIP_THRESHOLD: float = -0.50

    def __init__(self, symbol: str, vpin_bucket_count: int = 50) -> None:
        """
        Parameters
        ----------
        symbol:
            Instrument symbol. Passed to all sub-components for identification.
        vpin_bucket_count:
            Bucket count for VPINSignal. Default 50.
        """
        self.symbol: str = symbol
        self.vpin       = VPINSignal(symbol, bucket_count=vpin_bucket_count)
        self.order_flow = OrderFlowSignal(symbol)
        self.amihud     = AmihudSignal(symbol)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def composite_signal(self) -> float:
        """
        Compute and return the weighted composite microstructure signal.

        Returns a float in [-1.0, 1.0]. Positive values indicate favorable
        microstructure (liquid, directional flow confirmation). Negative
        values indicate adverse conditions (toxic flow, illiquidity,
        divergence).
        """
        s_vpin       = self.vpin.signal()
        s_order_flow = self.order_flow.signal()
        s_amihud     = self.amihud.signal()

        composite = (
            self.WEIGHT_VPIN       * s_vpin
            + self.WEIGHT_ORDER_FLOW * s_order_flow
            + self.WEIGHT_AMIHUD     * s_amihud
        )
        return _clamp(composite)

    def should_skip_entry(self) -> bool:
        """
        Return True if microstructure conditions are too adverse to enter.

        Entry is skipped when the composite signal is below SKIP_THRESHOLD.
        This is a fast, non-blocking check suitable for use in the hot path.
        """
        return self.composite_signal() < self.SKIP_THRESHOLD

    def component_signals(self) -> _ComponentSignals:
        """Return a snapshot of all three component signals."""
        return _ComponentSignals(
            vpin=self.vpin.signal(),
            order_flow=self.order_flow.signal(),
            amihud=self.amihud.signal(),
        )

    def summary(self) -> dict:
        """
        Return a dict suitable for logging or Prometheus push.

        Keys: symbol, composite, vpin_signal, order_flow_signal,
              amihud_signal, vpin_raw, illiquidity_ratio, skip_entry
        """
        cs = self.component_signals()
        return {
            "symbol":             self.symbol,
            "composite":          self.composite_signal(),
            "vpin_signal":        cs.vpin,
            "order_flow_signal":  cs.order_flow,
            "amihud_signal":      cs.amihud,
            "vpin_raw":           self.vpin.vpin,
            "illiquidity_ratio":  self.amihud.illiquidity_ratio(),
            "skip_entry":         self.should_skip_entry(),
        }

    def __repr__(self) -> str:
        return (
            f"MicrostructureComposite(symbol={self.symbol!r}, "
            f"composite={self.composite_signal():.4f}, "
            f"skip={self.should_skip_entry()})"
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_composites(symbols: list[str], vpin_buckets: int = 50) -> dict[str, MicrostructureComposite]:
    """
    Convenience factory: build a MicrostructureComposite for each symbol.

    Parameters
    ----------
    symbols:
        List of instrument symbols (e.g. ['BTC', 'ETH', 'SPY']).
    vpin_buckets:
        Bucket count passed to each VPINSignal. Default 50.

    Returns
    -------
    dict mapping symbol -> MicrostructureComposite
    """
    return {sym: MicrostructureComposite(sym, vpin_bucket_count=vpin_buckets) for sym in symbols}


# ---------------------------------------------------------------------------
# Stand-alone smoke test (not a pytest -- run directly for quick sanity check)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio as _asyncio

    async def _smoke() -> None:
        comp = MicrostructureComposite("BTC_TEST")

        # Inject clean buy flow
        for i in range(250):
            await comp.vpin.update(price=50000.0 + i, volume=1.0, side="buy")

        # Inject some bar data
        for i in range(30):
            await comp.order_flow.update(
                bar_buy_vol=1000.0 + i * 10,
                bar_sell_vol=800.0,
                close_price=50000.0 + i * 5,
            )
            await comp.amihud.update(abs_return=0.001, dollar_volume=1_000_000.0)

        print("Smoke test result:")
        print(comp.summary())
        print("Skip entry?", comp.should_skip_entry())

    _asyncio.run(_smoke())
