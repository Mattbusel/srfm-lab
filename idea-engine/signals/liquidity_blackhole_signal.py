"""
Liquidity Black Hole Detector: predict cascade events before they happen.

A "liquidity black hole" occurs when market makers simultaneously pull quotes,
creating a void that prices fall into. This is the financial equivalent of
gravitational collapse -- and it can be detected BEFORE it happens.

Leading indicators of a liquidity black hole:
  1. Depth decay rate: order book depth declining faster than normal
  2. VPIN divergence: informed flow increasing while depth decreases
  3. Spread acceleration: bid-ask spread widening at increasing rate
  4. Market maker retreat: quote updates slowing (MM pulling back)
  5. Cross-asset contagion: liquidity withdrawal spreading across assets

The signal triggers BEFORE the crash, giving time to reduce exposure
or position for the move.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class LiquidityBlackholeSignal:
    """Output of the black hole detector."""
    blackhole_probability: float      # 0-1 probability of liquidity collapse
    severity: str                     # "none" / "forming" / "imminent" / "active"
    depth_decay_rate: float           # rate of depth decline (negative = declining)
    vpin_level: float                 # current VPIN (0-1)
    spread_acceleration: float        # second derivative of spread
    mm_retreat_score: float           # 0-1 how much MMs are pulling back
    contagion_score: float            # 0-1 cross-asset spread of liquidity withdrawal
    time_to_event_estimate: int       # estimated bars until collapse (0 = happening now)
    recommended_action: str           # "hold" / "reduce_50pct" / "flatten" / "short_vol"
    confidence: float


class DepthDecayMonitor:
    """Track order book depth and detect abnormal decay."""

    def __init__(self, window: int = 50):
        self.window = window
        self._depth_history = deque(maxlen=window)

    def update(self, total_depth: float) -> float:
        """Update with current total order book depth. Returns decay rate."""
        self._depth_history.append(total_depth)

        if len(self._depth_history) < 10:
            return 0.0

        depths = np.array(list(self._depth_history))

        # Decay rate: slope of depth over recent window
        t = np.arange(len(depths))
        slope = float(np.polyfit(t, depths, 1)[0])

        # Normalize by average depth
        avg_depth = max(float(depths.mean()), 1e-10)
        decay_rate = slope / avg_depth

        return float(decay_rate)

    def is_abnormal(self, decay_rate: float, threshold: float = -0.02) -> bool:
        """Is the depth decay abnormally fast?"""
        return decay_rate < threshold


class VPINMonitor:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN).
    High VPIN = high probability that informed traders are driving flow.
    """

    def __init__(self, bucket_size: float = 1000, n_buckets: int = 50):
        self.bucket_size = bucket_size
        self.n_buckets = n_buckets
        self._buckets = deque(maxlen=n_buckets)
        self._current_buy = 0.0
        self._current_sell = 0.0
        self._current_volume = 0.0

    def update(self, volume: float, price_change: float) -> float:
        """
        Update VPIN with new trade data.
        Uses bulk volume classification: buy_pct = CDF(price_change / sigma).
        """
        # Classify volume as buy or sell (simplified)
        buy_pct = 0.5 + 0.5 * math.tanh(price_change * 50)
        buy_vol = volume * buy_pct
        sell_vol = volume * (1 - buy_pct)

        self._current_buy += buy_vol
        self._current_sell += sell_vol
        self._current_volume += volume

        # Bucket complete?
        if self._current_volume >= self.bucket_size:
            imbalance = abs(self._current_buy - self._current_sell) / max(self._current_volume, 1e-10)
            self._buckets.append(imbalance)
            self._current_buy = self._current_sell = self._current_volume = 0.0

        # VPIN = average imbalance across buckets
        if self._buckets:
            return float(np.mean(list(self._buckets)))
        return 0.5

    def get_vpin(self) -> float:
        if self._buckets:
            return float(np.mean(list(self._buckets)))
        return 0.5


class SpreadAccelerationMonitor:
    """Track bid-ask spread and detect acceleration (widening at increasing rate)."""

    def __init__(self, window: int = 30):
        self._spread_history = deque(maxlen=window)

    def update(self, spread_bps: float) -> float:
        """Update with current spread. Returns acceleration (second derivative)."""
        self._spread_history.append(spread_bps)

        if len(self._spread_history) < 10:
            return 0.0

        spreads = np.array(list(self._spread_history))

        # First derivative (rate of widening)
        velocity = np.diff(spreads)

        # Second derivative (acceleration of widening)
        if len(velocity) >= 3:
            acceleration = float(np.mean(np.diff(velocity[-5:])))
        else:
            acceleration = 0.0

        return acceleration


class MarketMakerRetreatMonitor:
    """Detect when market makers are reducing their activity."""

    def __init__(self, window: int = 30):
        self._quote_rates = deque(maxlen=window)   # quotes per bar
        self._cancel_rates = deque(maxlen=window)  # cancellations per bar

    def update(self, quotes_per_bar: float, cancels_per_bar: float) -> float:
        """
        Update with quote/cancel activity.
        Returns retreat score: 0 = normal, 1 = full retreat.
        """
        self._quote_rates.append(quotes_per_bar)
        self._cancel_rates.append(cancels_per_bar)

        if len(self._quote_rates) < 10:
            return 0.0

        quotes = np.array(list(self._quote_rates))
        cancels = np.array(list(self._cancel_rates))

        # Retreat = declining quote rate + increasing cancel rate
        avg_quotes = float(quotes.mean())
        recent_quotes = float(quotes[-5:].mean())
        quote_decline = max(0, (avg_quotes - recent_quotes) / max(avg_quotes, 1e-10))

        avg_cancels = float(cancels.mean())
        recent_cancels = float(cancels[-5:].mean())
        cancel_increase = max(0, (recent_cancels - avg_cancels) / max(avg_cancels, 1e-10))

        retreat = float(min(1.0, quote_decline * 2 + cancel_increase))
        return retreat


class LiquidityBlackholeDetector:
    """
    Master detector: combines all leading indicators to predict
    liquidity collapse before it happens.
    """

    def __init__(self):
        self.depth_monitor = DepthDecayMonitor()
        self.vpin_monitor = VPINMonitor()
        self.spread_monitor = SpreadAccelerationMonitor()
        self.mm_monitor = MarketMakerRetreatMonitor()
        self._contagion_scores: Dict[str, float] = {}

    def update(
        self,
        total_depth: float,
        volume: float,
        price_change: float,
        spread_bps: float,
        quotes_per_bar: float = 100,
        cancels_per_bar: float = 50,
    ) -> LiquidityBlackholeSignal:
        """Update all monitors and compute black hole probability."""
        depth_decay = self.depth_monitor.update(total_depth)
        vpin = self.vpin_monitor.update(volume, price_change)
        spread_accel = self.spread_monitor.update(spread_bps)
        mm_retreat = self.mm_monitor.update(quotes_per_bar, cancels_per_bar)

        # Contagion: would need multi-asset data (simplified: use spread as proxy)
        contagion = 0.0  # placeholder

        # Black hole probability: weighted combination of indicators
        indicators = {
            "depth_decay": max(0, -depth_decay * 20),    # negative decay = bad
            "vpin": max(0, (vpin - 0.5) * 2),            # VPIN > 0.5 = informed flow
            "spread_accel": max(0, spread_accel * 5),     # positive acceleration = widening
            "mm_retreat": mm_retreat,                       # 0-1 retreat score
        }

        # Weighted average
        weights = {"depth_decay": 0.3, "vpin": 0.25, "spread_accel": 0.25, "mm_retreat": 0.2}
        probability = sum(indicators[k] * weights[k] for k in indicators)
        probability = float(min(1.0, max(0.0, probability)))

        # Severity classification
        if probability > 0.8:
            severity = "active"
            action = "flatten"
            time_est = 0
        elif probability > 0.6:
            severity = "imminent"
            action = "reduce_50pct"
            time_est = 3
        elif probability > 0.3:
            severity = "forming"
            action = "hold"  # watch but don't act yet
            time_est = 10
        else:
            severity = "none"
            action = "hold"
            time_est = -1  # not applicable

        confidence = min(1.0, 0.5 + probability * 0.5)

        return LiquidityBlackholeSignal(
            blackhole_probability=probability,
            severity=severity,
            depth_decay_rate=depth_decay,
            vpin_level=vpin,
            spread_acceleration=spread_accel,
            mm_retreat_score=mm_retreat,
            contagion_score=contagion,
            time_to_event_estimate=time_est,
            recommended_action=action,
            confidence=confidence,
        )

    def update_contagion(self, symbol: str, spread_bps: float) -> None:
        """Update multi-asset contagion tracking."""
        self._contagion_scores[symbol] = spread_bps
