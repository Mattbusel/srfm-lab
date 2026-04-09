"""
Whale Detector: identify large player accumulation/distribution patterns.

Whales leave footprints in the order flow and on-chain data:
  1. Volume spikes at specific price levels (stealth accumulation)
  2. Unusually large dark pool prints
  3. Options flow: large block trades indicating directional bets
  4. On-chain: large wallet movements to/from exchanges
  5. Market microstructure: persistent one-sided order flow

The signal: when whales are accumulating, follow them.
When whales are distributing, get out ahead of retail.
"""

from __future__ import annotations
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class WhaleActivity:
    """Detected whale activity event."""
    timestamp: float
    activity_type: str        # "accumulation" / "distribution" / "repositioning"
    confidence: float
    estimated_size_usd: float
    price_level: float
    duration_bars: int
    stealth_score: float      # 0=obvious, 1=very hidden


@dataclass
class WhaleSignal:
    """Output signal from whale detection."""
    signal: float             # -1 to +1
    confidence: float
    whale_activity: str       # "accumulating" / "distributing" / "neutral"
    estimated_whale_position: float  # +1 = max long, -1 = max short
    smart_money_flow: float   # net smart money direction
    retail_divergence: float  # divergence between smart and retail flow
    narrative: str


class WhaleDetector:
    """Multi-signal whale detection and tracking."""

    def __init__(self, window: int = 50):
        self.window = window
        self._volume_history = deque(maxlen=window)
        self._price_history = deque(maxlen=window)
        self._trade_sizes = deque(maxlen=window * 10)
        self._flow_imbalance = deque(maxlen=window)
        self._whale_events: List[WhaleActivity] = []

    def update(
        self,
        price: float,
        volume: float,
        buy_volume: float,
        sell_volume: float,
        avg_trade_size: float,
        large_trade_count: int = 0,
        dark_pool_volume: float = 0.0,
    ) -> WhaleSignal:
        """Process one bar of data and detect whale activity."""
        self._volume_history.append(volume)
        self._price_history.append(price)
        self._trade_sizes.append(avg_trade_size)

        # Flow imbalance
        total = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / max(total, 1e-10)
        self._flow_imbalance.append(imbalance)

        if len(self._volume_history) < 20:
            return WhaleSignal(0.0, 0.0, "neutral", 0.0, 0.0, 0.0, "Insufficient data")

        volumes = np.array(list(self._volume_history))
        prices = np.array(list(self._price_history))
        flows = np.array(list(self._flow_imbalance))

        # 1. Volume anomaly (stealth accumulation detection)
        vol_z = (volume - float(volumes.mean())) / max(float(volumes.std()), 1e-10)
        price_change = abs(price - prices[-2]) / max(prices[-2], 1e-10) if len(prices) >= 2 else 0

        # Stealth: high volume + low price change = whale absorbing without moving price
        stealth_score = max(0, vol_z - price_change * 100) / max(vol_z, 1)

        # 2. Persistent flow imbalance (whale pushing one direction)
        recent_flow = flows[-10:]
        flow_persistence = float(np.mean(recent_flow))
        flow_consistency = float(np.mean(np.sign(recent_flow) == np.sign(flow_persistence)))

        # 3. Trade size anomaly (large trades = whale)
        sizes = np.array(list(self._trade_sizes))
        size_z = (avg_trade_size - float(sizes.mean())) / max(float(sizes.std()), 1e-10)

        # 4. Dark pool activity
        dark_ratio = dark_pool_volume / max(volume, 1e-10)

        # Composite whale activity score
        whale_score = (
            0.30 * float(np.tanh(stealth_score)) +
            0.25 * flow_persistence * flow_consistency +
            0.25 * float(np.tanh(size_z / 3)) +
            0.20 * float(np.tanh(dark_ratio * 5 - 1))
        )

        # Determine activity type
        if whale_score > 0.3 and flow_persistence > 0.2:
            activity = "accumulating"
            signal = min(1.0, whale_score)
        elif whale_score > 0.3 and flow_persistence < -0.2:
            activity = "distributing"
            signal = max(-1.0, -whale_score)
        else:
            activity = "neutral"
            signal = 0.0

        # Smart money vs retail divergence
        # If whale is accumulating but price is flat/down = divergence (bullish)
        recent_return = float(prices[-1] / prices[-10] - 1) if len(prices) >= 10 else 0
        if activity == "accumulating" and recent_return < 0:
            retail_div = 0.5  # smart money buying, retail selling = strong bullish
        elif activity == "distributing" and recent_return > 0:
            retail_div = -0.5  # smart money selling, retail buying = strong bearish
        else:
            retail_div = 0.0

        # Record whale events
        if abs(whale_score) > 0.5:
            self._whale_events.append(WhaleActivity(
                timestamp=float(len(self._volume_history)),
                activity_type=activity,
                confidence=min(1.0, abs(whale_score)),
                estimated_size_usd=volume * abs(flow_persistence),
                price_level=price,
                duration_bars=int(flow_consistency * 10),
                stealth_score=stealth_score,
            ))

        confidence = min(1.0, abs(whale_score) * flow_consistency)

        # Narrative
        parts = []
        if activity != "neutral":
            parts.append(f"Whale {activity}")
        if stealth_score > 0.5:
            parts.append("stealth mode detected")
        if abs(retail_div) > 0.3:
            parts.append(f"smart-retail divergence ({'bullish' if retail_div > 0 else 'bearish'})")
        if dark_ratio > 0.3:
            parts.append(f"high dark pool activity ({dark_ratio:.0%})")

        return WhaleSignal(
            signal=float(np.clip(signal + retail_div * 0.3, -1, 1)),
            confidence=confidence,
            whale_activity=activity,
            estimated_whale_position=float(flow_persistence),
            smart_money_flow=float(whale_score),
            retail_divergence=float(retail_div),
            narrative=" | ".join(parts) if parts else "No whale activity",
        )
