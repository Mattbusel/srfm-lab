"""
Volume Profile Signal: trade based on where volume actually happened.

Volume profile shows at which prices the most trading occurred.
High-volume nodes (HVN) are price levels with strong institutional interest.
Low-volume nodes (LVN) are gaps where price moves quickly through.

The signal:
  - Price approaching an HVN: expect support/resistance (slow down, consolidate)
  - Price in an LVN: expect fast moves (breakout potential)
  - POC (Point of Control): highest volume level = strongest attractor
  - Value Area (VA): 70% of volume range = fair value zone

Trading logic:
  - Outside VA + moving toward POC: mean reversion to fair value
  - At POC with volume drying up: breakout imminent
  - Breaking through HVN: strong trend signal (institutional conviction)
  - In LVN: momentum, ride the move
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class VolumeNode:
    """A node in the volume profile."""
    price_level: float
    volume: float
    is_hvn: bool = False          # High Volume Node
    is_lvn: bool = False          # Low Volume Node
    is_poc: bool = False          # Point of Control


@dataclass
class VolumeProfileState:
    """Current volume profile analysis."""
    poc_price: float              # Point of Control (highest volume price)
    value_area_high: float        # top of value area
    value_area_low: float         # bottom of value area
    current_zone: str             # "above_va" / "in_va" / "below_va" / "at_poc"
    nearest_hvn: Optional[float]  # nearest high volume support/resistance
    nearest_lvn: Optional[float]  # nearest low volume zone (fast move potential)
    signal: float                 # -1 to +1 trading signal
    confidence: float


class VolumeProfileSignal:
    """
    Volume profile-based trading signal.

    Maintains a rolling volume profile and generates signals from
    the relationship between current price and volume structure.
    """

    def __init__(self, n_bins: int = 50, lookback: int = 252):
        self.n_bins = n_bins
        self.lookback = lookback
        self._price_volume_history: deque = deque(maxlen=lookback)

    def update(self, price: float, volume: float) -> VolumeProfileState:
        """Update with new bar and compute signal."""
        self._price_volume_history.append((price, volume))

        if len(self._price_volume_history) < 30:
            return VolumeProfileState(price, price * 1.01, price * 0.99, "in_va", None, None, 0.0, 0.0)

        prices = np.array([p for p, v in self._price_volume_history])
        volumes = np.array([v for p, v in self._price_volume_history])

        # Build volume profile (histogram of volume at price levels)
        price_min, price_max = prices.min(), prices.max()
        bins = np.linspace(price_min, price_max, self.n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        vol_profile = np.zeros(self.n_bins)
        for p, v in self._price_volume_history:
            bin_idx = int(np.clip(np.digitize(p, bins) - 1, 0, self.n_bins - 1))
            vol_profile[bin_idx] += v

        # POC: highest volume bin
        poc_idx = int(np.argmax(vol_profile))
        poc_price = float(bin_centers[poc_idx])

        # Value area: 70% of total volume centered on POC
        total_vol = vol_profile.sum()
        target = total_vol * 0.7
        cumulative = vol_profile[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx

        while cumulative < target:
            expand_low = low_idx > 0
            expand_high = high_idx < self.n_bins - 1

            if expand_low and expand_high:
                if vol_profile[low_idx - 1] > vol_profile[high_idx + 1]:
                    low_idx -= 1
                    cumulative += vol_profile[low_idx]
                else:
                    high_idx += 1
                    cumulative += vol_profile[high_idx]
            elif expand_low:
                low_idx -= 1
                cumulative += vol_profile[low_idx]
            elif expand_high:
                high_idx += 1
                cumulative += vol_profile[high_idx]
            else:
                break

        va_high = float(bin_centers[high_idx])
        va_low = float(bin_centers[low_idx])

        # Identify HVN and LVN
        avg_vol = float(vol_profile[vol_profile > 0].mean()) if (vol_profile > 0).any() else 1
        hvn_threshold = avg_vol * 1.5
        lvn_threshold = avg_vol * 0.3

        nodes = []
        for i, (center, vol) in enumerate(zip(bin_centers, vol_profile)):
            is_hvn = vol > hvn_threshold
            is_lvn = vol < lvn_threshold and vol > 0
            nodes.append(VolumeNode(float(center), float(vol), is_hvn, is_lvn, i == poc_idx))

        # Current zone
        if price > va_high:
            zone = "above_va"
        elif price < va_low:
            zone = "below_va"
        elif abs(price - poc_price) / poc_price < 0.005:
            zone = "at_poc"
        else:
            zone = "in_va"

        # Nearest HVN and LVN
        hvns = [n.price_level for n in nodes if n.is_hvn and n.price_level != poc_price]
        lvns = [n.price_level for n in nodes if n.is_lvn]

        nearest_hvn = min(hvns, key=lambda l: abs(l - price)) if hvns else None
        nearest_lvn = min(lvns, key=lambda l: abs(l - price)) if lvns else None

        # Trading signal
        signal = self._compute_signal(price, poc_price, va_high, va_low, zone,
                                        nearest_hvn, nearest_lvn, volumes)

        return VolumeProfileState(
            poc_price=poc_price,
            value_area_high=va_high,
            value_area_low=va_low,
            current_zone=zone,
            nearest_hvn=nearest_hvn,
            nearest_lvn=nearest_lvn,
            signal=signal,
            confidence=min(1.0, len(self._price_volume_history) / self.lookback),
        )

    def _compute_signal(self, price, poc, va_high, va_low, zone,
                          nearest_hvn, nearest_lvn, volumes) -> float:
        signal = 0.0

        # Outside VA: mean revert toward POC
        if zone == "above_va":
            distance = (price - poc) / max(poc, 1e-10)
            signal = -float(np.tanh(distance * 10))  # short, revert to POC
        elif zone == "below_va":
            distance = (poc - price) / max(poc, 1e-10)
            signal = float(np.tanh(distance * 10))   # long, revert to POC

        # At POC with declining volume: breakout imminent (reduce signal, wait)
        elif zone == "at_poc":
            if len(volumes) >= 10:
                recent_vol = float(volumes[-5:].mean())
                avg_vol = float(volumes.mean())
                if recent_vol < avg_vol * 0.5:
                    signal = 0.0  # low vol at POC = breakout pending, no signal

        # In LVN: momentum (fast move zone)
        if nearest_lvn and abs(price - nearest_lvn) / max(price, 1e-10) < 0.01:
            # We're in a low volume zone: go with momentum
            if len(volumes) >= 3:
                recent_direction = float(np.sign(price - float(np.mean([p for p, _ in list(self._price_volume_history)[-5:]]))))
                signal = recent_direction * 0.5

        return float(np.clip(signal, -1, 1))
