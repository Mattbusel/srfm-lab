"""
Market Memory: the system remembers significant price levels and events.

Like a human trader who remembers "BTC bounced at 30K three times" or
"ETH rejected 4800 at the top", this module maintains a persistent
memory of price levels where significant things happened.

These remembered levels become gravitational attractors in the BH model:
  - Previous highs/lows create "mass" that attracts price
  - Levels with high volume create stronger attraction
  - Levels where reversals happened create "event horizons"
  - The memory decays over time (old levels matter less)

Integration: feeds into BH physics as additional mass sources at specific prices.
"""

from __future__ import annotations
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PriceMemory:
    """A remembered significant price level."""
    memory_id: str
    symbol: str
    price_level: float
    event_type: str           # "reversal" / "breakout" / "high_volume" / "gap" / "round_number"
    significance: float       # 0-1 how important
    created_at: float
    last_tested_at: float = 0.0
    times_tested: int = 0     # how many times price returned to this level
    times_held: int = 0       # how many times the level held (bounce)
    times_broken: int = 0     # how many times price broke through
    volume_at_level: float = 0.0
    decay_rate: float = 0.01  # significance decays per day

    @property
    def strength(self) -> float:
        """Current strength accounting for decay and test history."""
        age_days = (time.time() - self.created_at) / 86400
        decay = math.exp(-self.decay_rate * age_days)
        # Levels that hold get stronger, levels that break get weaker
        hold_ratio = self.times_held / max(self.times_tested, 1)
        return float(self.significance * decay * (0.5 + 0.5 * hold_ratio))

    @property
    def gravitational_mass(self) -> float:
        """Mass for BH model integration."""
        return self.strength * (1 + math.log(1 + self.volume_at_level / 1e6))


class MarketMemoryEngine:
    """
    Persistent memory of significant price levels.

    Automatically identifies and remembers:
    - All-time highs and lows
    - Local reversal points
    - High-volume price levels (volume profile peaks)
    - Round numbers (psychological levels)
    - Gap levels (unfilled gaps)
    - Previous support that became resistance (and vice versa)
    """

    def __init__(self, max_memories_per_symbol: int = 50):
        self.max_memories = max_memories_per_symbol
        self._memories: Dict[str, List[PriceMemory]] = defaultdict(list)
        self._counter = 0
        self._price_history: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)  # (price, volume, timestamp)

    def _next_id(self) -> str:
        self._counter += 1
        return f"mem_{self._counter:06d}"

    def record_bar(self, symbol: str, open_p: float, high: float, low: float,
                    close: float, volume: float) -> List[PriceMemory]:
        """
        Process a new bar and identify any new significant levels.
        Returns list of newly created memories.
        """
        self._price_history[symbol].append((close, volume, time.time()))
        new_memories = []
        history = self._price_history[symbol]

        # Only analyze after enough data
        if len(history) < 20:
            return new_memories

        prices = np.array([h[0] for h in history])
        volumes = np.array([h[1] for h in history])

        # 1. Detect reversals (local extrema)
        if len(prices) >= 5:
            # Local high
            if prices[-3] > prices[-4] and prices[-3] > prices[-2] and prices[-3] > prices[-5]:
                mem = self._create_memory(symbol, float(prices[-3]), "reversal",
                                           0.6, float(volumes[-3]))
                new_memories.append(mem)

            # Local low
            if prices[-3] < prices[-4] and prices[-3] < prices[-2] and prices[-3] < prices[-5]:
                mem = self._create_memory(symbol, float(prices[-3]), "reversal",
                                           0.6, float(volumes[-3]))
                new_memories.append(mem)

        # 2. High volume levels (volume profile peak)
        if len(volumes) >= 20:
            recent_vol = volumes[-20:]
            vol_threshold = float(np.percentile(recent_vol, 90))
            if volume > vol_threshold:
                mem = self._create_memory(symbol, close, "high_volume",
                                           0.5, volume)
                new_memories.append(mem)

        # 3. Round numbers
        if close > 10:
            round_levels = self._get_round_numbers(close)
            for level in round_levels:
                if abs(close - level) / close < 0.005:  # within 0.5% of round
                    # Only add if not already remembered
                    existing = [m for m in self._memories[symbol]
                                if abs(m.price_level - level) / level < 0.001]
                    if not existing:
                        mem = self._create_memory(symbol, level, "round_number", 0.4, 0)
                        new_memories.append(mem)

        # 4. Gap detection
        if len(prices) >= 2:
            gap = abs(open_p - prices[-2]) / max(prices[-2], 1e-10)
            if gap > 0.02:  # > 2% gap
                gap_level = (open_p + float(prices[-2])) / 2
                mem = self._create_memory(symbol, gap_level, "gap", 0.7, volume)
                new_memories.append(mem)

        # 5. All-time high / low
        if close == float(prices.max()):
            mem = self._create_memory(symbol, close, "all_time_high", 0.9, volume)
            new_memories.append(mem)
        if close == float(prices.min()):
            mem = self._create_memory(symbol, close, "all_time_low", 0.9, volume)
            new_memories.append(mem)

        # Update existing memories: is price testing a remembered level?
        for mem in self._memories[symbol]:
            distance_pct = abs(close - mem.price_level) / max(mem.price_level, 1e-10)
            if distance_pct < 0.01:  # within 1% of level
                mem.times_tested += 1
                mem.last_tested_at = time.time()

                # Did it hold or break?
                if len(prices) >= 3:
                    bounced = (close > mem.price_level and prices[-2] < mem.price_level) or \
                              (close < mem.price_level and prices[-2] > mem.price_level)
                    if bounced:
                        mem.times_held += 1
                    else:
                        # Price sitting at level, check next bar
                        pass

        # Prune weak memories
        self._prune(symbol)

        return new_memories

    def _create_memory(self, symbol: str, price: float, event_type: str,
                        significance: float, volume: float) -> PriceMemory:
        mem = PriceMemory(
            memory_id=self._next_id(),
            symbol=symbol,
            price_level=price,
            event_type=event_type,
            significance=significance,
            created_at=time.time(),
            volume_at_level=volume,
        )
        self._memories[symbol].append(mem)
        return mem

    def _prune(self, symbol: str) -> None:
        """Remove weakest memories to stay within limit."""
        memories = self._memories[symbol]
        if len(memories) > self.max_memories:
            memories.sort(key=lambda m: m.strength, reverse=True)
            self._memories[symbol] = memories[:self.max_memories]

    def _get_round_numbers(self, price: float) -> List[float]:
        """Get nearby round number levels."""
        magnitude = 10 ** int(math.log10(max(price, 1)))
        rounds = []
        for mult in [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:
            level = round(price / (magnitude * mult)) * (magnitude * mult)
            if level > 0:
                rounds.append(level)
        return list(set(rounds))

    def get_nearby_levels(self, symbol: str, current_price: float,
                           range_pct: float = 0.05) -> List[PriceMemory]:
        """Get remembered levels near the current price."""
        nearby = []
        for mem in self._memories[symbol]:
            dist = abs(current_price - mem.price_level) / max(current_price, 1e-10)
            if dist < range_pct:
                nearby.append(mem)
        nearby.sort(key=lambda m: abs(current_price - m.price_level))
        return nearby

    def get_gravitational_field(self, symbol: str, current_price: float) -> float:
        """
        Compute the net gravitational pull from all remembered levels.

        Positive = price is attracted upward (support below is stronger than resistance above)
        Negative = price is attracted downward (resistance above is stronger)

        This integrates directly with the BH physics model.
        """
        memories = self._memories.get(symbol, [])
        if not memories:
            return 0.0

        net_force = 0.0
        for mem in memories:
            if mem.strength < 0.01:
                continue

            distance = (mem.price_level - current_price) / max(current_price, 1e-10)
            if abs(distance) < 0.001:
                continue  # too close, skip (would divide by ~zero)

            # Gravitational force: F = G * M / r^2
            # Direction: toward the level (positive if level is above, negative if below)
            force = mem.gravitational_mass / (distance ** 2) * np.sign(distance)

            # Cap to prevent singularities near levels
            force = max(-5.0, min(5.0, force))
            net_force += force

        return float(np.tanh(net_force * 0.1))  # normalize to [-1, 1]

    def get_support_resistance(self, symbol: str, current_price: float) -> Dict:
        """Identify nearest support and resistance from memory."""
        memories = self._memories.get(symbol, [])

        support = None
        resistance = None
        support_strength = 0.0
        resistance_strength = 0.0

        for mem in memories:
            if mem.strength < 0.1:
                continue
            if mem.price_level < current_price:
                if support is None or (current_price - mem.price_level < current_price - support):
                    support = mem.price_level
                    support_strength = mem.strength
            elif mem.price_level > current_price:
                if resistance is None or (mem.price_level - current_price < resistance - current_price):
                    resistance = mem.price_level
                    resistance_strength = mem.strength

        return {
            "nearest_support": support,
            "support_strength": support_strength,
            "nearest_resistance": resistance,
            "resistance_strength": resistance_strength,
            "support_distance_pct": (current_price - support) / current_price * 100 if support else None,
            "resistance_distance_pct": (resistance - current_price) / current_price * 100 if resistance else None,
        }

    def get_all_memories(self, symbol: str) -> List[Dict]:
        """Get all memories for a symbol."""
        return [
            {
                "price": m.price_level,
                "type": m.event_type,
                "strength": m.strength,
                "gravitational_mass": m.gravitational_mass,
                "times_tested": m.times_tested,
                "times_held": m.times_held,
                "age_days": (time.time() - m.created_at) / 86400,
            }
            for m in sorted(self._memories.get(symbol, []), key=lambda m: m.strength, reverse=True)
        ]
