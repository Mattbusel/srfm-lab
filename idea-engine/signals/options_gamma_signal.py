"""
Options Gamma Exposure Signal: trade the dealer hedging flow.

When options dealers are short gamma, they MUST hedge by buying when
prices rise and selling when prices fall. This amplifies moves.

When dealers are long gamma, they do the opposite: sell into rallies
and buy dips. This dampens moves and pins price to strikes.

Gamma Exposure (GEX) at each strike tells you WHERE price is likely
to be pinned and WHERE it will accelerate:

  - Positive GEX zone: price is pinned, mean reversion works
  - Negative GEX zone: price accelerates, momentum works
  - GEX flip level: the price where dealer hedging switches from
    dampening to amplifying. Cross this level and expect a big move.

This is one of the most reliable institutional signals because
dealer hedging is MECHANICAL: they MUST do it regardless of view.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class GammaExposureSnapshot:
    """Gamma exposure data at a point in time."""
    spot_price: float
    gex_by_strike: Dict[float, float]  # strike -> GEX (positive = long gamma)
    total_gex: float                    # net gamma exposure
    gex_flip_level: float              # price where GEX changes sign
    put_wall: float                    # highest put OI strike (support)
    call_wall: float                   # highest call OI strike (resistance)
    max_pain: float                    # strike that maximizes option seller profit
    implied_move: float                # expected 1-day move from ATM straddle
    dealer_position: str               # "long_gamma" / "short_gamma" / "neutral"


@dataclass
class GammaSignal:
    """Trading signal derived from gamma exposure."""
    signal: float                # -1 to +1
    confidence: float
    regime: str                  # "pinned" / "accelerating" / "transitioning"
    key_levels: Dict[str, float]
    narrative: str
    strategy_recommendation: str  # "mean_reversion" / "momentum" / "straddle"


class OptionsGammaSignal:
    """
    Generate trading signals from options gamma exposure.

    The key insight: dealer hedging flow is PREDICTABLE because it's
    mechanical. Short gamma dealers MUST buy high and sell low.
    Long gamma dealers MUST sell high and buy low.
    """

    def __init__(self):
        self._gex_history: deque = deque(maxlen=50)
        self._flip_history: deque = deque(maxlen=50)

    def compute_signal(self, snapshot: GammaExposureSnapshot) -> GammaSignal:
        """Compute the gamma-based trading signal."""
        self._gex_history.append(snapshot.total_gex)
        self._flip_history.append(snapshot.gex_flip_level)

        spot = snapshot.spot_price
        gex = snapshot.total_gex
        flip = snapshot.gex_flip_level

        # 1. Determine dealer position
        if gex > 0:
            # Long gamma: dealers dampen moves (sell rallies, buy dips)
            # Strategy: mean reversion, price is pinned
            regime = "pinned"
            strategy = "mean_reversion"

            # Signal: fade moves away from max_pain
            distance_from_pain = (spot - snapshot.max_pain) / max(spot, 1e-10)
            signal = -float(np.tanh(distance_from_pain * 20))
            confidence = min(1.0, abs(gex) / 1e9)

        elif gex < 0:
            # Short gamma: dealers amplify moves (buy rallies, sell dips)
            # Strategy: momentum, expect acceleration
            regime = "accelerating"
            strategy = "momentum"

            # Signal: go with the trend, expect amplification
            if len(self._gex_history) >= 3:
                recent_price_move = spot - snapshot.gex_flip_level
                signal = float(np.tanh(recent_price_move / max(spot, 1e-10) * 100))
            else:
                signal = 0.0
            confidence = min(1.0, abs(gex) / 1e9)

        else:
            regime = "transitioning"
            strategy = "straddle"
            signal = 0.0
            confidence = 0.3

        # 2. GEX flip level proximity
        # If price is near the flip level, expect a regime change
        flip_distance_pct = abs(spot - flip) / max(spot, 1e-10) * 100
        if flip_distance_pct < 1.0:
            # Very close to flip: high uncertainty, consider straddle
            strategy = "straddle"
            signal *= 0.5  # reduce directional conviction
            confidence *= 0.7

        # 3. Put/Call wall awareness
        put_distance = (spot - snapshot.put_wall) / max(spot, 1e-10) * 100
        call_distance = (snapshot.call_wall - spot) / max(spot, 1e-10) * 100

        if put_distance < 1.0:
            # Near put wall: strong support, lean long
            signal = max(signal, 0.3)
        if call_distance < 1.0:
            # Near call wall: strong resistance, lean short
            signal = min(signal, -0.3)

        # 4. Max pain magnet (OpEx approaching = price gravitates to max pain)
        pain_distance = (spot - snapshot.max_pain) / max(spot, 1e-10)

        # Key levels
        key_levels = {
            "gex_flip": flip,
            "put_wall": snapshot.put_wall,
            "call_wall": snapshot.call_wall,
            "max_pain": snapshot.max_pain,
            "implied_move_pct": snapshot.implied_move * 100,
        }

        # Narrative
        parts = []
        parts.append(f"Dealers are {snapshot.dealer_position.replace('_', ' ')}")
        parts.append(f"Regime: {regime}")
        if flip_distance_pct < 2:
            parts.append(f"Near GEX flip ({flip:.0f})")
        parts.append(f"Strategy: {strategy}")
        if put_distance < 2:
            parts.append(f"Near put wall ({snapshot.put_wall:.0f})")
        if call_distance < 2:
            parts.append(f"Near call wall ({snapshot.call_wall:.0f})")

        return GammaSignal(
            signal=float(np.clip(signal, -1, 1)),
            confidence=float(confidence),
            regime=regime,
            key_levels=key_levels,
            narrative=" | ".join(parts),
            strategy_recommendation=strategy,
        )

    @staticmethod
    def estimate_gex_from_oi(
        strikes: np.ndarray,
        call_oi: np.ndarray,
        put_oi: np.ndarray,
        spot: float,
        days_to_expiry: int = 30,
        risk_free: float = 0.05,
        vol: float = 0.3,
    ) -> GammaExposureSnapshot:
        """
        Estimate gamma exposure from options open interest data.

        GEX per strike = gamma * OI * 100 * spot^2 * 0.01

        Positive GEX from calls (dealers are short calls = long gamma on calls)
        Negative GEX from puts (dealers are short puts = short gamma on puts)
        """
        n = len(strikes)
        T = days_to_expiry / 365

        gex_by_strike = {}
        total_gex = 0.0

        for i in range(n):
            K = strikes[i]
            # BS gamma approximation
            d1 = (math.log(spot / K) + (risk_free + vol**2 / 2) * T) / (vol * math.sqrt(T) + 1e-10)
            gamma = math.exp(-d1**2 / 2) / (spot * vol * math.sqrt(2 * math.pi * T) + 1e-10)

            # Call GEX (dealers long gamma from short calls)
            call_gex = gamma * call_oi[i] * 100 * spot**2 * 0.01
            # Put GEX (dealers short gamma from short puts)
            put_gex = -gamma * put_oi[i] * 100 * spot**2 * 0.01

            strike_gex = call_gex + put_gex
            gex_by_strike[float(K)] = float(strike_gex)
            total_gex += strike_gex

        # GEX flip level: where cumulative GEX changes sign
        sorted_strikes = sorted(gex_by_strike.items())
        cum_gex = 0.0
        flip_level = spot
        for strike, gex in sorted_strikes:
            prev_cum = cum_gex
            cum_gex += gex
            if prev_cum * cum_gex < 0:  # sign change
                flip_level = strike
                break

        # Put wall (highest put OI)
        put_wall = float(strikes[np.argmax(put_oi)])

        # Call wall (highest call OI)
        call_wall = float(strikes[np.argmax(call_oi)])

        # Max pain (strike where OI pain is maximized for buyers)
        total_pain = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if strikes[j] < strikes[i]:
                    total_pain[i] += put_oi[j] * (strikes[i] - strikes[j])
                elif strikes[j] > strikes[i]:
                    total_pain[i] += call_oi[j] * (strikes[j] - strikes[i])
        max_pain = float(strikes[np.argmin(total_pain)])

        # Implied move from ATM straddle
        atm_idx = int(np.argmin(np.abs(strikes - spot)))
        implied_move = vol * math.sqrt(T)

        dealer_pos = "long_gamma" if total_gex > 0 else "short_gamma" if total_gex < 0 else "neutral"

        return GammaExposureSnapshot(
            spot_price=spot,
            gex_by_strike=gex_by_strike,
            total_gex=total_gex,
            gex_flip_level=flip_level,
            put_wall=put_wall,
            call_wall=call_wall,
            max_pain=max_pain,
            implied_move=implied_move,
            dealer_position=dealer_pos,
        )
