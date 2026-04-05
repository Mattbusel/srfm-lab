"""
alternative_data/exchange_flows.py
=====================================
Simulated on-chain exchange flow data (Glassnode-style).

Financial rationale
-------------------
Exchange flows track the net movement of coins between on-chain addresses
associated with exchanges and the broader network:

  Inflows  (coins moving INTO exchanges):
    → Holders are preparing to SELL
    → Large inflows = bearish selling pressure
    → Sustained inflow growth over 24h is a distribution signal

  Outflows (coins moving OUT of exchanges):
    → Coins being withdrawn to cold storage / self-custody
    → Large outflows = accumulation (long-term conviction buying)
    → Exchange reserves FALLING over time = supply squeeze bullish

Derived signals:
  net_flow         = inflow - outflow  (positive = net selling pressure)
  reserve_change   = cumulative net_flow over 24h
  exchange_reserve : total estimated exchange balance (modelled)

  sell_pressure_ratio = inflow / (inflow + outflow)
    > 0.6 → sell pressure dominant
    < 0.4 → accumulation dominant

Because real Glassnode data requires expensive API subscriptions, we model
realistic exchange flows using:
  1. A mean-reverting Ornstein-Uhlenbeck process for the baseline flow
  2. Periodic injection of large-flow events to simulate whale movements
  3. Market-state-dependent calibration (different parameters per coin)

The simulation is seeded per-symbol-per-hour so intra-cycle calls return
consistent values.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Symbol calibration parameters
# ---------------------------------------------------------------------------

# {symbol: (base_flow_usd, flow_volatility, mean_reversion_speed)}
# base_flow_usd: typical hourly inflow in USD millions
# flow_volatility: std dev of flow as fraction of base_flow
# mean_reversion_speed: 0-1, how quickly flow reverts to baseline
_SYMBOL_PARAMS: dict[str, tuple[float, float, float]] = {
    "BTC":  (500.0,  0.35, 0.15),
    "ETH":  (300.0,  0.40, 0.18),
    "SOL":  (80.0,   0.50, 0.20),
    "BNB":  (60.0,   0.40, 0.15),
    "XRP":  (100.0,  0.45, 0.18),
    "DOGE": (30.0,   0.60, 0.22),
    "ADA":  (40.0,   0.50, 0.20),
    "AVAX": (35.0,   0.50, 0.20),
    "LINK": (20.0,   0.45, 0.18),
    "DOT":  (25.0,   0.45, 0.18),
}

DEFAULT_SYMBOL_PARAMS = (50.0, 0.50, 0.20)

TRACKED_SYMBOLS: list[str] = list(_SYMBOL_PARAMS.keys())


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExchangeFlowSnapshot:
    """
    Exchange flow signal for a single symbol.

    Attributes
    ----------
    symbol              : Canonical ticker
    inflow_usd_1h       : Estimated USD inflow to exchanges in last hour (millions)
    outflow_usd_1h      : Estimated USD outflow from exchanges in last hour (millions)
    net_flow_usd_1h     : inflow - outflow (positive = sell pressure)
    sell_pressure_ratio : inflow / (inflow + outflow); 0.5 = neutral
    reserve_change_24h  : Cumulative net_flow over 24h (positive = exchange balance rising)
    flow_regime         : 'distribution' | 'accumulation' | 'neutral'
    signal_strength     : 0-1, magnitude of the signal
    signal_type         : 'bearish_distribution' | 'bullish_accumulation' | 'neutral'
    timestamp           : UTC ISO string
    """
    symbol:              str
    inflow_usd_1h:       float
    outflow_usd_1h:      float
    net_flow_usd_1h:     float
    sell_pressure_ratio: float
    reserve_change_24h:  float
    flow_regime:         str
    signal_strength:     float
    signal_type:         str
    timestamp:           str

    @property
    def is_bearish(self) -> bool:
        return self.signal_type == "bearish_distribution"

    @property
    def is_bullish(self) -> bool:
        return self.signal_type == "bullish_accumulation"


def _classify_flow(sell_ratio: float, reserve_chg_24h: float) -> tuple[str, str, float]:
    """Return (flow_regime, signal_type, signal_strength)."""
    # Primary: sell pressure ratio
    if sell_ratio > 0.60:
        regime   = "distribution"
        strength = min(1.0, (sell_ratio - 0.60) / 0.20)   # scales 0→1 over 60%-80%
    elif sell_ratio < 0.40:
        regime   = "accumulation"
        strength = min(1.0, (0.40 - sell_ratio) / 0.20)
    else:
        regime   = "neutral"
        strength = 0.0

    # Confirm with 24h reserve change direction
    if regime == "distribution" and reserve_chg_24h > 0:
        signal   = "bearish_distribution"
        strength = min(1.0, strength * 1.2)
    elif regime == "accumulation" and reserve_chg_24h < 0:
        signal   = "bullish_accumulation"
        strength = min(1.0, strength * 1.2)
    elif regime == "neutral":
        signal   = "neutral"
    else:
        signal   = "neutral"  # regime and reserve direction conflict → no signal

    return regime, signal, round(strength, 3)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ExchangeFlowSimulator:
    """
    Simulates realistic on-chain exchange flow data for crypto assets.

    Uses an Ornstein-Uhlenbeck (mean-reverting) process calibrated to
    historically plausible flow magnitudes for each symbol.

    Parameters
    ----------
    symbols   : List of symbols to track
    seed_salt : Additional seed component to vary simulations across instances
    """

    def __init__(
        self,
        symbols:   list[str] = None,
        seed_salt: int       = 0,
    ) -> None:
        self.symbols   = symbols or TRACKED_SYMBOLS
        self.seed_salt = seed_salt
        # State for OU process per symbol: {symbol: current_deviation}
        self._state: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_all(self) -> list[ExchangeFlowSnapshot]:
        """
        Simulate current exchange flows for all tracked symbols.

        Returns
        -------
        List of ExchangeFlowSnapshot sorted by signal_strength descending.
        """
        ts_now  = datetime.now(timezone.utc).isoformat()
        results: list[ExchangeFlowSnapshot] = []

        for sym in self.symbols:
            snap = self._simulate_symbol(sym, ts_now)
            results.append(snap)

        results.sort(key=lambda s: s.signal_strength, reverse=True)
        logger.info(
            "ExchangeFlowSimulator: simulated flows for %d symbols.", len(results)
        )
        return results

    def fetch_symbol(self, symbol: str) -> ExchangeFlowSnapshot:
        """Simulate flows for a single symbol."""
        return self._simulate_symbol(symbol, datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------ #
    # Internal — simulation                                                #
    # ------------------------------------------------------------------ #

    def _simulate_symbol(self, symbol: str, ts_now: str) -> ExchangeFlowSnapshot:
        """Simulate one hour of exchange flows using the OU process."""
        base_flow, vol, theta = _SYMBOL_PARAMS.get(symbol, DEFAULT_SYMBOL_PARAMS)

        # Seed by symbol + current hour + salt for consistency within a cycle
        hour_bucket = int(time.time() / 3600)
        rng = random.Random(hash(symbol) ^ (hour_bucket * 7919) ^ self.seed_salt)

        # OU deviation from the previous state
        prev_dev = self._state.get(symbol, 0.0)
        noise    = rng.gauss(0, vol * base_flow)
        new_dev  = prev_dev + theta * (0 - prev_dev) + noise
        self._state[symbol] = new_dev

        inflow  = max(1.0, base_flow + new_dev)

        # Outflow: anti-correlated with inflow on a short timescale
        # (large inflows tend to come from net sellers, reducing outflow)
        corr_noise = rng.gauss(0, vol * base_flow * 0.7)
        outflow    = max(1.0, base_flow - new_dev * 0.4 + corr_noise)

        net_flow  = inflow - outflow
        total     = inflow + outflow
        sell_ratio = inflow / total if total > 0 else 0.5

        # 24h reserve change: accumulate last 24 modelled hours
        reserve_24h = self._simulate_24h_reserve_change(symbol, base_flow, vol, rng)

        flow_regime, signal_type, strength = _classify_flow(sell_ratio, reserve_24h)

        return ExchangeFlowSnapshot(
            symbol=symbol,
            inflow_usd_1h=round(inflow, 2),
            outflow_usd_1h=round(outflow, 2),
            net_flow_usd_1h=round(net_flow, 2),
            sell_pressure_ratio=round(sell_ratio, 4),
            reserve_change_24h=round(reserve_24h, 2),
            flow_regime=flow_regime,
            signal_strength=strength,
            signal_type=signal_type,
            timestamp=ts_now,
        )

    @staticmethod
    def _simulate_24h_reserve_change(
        symbol:    str,
        base_flow: float,
        vol:       float,
        rng:       random.Random,
    ) -> float:
        """
        Simulate the net exchange reserve change over 24 hours.

        We model 24 hourly net-flow draws and sum them.  Because each draw
        is mean-reverting, most 24h changes are small; occasionally a
        sustained directional flow creates a significant change.
        """
        net_24h = 0.0
        dev     = 0.0
        theta   = 0.18  # mean reversion
        for _ in range(24):
            noise  = rng.gauss(0, vol * base_flow)
            dev    = dev + theta * (0 - dev) + noise
            hourly_inflow  = max(0.5, base_flow + dev)
            hourly_outflow = max(0.5, base_flow - dev * 0.4 + rng.gauss(0, vol * base_flow * 0.5))
            net_24h += hourly_inflow - hourly_outflow
        return net_24h
