"""
Cross-Exchange Flow Signal: detect smart money movement between exchanges.

When large amounts of crypto move between exchanges, it signals intent:
  - Exchange inflow -> selling pressure (moving to sell)
  - Exchange outflow -> accumulation (moving to cold storage)
  - Inter-exchange transfer -> arbitrage or positioning

This module tracks flow patterns and generates signals from:
  1. Net flow direction (inflow vs outflow aggregate)
  2. Flow velocity (rate of change of flows)
  3. Whale concentration (are flows from few large wallets or many small?)
  4. Exchange-specific flow patterns (Binance vs Coinbase vs Kraken)
  5. Stablecoin flow ratio (USDT/USDC inflow = dry powder for buying)
"""

from __future__ import annotations
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ExchangeFlowSnapshot:
    """Flow data for one exchange at one timestamp."""
    exchange: str
    timestamp: float
    inflow_usd: float           # USD value flowing INTO exchange
    outflow_usd: float          # USD value flowing OUT of exchange
    net_flow_usd: float         # inflow - outflow (positive = net inflow = bearish)
    n_transactions: int
    avg_transaction_size: float
    whale_pct: float            # % of flow from top 10 addresses
    stablecoin_inflow_usd: float


@dataclass
class FlowSignal:
    """Output signal from cross-exchange flow analysis."""
    signal: float               # -1 (max bearish) to +1 (max bullish)
    confidence: float
    net_flow_direction: str     # "inflow" / "outflow" / "neutral"
    flow_velocity: float        # rate of change (positive = accelerating inflow)
    whale_activity: str         # "accumulating" / "distributing" / "neutral"
    stablecoin_dry_powder: float  # stablecoin reserves ratio (high = buying power)
    dominant_exchange: str      # which exchange is driving the flow
    narrative: str


class CrossExchangeFlowDetector:
    """
    Detect smart money movement between exchanges.

    Tracks flows across multiple exchanges and generates aggregate signals.
    """

    def __init__(self, exchanges: List[str] = None, window: int = 24):
        self.exchanges = exchanges or ["binance", "coinbase", "kraken", "okx", "bybit"]
        self.window = window
        self._flow_history: Dict[str, deque] = {
            ex: deque(maxlen=window) for ex in self.exchanges
        }
        self._aggregate_flow: deque = deque(maxlen=window * 5)
        self._stablecoin_ratio: deque = deque(maxlen=window)

    def update(self, snapshots: List[ExchangeFlowSnapshot]) -> FlowSignal:
        """Process new flow snapshots from all exchanges."""
        total_inflow = 0.0
        total_outflow = 0.0
        total_stablecoin = 0.0
        whale_activity_sum = 0.0
        dominant_ex = ""
        max_abs_flow = 0.0

        for snap in snapshots:
            if snap.exchange in self._flow_history:
                self._flow_history[snap.exchange].append(snap)

            total_inflow += snap.inflow_usd
            total_outflow += snap.outflow_usd
            total_stablecoin += snap.stablecoin_inflow_usd
            whale_activity_sum += snap.whale_pct * abs(snap.net_flow_usd)

            if abs(snap.net_flow_usd) > max_abs_flow:
                max_abs_flow = abs(snap.net_flow_usd)
                dominant_ex = snap.exchange

        net_flow = total_inflow - total_outflow
        self._aggregate_flow.append(net_flow)

        # Stablecoin ratio: inflow of stablecoins as fraction of total
        total_flow = total_inflow + total_outflow
        sc_ratio = total_stablecoin / max(total_flow, 1e-10)
        self._stablecoin_ratio.append(sc_ratio)

        # 1. Net flow signal (inflow = bearish, outflow = bullish)
        if len(self._aggregate_flow) >= 5:
            recent = np.array(list(self._aggregate_flow)[-self.window:])
            avg_flow = float(recent.mean())
            flow_std = max(float(recent.std()), 1e-10)
            flow_z = avg_flow / flow_std

            # Negative z = net outflow = bullish
            net_flow_signal = float(-np.tanh(flow_z))
        else:
            net_flow_signal = 0.0

        # 2. Flow velocity (rate of change)
        if len(self._aggregate_flow) >= 5:
            flows = list(self._aggregate_flow)
            velocity = float(np.mean(np.diff(flows[-5:])))
        else:
            velocity = 0.0

        # 3. Whale activity
        total_abs_flow = total_inflow + total_outflow
        if total_abs_flow > 0:
            whale_weighted = whale_activity_sum / total_abs_flow
        else:
            whale_weighted = 0.5

        if whale_weighted > 0.6 and net_flow < 0:
            whale_status = "accumulating"
            whale_signal = 0.3
        elif whale_weighted > 0.6 and net_flow > 0:
            whale_status = "distributing"
            whale_signal = -0.3
        else:
            whale_status = "neutral"
            whale_signal = 0.0

        # 4. Stablecoin dry powder
        avg_sc = float(np.mean(list(self._stablecoin_ratio))) if self._stablecoin_ratio else 0
        sc_signal = float(np.tanh((avg_sc - 0.3) * 5))  # high stablecoin = buying power

        # 5. Composite signal
        signal = (
            0.40 * net_flow_signal +
            0.25 * whale_signal +
            0.20 * sc_signal +
            0.15 * float(-np.tanh(velocity / max(abs(total_flow), 1e-10) * 100))
        )
        signal = float(np.clip(signal, -1, 1))

        confidence = min(1.0, abs(signal) * 2)

        # Direction
        if net_flow > total_flow * 0.1:
            direction = "inflow"
        elif net_flow < -total_flow * 0.1:
            direction = "outflow"
        else:
            direction = "neutral"

        # Narrative
        parts = []
        if abs(net_flow_signal) > 0.3:
            parts.append(f"Net exchange {'inflow (bearish)' if net_flow > 0 else 'outflow (bullish)'}")
        if whale_status != "neutral":
            parts.append(f"Whales are {whale_status}")
        if avg_sc > 0.4:
            parts.append(f"High stablecoin reserves (dry powder)")
        narrative = " | ".join(parts) if parts else "Neutral exchange flow"

        return FlowSignal(
            signal=signal,
            confidence=confidence,
            net_flow_direction=direction,
            flow_velocity=velocity,
            whale_activity=whale_status,
            stablecoin_dry_powder=avg_sc,
            dominant_exchange=dominant_ex,
            narrative=narrative,
        )
