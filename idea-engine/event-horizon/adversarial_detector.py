"""
Adversarial Detector: detect when other algorithms are front-running you.

In live markets, other algorithms can detect your order patterns and
trade against you. This module:
1. Monitors slippage patterns for anomalies (consistently worse fills = front-running)
2. Detects order flow signature matching (someone is trading the same signal)
3. Identifies crowded trade detection (too many algos on the same side)
4. Measures information leakage (price moves before your order hits)
5. Adapts execution strategy to avoid detection

If you're being systematically front-run, the module recommends:
  - Randomizing execution timing
  - Using dark pools instead of lit venues
  - Breaking orders into smaller pieces
  - Temporarily pausing the compromised signal
"""

from __future__ import annotations
import math
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FrontRunSignal:
    """Evidence of front-running or information leakage."""
    signal_type: str              # "slippage_anomaly" / "signature_match" / "crowding" / "leakage"
    severity: float               # 0-1
    description: str
    affected_symbol: str
    evidence: Dict = field(default_factory=dict)
    recommended_action: str = ""


@dataclass
class ExecutionFingerprint:
    """The detectable pattern of our execution behavior."""
    avg_order_size: float
    avg_execution_time_ms: float
    preferred_hour: int
    rebalance_frequency_bars: int
    signal_delay_bars: int         # how many bars after signal do we trade?
    venue_preference: str


class AdversarialDetector:
    """
    Detect when other market participants are exploiting your trading patterns.
    """

    def __init__(self, slippage_window: int = 100, leakage_window: int = 50):
        self.slippage_window = slippage_window
        self.leakage_window = leakage_window

        # Per-symbol tracking
        self._slippage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=slippage_window))
        self._pre_trade_moves: Dict[str, deque] = defaultdict(lambda: deque(maxlen=leakage_window))
        self._order_sizes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=slippage_window))
        self._execution_times: deque = deque(maxlen=slippage_window)

        self._alerts: List[FrontRunSignal] = []

    def record_execution(
        self,
        symbol: str,
        intended_price: float,       # price when signal was generated
        execution_price: float,      # actual fill price
        side: str,                   # "buy" / "sell"
        size: float,
        pre_signal_price_5bar: float,  # price 5 bars before our signal
        execution_time_ms: float,
    ) -> List[FrontRunSignal]:
        """
        Record an execution and check for adversarial patterns.
        Returns list of front-running signals (empty if clean).
        """
        signals = []

        # Slippage: how much worse was the fill vs intended?
        if side == "buy":
            slippage_bps = (execution_price - intended_price) / max(intended_price, 1e-10) * 10000
        else:
            slippage_bps = (intended_price - execution_price) / max(intended_price, 1e-10) * 10000

        self._slippage_history[symbol].append(slippage_bps)
        self._order_sizes[symbol].append(size)
        self._execution_times.append(execution_time_ms)

        # Pre-trade move: did price move in our direction BEFORE we traded?
        # This is the classic front-running signature
        if side == "buy":
            pre_move_bps = (intended_price - pre_signal_price_5bar) / max(pre_signal_price_5bar, 1e-10) * 10000
        else:
            pre_move_bps = (pre_signal_price_5bar - intended_price) / max(pre_signal_price_5bar, 1e-10) * 10000

        self._pre_trade_moves[symbol].append(pre_move_bps)

        # 1. Slippage anomaly detection
        slippage_signal = self._check_slippage_anomaly(symbol)
        if slippage_signal:
            signals.append(slippage_signal)

        # 2. Information leakage detection
        leakage_signal = self._check_information_leakage(symbol)
        if leakage_signal:
            signals.append(leakage_signal)

        # 3. Crowding detection (from slippage trend)
        crowding_signal = self._check_crowding(symbol)
        if crowding_signal:
            signals.append(crowding_signal)

        self._alerts.extend(signals)
        return signals

    def _check_slippage_anomaly(self, symbol: str) -> Optional[FrontRunSignal]:
        """Detect abnormally high slippage (someone front-running our orders)."""
        history = list(self._slippage_history[symbol])
        if len(history) < 20:
            return None

        recent = np.array(history[-10:])
        baseline = np.array(history[:-10])

        recent_avg = float(recent.mean())
        baseline_avg = float(baseline.mean())
        baseline_std = float(baseline.std())

        if baseline_std < 1e-10:
            return None

        z_score = (recent_avg - baseline_avg) / baseline_std

        if z_score > 2.0:  # slippage is 2+ std above normal
            return FrontRunSignal(
                signal_type="slippage_anomaly",
                severity=min(1.0, z_score / 4),
                description=f"{symbol}: slippage {recent_avg:.1f} bps vs baseline {baseline_avg:.1f} bps "
                            f"(z={z_score:.1f}). Possible front-running.",
                affected_symbol=symbol,
                evidence={"recent_avg_bps": recent_avg, "baseline_avg_bps": baseline_avg, "z_score": z_score},
                recommended_action="Switch to dark pool execution or randomize timing",
            )
        return None

    def _check_information_leakage(self, symbol: str) -> Optional[FrontRunSignal]:
        """Detect if price consistently moves in our direction BEFORE we trade."""
        moves = list(self._pre_trade_moves[symbol])
        if len(moves) < 20:
            return None

        recent = np.array(moves[-20:])

        # If pre-trade moves are consistently positive (price moves in our direction
        # before we trade), someone knows our signal or is faster
        avg_pre_move = float(recent.mean())
        pct_favorable = float(np.mean(recent > 0))

        if avg_pre_move > 3.0 and pct_favorable > 0.6:
            return FrontRunSignal(
                signal_type="leakage",
                severity=min(1.0, avg_pre_move / 10),
                description=f"{symbol}: price moves {avg_pre_move:.1f} bps in our direction "
                            f"BEFORE our order ({pct_favorable:.0%} of the time). Information leakage detected.",
                affected_symbol=symbol,
                evidence={"avg_pre_move_bps": avg_pre_move, "favorable_pct": pct_favorable},
                recommended_action="Increase signal delay by 1-2 bars, use iceberg orders, rotate venues",
            )
        return None

    def _check_crowding(self, symbol: str) -> Optional[FrontRunSignal]:
        """Detect if too many algorithms are on the same trade (crowding)."""
        slippage = list(self._slippage_history[symbol])
        sizes = list(self._order_sizes[symbol])

        if len(slippage) < 30 or len(sizes) < 30:
            return None

        slip = np.array(slippage[-30:])
        sz = np.array(sizes[-30:])

        # Crowding signal: slippage increasing while order size is constant
        # (other algos competing for same liquidity)
        if sz.std() < sz.mean() * 0.1:  # our size is consistent
            slip_trend = float(np.polyfit(range(len(slip)), slip, 1)[0])
            if slip_trend > 0.1:  # slippage trending up
                return FrontRunSignal(
                    signal_type="crowding",
                    severity=min(1.0, slip_trend),
                    description=f"{symbol}: slippage increasing {slip_trend:.2f} bps/trade while our "
                                f"size is constant. Multiple algorithms competing for same liquidity.",
                    affected_symbol=symbol,
                    evidence={"slippage_trend": slip_trend, "avg_size": float(sz.mean())},
                    recommended_action="Consider pausing this signal or trading different instruments",
                )
        return None

    def get_stealth_recommendations(self, symbol: str) -> Dict:
        """Recommend execution modifications to avoid detection."""
        recent_alerts = [a for a in self._alerts if a.affected_symbol == symbol]

        if not recent_alerts:
            return {"status": "clean", "modifications": []}

        modifications = []
        max_severity = max(a.severity for a in recent_alerts)

        if max_severity > 0.7:
            modifications.extend([
                "Switch to 100% dark pool execution",
                "Randomize order timing by 1-5 minutes",
                "Split orders into 3-5 child orders",
                "Add 1 bar delay before execution",
            ])
        elif max_severity > 0.3:
            modifications.extend([
                "Increase dark pool allocation to 60%",
                "Add random jitter (0-60s) to execution timing",
                "Use iceberg orders (show 20% of full size)",
            ])
        else:
            modifications.append("Monitor but no immediate action needed")

        return {
            "status": "compromised" if max_severity > 0.5 else "monitoring",
            "max_severity": float(max_severity),
            "n_alerts": len(recent_alerts),
            "modifications": modifications,
        }

    def get_report(self) -> Dict:
        """Full adversarial detection report."""
        by_symbol = defaultdict(list)
        for alert in self._alerts:
            by_symbol[alert.affected_symbol].append(alert)

        return {
            "total_alerts": len(self._alerts),
            "by_symbol": {
                sym: {
                    "n_alerts": len(alerts),
                    "max_severity": max(a.severity for a in alerts),
                    "types": list(set(a.signal_type for a in alerts)),
                }
                for sym, alerts in by_symbol.items()
            },
            "most_compromised": max(by_symbol.items(), key=lambda x: len(x[1]))[0]
            if by_symbol else None,
            "overall_status": "clean" if not self._alerts else
                              "compromised" if any(a.severity > 0.5 for a in self._alerts) else "monitoring",
        }
