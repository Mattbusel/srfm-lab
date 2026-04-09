"""
Spacetime Arbitrage Detector: exploit finite information propagation speed across exchanges.

In Special Relativity, events are connected by light cones. Nothing travels
faster than c. In financial markets, information propagates at finite speed
too: network latency, order book update times, and human reaction time create
a "financial speed of light" between exchanges.

A price shock on Exchange A creates a light cone of predictable price action
on Exchange B. If the shock on A is classified as SPACELIKE (beta >= 1),
it has moved faster than the local causal mechanism -- the wavefront has not
yet arrived at B.

This module:
  1. Maintains Minkowski metrics per exchange pair
  2. Detects when a price move on one exchange is SPACELIKE relative to another
  3. Predicts the arrival time and magnitude of the wavefront on the target exchange
  4. Generates arbitrage signals during the propagation window

The "speed of light" CF is calibrated per exchange pair from historical
latency and correlation data.
"""

from __future__ import annotations
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Exchange Observer: one "reference frame" in the multi-observer model
# ---------------------------------------------------------------------------

@dataclass
class ExchangeObserver:
    """
    An exchange as a reference frame in financial spacetime.
    Each exchange has its own local time (tick rate) and price (position).
    """
    name: str
    latency_ms: float = 0.0       # network latency to this exchange
    tick_rate_hz: float = 1.0     # update frequency
    last_price: float = 0.0
    last_timestamp: float = 0.0
    price_history: deque = field(default_factory=lambda: deque(maxlen=200))
    timestamp_history: deque = field(default_factory=lambda: deque(maxlen=200))

    def update(self, price: float, timestamp: float) -> None:
        self.price_history.append(price)
        self.timestamp_history.append(timestamp)
        self.last_price = price
        self.last_timestamp = timestamp


# ---------------------------------------------------------------------------
# Minkowski Interval Between Exchanges
# ---------------------------------------------------------------------------

@dataclass
class SpacetimeInterval:
    """
    The Minkowski interval between two events on different exchanges.

    ds^2 = -CF^2 * dt^2 + (d ln P)^2

    If ds^2 < 0: TIMELIKE (causal, information has propagated)
    If ds^2 > 0: SPACELIKE (acausal, wavefront hasn't arrived yet)
    If ds^2 = 0: LIGHTLIKE (wavefront is arriving right now)
    """
    exchange_a: str
    exchange_b: str
    dt: float               # time difference (seconds)
    d_ln_price: float       # log price difference
    ds_squared: float       # Minkowski interval
    beta: float             # velocity parameter (d_ln_P / (CF * dt))
    classification: str     # TIMELIKE / SPACELIKE / LIGHTLIKE
    cf: float               # speed of light for this exchange pair


def compute_spacetime_interval(
    price_a: float,
    price_b: float,
    time_a: float,
    time_b: float,
    cf: float,
) -> SpacetimeInterval:
    """
    Compute the Minkowski interval between two price events on different exchanges.

    cf: the "speed of light" for this exchange pair (calibrated from historical data).
    """
    dt = abs(time_b - time_a) + 1e-10  # seconds
    d_ln_p = abs(math.log(max(price_b, 1e-10)) - math.log(max(price_a, 1e-10)))

    ds2 = -(cf**2) * (dt**2) + d_ln_p**2
    beta = d_ln_p / (cf * dt)

    if ds2 < -1e-12:
        classification = "TIMELIKE"
    elif ds2 > 1e-12:
        classification = "SPACELIKE"
    else:
        classification = "LIGHTLIKE"

    return SpacetimeInterval(
        exchange_a="A",
        exchange_b="B",
        dt=dt,
        d_ln_price=d_ln_p,
        ds_squared=ds2,
        beta=beta,
        classification=classification,
        cf=cf,
    )


# ---------------------------------------------------------------------------
# Light Cone: the causal boundary of price information
# ---------------------------------------------------------------------------

@dataclass
class LightCone:
    """
    The light cone of a price event: defines which future events are
    causally accessible from this event.

    Inside the cone: price has propagated (no arb opportunity)
    Outside the cone: price hasn't arrived yet (potential arb)
    On the cone: wavefront arriving (maximum urgency)
    """
    origin_exchange: str
    origin_price: float
    origin_time: float
    origin_return: float         # the magnitude of the shock
    cf: float

    def arrival_time(self, target_exchange_latency_ms: float) -> float:
        """
        Predict when the wavefront arrives at the target exchange.
        arrival = origin_time + |d_ln_P| / CF + latency
        """
        propagation_time = abs(self.origin_return) / max(self.cf, 1e-10)
        latency_seconds = target_exchange_latency_ms / 1000.0
        return self.origin_time + propagation_time + latency_seconds

    def expected_impact(self, target_correlation: float) -> float:
        """
        Expected price impact on the target exchange.
        impact = origin_return * correlation * decay_factor
        """
        # Decay: impact diminishes with time (like gravitational wave amplitude)
        decay = 0.8  # 80% of the move propagates
        return self.origin_return * target_correlation * decay

    def is_expired(self, current_time: float, max_window_seconds: float = 5.0) -> bool:
        """Has the light cone window closed?"""
        return (current_time - self.origin_time) > max_window_seconds


# ---------------------------------------------------------------------------
# CF Calibrator: learn the "speed of light" per exchange pair
# ---------------------------------------------------------------------------

class CFCalibrator:
    """
    Calibrate the financial speed of light (CF) for each exchange pair.

    CF is estimated from historical data as:
      CF = median(|d_ln_P| / dt) for events where exchanges are correlated

    A higher CF means information propagates faster between these exchanges
    (tighter coupling, less arb opportunity).
    """

    def __init__(self, window: int = 500):
        self.window = window
        self._observations: Dict[Tuple[str, str], deque] = {}

    def record(self, exchange_a: str, exchange_b: str,
               price_a: float, price_b: float,
               time_a: float, time_b: float) -> None:
        """Record a pair of contemporaneous price observations."""
        key = tuple(sorted([exchange_a, exchange_b]))
        if key not in self._observations:
            self._observations[key] = deque(maxlen=self.window)

        dt = abs(time_b - time_a) + 1e-6
        d_ln_p = abs(math.log(max(price_b, 1e-10)) - math.log(max(price_a, 1e-10)))

        if dt > 0 and d_ln_p > 0:
            self._observations[key].append(d_ln_p / dt)

    def get_cf(self, exchange_a: str, exchange_b: str) -> float:
        """Get calibrated CF for an exchange pair."""
        key = tuple(sorted([exchange_a, exchange_b]))
        obs = self._observations.get(key, [])
        if len(obs) < 10:
            return 0.001  # default: slow propagation (conservative)
        return float(np.median(list(obs)))


# ---------------------------------------------------------------------------
# Arbitrage Signal Generator
# ---------------------------------------------------------------------------

@dataclass
class ArbSignal:
    """A spacetime arbitrage signal."""
    signal_id: str
    source_exchange: str
    target_exchange: str
    direction: float              # +1 buy target, -1 sell target
    magnitude: float              # expected move on target (log return)
    urgency: float                # 0-1, how close to wavefront arrival
    confidence: float             # 0-1, based on historical accuracy
    window_remaining_ms: float    # milliseconds before wavefront arrives
    classification: str           # SPACELIKE / LIGHTLIKE
    beta: float                   # superluminal velocity


class SpacetimeArbitrageDetector:
    """
    Detect and trade price discrepancies that propagate at finite speed
    across exchanges.

    The detector:
    1. Monitors price feeds from multiple exchanges
    2. Computes Minkowski intervals between exchange pairs
    3. Detects SPACELIKE events (information hasn't propagated yet)
    4. Generates arbitrage signals during the propagation window
    5. Tracks accuracy and auto-calibrates CF
    """

    def __init__(self, exchanges: List[str], symbol: str = "BTC"):
        self.symbol = symbol
        self.observers = {name: ExchangeObserver(name=name) for name in exchanges}
        self.calibrator = CFCalibrator()
        self.active_cones: List[LightCone] = []
        self._signal_counter = 0
        self._accuracy_history: deque = deque(maxlen=200)

        # Cross-exchange correlations (learned)
        self._correlations: Dict[Tuple[str, str], float] = {}
        self._return_histories: Dict[str, deque] = {
            name: deque(maxlen=100) for name in exchanges
        }

    def update_price(self, exchange: str, price: float, timestamp: float) -> List[ArbSignal]:
        """
        Update price for one exchange and check for arbitrage signals
        against all other exchanges.
        """
        signals = []

        if exchange not in self.observers:
            return signals

        observer = self.observers[exchange]
        prev_price = observer.last_price

        # Record return
        if prev_price > 0:
            ret = math.log(price / prev_price)
            self._return_histories[exchange].append(ret)
        else:
            ret = 0.0

        observer.update(price, timestamp)

        # Check against all other exchanges
        for other_name, other_obs in self.observers.items():
            if other_name == exchange or other_obs.last_price <= 0:
                continue

            # Calibrate CF
            self.calibrator.record(exchange, other_name,
                                    price, other_obs.last_price,
                                    timestamp, other_obs.last_timestamp)

            cf = self.calibrator.get_cf(exchange, other_name)

            # Compute Minkowski interval
            interval = compute_spacetime_interval(
                price, other_obs.last_price,
                timestamp, other_obs.last_timestamp,
                cf,
            )
            interval.exchange_a = exchange
            interval.exchange_b = other_name

            # Detect SPACELIKE events: the price shock hasn't arrived at the other exchange yet
            if interval.classification == "SPACELIKE" and abs(ret) > 0.001:
                # This is an arb opportunity
                correlation = self._get_correlation(exchange, other_name)

                # Create light cone
                cone = LightCone(
                    origin_exchange=exchange,
                    origin_price=price,
                    origin_time=timestamp,
                    origin_return=ret,
                    cf=cf,
                )

                arrival = cone.arrival_time(other_obs.latency_ms)
                expected_impact = cone.expected_impact(correlation)
                window_ms = max(0, (arrival - timestamp) * 1000)

                if abs(expected_impact) > 0.0005 and window_ms > 10:
                    self._signal_counter += 1
                    signals.append(ArbSignal(
                        signal_id=f"arb_{self._signal_counter:06d}",
                        source_exchange=exchange,
                        target_exchange=other_name,
                        direction=float(np.sign(expected_impact)),
                        magnitude=abs(expected_impact),
                        urgency=min(1.0, 1.0 / max(window_ms / 100, 0.1)),
                        confidence=min(0.9, abs(correlation) * abs(interval.beta)),
                        window_remaining_ms=window_ms,
                        classification=interval.classification,
                        beta=interval.beta,
                    ))

                self.active_cones.append(cone)

        # Clean expired cones
        self.active_cones = [c for c in self.active_cones
                              if not c.is_expired(timestamp)]

        return signals

    def _get_correlation(self, exchange_a: str, exchange_b: str) -> float:
        """Get rolling correlation between two exchanges."""
        key = tuple(sorted([exchange_a, exchange_b]))
        if key in self._correlations:
            return self._correlations[key]

        hist_a = list(self._return_histories.get(exchange_a, []))
        hist_b = list(self._return_histories.get(exchange_b, []))
        n = min(len(hist_a), len(hist_b))
        if n < 20:
            return 0.8  # default: assume high correlation for same asset

        a = np.array(hist_a[-n:])
        b = np.array(hist_b[-n:])
        if a.std() > 1e-10 and b.std() > 1e-10:
            corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = 0.8

        self._correlations[key] = corr
        return corr

    def record_outcome(self, signal_id: str, was_profitable: bool) -> None:
        """Track accuracy for self-calibration."""
        self._accuracy_history.append(1.0 if was_profitable else 0.0)

    def get_accuracy(self) -> float:
        if not self._accuracy_history:
            return 0.5
        return float(np.mean(list(self._accuracy_history)))

    def get_diagnostics(self) -> Dict:
        """Full system diagnostics."""
        cf_estimates = {}
        for key, obs in self.calibrator._observations.items():
            if obs:
                cf_estimates[f"{key[0]}-{key[1]}"] = float(np.median(list(obs)))

        return {
            "exchanges": list(self.observers.keys()),
            "active_light_cones": len(self.active_cones),
            "cf_estimates": cf_estimates,
            "correlations": {f"{k[0]}-{k[1]}": v for k, v in self._correlations.items()},
            "signal_accuracy": self.get_accuracy(),
            "total_signals_generated": self._signal_counter,
        }
