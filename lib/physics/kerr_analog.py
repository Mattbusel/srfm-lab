"""
Kerr metric analog for rotating/spinning market black holes.

The standard BH analog (Schwarzschild) models a non-rotating attractor.
The Kerr analog extends this with angular momentum J (spin) arising from
directional order flow imbalance — the market "spins" when buy/sell flow
is consistently asymmetric.

Key physics:
  - Ergosphere: region where market "drags" all participants (trend regime)
  - Frame dragging: position relative to BH is dragged in direction of spin
  - Penrose process: extract energy from ergosphere (momentum fade trades)
  - Innermost stable circular orbit (ISCO): minimum hold time in spinning market
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Kerr parameters ───────────────────────────────────────────────────────────

@dataclass
class KerrMarketState:
    """
    Kerr BH analog state for a market instrument.
    M = total BH mass (from BHPhysicsEngine)
    J = angular momentum = cumulative order flow imbalance
    a = spin parameter = J/M (dimensionless: a in [0, 1))
    """
    mass: float = 0.0
    spin: float = 0.0          # |a| <= 1 (0=Schwarzschild, 1=extremal Kerr)
    spin_sign: int = 1         # +1 = bullish spin, -1 = bearish spin
    frame_drag_rate: float = 0.0  # omega_H = a / (2*r_H)

    def update(self, mass: float, flow_imbalance: float, decay: float = 0.97) -> None:
        """
        Update Kerr state from new mass and order flow imbalance.
        flow_imbalance: (buy_vol - sell_vol) / total_vol, range [-1, 1]
        """
        self.mass = mass
        # Angular momentum accumulates like mass
        angular_momentum = self.spin * self.mass * decay + (1 - decay) * flow_imbalance
        # Normalize to [0, 1)
        a = abs(angular_momentum)
        self.spin = float(min(a, 0.998))  # never reach extremal
        self.spin_sign = 1 if angular_momentum >= 0 else -1
        if mass > 0:
            self.frame_drag_rate = self.spin / max(2 * self._horizon_radius(), 1e-10)

    def _horizon_radius(self) -> float:
        """Event horizon r+ = M + sqrt(M^2 - a^2) (M=1 normalized)."""
        a = self.spin
        if a > 1:
            return 1.0
        return 1 + math.sqrt(max(1 - a ** 2, 0))

    @property
    def ergosphere_radius(self) -> float:
        """
        Ergosphere outer boundary in equatorial plane:
        r_ergo = M + sqrt(M^2 - a^2*cos^2(theta)) (theta=pi/2 at equator)
        = M + M = 2M (at equator)
        In our analog: ergosphere = mass * 2
        """
        return self.mass * 2.0

    @property
    def isco_radius(self) -> float:
        """
        Innermost Stable Circular Orbit (ISCO) in Kerr metric.
        For prograde orbits: r_isco = M * (3 + Z2 - sqrt((3-Z1)*(3+Z1+2*Z2)))
        Simplified: ranges from 6M (Schwarzschild) to M (extremal prograde).
        """
        a = self.spin
        # Full Kerr ISCO formula
        Z1 = 1 + (1 - a ** 2) ** (1 / 3) * (
            (1 + a) ** (1 / 3) + (1 - a) ** (1 / 3)
        )
        Z2 = math.sqrt(3 * a ** 2 + Z1 ** 2)
        r_isco = self.mass * (3 + Z2 - math.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2)))
        return max(r_isco, self.mass)

    @property
    def in_ergosphere(self) -> bool:
        """Is market radius < ergosphere? (strong trend drag region)."""
        return self.mass > 1.0  # proxy: high mass = inside ergosphere

    @property
    def penrose_efficiency(self) -> float:
        """
        Penrose process efficiency: energy extracted from ergosphere.
        eta = 1 - sqrt((1 - a/2)) ~ a/4 for small a.
        In trading: profit available from momentum fade in spinning market.
        """
        return float(1 - math.sqrt(max(1 - self.spin / 2, 0)))


# ── Frame dragging signal ─────────────────────────────────────────────────────

class KerrFrameDragSignal:
    """
    Frame dragging in Kerr geometry: nearby particles are dragged
    along the spin direction regardless of their own motion.

    In market terms: in a strongly spinning (trending) market, all
    positions are dragged toward the spin direction. Anti-spin trades
    require extra energy to maintain.
    """

    def __init__(
        self,
        spin_ema_fast: int = 12,
        spin_ema_slow: int = 48,
        imbalance_window: int = 20,
    ):
        self.state = KerrMarketState()
        self._fast = spin_ema_fast
        self._slow = spin_ema_slow
        self._imbalance_window = imbalance_window
        self._prices: list[float] = []
        self._volumes: list[float] = []
        self._buy_vols: list[float] = []

    def update(self, price: float, volume: float, buy_fraction: float = 0.5) -> None:
        """
        Update with new bar.
        buy_fraction: estimated fraction of volume that was buyer-initiated.
        """
        self._prices.append(price)
        self._volumes.append(volume)
        self._buy_vols.append(buy_fraction * volume)

    def compute(self, bh_mass: float) -> dict:
        """Compute Kerr signal metrics."""
        if len(self._prices) < self._slow:
            return {"frame_drag": 0.0, "spin": 0.0, "ergosphere": False}

        # Flow imbalance
        w = self._imbalance_window
        recent_buy = sum(self._buy_vols[-w:])
        recent_total = sum(self._volumes[-w:]) + 1e-10
        imbalance = 2 * recent_buy / recent_total - 1  # [-1, 1]

        self.state.update(bh_mass, imbalance)

        # Frame drag force = omega_H * angular_momentum_proxy
        drag = self.state.frame_drag_rate * self.state.spin_sign

        # In-ergosphere: trade WITH spin direction
        in_ergo = self.state.in_ergosphere
        direction_bias = self.state.spin_sign if in_ergo else 0

        return {
            "frame_drag": float(drag),
            "spin": float(self.state.spin),
            "spin_sign": int(self.state.spin_sign),
            "in_ergosphere": bool(in_ergo),
            "direction_bias": int(direction_bias),
            "penrose_efficiency": float(self.state.penrose_efficiency),
            "isco_bars": float(self.state.isco_radius / max(bh_mass, 1e-10) * 6),  # ~min hold
            "flow_imbalance": float(imbalance),
        }


# ── Kerr geodesic deviation ────────────────────────────────────────────────────

def kerr_geodesic_deviation(
    r: float,
    a: float,
    M: float = 1.0,
    prograde: bool = True,
) -> float:
    """
    Geodesic deviation in Kerr spacetime: how much a circular orbit
    deviates from Keplerian due to frame dragging.

    r: orbital radius
    a: spin parameter
    prograde: if True, orbit co-rotates with spin (lower energy needed)

    Returns effective orbital frequency (analog: optimal trade hold time).
    """
    if r <= 0 or M <= 0:
        return 0.0
    # Kerr orbital frequency for circular equatorial orbit
    sign = 1 if prograde else -1
    omega = math.sqrt(M / r ** 3) / (1 + sign * a * math.sqrt(M / r ** 3))
    return float(omega)


def frame_drag_correction(
    raw_signal: float,
    spin: float,
    spin_sign: int,
    correction_strength: float = 0.3,
) -> float:
    """
    Apply Kerr frame dragging correction to a raw BH signal.
    Amplifies signals aligned with spin; attenuates opposing signals.
    """
    alignment = raw_signal * spin_sign
    if alignment > 0:
        # Signal aligned with spin: amplify
        return raw_signal * (1 + correction_strength * spin)
    else:
        # Signal opposing spin: attenuate
        return raw_signal * (1 - correction_strength * spin)
