"""
srfm_core.py — Special Relativistic Financial Mechanics: core physics library.

All SRFM physics is extracted here so any strategy can import and reuse it.
LARSA and future strategies are thin orchestration layers on top of these classes.
"""

from __future__ import annotations
import math
from collections import deque
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Causal(Enum):
    TIMELIKE  = "TIMELIKE"
    SPACELIKE = "SPACELIKE"
    LIGHTLIKE = "LIGHTLIKE"


class BHState(Enum):
    ABSENT   = "ABSENT"
    FORMING  = "FORMING"
    ACTIVE   = "ACTIVE"
    COLLAPSE = "COLLAPSE"


class MarketRegime(Enum):
    TRENDING  = "TRENDING"
    RANGING   = "RANGING"
    CRISIS    = "CRISIS"
    RECOVERY  = "RECOVERY"


# ─────────────────────────────────────────────────────────────────────────────
# MinkowskiClassifier
# ─────────────────────────────────────────────────────────────────────────────

class MinkowskiClassifier:
    """
    Classify a price move as TIMELIKE, SPACELIKE, or LIGHTLIKE using the
    Minkowski spacetime interval: ds² = c²dt² − dx²

    Parameters
    ----------
    cf : float
        'Speed of light' scaling factor.  Controls how steep a move must be
        to be considered TIMELIKE.  Smaller cf → more moves are spacelike.
    window : int
        Rolling window for normalising the spatial displacement.
    lightlike_tol : float
        Fractional tolerance around zero for LIGHTLIKE classification.
    """

    def __init__(self, cf: float = 1.0, window: int = 20, lightlike_tol: float = 0.02):
        self.cf = cf
        self.window = window
        self.lightlike_tol = lightlike_tol
        self._returns: deque = deque(maxlen=window)

    def update(self, price_return: float, time_delta: float = 1.0) -> Causal:
        """Feed one bar's return; get back a Causal classification."""
        self._returns.append(price_return)
        return self.classify(price_return, time_delta)

    def classify(self, dx: float, dt: float = 1.0) -> Causal:
        """
        dx : normalised spatial displacement (price return or log-return).
        dt : time step (typically 1 bar).
        """
        interval = (self.cf * dt) ** 2 - dx ** 2
        if abs(interval) < self.lightlike_tol * (self.cf * dt) ** 2:
            return Causal.LIGHTLIKE
        return Causal.TIMELIKE if interval > 0 else Causal.SPACELIKE

    @property
    def recent_causal_fraction(self) -> float:
        """Fraction of recent bars classified TIMELIKE (ordered, causal flow)."""
        if not self._returns:
            return 0.5
        timelike = sum(
            1 for r in self._returns
            if self.classify(r) == Causal.TIMELIKE
        )
        return timelike / len(self._returns)


# ─────────────────────────────────────────────────────────────────────────────
# BlackHoleDetector
# ─────────────────────────────────────────────────────────────────────────────

class BlackHoleDetector:
    """
    Detect Black Hole (BH) formation in price data.

    A BH forms when a cluster of strongly SPACELIKE moves (high-velocity,
    anomalous price displacement) accumulates sufficient 'gravitational mass'.
    The well collapses when volatility reverts or momentum exhausts.

    Parameters
    ----------
    bh_form_threshold : float
        Minimum cumulative mass to declare a BH active.
    bh_collapse_threshold : float
        Mass below which an active BH collapses.
    mass_decay : float
        Per-bar exponential decay applied to accumulated mass.
    max_memory : int
        How many prior well events to remember.
    """

    def __init__(
        self,
        bh_form_threshold: float = 1.5,
        bh_collapse_threshold: float = 0.4,
        mass_decay: float = 0.92,
        max_memory: int = 10,
    ):
        self.bh_form_threshold = bh_form_threshold
        self.bh_collapse_threshold = bh_collapse_threshold
        self.mass_decay = mass_decay

        self.state: BHState = BHState.ABSENT
        self.mass: float = 0.0
        self.direction: int = 0       # +1 long well, -1 short well
        self.bars_active: int = 0
        self.formation_price: Optional[float] = None

        # Well memory: list of dicts {direction, mass_peak, duration, price}
        self.well_history: deque = deque(maxlen=max_memory)
        self._current_peak: float = 0.0

    # ------------------------------------------------------------------
    def update(self, causal: Causal, price_return: float) -> BHState:
        """
        Feed one bar's Causal classification and raw return.
        Returns the current BHState after the update.
        """
        # Accrete mass on spacelike moves
        if causal == Causal.SPACELIKE:
            self.mass += abs(price_return)
            # Direction from most recent spacelike move
            if price_return != 0:
                self.direction = 1 if price_return > 0 else -1
        else:
            self.mass *= self.mass_decay

        self.mass = max(self.mass, 0.0)

        # State machine
        if self.state == BHState.ABSENT:
            if self.mass >= self.bh_form_threshold:
                self.state = BHState.FORMING
                self._current_peak = self.mass

        elif self.state == BHState.FORMING:
            if self.mass > self._current_peak:
                self._current_peak = self.mass
            if self._current_peak >= self.bh_form_threshold:
                self.state = BHState.ACTIVE
                self.bars_active = 0

        elif self.state == BHState.ACTIVE:
            self.bars_active += 1
            if self.mass > self._current_peak:
                self._current_peak = self.mass
            if self.mass <= self.bh_collapse_threshold:
                self.state = BHState.COLLAPSE

        elif self.state == BHState.COLLAPSE:
            self._record_well()
            self.state = BHState.ABSENT
            self.mass = 0.0
            self.bars_active = 0
            self._current_peak = 0.0
            self.direction = 0

        return self.state

    # ------------------------------------------------------------------
    def _record_well(self):
        self.well_history.append({
            "direction": self.direction,
            "mass_peak": self._current_peak,
            "duration_bars": self.bars_active,
        })

    # ------------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self.state == BHState.ACTIVE

    @property
    def hawking_temperature(self) -> float:
        """Proxy: higher mass → lower Hawking temperature (more stable well)."""
        if self.mass <= 0:
            return float("inf")
        return 1.0 / (8.0 * math.pi * self.mass)

    @property
    def schwarzschild_radius(self) -> float:
        """Proportional to accumulated mass."""
        return 2.0 * self.mass  # G = c = 1 units


# ─────────────────────────────────────────────────────────────────────────────
# GeodesicAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class GeodesicAnalyzer:
    """
    Measure geodesic deviation in price space — how far actual price paths
    deviate from 'straight-line' (free-fall) trajectories.

    Also computes rapidity (relativistic velocity) and causal fraction.
    """

    def __init__(self, window: int = 30, c: float = 1.0):
        self.window = window
        self.c = c
        self._returns: deque = deque(maxlen=window)
        self._causals: deque = deque(maxlen=window)

    def update(self, ret: float, causal: Causal):
        self._returns.append(ret)
        self._causals.append(causal)

    @property
    def rapidity(self) -> float:
        """
        Lorentzian rapidity: φ = atanh(v/c), where v = mean |return| / c.
        Bounded: rapidity → ∞ as velocity → c.
        """
        if not self._returns:
            return 0.0
        v = min(abs(sum(self._returns) / len(self._returns)), 0.9999 * self.c)
        return math.atanh(v / self.c)

    @property
    def causal_fraction(self) -> float:
        if not self._causals:
            return 0.5
        return sum(1 for c in self._causals if c == Causal.TIMELIKE) / len(self._causals)

    @property
    def geodesic_deviation(self) -> float:
        """
        RMS deviation of returns from their rolling mean — measures curvature
        of the price geodesic.  High deviation → spacetime is curved (crisis).
        """
        if len(self._returns) < 2:
            return 0.0
        mean = sum(self._returns) / len(self._returns)
        variance = sum((r - mean) ** 2 for r in self._returns) / len(self._returns)
        return math.sqrt(variance)

    @property
    def lorentz_factor(self) -> float:
        """γ = 1 / sqrt(1 - v²/c²)"""
        if not self._returns:
            return 1.0
        v = min(abs(sum(self._returns) / len(self._returns)), 0.9999 * self.c)
        return 1.0 / math.sqrt(1.0 - (v / self.c) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# HawkingMonitor
# ─────────────────────────────────────────────────────────────────────────────

class HawkingMonitor:
    """
    Track Hawking radiation temperature for one or more black holes.

    Hawking temperature: T_H = ℏc³ / (8πGMk_B)
    In our units (ℏ = G = c = k_B = 1): T_H = 1 / (8πM)

    Low temperature → large stable well → high confidence trade signal.
    High temperature → small unstable well → avoid or reduce size.
    """

    def __init__(self, stability_threshold: float = 0.05):
        self.stability_threshold = stability_threshold

    def temperature(self, mass: float) -> float:
        if mass <= 0:
            return float("inf")
        return 1.0 / (8.0 * math.pi * mass)

    def is_stable(self, mass: float) -> bool:
        return self.temperature(mass) < self.stability_threshold

    def size_scalar(self, mass: float, max_scalar: float = 1.0) -> float:
        """
        Position size scalar based on well stability.
        Stable (cold) well → full size.  Hot well → reduce size.
        """
        t = self.temperature(mass)
        if t >= self.stability_threshold * 10:
            return 0.0
        return max_scalar * max(0.0, 1.0 - t / (self.stability_threshold * 10))


# ─────────────────────────────────────────────────────────────────────────────
# GravitationalLens
# ─────────────────────────────────────────────────────────────────────────────

class GravitationalLens:
    """
    Model gravitational lensing amplification of a signal.

    In GR, a massive body bends light and amplifies background sources.
    Here, a BH 'bends' the signal from other indicators, amplifying
    or suppressing them based on proximity to the Einstein radius.

    Parameters
    ----------
    einstein_radius_scale : float
        Controls how much mass is needed to produce strong lensing.
    """

    def __init__(self, einstein_radius_scale: float = 1.0):
        self.scale = einstein_radius_scale

    def einstein_radius(self, mass: float, distance: float = 1.0) -> float:
        """θ_E ∝ sqrt(M / D_L)"""
        if mass <= 0 or distance <= 0:
            return 0.0
        return self.scale * math.sqrt(mass / distance)

    def mu_amplification(self, mass: float, source_offset: float = 0.5) -> float:
        """
        Magnification μ = (u² + 2) / (u * sqrt(u² + 4))
        where u = source_offset / einstein_radius.
        """
        r_e = self.einstein_radius(mass)
        if r_e <= 0:
            return 1.0
        u = source_offset / r_e
        if u <= 0:
            return 1.0
        return (u ** 2 + 2.0) / (u * math.sqrt(u ** 2 + 4.0))

    def amplify_signal(self, raw_signal: float, mass: float, source_offset: float = 0.5) -> float:
        """Apply lensing amplification to any raw signal."""
        return raw_signal * self.mu_amplification(mass, source_offset)


# ─────────────────────────────────────────────────────────────────────────────
# ProperTimeClock
# ─────────────────────────────────────────────────────────────────────────────

class ProperTimeClock:
    """
    Accumulate proper time τ along the price trajectory.

    Proper time: dτ² = dt² − (dx/c)²
    An observer moving through price space ages more slowly than coordinate time
    when making large spacelike moves.  Used to gate entries and exits.

    Parameters
    ----------
    c : float
        Speed of light in our units.
    min_proper_time : float
        Minimum proper time before a new entry is allowed.
    """

    def __init__(self, c: float = 1.0, min_proper_time: float = 5.0):
        self.c = c
        self.min_proper_time = min_proper_time
        self.proper_time: float = 0.0
        self._last_gate_time: float = 0.0

    def tick(self, price_return: float, dt: float = 1.0):
        """Advance proper time by one bar."""
        dx = price_return
        interval = (self.c * dt) ** 2 - dx ** 2
        if interval > 0:
            self.proper_time += math.sqrt(interval)
        # Spacelike moves do NOT advance proper time

    def gate_passed(self) -> bool:
        """True if enough proper time has elapsed since last gate reset."""
        return (self.proper_time - self._last_gate_time) >= self.min_proper_time

    def reset_gate(self):
        """Call when an entry is taken."""
        self._last_gate_time = self.proper_time

    @property
    def elapsed_since_gate(self) -> float:
        return self.proper_time - self._last_gate_time
