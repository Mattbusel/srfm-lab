"""
BH Physics Extensions: advanced GR concepts for trading signals.

Extends the core BH engine with three new physics-derived signals:
1. Lense-Thirring Frame Dragging: cross-asset momentum transfer prediction
2. Penrose Process Exit: dynamic exit strategy using ergosphere boundaries
3. Riemann Curvature Regime Detection: covariance-based regime classification

All functions are pure Python/numpy, designed to integrate with the
existing BHState class in tools/live_trader_alpaca.py.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Lense-Thirring Frame Dragging
# ---------------------------------------------------------------------------

@dataclass
class FrameDraggingState:
    """
    Models momentum transfer from a dominant asset (primary BH) to
    correlated assets (test particles) via frame dragging.

    In GR: a rotating mass drags spacetime, causing nearby objects
    to precess. In markets: a high-momentum leader (BTC) "drags"
    correlated assets in its direction before their own momentum builds.
    """
    primary_symbol: str = "BTC"
    window: int = 24            # bars for spin/mass estimation
    primary_mass: float = 0.0   # BH mass of primary asset
    primary_spin: float = 0.0   # angular momentum (momentum of momentum)
    dragging_coefficients: Dict[str, float] = field(default_factory=dict)

    # Rolling buffers
    primary_returns: deque = field(default_factory=lambda: deque(maxlen=24))
    satellite_returns: Dict[str, deque] = field(default_factory=dict)


class LenseThirringPredictor:
    """
    Predict momentum transfer from a dominant asset to correlated assets.

    The "frame dragging" coefficient omega measures how much the primary
    asset's angular momentum (change in momentum) predicts the satellite
    asset's next-bar return.

    omega = correlation(d(momentum_primary), return_satellite, lag=1)

    High omega -> strong frame dragging -> satellite follows primary.
    """

    def __init__(self, primary: str = "BTC", satellites: List[str] = None,
                 mass_window: int = 24, spin_window: int = 12):
        self.primary = primary
        self.satellites = satellites or []
        self.mass_window = mass_window
        self.spin_window = spin_window

        self._primary_returns = deque(maxlen=mass_window)
        self._primary_momentum = deque(maxlen=mass_window)
        self._sat_returns: Dict[str, deque] = {
            s: deque(maxlen=mass_window) for s in self.satellites
        }
        self._omega: Dict[str, float] = {s: 0.0 for s in self.satellites}

    def update(self, primary_return: float, satellite_returns: Dict[str, float]) -> None:
        """Update with new bar data."""
        self._primary_returns.append(primary_return)

        # Primary momentum = sum of recent returns (mass * velocity analog)
        if len(self._primary_returns) >= 3:
            recent = list(self._primary_returns)[-3:]
            momentum = sum(recent)
        else:
            momentum = primary_return
        self._primary_momentum.append(momentum)

        for sym, ret in satellite_returns.items():
            if sym in self._sat_returns:
                self._sat_returns[sym].append(ret)

        # Recompute dragging coefficients
        if len(self._primary_momentum) >= self.spin_window:
            # Angular momentum = d(momentum)/dt (second derivative of price)
            mom_list = list(self._primary_momentum)
            spin = [mom_list[i] - mom_list[i-1] for i in range(1, len(mom_list))]

            for sym in self.satellites:
                sat_rets = list(self._sat_returns.get(sym, []))
                n = min(len(spin), len(sat_rets) - 1)
                if n >= 5:
                    # Lagged correlation: spin[t] predicts satellite_return[t+1]
                    x = np.array(spin[-n:])
                    y = np.array(sat_rets[-n:])
                    if x.std() > 1e-10 and y.std() > 1e-10:
                        self._omega[sym] = float(np.corrcoef(x, y)[0, 1])

    def get_signal(self, symbol: str) -> float:
        """
        Get the frame-dragging signal for a satellite asset.

        Returns: float in [-1, 1]
          Positive: primary's angular momentum predicts satellite moving up
          Negative: primary's angular momentum predicts satellite moving down
          Near zero: no dragging effect detected
        """
        omega = self._omega.get(symbol, 0.0)

        # Scale by current spin magnitude
        if len(self._primary_momentum) >= 3:
            mom_list = list(self._primary_momentum)
            current_spin = mom_list[-1] - mom_list[-2]
            # Signal = dragging_coefficient * current_spin_direction
            signal = omega * math.tanh(current_spin * 50)
        else:
            signal = 0.0

        return float(np.clip(signal, -1, 1))

    def get_all_signals(self) -> Dict[str, float]:
        return {sym: self.get_signal(sym) for sym in self.satellites}

    def get_diagnostics(self) -> Dict:
        return {
            "primary": self.primary,
            "omega_coefficients": dict(self._omega),
            "primary_momentum_current": float(self._primary_momentum[-1]) if self._primary_momentum else 0.0,
            "n_observations": len(self._primary_returns),
        }


# ---------------------------------------------------------------------------
# 2. Penrose Process Exit Strategy
# ---------------------------------------------------------------------------

@dataclass
class PenroseExitSignal:
    """Output of the Penrose process exit analysis."""
    in_ergosphere: bool         # True if price is in the ergosphere zone
    exit_action: str            # "hold" / "split" / "full_exit"
    tight_stop_distance: float  # distance for the loss component (% of price)
    runner_trail_distance: float # distance for the gain component (% of price)
    ergosphere_depth: float     # how deep into the ergosphere (0=boundary, 1=horizon)
    energy_extraction_potential: float  # expected gain from Penrose split


class PenroseExitStrategy:
    """
    The Penrose Process: extract energy from a rotating black hole's ergosphere.

    In trading terms:
    - The "horizon" is the core BH signal (mass > BH_FORM)
    - The "ergosphere" is the zone where momentum is high but volatile
      (high spin, BH mass near formation threshold)
    - The "Penrose split" splits a position into:
      - Component A: tight stop (negative energy, sacrificed to the BH)
      - Component B: trailing runner (positive energy, extracted profit)

    The ratio of tight_stop to trailing_stop depends on the spin-to-mass ratio.
    High spin/mass = more energy extractable = wider trail, tighter stop.
    """

    def __init__(self, mass_threshold: float = 1.92, spin_threshold: float = 0.5):
        self.mass_threshold = mass_threshold
        self.spin_threshold = spin_threshold

    def analyze(
        self,
        bh_mass: float,
        bh_spin: float,       # kerr spin parameter (angular momentum / mass)
        atr: float,           # current ATR for sizing stops
        unrealized_pnl_pct: float,
    ) -> PenroseExitSignal:
        """
        Determine if the current state is in the ergosphere and compute
        the optimal Penrose exit strategy.

        bh_mass: current BH mass from BHState
        bh_spin: spin parameter (momentum / mass, or kerr_spin from composite)
        atr: average true range for stop distance calibration
        unrealized_pnl_pct: current position P&L as fraction of entry
        """
        # Ergosphere boundary: r_ergo = M + sqrt(M^2 - a^2*cos^2(theta))
        # Simplified: ergosphere exists when mass > threshold * (1 - spin^2)
        # Higher spin -> larger ergosphere -> more extractable energy
        effective_threshold = self.mass_threshold * (1 - min(bh_spin**2, 0.99))
        in_ergosphere = bh_mass > effective_threshold and abs(bh_spin) > self.spin_threshold

        if not in_ergosphere:
            return PenroseExitSignal(
                in_ergosphere=False,
                exit_action="hold",
                tight_stop_distance=0.0,
                runner_trail_distance=0.0,
                ergosphere_depth=0.0,
                energy_extraction_potential=0.0,
            )

        # Ergosphere depth: 0 at boundary, 1 at horizon
        if self.mass_threshold > 0:
            depth = min(1.0, (bh_mass - effective_threshold) / max(self.mass_threshold * 0.5, 1e-6))
        else:
            depth = 0.5

        # Spin-to-mass ratio determines energy extraction efficiency
        # In GR: max extraction = 29% of mass for a maximally spinning BH
        spin_mass_ratio = abs(bh_spin) / max(bh_mass, 1e-6)
        extraction_efficiency = min(0.29, spin_mass_ratio * 0.3)
        energy_potential = extraction_efficiency * abs(unrealized_pnl_pct)

        # Stop distances: calibrated by ATR and ergosphere depth
        # Deeper in ergosphere = tighter stop (more committed to extraction)
        tight_stop = atr * max(0.3, 1.0 - depth * 0.7)
        runner_trail = atr * (1.5 + depth * 1.5)  # wider trail deeper in ergosphere

        # Action decision
        if unrealized_pnl_pct > 0.01 and depth > 0.3:
            action = "split"  # profitable + deep in ergosphere -> split position
        elif depth > 0.8:
            action = "full_exit"  # too deep, approaching horizon
        else:
            action = "hold"  # in ergosphere but not deep enough to split

        return PenroseExitSignal(
            in_ergosphere=True,
            exit_action=action,
            tight_stop_distance=float(tight_stop),
            runner_trail_distance=float(runner_trail),
            ergosphere_depth=float(depth),
            energy_extraction_potential=float(energy_potential),
        )


# ---------------------------------------------------------------------------
# 3. Riemann Curvature Regime Detection
# ---------------------------------------------------------------------------

@dataclass
class CurvatureRegime:
    """Regime classified by spacetime curvature."""
    scalar_curvature: float     # R: positive = converging, negative = diverging
    regime: str                 # "converging" / "flat" / "diverging" / "singularity"
    curvature_trend: float      # dR/dt: rate of change
    metric_stability: float     # 0-1: how stable the covariance structure is
    geodesic_deviation: float   # how fast nearby trajectories are separating


class RiemannCurvatureDetector:
    """
    Detect market regime from the curvature of the asset correlation manifold.

    In GR:
    - Positive curvature (R > 0): geodesics converge (sphere-like) = herding
    - Flat curvature (R ~ 0): geodesics parallel (Euclidean) = normal market
    - Negative curvature (R < 0): geodesics diverge (hyperbolic) = dispersion
    - Singularity (|R| -> inf): breakdown of smooth structure = crisis

    We estimate curvature from the RATE OF CHANGE of the covariance matrix.
    If correlations are increasing (converging), curvature is positive.
    If correlations are decreasing (diverging), curvature is negative.
    """

    def __init__(self, window: int = 21, n_assets: int = 5):
        self.window = window
        self.n_assets = n_assets
        self._returns_buffer: deque = deque(maxlen=window * 2)
        self._curvature_history: deque = deque(maxlen=63)

    def update(self, returns_vector: np.ndarray) -> None:
        """
        Update with a new bar of returns across multiple assets.
        returns_vector: (n_assets,) array of returns for this bar.
        """
        self._returns_buffer.append(returns_vector.copy())

    def compute(self) -> CurvatureRegime:
        """Compute the current curvature regime."""
        n = len(self._returns_buffer)
        if n < self.window + 5:
            return CurvatureRegime(0.0, "insufficient_data", 0.0, 0.5, 0.0)

        returns = np.array(list(self._returns_buffer))
        T = returns.shape[0]

        # Compute covariance in two halves (early and late window)
        mid = T // 2
        cov_early = np.cov(returns[:mid].T) + np.eye(returns.shape[1]) * 1e-8
        cov_late = np.cov(returns[mid:].T) + np.eye(returns.shape[1]) * 1e-8

        # Metric tensor = covariance matrix
        # Curvature ~ rate of change of metric
        # Simplified: use Frobenius norm of (cov_late - cov_early) / cov_early
        delta_cov = cov_late - cov_early
        frobenius_change = float(np.linalg.norm(delta_cov, "fro"))
        frobenius_base = float(np.linalg.norm(cov_early, "fro"))
        relative_change = frobenius_change / max(frobenius_base, 1e-10)

        # Direction of curvature: are correlations increasing or decreasing?
        # Average off-diagonal correlation change
        n_assets = returns.shape[1]
        if n_assets >= 2:
            mask = ~np.eye(n_assets, dtype=bool)
            avg_corr_early = float(np.corrcoef(returns[:mid].T)[mask].mean())
            avg_corr_late = float(np.corrcoef(returns[mid:].T)[mask].mean())
            corr_change = avg_corr_late - avg_corr_early
        else:
            corr_change = 0.0

        # Scalar curvature: positive = converging, negative = diverging
        scalar_curvature = float(corr_change * 10)  # scale for interpretability

        # Metric stability: inverse of relative change (stable = high)
        stability = float(max(0, 1 - relative_change * 5))

        # Geodesic deviation: eigenvalue dispersion of covariance
        try:
            eigvals = np.linalg.eigvalsh(cov_late)
            eigvals = eigvals[eigvals > 0]
            if len(eigvals) >= 2:
                # Absorption ratio: fraction explained by top eigenvalue
                geo_deviation = float(eigvals[-1] / eigvals.sum())
            else:
                geo_deviation = 0.5
        except:
            geo_deviation = 0.5

        # Curvature trend
        self._curvature_history.append(scalar_curvature)
        if len(self._curvature_history) >= 5:
            recent = list(self._curvature_history)
            trend = (recent[-1] - recent[-5]) / 5
        else:
            trend = 0.0

        # Regime classification
        if abs(scalar_curvature) > 2.0 and stability < 0.2:
            regime = "singularity"  # crisis: correlation structure breaking down
        elif scalar_curvature > 0.5:
            regime = "converging"   # herding: assets moving together
        elif scalar_curvature < -0.5:
            regime = "diverging"    # dispersion: assets decorrelating
        else:
            regime = "flat"         # normal market: stable correlations

        return CurvatureRegime(
            scalar_curvature=scalar_curvature,
            regime=regime,
            curvature_trend=float(trend),
            metric_stability=stability,
            geodesic_deviation=geo_deviation,
        )

    def regime_signal(self) -> Dict:
        """Convert curvature regime to trading signal adjustments."""
        regime = self.compute()

        adjustments = {
            "converging": {
                "momentum_boost": 1.2,    # herding amplifies trends
                "reversion_dampen": 0.5,  # mean reversion fails in herding
                "position_reduce": 0.8,   # reduce for correlation risk
                "vol_expect": "rising",
            },
            "flat": {
                "momentum_boost": 1.0,
                "reversion_dampen": 1.0,
                "position_reduce": 1.0,
                "vol_expect": "stable",
            },
            "diverging": {
                "momentum_boost": 0.7,    # momentum weakens in dispersion
                "reversion_dampen": 1.3,  # mean reversion works better
                "position_reduce": 1.0,
                "vol_expect": "falling",
            },
            "singularity": {
                "momentum_boost": 0.3,    # everything breaks down
                "reversion_dampen": 0.3,
                "position_reduce": 0.3,   # massive risk reduction
                "vol_expect": "extreme",
            },
        }

        adj = adjustments.get(regime.regime, adjustments["flat"])
        adj["curvature"] = regime.scalar_curvature
        adj["regime"] = regime.regime
        adj["stability"] = regime.metric_stability
        adj["geodesic_deviation"] = regime.geodesic_deviation
        return adj
