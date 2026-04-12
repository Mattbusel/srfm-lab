"""
StabilityAnalyzer — Market stability as function of agent population.

Metrics:
  - Lyapunov exponent estimation: is agent dynamics locally stable?
  - Bifurcation analysis: stability change as population mix changes
  - AttractorMapper: fixed points and limit cycles in strategy space
  - RegimePhasePortrait: agent aggregate behavior in 2D phase space
  - CrisisContagion: how does a single agent failure propagate?
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.linalg import svd


# ============================================================
# Lyapunov Exponent Estimator
# ============================================================

class LyapunovEstimator:
    """
    Estimates the largest Lyapunov exponent (LLE) of agent dynamics.

    Positive LLE → chaotic / unstable dynamics.
    Negative LLE → convergent / stable dynamics.
    Zero LLE     → neutral stability.

    Method: Rosenstein et al. (1993) — track divergence of nearby
    trajectories in the system's state space.
    """

    def __init__(
        self,
        embedding_dim: int   = 5,
        tau:           int   = 1,
        min_n:         int   = 50,
        diverge_window: int  = 10,
    ) -> None:
        self.m      = embedding_dim
        self.tau    = tau
        self.min_n  = min_n
        self.div_w  = diverge_window
        self._states: deque = deque(maxlen=500)

    def update(self, state: float) -> None:
        """Update with scalar market state (e.g., mid-price)."""
        self._states.append(state)

    def estimate_lle(self) -> float:
        """
        Estimate the largest Lyapunov exponent.

        Returns LLE ≈ 0 if insufficient data.
        """
        data = np.array(list(self._states))
        n    = len(data)
        if n < self.min_n + self.m * self.tau:
            return 0.0

        # Build delay embedding
        N     = n - (self.m - 1) * self.tau
        emb   = np.column_stack([
            data[i * self.tau: i * self.tau + N]
            for i in range(self.m)
        ])

        # Find nearest neighbors for each point (excluding temporal neighbors)
        lles = []
        for i in range(N):
            dists = np.linalg.norm(emb - emb[i], axis=1)
            # Exclude temporal neighbors (within 10 steps)
            dists[:max(0, i - 10)] = np.inf
            dists[i: min(N, i + 10)] = np.inf
            if np.isinf(dists).all():
                continue
            j = int(np.argmin(dists))
            d0 = max(dists[j], 1e-8)

            # Track divergence
            divs = []
            for k in range(1, min(self.div_w, N - max(i, j))):
                d_k = np.linalg.norm(emb[i + k] - emb[j + k])
                if d_k > 0:
                    divs.append(math.log(d_k / d0))

            if divs:
                lles.append(np.mean(divs))

        if not lles:
            return 0.0
        return float(np.mean(lles))

    def is_stable(self, threshold: float = 0.01) -> bool:
        return self.estimate_lle() < threshold


# ============================================================
# Bifurcation Analyzer
# ============================================================

class BifurcationAnalyzer:
    """
    Analyzes how market stability changes as population mix changes.

    Tracks (MM_fraction, momentum_fraction, volatility) triplets
    and identifies bifurcation points — mixes where stability
    suddenly changes.
    """

    def __init__(self) -> None:
        # Records: (mm_frac, mom_frac, vol, lyapunov)
        self._records: List[Tuple[float, float, float, float]] = []

    def record(
        self,
        mm_fraction:  float,
        mom_fraction: float,
        volatility:   float,
        lyapunov:     float,
    ) -> None:
        self._records.append((mm_fraction, mom_fraction, volatility, lyapunov))

    def stability_surface(
        self, bins: int = 10
    ) -> Dict[Tuple[float, float], float]:
        """
        Returns stability (mean LLE) for each (MM_frac, mom_frac) bin.
        """
        if not self._records:
            return {}

        arr   = np.array(self._records)
        mm_f  = arr[:, 0]
        mo_f  = arr[:, 1]
        lle   = arr[:, 3]

        # Bin the fractions
        mm_edges = np.linspace(0, 1, bins + 1)
        mo_edges = np.linspace(0, 1, bins + 1)
        result:  Dict[Tuple[float, float], float] = {}

        for i in range(bins):
            for j in range(bins):
                mask = (
                    (mm_f >= mm_edges[i]) & (mm_f < mm_edges[i + 1]) &
                    (mo_f >= mo_edges[j]) & (mo_f < mo_edges[j + 1])
                )
                if mask.sum() > 0:
                    key = (
                        float((mm_edges[i] + mm_edges[i + 1]) / 2),
                        float((mo_edges[j] + mo_edges[j + 1]) / 2),
                    )
                    result[key] = float(np.mean(lle[mask]))

        return result

    def find_bifurcation_points(self) -> List[Tuple[float, float]]:
        """
        Identify (mm_frac, mom_frac) where stability flips sign.
        """
        surface = self.stability_surface(bins=5)
        if len(surface) < 2:
            return []

        bifurcations = []
        items = sorted(surface.items())
        prev_lle = None
        for (mm, mo), lle in items:
            if prev_lle is not None:
                if (prev_lle < 0) != (lle < 0):
                    bifurcations.append((mm, mo))
            prev_lle = lle
        return bifurcations


# ============================================================
# Attractor Mapper
# ============================================================

class AttractorMapper:
    """
    Identifies fixed points and limit cycles in agent strategy space.

    Uses PCA to reduce strategy space to 2D, then identifies
    clusters / cycles via density estimation.
    """

    def __init__(
        self,
        window:       int = 200,
        n_components: int = 2,
    ) -> None:
        self.window       = window
        self.n_components = n_components
        self._trajectories: deque = deque(maxlen=window)

    def record_state(self, state_vector: np.ndarray) -> None:
        """Record high-dimensional agent policy state."""
        self._trajectories.append(state_vector)

    def pca_projection(self) -> Optional[np.ndarray]:
        """
        Project trajectory onto first 2 principal components.
        Returns (T, 2) array or None if insufficient data.
        """
        if len(self._trajectories) < 10:
            return None

        X = np.array(list(self._trajectories), dtype=np.float32)
        X = X - X.mean(axis=0)

        try:
            U, S, Vt = svd(X, full_matrices=False)
            return U[:, :self.n_components] * S[:self.n_components]
        except Exception:
            return None

    def detect_fixed_point(
        self, proj: Optional[np.ndarray] = None, radius: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Detect approximate fixed point: region where trajectory stays.
        Returns center of fixed point cluster or None.
        """
        if proj is None:
            proj = self.pca_projection()
        if proj is None:
            return None

        # Check if recent trajectory is within small radius of mean
        recent = proj[-20:]
        center = recent.mean(axis=0)
        dists  = np.linalg.norm(recent - center, axis=1)
        if dists.max() < radius:
            return center
        return None

    def detect_limit_cycle(
        self, proj: Optional[np.ndarray] = None, n_laps: int = 3
    ) -> bool:
        """
        Detect limit cycle: trajectory that returns to a region periodically.
        """
        if proj is None:
            proj = self.pca_projection()
        if proj is None or len(proj) < 30:
            return False

        # FFT-based periodicity detection on first PC
        x    = proj[:, 0]
        x    = x - x.mean()
        fft  = np.abs(np.fft.rfft(x))
        freq = np.fft.rfftfreq(len(x))

        # Strong periodic peak at non-zero frequency → limit cycle
        if len(fft) < 3:
            return False
        fft[0] = 0  # remove DC
        peak_ratio = fft.max() / (fft.mean() + 1e-8)
        return bool(peak_ratio > 5.0)

    def trajectory_summary(self) -> Dict[str, Any]:
        proj = self.pca_projection()
        if proj is None:
            return {"status": "insufficient_data"}

        fp    = self.detect_fixed_point(proj)
        cycle = self.detect_limit_cycle(proj)
        return {
            "n_points":      len(proj),
            "fixed_point":   fp.tolist() if fp is not None else None,
            "limit_cycle":   cycle,
            "variance_pc1":  float(np.var(proj[:, 0])),
            "variance_pc2":  float(np.var(proj[:, 1])) if proj.shape[1] > 1 else 0.0,
        }


# ============================================================
# Regime Phase Portrait
# ============================================================

class RegimePhasePortrait:
    """
    Plots agent aggregate behavior in 2D phase space.

    Axes:
      X: net buy pressure (fraction buying - fraction selling)
      Y: price volatility (rolling std of returns)

    Quadrants:
      (+, +) → bullish + volatile
      (+, -) → bullish + calm
      (-, +) → bearish + volatile
      (-, -) → bearish + calm
    """

    def __init__(self, vol_window: int = 20) -> None:
        self.vol_window  = vol_window
        self._buy_press: List[float] = []
        self._vol:       List[float] = []
        self._prices:    deque       = deque(maxlen=vol_window + 5)

    def update(
        self,
        agent_actions: Dict[str, int],
        mid_price: float,
    ) -> Tuple[float, float]:
        """
        Update phase portrait with current agent actions and price.
        Returns (buy_pressure, volatility) point.
        """
        actions = list(agent_actions.values())
        n       = max(len(actions), 1)
        buy_p   = (sum(1 for a in actions if a > 0) - sum(1 for a in actions if a < 0)) / n
        self._buy_press.append(buy_p)

        self._prices.append(mid_price)
        if len(self._prices) >= 5:
            prices  = np.array(list(self._prices))
            returns = np.diff(np.log(prices + 1e-8))
            vol     = float(np.std(returns)) if len(returns) > 1 else 0.0
        else:
            vol = 0.0
        self._vol.append(vol)
        return buy_p, vol

    def trajectory(self) -> np.ndarray:
        """Return (T, 2) phase space trajectory."""
        n = min(len(self._buy_press), len(self._vol))
        return np.column_stack([self._buy_press[:n], self._vol[:n]])

    def current_quadrant(self) -> str:
        if not self._buy_press or not self._vol:
            return "unknown"
        bp  = self._buy_press[-1]
        vol = self._vol[-1]
        q = "bullish" if bp > 0 else "bearish"
        q += "_volatile" if vol > float(np.mean(self._vol) if self._vol else 0) else "_calm"
        return q


# ============================================================
# Crisis Contagion
# ============================================================

class CrisisContagion:
    """
    Models how a single agent's distress propagates through the network.

    Simulation:
      1. Inject "failure" shock to one agent (force large position unwind)
      2. Track how price impact propagates to connected agents
      3. Measure amplification: final volatility / initial shock magnitude
    """

    def __init__(self, agent_ids: List[str]) -> None:
        self.agent_ids = agent_ids
        self._contagion_log: List[Dict] = []

    def simulate_failure(
        self,
        failed_agent:     str,
        shock_magnitude:  float,
        neighbor_map:     Dict[str, List[str]],
        n_hops:           int = 3,
    ) -> Dict[str, float]:
        """
        Simulate contagion from a failing agent.

        Uses simple SI (susceptible-infected) model:
        Neighbors of failing agent experience fraction of shock.
        Their neighbors experience a smaller fraction, etc.

        Returns dict of agent → exposure level.
        """
        exposure: Dict[str, float] = {failed_agent: shock_magnitude}
        infected = {failed_agent}

        for hop in range(n_hops):
            new_infections = {}
            for infected_id in list(infected):
                neighbors = neighbor_map.get(infected_id, [])
                hop_decay = 0.5 ** (hop + 1)
                for nb in neighbors:
                    if nb not in exposure:
                        new_infections[nb] = exposure[infected_id] * hop_decay
                    else:
                        new_infections[nb] = max(
                            exposure[nb], exposure[infected_id] * hop_decay
                        )
            exposure.update(new_infections)
            infected.update(new_infections.keys())

        self._contagion_log.append({
            "failed_agent":    failed_agent,
            "shock_magnitude": shock_magnitude,
            "n_exposed":       len(exposure),
            "max_exposure":    max(exposure.values()),
            "mean_exposure":   float(np.mean(list(exposure.values()))),
        })
        return exposure

    def amplification_ratio(self) -> float:
        """Ratio of total exposure to initial shock magnitude."""
        if not self._contagion_log:
            return 0.0
        last = self._contagion_log[-1]
        total_exp = last["mean_exposure"] * last["n_exposed"]
        return float(total_exp / (last["shock_magnitude"] + 1e-8))

    def contagion_log(self) -> List[Dict]:
        return list(self._contagion_log)


# ============================================================
# StabilityAnalyzer (composite)
# ============================================================

class StabilityAnalyzer:
    """
    Composite stability analyzer.
    """

    def __init__(
        self,
        agent_ids:    List[str],
        neighbor_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.agent_ids    = agent_ids
        self.neighbor_map = neighbor_map or {}

        self.lyapunov     = LyapunovEstimator()
        self.bifurcation  = BifurcationAnalyzer()
        self.attractor    = AttractorMapper()
        self.phase        = RegimePhasePortrait()
        self.contagion    = CrisisContagion(agent_ids)

        self._step = 0

    def step(
        self,
        mid_price:     float,
        agent_actions: Dict[str, int],
        state_vector:  Optional[np.ndarray] = None,
        mm_fraction:   Optional[float] = None,
        mom_fraction:  Optional[float] = None,
    ) -> Dict[str, Any]:
        self._step += 1

        self.lyapunov.update(mid_price)
        bp, vol = self.phase.update(agent_actions, mid_price)

        if state_vector is not None:
            self.attractor.record_state(state_vector)

        lle = self.lyapunov.estimate_lle()
        if mm_fraction is not None and mom_fraction is not None:
            self.bifurcation.record(mm_fraction, mom_fraction, vol, lle)

        return {
            "step":         self._step,
            "lyapunov_lle": lle,
            "is_stable":    self.lyapunov.is_stable(),
            "buy_pressure": bp,
            "volatility":   vol,
            "phase_quadrant": self.phase.current_quadrant(),
        }

    def full_report(self) -> Dict[str, Any]:
        traj_summary = self.attractor.trajectory_summary()
        return {
            "total_steps":     self._step,
            "lyapunov_lle":    self.lyapunov.estimate_lle(),
            "is_stable":       self.lyapunov.is_stable(),
            "bifurcation_pts": self.bifurcation.find_bifurcation_points(),
            "attractor":       traj_summary,
            "phase_quadrant":  self.phase.current_quadrant(),
            "contagion_log":   self.contagion.contagion_log(),
        }

    def simulate_crisis_contagion(
        self, failed_agent: str, shock: float = 1.0
    ) -> Dict[str, float]:
        return self.contagion.simulate_failure(
            failed_agent, shock, self.neighbor_map
        )
