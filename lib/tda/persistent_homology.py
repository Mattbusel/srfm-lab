"""
Topological Data Analysis — Persistent Homology (T3-6)
Applies TDA to the N-dimensional feature manifold to detect market regime transitions.

Betti numbers:
  β₀ = connected components (fragmentation)
  β₁ = loops/cycles (mean-reversion cycles)

Uses a fast Vietoris-Rips approximation without external dependencies.
Regime signals:
  high β₁ (persistent loops) → cyclic regime → mean reversion
  topological phase transition → regime change alert
"""
import math
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

log = logging.getLogger(__name__)

@dataclass
class TDAConfig:
    window: int = 100           # point cloud window size
    n_features: int = 5         # (mass, beta, cf, hurst, garch_vol)
    max_filtration: float = 2.0 # max edge length in Vietoris-Rips
    n_filtration_steps: int = 20
    persistence_threshold: float = 0.1  # min feature lifespan to count
    regime_change_threshold: float = 0.4  # delta in Betti numbers to flag

class PersistentHomologyAnalyzer:
    """
    Computes simplified persistent homology (β₀, β₁) for market regime detection.

    Uses Union-Find for β₀ (connected components).
    Approximates β₁ using cycle counting in the Rips complex.

    Usage:
        analyzer = PersistentHomologyAnalyzer()
        result = analyzer.update(mass, beta, cf, hurst, garch_vol)
        if result['regime_change']:
            # Potential regime transition detected
    """

    def __init__(self, cfg: TDAConfig = None):
        self.cfg = cfg or TDAConfig()
        self._points: list[list[float]] = []
        self._prev_betti: tuple = (1, 0)  # (β₀, β₁)
        self.betti: tuple = (1, 0)
        self._regime_change_ema: float = 0.0

    def update(self, mass: float, beta: float, cf: float, hurst: float, garch_vol: float) -> dict:
        """
        Feed one bar's feature vector.
        Returns: {betti_0, betti_1, regime_change, regime_change_score, cycle_regime}
        """
        point = [mass, beta, cf, hurst, garch_vol]
        self._points.append(point)
        if len(self._points) > self.cfg.window:
            self._points.pop(0)

        if len(self._points) < 20:
            return self._default_output()

        # Only recompute TDA every 5 bars for performance
        if len(self._points) % 5 == 0:
            self._prev_betti = self.betti
            self.betti = self._compute_betti()

        # Detect topology change
        db0 = abs(self.betti[0] - self._prev_betti[0])
        db1 = abs(self.betti[1] - self._prev_betti[1])
        topo_delta = (db0 + db1) / (max(self._prev_betti[0], 1))

        self._regime_change_ema = 0.9 * self._regime_change_ema + 0.1 * topo_delta
        regime_change = self._regime_change_ema > self.cfg.regime_change_threshold

        # Cycle regime: high β₁ → cyclic/mean-reverting market
        cycle_regime = self.betti[1] >= 2

        return {
            "betti_0": self.betti[0],
            "betti_1": self.betti[1],
            "regime_change": regime_change,
            "regime_change_score": self._regime_change_ema,
            "cycle_regime": cycle_regime,
        }

    def _compute_betti(self) -> tuple:
        """Compute simplified (β₀, β₁) via Union-Find on Vietoris-Rips complex."""
        pts = np.array(self._points[-self.cfg.window:], dtype=float)
        n = len(pts)

        # Normalize features to [0, 1]
        for col in range(pts.shape[1]):
            col_min = pts[:, col].min()
            col_max = pts[:, col].max()
            if col_max > col_min:
                pts[:, col] = (pts[:, col] - col_min) / (col_max - col_min)

        # Compute pairwise distances
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                dists[i, j] = d
                dists[j, i] = d

        # Find optimal filtration radius (median distance)
        upper_tri = dists[np.triu_indices(n, k=1)]
        if len(upper_tri) == 0:
            return (1, 0)
        eps = float(np.median(upper_tri)) * 0.7

        # Union-Find for β₀ (connected components)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if dists[i, j] <= eps:
                    union(i, j)
                    edges.append((i, j))

        b0 = len(set(find(i) for i in range(n)))

        # Approximate β₁: count independent cycles using Euler characteristic
        # For a simplicial complex: χ = V - E + F = β₀ - β₁ + β₂
        # Simplified: β₁ ≈ E - V + β₀ (ignoring triangles/β₂)
        b1 = max(0, len(edges) - n + b0)
        # Cap to avoid spurious high values
        b1 = min(b1, 5)

        return (b0, b1)

    def _default_output(self) -> dict:
        return {
            "betti_0": 1, "betti_1": 0,
            "regime_change": False, "regime_change_score": 0.0,
            "cycle_regime": False,
        }
