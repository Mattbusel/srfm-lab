"""
QUBO Portfolio Optimization (T3-2)
Reformulates instrument selection as a Quadratic Unconstrained Binary Optimization problem.

QUBO formulation:
  min: -Σᵢ (αᵢ xᵢ) + λ Σᵢ Σⱼ (ρᵢⱼ xᵢ xⱼ) + penalty(|Σxᵢ - K|²)

Where:
  αᵢ = signal strength for instrument i (from BH physics + signal stack)
  ρᵢⱼ = pairwise correlation (penalizes redundant/concentrated positions)
  K = target portfolio cardinality (number of active instruments)
  xᵢ ∈ {0, 1} = binary allocation decision

Solved via Simulated Annealing (SA) — trivially fast for N≤25 instruments.
"""
import math
import random
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class QUBOConfig:
    target_cardinality: int = 8      # target number of active instruments
    cardinality_penalty: float = 2.0  # λ_card for cardinality constraint
    correlation_penalty: float = 0.5  # λ_corr for correlation penalty
    sa_temperature_init: float = 2.0
    sa_temperature_final: float = 0.01
    sa_steps: int = 5000
    sa_seed: int = 42

class QUBOPortfolioOptimizer:
    """
    Solves the QUBO portfolio selection problem using Simulated Annealing.

    Usage:
        optimizer = QUBOPortfolioOptimizer()
        selected = optimizer.solve(
            symbols=["BTC", "ETH", "AVAX", ...],
            alphas={"BTC": 0.8, "ETH": 0.6, ...},  # signal strengths
            corr_matrix={"BTC": {"ETH": 0.7, ...}, ...}
        )
        # selected = ["BTC", "ETH", "LTC"] — optimal subset
    """

    def __init__(self, cfg: QUBOConfig = None):
        self.cfg = cfg or QUBOConfig()

    def solve(
        self,
        symbols: list[str],
        alphas: dict[str, float],  # signal strength per instrument
        corr_matrix: dict[str, dict[str, float]],  # pairwise correlations
    ) -> list[str]:
        """
        Returns optimal subset of instruments to trade.

        symbols: all candidate instruments
        alphas: {sym: signal_strength} — higher = stronger BH/signal
        corr_matrix: {sym_a: {sym_b: corr}} — pairwise correlations
        """
        n = len(symbols)
        if n == 0:
            return []

        K = min(self.cfg.target_cardinality, n)

        # Build QUBO matrix Q where objective = x^T Q x + linear terms
        alpha_vec = np.array([alphas.get(s, 0.0) for s in symbols])

        # Correlation penalty matrix
        corr_arr = np.zeros((n, n))
        for i, si in enumerate(symbols):
            for j, sj in enumerate(symbols):
                if i != j:
                    c = corr_matrix.get(si, {}).get(sj, corr_matrix.get(sj, {}).get(si, 0.0))
                    corr_arr[i, j] = abs(c)  # penalize any high correlation

        def objective(x: np.ndarray) -> float:
            """QUBO objective: minimize = -alpha'x + λ_corr * x'Cx + λ_card * (Σx-K)²"""
            alpha_term = -np.dot(alpha_vec, x)
            corr_term = self.cfg.correlation_penalty * float(x @ corr_arr @ x)
            card_penalty = self.cfg.cardinality_penalty * (np.sum(x) - K) ** 2
            return alpha_term + corr_term + card_penalty

        # Simulated Annealing
        rng = random.Random(self.cfg.sa_seed)

        # Initialize: select top-K by alpha
        sorted_by_alpha = sorted(range(n), key=lambda i: -alpha_vec[i])
        x = np.zeros(n)
        for i in sorted_by_alpha[:K]:
            x[i] = 1.0

        best_x = x.copy()
        best_obj = objective(x)
        current_obj = best_obj

        T = self.cfg.sa_temperature_init
        T_final = self.cfg.sa_temperature_final
        cooling = (T_final / T) ** (1.0 / self.cfg.sa_steps)

        for step in range(self.cfg.sa_steps):
            # Random move: flip one bit
            idx = rng.randint(0, n - 1)
            x_new = x.copy()
            x_new[idx] = 1.0 - x_new[idx]

            new_obj = objective(x_new)
            delta = new_obj - current_obj

            if delta < 0 or rng.random() < math.exp(-delta / (T + 1e-12)):
                x = x_new
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_x = x.copy()

            T *= cooling

        selected = [symbols[i] for i in range(n) if best_x[i] > 0.5]
        log.info("QUBO: selected %d/%d instruments: %s (obj=%.4f)",
                 len(selected), n, selected, best_obj)
        return selected

    def get_instrument_weights(
        self,
        symbols: list[str],
        alphas: dict[str, float],
        corr_matrix: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """
        Solve QUBO then return normalized weights for selected instruments.
        Selected instruments get equal weight (1/K). Unselected get 0.
        """
        selected = self.solve(symbols, alphas, corr_matrix)
        k = len(selected)
        if k == 0:
            return {s: 0.0 for s in symbols}
        weight = 1.0 / k
        return {s: weight if s in selected else 0.0 for s in symbols}
