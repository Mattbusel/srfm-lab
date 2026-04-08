"""
Hypothesis Portfolio — treats trading hypotheses like assets in a portfolio.

Applies portfolio construction theory to allocate capital across hypotheses:
  - Each hypothesis = an asset with expected return (alpha), variance, correlations
  - Optimize: max Sharpe of hypothesis portfolio
  - Constraints: diversification, regime limits, correlation limits
  - Risk budgeting: ERC across independent hypotheses
  - Kelly sizing: fractional Kelly for hypothesis mix
  - Dynamic rebalancing: add/remove hypotheses as regime changes
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HypothesisAsset:
    id: str
    name: str
    expected_sharpe: float       # estimated Sharpe ratio
    sharpe_uncertainty: float    # std dev of Sharpe estimate
    edge_type: str               # momentum, mean_reversion, etc.
    regime: str                  # optimal regime
    capacity_usd: float          # max capital this hypothesis can absorb
    correlation_tags: list[str]  # tags for correlation grouping
    active: bool = True
    last_ic: float = 0.0         # most recent information coefficient


@dataclass
class HypothesisAllocation:
    id: str
    weight: float                # fraction of total capital
    dollar_amount: float
    rationale: str


class HypothesisPortfolio:
    """
    Maintains a portfolio of trading hypotheses with optimal allocation.
    """

    def __init__(self, total_capital: float = 1_000_000.0):
        self.total_capital = total_capital
        self.hypotheses: dict[str, HypothesisAsset] = {}
        self.allocation_history: list[dict] = []
        self._corr_matrix: Optional[np.ndarray] = None
        self._sharpe_history: dict[str, list[float]] = {}

    def add_hypothesis(self, h: HypothesisAsset) -> None:
        self.hypotheses[h.id] = h
        self._sharpe_history[h.id] = []
        self._corr_matrix = None  # invalidate cache

    def remove_hypothesis(self, h_id: str) -> None:
        self.hypotheses.pop(h_id, None)
        self._sharpe_history.pop(h_id, None)
        self._corr_matrix = None

    def update_performance(self, h_id: str, realized_sharpe: float) -> None:
        """Record realized performance for a hypothesis."""
        if h_id in self._sharpe_history:
            self._sharpe_history[h_id].append(realized_sharpe)
        if h_id in self.hypotheses:
            # Exponential moving average update
            alpha = 0.2
            old = self.hypotheses[h_id].expected_sharpe
            self.hypotheses[h_id].expected_sharpe = (1 - alpha) * old + alpha * realized_sharpe

    def estimate_correlation_matrix(self) -> np.ndarray:
        """
        Estimate correlation matrix between hypotheses based on:
        - Same edge type → higher correlation
        - Same regime → moderate correlation
        - Same tags → moderate correlation
        - Different everything → low baseline correlation
        """
        ids = [h.id for h in self.hypotheses.values() if h.active]
        n = len(ids)
        if n == 0:
            return np.eye(1)

        corr = np.eye(n)
        hyps = [self.hypotheses[i] for i in ids]

        for i in range(n):
            for j in range(i + 1, n):
                a, b = hyps[i], hyps[j]
                c = 0.05  # baseline

                # Same edge type
                if a.edge_type == b.edge_type:
                    c += 0.40

                # Same regime
                if a.regime == b.regime:
                    c += 0.15

                # Shared correlation tags
                shared_tags = set(a.correlation_tags) & set(b.correlation_tags)
                c += len(shared_tags) * 0.10

                c = float(min(c, 0.95))
                corr[i, j] = corr[j, i] = c

        self._corr_matrix = corr
        return corr

    def expected_returns(self) -> tuple[list[str], np.ndarray]:
        """Return active hypothesis IDs and their expected Sharpe vector."""
        active = [(id_, h) for id_, h in self.hypotheses.items() if h.active]
        ids = [a[0] for a in active]
        mu = np.array([a[1].expected_sharpe for a in active])
        return ids, mu

    def optimize_max_sharpe(
        self,
        min_weight: float = 0.02,
        max_weight: float = 0.40,
        max_edge_concentration: float = 0.50,
    ) -> dict[str, HypothesisAllocation]:
        """
        Maximum Sharpe allocation across active hypotheses.
        Uses correlation-adjusted optimization.
        """
        ids, mu = self.expected_returns()
        n = len(ids)
        if n == 0:
            return {}

        corr = self.estimate_correlation_matrix()
        uncertainties = np.array([
            self.hypotheses[id_].sharpe_uncertainty for id_ in ids
        ])
        # Covariance = correlation * sigma_i * sigma_j
        Sigma = corr * np.outer(uncertainties, uncertainties) + np.eye(n) * 1e-6

        # Analytical max Sharpe (long-only via iterative)
        from scipy.optimize import minimize

        def neg_sharpe(w):
            w = np.maximum(w, 0)
            w /= w.sum() + 1e-10
            port_ret = float(w @ mu)
            port_var = float(w @ Sigma @ w)
            return -(port_ret / math.sqrt(max(port_var, 1e-10)))

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(min_weight, max_weight)] * n

        w0 = np.ones(n) / n
        result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        w = result.x if result.success else w0

        # Edge concentration constraint
        edge_weights = {}
        for i, id_ in enumerate(ids):
            et = self.hypotheses[id_].edge_type
            edge_weights[et] = edge_weights.get(et, 0.0) + w[i]

        for et, ew in edge_weights.items():
            if ew > max_edge_concentration:
                # Scale down over-concentrated edge
                scale = max_edge_concentration / ew
                for i, id_ in enumerate(ids):
                    if self.hypotheses[id_].edge_type == et:
                        w[i] *= scale

        # Renormalize
        w = np.maximum(w, 0)
        w /= w.sum() + 1e-10

        allocations = {}
        for i, id_ in enumerate(ids):
            allocations[id_] = HypothesisAllocation(
                id=id_,
                weight=float(w[i]),
                dollar_amount=float(w[i] * self.total_capital),
                rationale=f"Max Sharpe optimization, edge={self.hypotheses[id_].edge_type}",
            )

        return allocations

    def risk_parity_allocation(self) -> dict[str, HypothesisAllocation]:
        """
        Equal Risk Contribution allocation across hypotheses.
        Each hypothesis contributes equal Sharpe-risk to portfolio.
        """
        ids, mu = self.expected_returns()
        n = len(ids)
        if n == 0:
            return {}

        corr = self.estimate_correlation_matrix()
        uncertainties = np.array([self.hypotheses[id_].sharpe_uncertainty for id_ in ids])
        Sigma = corr * np.outer(uncertainties, uncertainties) + np.eye(n) * 1e-6

        # Newton-Raphson ERC
        w = np.ones(n) / n
        for _ in range(200):
            var = float(w @ Sigma @ w)
            if var < 1e-12:
                break
            mrc = Sigma @ w
            rc = w * mrc / var
            target = 1.0 / n
            grad = 2 * (rc - target) * mrc / var
            hess = 2 * (mrc / var) ** 2
            w_new = w - 0.5 * grad / (hess + 1e-10)
            w_new = np.maximum(w_new, 1e-8)
            w_new /= w_new.sum()
            if np.max(np.abs(w_new - w)) < 1e-6:
                w = w_new
                break
            w = w_new

        allocations = {}
        for i, id_ in enumerate(ids):
            allocations[id_] = HypothesisAllocation(
                id=id_,
                weight=float(w[i]),
                dollar_amount=float(w[i] * self.total_capital),
                rationale="Equal Risk Contribution",
            )
        return allocations

    def kelly_allocation(self, fraction: float = 0.25) -> dict[str, HypothesisAllocation]:
        """
        Fractional Kelly allocation: w* = fraction * Sigma^{-1} * mu
        """
        ids, mu = self.expected_returns()
        n = len(ids)
        if n == 0:
            return {}

        corr = self.estimate_correlation_matrix()
        uncertainties = np.array([self.hypotheses[id_].sharpe_uncertainty for id_ in ids])
        Sigma = corr * np.outer(uncertainties, uncertainties) + np.eye(n) * 1e-6

        try:
            w_raw = fraction * np.linalg.solve(Sigma, mu)
        except np.linalg.LinAlgError:
            w_raw = np.ones(n) / n

        # Long only, normalize
        w = np.maximum(w_raw, 0)
        if w.sum() > 1e-10:
            w /= w.sum()
        else:
            w = np.ones(n) / n

        allocations = {}
        for i, id_ in enumerate(ids):
            allocations[id_] = HypothesisAllocation(
                id=id_,
                weight=float(w[i]),
                dollar_amount=float(w[i] * self.total_capital),
                rationale=f"Fractional Kelly ({fraction:.0%})",
            )
        return allocations

    def regime_filtered_allocation(
        self,
        current_regime: str,
        regime_affinity_map: Optional[dict] = None,
    ) -> dict[str, HypothesisAllocation]:
        """
        Allocate only to hypotheses favorable in current regime.
        """
        if regime_affinity_map is None:
            from idea_engine.debate_system.agents.regime_expert import REGIME_SIGNAL_AFFINITY
            regime_affinity_map = REGIME_SIGNAL_AFFINITY

        # Temporarily disable unfavorable hypotheses
        for h in self.hypotheses.values():
            affinity = regime_affinity_map.get(h.edge_type, {}).get(current_regime, 0.5)
            h.active = affinity >= 0.4  # only use if at least marginally favorable

        alloc = self.optimize_max_sharpe()

        # Restore
        for h in self.hypotheses.values():
            h.active = True

        return alloc

    def portfolio_metrics(self, allocations: dict[str, HypothesisAllocation]) -> dict:
        """Compute portfolio-level statistics for an allocation."""
        ids = list(allocations.keys())
        w = np.array([allocations[id_].weight for id_ in ids])
        mu_arr = np.array([self.hypotheses[id_].expected_sharpe for id_ in ids])

        if len(ids) == 0:
            return {"expected_sharpe": 0.0, "diversification_ratio": 1.0}

        corr = self.estimate_correlation_matrix()
        unc = np.array([self.hypotheses[id_].sharpe_uncertainty for id_ in ids])
        Sigma = corr * np.outer(unc, unc) + np.eye(len(ids)) * 1e-6

        port_ret = float(w @ mu_arr)
        port_var = float(w @ Sigma @ w)
        port_sharpe = port_ret / math.sqrt(max(port_var, 1e-10))

        # Diversification ratio: weighted avg vol / portfolio vol
        weighted_vol = float(w @ unc)
        port_vol = math.sqrt(max(port_var, 1e-10))
        div_ratio = weighted_vol / max(port_vol, 1e-10)

        # Effective N (1 / HHI)
        hhi = float((w**2).sum())
        effective_n = 1.0 / max(hhi, 1e-10)

        return {
            "expected_sharpe": float(port_sharpe),
            "diversification_ratio": float(div_ratio),
            "effective_n": float(effective_n),
            "portfolio_variance": float(port_var),
            "largest_position": float(w.max()) if len(w) > 0 else 0.0,
            "n_active": len(ids),
        }
