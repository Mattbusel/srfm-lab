"""
portfolio_allocator.py
----------------------
Dynamic portfolio allocation across active strategies.
Supports multi-strategy Kelly, regime-aware weights, drawdown deleveraging,
volatility targeting, correlation clustering, risk budgeting, and rebalancing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_inv(mat: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Invert a covariance matrix with ridge regularisation."""
    n = mat.shape[0]
    return np.linalg.inv(mat + np.eye(n) * ridge)


def _normalize(w: np.ndarray) -> np.ndarray:
    total = w.sum()
    if total == 0:
        return np.ones(len(w)) / len(w)
    return w / total


# ---------------------------------------------------------------------------
# Input / output data classes
# ---------------------------------------------------------------------------

@dataclass
class StrategySignal:
    """Caller-supplied data for one strategy at the allocation step."""
    id: str
    name: str
    expected_return: float       # annualised expected return
    volatility: float            # annualised volatility (> 0)
    sharpe: float                # recent Sharpe ratio
    current_drawdown: float      # drawdown from peak, expressed as positive fraction
    returns: list[float] = field(default_factory=list)   # recent periodic returns
    regime_scores: dict[str, float] = field(default_factory=dict)  # {regime: score}
    rolling_rank: float = 0.5    # rolling performance rank [0, 1], higher is better
    tags: list[str] = field(default_factory=list)


@dataclass
class AllocationResult:
    """Output of the allocator for one rebalance cycle."""
    timestamp: str
    weights: dict[str, float]                    # strategy id -> weight
    risk_contributions: dict[str, float]         # strategy id -> risk contribution
    expected_portfolio_return: float
    expected_portfolio_vol: float
    expected_sharpe: float
    leverage: float
    regime: str
    deleveraged: bool
    rebalance_triggered: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Correlation Clustering
# ---------------------------------------------------------------------------

class CorrelationClusterer:
    """
    Group strategies into clusters using single-linkage agglomerative
    clustering on pairwise return correlations.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold   # merge if |corr| > threshold

    def fit(self, returns_matrix: np.ndarray, ids: list[str]) -> dict[int, list[str]]:
        """
        *returns_matrix*: shape (T, N) — T periods, N strategies.
        Returns {cluster_id: [strategy_ids]}.
        """
        n = returns_matrix.shape[1]
        if n == 0:
            return {}
        if n == 1:
            return {0: [ids[0]]}

        corr = np.corrcoef(returns_matrix.T)
        np.nan_to_num(corr, copy=False, nan=0.0)

        # Simple greedy single-linkage
        assigned = [-1] * n
        cluster_id = 0
        for i in range(n):
            if assigned[i] != -1:
                continue
            assigned[i] = cluster_id
            for j in range(i + 1, n):
                if assigned[j] == -1 and abs(corr[i, j]) > self.threshold:
                    assigned[j] = cluster_id
            cluster_id += 1

        clusters: dict[int, list[str]] = {}
        for idx, cid in enumerate(assigned):
            clusters.setdefault(cid, []).append(ids[idx])
        return clusters


# ---------------------------------------------------------------------------
# Risk Budget Allocator
# ---------------------------------------------------------------------------

class RiskBudgetAllocator:
    """
    Equal-risk-contribution (ERC) weights given a covariance matrix.
    Solves iteratively via Newton gradient descent.
    """

    def __init__(self, max_iter: int = 200, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, cov: np.ndarray, budgets: Optional[np.ndarray] = None) -> np.ndarray:
        n = cov.shape[0]
        if budgets is None:
            budgets = np.ones(n) / n

        w = np.ones(n) / n
        for _ in range(self.max_iter):
            sigma_w = cov @ w
            port_vol = np.sqrt(w @ sigma_w)
            risk_contributions = w * sigma_w / (port_vol + 1e-12)
            diff = risk_contributions - budgets * port_vol
            if np.max(np.abs(diff)) < self.tol:
                break
            # Gradient step
            grad = 2 * cov @ w - 2 * budgets * port_vol / (w + 1e-12)
            step = 0.01 / (np.linalg.norm(grad) + 1e-12)
            w = w - step * grad
            w = np.clip(w, 1e-8, None)
            w /= w.sum()
        return w


# ---------------------------------------------------------------------------
# Main Allocator
# ---------------------------------------------------------------------------

class PortfolioAllocator:
    """
    Multi-strategy portfolio allocator.

    Parameters
    ----------
    target_vol : float
        Annualised portfolio volatility target.
    max_dd_limit : float
        Maximum tolerated portfolio drawdown before hard deleveraging.
    kelly_fraction : float
        Fraction of full Kelly to use (e.g. 0.5 = half-Kelly).
    rebalance_mode : str
        ``'threshold'`` (drift-based) or ``'calendar'`` (period-based).
    rebalance_threshold : float
        Maximum allowed weight drift before rebalancing (threshold mode).
    rebalance_period : int
        Number of periods between rebalances (calendar mode).
    rotation_top_k : int
        Only allocate to the top-K strategies by rolling rank.
    dd_deleverage_factor : float
        Multiplier applied to all weights when a strategy exceeds its
        individual drawdown limit.
    max_single_strategy_weight : float
        Hard cap on any single strategy's weight.
    """

    REGIME_WEIGHTS: dict[str, dict[str, float]] = {
        "trending": {"momentum": 1.4, "reversion": 0.6, "arbitrage": 1.0, "macro": 1.2, "micro": 0.8},
        "mean_reverting": {"momentum": 0.6, "reversion": 1.4, "arbitrage": 1.1, "macro": 0.8, "micro": 1.2},
        "high_vol": {"momentum": 0.7, "reversion": 0.7, "arbitrage": 1.3, "macro": 0.9, "micro": 0.9},
        "low_vol": {"momentum": 1.1, "reversion": 1.1, "arbitrage": 0.9, "macro": 1.0, "micro": 1.0},
        "crisis": {"momentum": 0.5, "reversion": 0.5, "arbitrage": 0.8, "macro": 1.5, "micro": 0.5},
        "unknown": {"momentum": 1.0, "reversion": 1.0, "arbitrage": 1.0, "macro": 1.0, "micro": 1.0},
    }

    def __init__(
        self,
        target_vol: float = 0.10,
        max_dd_limit: float = 0.15,
        kelly_fraction: float = 0.5,
        rebalance_mode: str = "threshold",
        rebalance_threshold: float = 0.05,
        rebalance_period: int = 20,
        rotation_top_k: Optional[int] = None,
        dd_deleverage_factor: float = 0.5,
        max_single_strategy_weight: float = 0.40,
        correlation_threshold: float = 0.60,
        use_risk_budget: bool = False,
    ) -> None:
        self.target_vol = target_vol
        self.max_dd_limit = max_dd_limit
        self.kelly_fraction = kelly_fraction
        self.rebalance_mode = rebalance_mode
        self.rebalance_threshold = rebalance_threshold
        self.rebalance_period = rebalance_period
        self.rotation_top_k = rotation_top_k
        self.dd_deleverage_factor = dd_deleverage_factor
        self.max_single_strategy_weight = max_single_strategy_weight
        self.correlation_threshold = correlation_threshold
        self.use_risk_budget = use_risk_budget

        self._prev_weights: dict[str, float] = {}
        self._period_counter: int = 0
        self._history: list[AllocationResult] = []
        self._clusterer = CorrelationClusterer(threshold=correlation_threshold)
        self._risk_budget_solver = RiskBudgetAllocator()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def allocate(
        self,
        signals: list[StrategySignal],
        regime: str = "unknown",
        portfolio_drawdown: float = 0.0,
    ) -> AllocationResult:
        """
        Compute weights for the current period.

        Parameters
        ----------
        signals : list[StrategySignal]
            Per-strategy inputs for this period.
        regime : str
            Current market regime label.
        portfolio_drawdown : float
            Current portfolio drawdown from equity peak (positive fraction).
        """
        notes: list[str] = []
        self._period_counter += 1

        if not signals:
            return self._empty_result(regime, notes)

        # 1. Strategy rotation filter
        signals = self._apply_rotation_filter(signals, notes)

        # 2. Build correlation matrix from return histories
        ids = [s.id for s in signals]
        cov, corr = self._build_covariance(signals)

        # 3. Cluster and elect representatives
        clusters = self._build_clusters(signals, corr, ids)

        # 4. Compute raw Kelly weights
        raw_weights = self._kelly_weights(signals, cov)

        # 5. Regime adjustment
        raw_weights = self._regime_adjust(raw_weights, signals, regime, notes)

        # 6. Risk budget override (optional)
        if self.use_risk_budget:
            raw_weights = self._risk_budget_weights(signals, cov, notes)

        # 7. Cluster-level equalisation
        raw_weights = self._cluster_equalise(raw_weights, ids, clusters, notes)

        # 8. Per-strategy drawdown deleveraging
        raw_weights, deleveraged_local = self._strategy_dd_deleverage(
            raw_weights, signals, notes
        )

        # 9. Normalise and cap
        raw_weights = _normalize(np.array([raw_weights[i] for i in range(len(signals))]))
        raw_weights = np.clip(raw_weights, 0, self.max_single_strategy_weight)
        raw_weights = _normalize(raw_weights)

        weights_dict = {sid: float(raw_weights[i]) for i, sid in enumerate(ids)}

        # 10. Volatility targeting
        port_vol = self._portfolio_vol(raw_weights, cov)
        vol_scale = self.target_vol / (port_vol + 1e-9)
        leverage = float(np.clip(vol_scale, 0.0, 3.0))
        scaled_weights = {sid: w * leverage for sid, w in weights_dict.items()}

        # 11. Portfolio-level drawdown hard stop
        deleveraged_global = False
        if portfolio_drawdown > self.max_dd_limit:
            scaled_weights = {sid: w * self.dd_deleverage_factor
                              for sid, w in scaled_weights.items()}
            leverage *= self.dd_deleverage_factor
            deleveraged_global = True
            notes.append(f"Portfolio DD {portfolio_drawdown:.1%} > limit {self.max_dd_limit:.1%}. "
                         f"All weights halved.")

        # 12. Rebalance gating
        rebalance_triggered = self._check_rebalance(weights_dict)

        # 13. Risk contributions
        risk_contribs = self._risk_contributions(
            np.array([scaled_weights[sid] for sid in ids]), cov
        )
        risk_contrib_dict = {sid: float(risk_contribs[i]) for i, sid in enumerate(ids)}

        # 14. Portfolio stats
        port_ret = float(sum(
            scaled_weights[sid] * s.expected_return
            for sid, s in zip(ids, signals)
        ))
        final_vol = self._portfolio_vol(
            np.array([scaled_weights[sid] for sid in ids]), cov
        )
        sharpe = port_ret / (final_vol + 1e-9)

        result = AllocationResult(
            timestamp=_utcnow(),
            weights=scaled_weights,
            risk_contributions=risk_contrib_dict,
            expected_portfolio_return=round(port_ret, 6),
            expected_portfolio_vol=round(float(final_vol), 6),
            expected_sharpe=round(float(sharpe), 4),
            leverage=round(leverage, 4),
            regime=regime,
            deleveraged=deleveraged_global or deleveraged_local,
            rebalance_triggered=rebalance_triggered,
            notes=notes,
        )
        self._prev_weights = dict(weights_dict)
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_rotation_filter(
        self, signals: list[StrategySignal], notes: list[str]
    ) -> list[StrategySignal]:
        if self.rotation_top_k is None or self.rotation_top_k >= len(signals):
            return signals
        ranked = sorted(signals, key=lambda s: s.rolling_rank, reverse=True)
        kept = ranked[: self.rotation_top_k]
        dropped = [s.name for s in ranked[self.rotation_top_k:]]
        notes.append(f"Rotation: dropped {dropped}")
        return kept

    def _build_covariance(
        self, signals: list[StrategySignal]
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(signals)
        min_len = min((len(s.returns) for s in signals), default=0)

        if min_len >= 4:
            mat = np.column_stack([s.returns[-min_len:] for s in signals])
            cov = np.cov(mat.T)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            std = np.sqrt(np.diag(cov))
            outer = np.outer(std, std)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where(outer > 0, cov / outer, 0.0)
        else:
            # Fallback: diagonal covariance from reported vol
            vols = np.array([s.volatility for s in signals])
            cov = np.diag(vols ** 2)
            corr = np.eye(n)
        return cov, corr

    def _build_clusters(
        self,
        signals: list[StrategySignal],
        corr: np.ndarray,
        ids: list[str],
    ) -> dict[int, list[str]]:
        n = len(signals)
        if n < 2:
            return {0: list(ids)}
        min_len = min((len(s.returns) for s in signals), default=0)
        if min_len >= 4:
            mat = np.column_stack([s.returns[-min_len:] for s in signals])
            return self._clusterer.fit(mat, ids)
        return {i: [sid] for i, sid in enumerate(ids)}

    def _kelly_weights(
        self, signals: list[StrategySignal], cov: np.ndarray
    ) -> np.ndarray:
        mu = np.array([s.expected_return for s in signals])
        inv_cov = _safe_inv(cov)
        w = inv_cov @ mu
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            w = _normalize(w)
        else:
            w = np.ones(len(signals)) / len(signals)
        return w * self.kelly_fraction + (1 - self.kelly_fraction) / len(signals)

    def _regime_adjust(
        self,
        weights: np.ndarray,
        signals: list[StrategySignal],
        regime: str,
        notes: list[str],
    ) -> np.ndarray:
        regime_map = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS["unknown"])
        adjustments = np.ones(len(signals))
        for i, s in enumerate(signals):
            multipliers = [regime_map.get(tag, 1.0) for tag in s.tags]
            adjustments[i] = float(np.mean(multipliers)) if multipliers else 1.0
        adjusted = weights * adjustments
        if adjusted.sum() > 0:
            adjusted = _normalize(adjusted)
        notes.append(f"Regime '{regime}' applied.")
        return adjusted

    def _risk_budget_weights(
        self,
        signals: list[StrategySignal],
        cov: np.ndarray,
        notes: list[str],
    ) -> np.ndarray:
        sharpes = np.array([max(s.sharpe, 0.01) for s in signals])
        budgets = sharpes / sharpes.sum()
        w = self._risk_budget_solver.solve(cov, budgets)
        notes.append("Risk budget weights applied.")
        return w

    def _cluster_equalise(
        self,
        weights: np.ndarray,
        ids: list[str],
        clusters: dict[int, list[str]],
        notes: list[str],
    ) -> np.ndarray:
        """
        Re-distribute weights so each cluster receives equal total weight,
        then divide equally within the cluster.
        """
        if len(clusters) <= 1:
            return weights
        id_to_idx = {sid: i for i, sid in enumerate(ids)}
        n_clusters = len(clusters)
        equal_cluster_weight = 1.0 / n_clusters
        new_weights = np.zeros(len(ids))
        for members in clusters.values():
            per_member = equal_cluster_weight / len(members)
            for sid in members:
                if sid in id_to_idx:
                    new_weights[id_to_idx[sid]] = per_member
        notes.append(f"Cluster equalisation across {n_clusters} clusters.")
        # Blend 50/50 with original Kelly weights
        blended = 0.5 * weights + 0.5 * new_weights
        return _normalize(blended)

    def _strategy_dd_deleverage(
        self,
        weights: np.ndarray,
        signals: list[StrategySignal],
        notes: list[str],
    ) -> tuple[np.ndarray, bool]:
        deleveraged = False
        for i, s in enumerate(signals):
            if s.current_drawdown > 0.10:
                scale = max(0.2, 1.0 - s.current_drawdown * 4)
                weights[i] *= scale
                deleveraged = True
                notes.append(
                    f"Strategy '{s.name}' DD={s.current_drawdown:.1%}; "
                    f"weight scaled by {scale:.2f}"
                )
        return weights, deleveraged

    def _portfolio_vol(self, weights: np.ndarray, cov: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float)
        return float(np.sqrt(np.clip(w @ cov @ w, 0, None)))

    def _risk_contributions(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        sigma_w = cov @ weights
        port_vol = self._portfolio_vol(weights, cov)
        if port_vol < 1e-10:
            return weights / (weights.sum() + 1e-12)
        return weights * sigma_w / port_vol

    def _check_rebalance(self, new_weights: dict[str, float]) -> bool:
        if not self._prev_weights:
            return True
        if self.rebalance_mode == "calendar":
            return self._period_counter % self.rebalance_period == 0
        # Threshold mode
        all_ids = set(new_weights) | set(self._prev_weights)
        for sid in all_ids:
            drift = abs(new_weights.get(sid, 0.0) - self._prev_weights.get(sid, 0.0))
            if drift > self.rebalance_threshold:
                return True
        return False

    def _empty_result(self, regime: str, notes: list[str]) -> AllocationResult:
        notes.append("No signals provided.")
        return AllocationResult(
            timestamp=_utcnow(),
            weights={},
            risk_contributions={},
            expected_portfolio_return=0.0,
            expected_portfolio_vol=0.0,
            expected_sharpe=0.0,
            leverage=0.0,
            regime=regime,
            deleveraged=False,
            rebalance_triggered=False,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def allocation_report(self, result: AllocationResult, signals: list[StrategySignal]) -> str:
        """Return a formatted allocation report string."""
        id_to_name = {s.id: s.name for s in signals}
        lines = [
            "=" * 60,
            f"  ALLOCATION REPORT  [{result.timestamp[:19]}]",
            "=" * 60,
            f"  Regime         : {result.regime}",
            f"  Leverage       : {result.leverage:.3f}x",
            f"  Expected Return: {result.expected_portfolio_return:.2%}",
            f"  Expected Vol   : {result.expected_portfolio_vol:.2%}",
            f"  Expected Sharpe: {result.expected_sharpe:.3f}",
            f"  Deleveraged    : {result.deleveraged}",
            f"  Rebalanced     : {result.rebalance_triggered}",
            "-" * 60,
            f"  {'Strategy':<30} {'Weight':>8}  {'Risk Contrib':>13}",
            "-" * 60,
        ]
        for sid, w in sorted(result.weights.items(), key=lambda x: -x[1]):
            name = id_to_name.get(sid, sid[:8])
            rc = result.risk_contributions.get(sid, 0.0)
            lines.append(f"  {name:<30} {w:>7.2%}  {rc:>12.2%}")
        if result.notes:
            lines.append("-" * 60)
            for note in result.notes:
                lines.append(f"  NOTE: {note}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def history_to_json(self) -> str:
        return json.dumps([r.to_dict() for r in self._history], indent=2)

    def marginal_sharpe(
        self, candidate: StrategySignal, current_signals: list[StrategySignal]
    ) -> float:
        """
        Estimate the marginal Sharpe improvement from adding *candidate*
        to the current portfolio.
        """
        if not current_signals:
            return candidate.sharpe

        all_signals = current_signals + [candidate]
        result_without = self.allocate(current_signals)
        result_with = self.allocate(all_signals)
        return result_with.expected_sharpe - result_without.expected_sharpe


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    rng = np.random.default_rng(42)

    def make_returns(sharpe: float, vol: float = 0.15, n: int = 60) -> list[float]:
        daily_ret = sharpe * vol / 252
        daily_vol = vol / np.sqrt(252)
        return (rng.normal(daily_ret, daily_vol, n)).tolist()

    signals = [
        StrategySignal(
            id="s1", name="MomentumAlpha",
            expected_return=0.12, volatility=0.14, sharpe=1.1,
            current_drawdown=0.03, returns=make_returns(1.1),
            rolling_rank=0.85, tags=["momentum"],
        ),
        StrategySignal(
            id="s2", name="MeanReversionBeta",
            expected_return=0.09, volatility=0.10, sharpe=0.9,
            current_drawdown=0.08, returns=make_returns(0.9),
            rolling_rank=0.60, tags=["reversion"],
        ),
        StrategySignal(
            id="s3", name="StatArbGamma",
            expected_return=0.07, volatility=0.08, sharpe=0.8,
            current_drawdown=0.13, returns=make_returns(0.8),
            rolling_rank=0.40, tags=["arbitrage"],
        ),
        StrategySignal(
            id="s4", name="MacroDelta",
            expected_return=0.15, volatility=0.18, sharpe=1.3,
            current_drawdown=0.01, returns=make_returns(1.3),
            rolling_rank=0.92, tags=["macro"],
        ),
    ]

    allocator = PortfolioAllocator(
        target_vol=0.10,
        max_dd_limit=0.20,
        kelly_fraction=0.5,
        rebalance_mode="threshold",
        rotation_top_k=3,
    )
    result = allocator.allocate(signals, regime="trending", portfolio_drawdown=0.05)
    print(allocator.allocation_report(result, signals))
    print(f"\nMarginal Sharpe of MacroDelta: "
          f"{allocator.marginal_sharpe(signals[3], signals[:3]):.4f}")


if __name__ == "__main__":
    _smoke_test()
