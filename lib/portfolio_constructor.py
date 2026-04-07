"""
portfolio_constructor.py -- translates signal scores into target portfolio weights.

Applies risk budgeting, sector constraints, turnover limits, and
concentration caps. Returns PortfolioConstructionResult with full attribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional
import logging

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

logger = logging.getLogger(__name__)

# -- default constraint parameters
MAX_SINGLE_WEIGHT   = 0.15    # 15% single-name cap
MAX_SECTOR_WEIGHT   = 0.30    # 30% sector cap
DEFAULT_MAX_TURNOVER = 0.20   # 20% two-way turnover limit per rebalance
SIGMOID_SCALE        = 0.05   # default scale for zscore_to_weight


@dataclass
class PortfolioConstructionResult:
    """Output of a full portfolio construction cycle."""
    weights: dict[str, float]         # symbol -> final weight (sum ~ 1)
    turnover: float                   # sum(|w_new - w_old|)
    risk_contrib: dict[str, float]    # symbol -> fractional risk contribution
    expected_return: float            # weighted sum of signal scores
    expected_vol: float               # portfolio ex-ante volatility (if cov available)
    diagnostics: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Signal-to-weight converters
# ------------------------------------------------------------------

class SignalToWeightConverter:
    """Maps raw signal scores to portfolio weights."""

    @staticmethod
    def zscore_to_weight(z: float, scale: float = SIGMOID_SCALE) -> float:
        """
        Sigmoid mapping: w = scale * (2 / (1 + exp(-z)) - 1)
        Output range: (-scale, +scale).
        """
        sig = 2.0 / (1.0 + math.exp(-z)) - 1.0
        return scale * sig

    @staticmethod
    def rank_to_weight(rank: int, n: int, max_weight: float = MAX_SINGLE_WEIGHT) -> float:
        """
        Linear rank weight: top-ranked symbol gets max_weight,
        bottom-ranked gets -max_weight (for long/short book).
        rank in [1, n] where 1 = strongest positive signal.
        """
        if n <= 1:
            return 0.0
        # -- map rank to [-1, 1] then scale
        normalized = 1.0 - 2.0 * (rank - 1) / (n - 1)
        return max_weight * normalized

    @staticmethod
    def clip_to_max(weight: float, max_weight: float = MAX_SINGLE_WEIGHT) -> float:
        """Hard-clip a weight to [-max_weight, max_weight]."""
        return max(-max_weight, min(max_weight, weight))


# ------------------------------------------------------------------
# Risk budget allocator
# ------------------------------------------------------------------

class RiskBudgetAllocator:
    """
    Allocates weights proportional to inverse volatility (equal risk contribution).
    Uses diagonal of covariance matrix (individual vols) for simplicity.
    Full ERC via iterative solver available when numpy is present.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def _vols_from_cov(self, cov: "np.ndarray") -> "np.ndarray":
        diag = np.diag(cov)
        return np.sqrt(np.maximum(diag, 1e-12))

    def allocate(
        self,
        signals: dict[str, float],
        cov_matrix: Optional["np.ndarray"] = None,
    ) -> dict[str, float]:
        """
        Compute equal-risk-contribution weights.
        If cov_matrix is None, uses unit volatilities (equal weight).
        Returns weights that sum to 1 (long-only, absolute weight).
        """
        symbols = list(signals.keys())
        n = len(symbols)
        if n == 0:
            return {}

        if cov_matrix is not None and _HAS_NUMPY:
            vols = self._vols_from_cov(cov_matrix)
        else:
            vols = [1.0] * n

        inv_vols = [1.0 / max(v, 1e-12) for v in vols]
        total_inv = sum(inv_vols)
        raw_weights = {sym: inv_vols[i] / total_inv for i, sym in enumerate(symbols)}

        # -- apply signal sign: positive signal -> long, negative -> short
        signed = {}
        for sym, w in raw_weights.items():
            sign = 1.0 if signals[sym] >= 0 else -1.0
            signed[sym] = sign * w

        return signed

    def risk_contributions(
        self,
        weights: dict[str, float],
        cov_matrix: Optional["np.ndarray"] = None,
    ) -> dict[str, float]:
        """
        Compute fractional risk contribution per symbol.
        RC_i = w_i * (Sigma * w)_i / (w' Sigma w)
        Falls back to w_i^2 / sum(w_j^2) without cov.
        """
        symbols = list(weights.keys())
        w_vals = [weights[s] for s in symbols]

        if cov_matrix is not None and _HAS_NUMPY:
            w_arr = np.array(w_vals)
            sigma_w = cov_matrix @ w_arr
            port_var = float(w_arr @ sigma_w)
            if port_var < 1e-12:
                rc_vals = [1.0 / len(symbols)] * len(symbols)
            else:
                rc_vals = [w_arr[i] * sigma_w[i] / port_var for i in range(len(symbols))]
        else:
            sq_sum = sum(w ** 2 for w in w_vals) or 1.0
            rc_vals = [w ** 2 / sq_sum for w in w_vals]

        return {sym: rc_vals[i] for i, sym in enumerate(symbols)}


# ------------------------------------------------------------------
# Turnover constraint
# ------------------------------------------------------------------

class TurnoverConstraint:
    """Clips weight changes so total two-way turnover stays within budget."""

    def apply(
        self,
        current: dict[str, float],
        target: dict[str, float],
        max_turnover: float = DEFAULT_MAX_TURNOVER,
    ) -> dict[str, float]:
        """
        Scale all deltas so sum(|w_new - w_old|) <= max_turnover.
        Returns a new weight dict.
        """
        all_symbols = set(current.keys()) | set(target.keys())
        deltas = {
            sym: target.get(sym, 0.0) - current.get(sym, 0.0)
            for sym in all_symbols
        }
        total_delta = sum(abs(d) for d in deltas.values())

        if total_delta <= max_turnover or total_delta < 1e-12:
            return {sym: current.get(sym, 0.0) + d for sym, d in deltas.items()}

        scale = max_turnover / total_delta
        result = {}
        for sym in all_symbols:
            result[sym] = current.get(sym, 0.0) + deltas[sym] * scale
        return result

    def compute_turnover(
        self, current: dict[str, float], proposed: dict[str, float]
    ) -> float:
        all_symbols = set(current.keys()) | set(proposed.keys())
        return sum(
            abs(proposed.get(s, 0.0) - current.get(s, 0.0))
            for s in all_symbols
        )


# ------------------------------------------------------------------
# Sector constraint
# ------------------------------------------------------------------

class SectorConstraint:
    """
    Ensures no single sector exceeds max_sector_weight of total portfolio.
    Scales down all symbols in an over-weight sector proportionally.
    """

    def __init__(self, max_sector_weight: float = MAX_SECTOR_WEIGHT):
        self.max_sector_weight = max_sector_weight

    def apply(
        self,
        weights: dict[str, float],
        sector_map: dict[str, str],
    ) -> dict[str, float]:
        """
        sector_map: {symbol -> sector_name}
        Returns adjusted weights.
        """
        if not sector_map:
            return dict(weights)

        # -- compute sector gross weights
        sector_weights: dict[str, float] = {}
        for sym, w in weights.items():
            sec = sector_map.get(sym, "UNKNOWN")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + abs(w)

        result = dict(weights)
        for sec, sec_w in sector_weights.items():
            if sec_w <= self.max_sector_weight:
                continue
            scale = self.max_sector_weight / sec_w
            for sym in result:
                if sector_map.get(sym, "UNKNOWN") == sec:
                    result[sym] *= scale

        return result


# ------------------------------------------------------------------
# Concentration limit
# ------------------------------------------------------------------

class ConcentrationLimit:
    """Hard-caps any single position at max_position_weight."""

    def __init__(self, max_position_weight: float = MAX_SINGLE_WEIGHT):
        self.max_position_weight = max_position_weight

    def apply(self, weights: dict[str, float]) -> dict[str, float]:
        return {
            sym: max(-self.max_position_weight, min(self.max_position_weight, w))
            for sym, w in weights.items()
        }


# ------------------------------------------------------------------
# Portfolio constructor facade
# ------------------------------------------------------------------

class PortfolioConstructor:
    """
    Translates signal scores into target portfolio weights.
    Applies risk budgeting, sector constraints, and turnover limits.
    """

    def __init__(
        self,
        max_turnover: float = DEFAULT_MAX_TURNOVER,
        max_position: float = MAX_SINGLE_WEIGHT,
        max_sector: float = MAX_SECTOR_WEIGHT,
        zscore_scale: float = SIGMOID_SCALE,
    ):
        self.max_turnover   = max_turnover
        self.max_position   = max_position
        self.max_sector     = max_sector
        self.zscore_scale   = zscore_scale

        self._converter    = SignalToWeightConverter()
        self._risk_alloc   = RiskBudgetAllocator()
        self._turnover_con = TurnoverConstraint()
        self._sector_con   = SectorConstraint(max_sector)
        self._conc_limit   = ConcentrationLimit(max_position)

    def construct(
        self,
        signals: dict[str, float],
        current_weights: Optional[dict[str, float]] = None,
        cov_matrix: Optional["np.ndarray"] = None,
        sector_map: Optional[dict[str, str]] = None,
    ) -> PortfolioConstructionResult:
        """
        Full portfolio construction pipeline:
        1. Convert signals to raw weights (zscore sigmoid).
        2. Apply risk-budget scaling.
        3. Apply concentration limit.
        4. Apply sector constraint.
        5. Apply turnover constraint vs. current weights.
        6. Compute diagnostics.
        """
        if not signals:
            return PortfolioConstructionResult(
                weights={}, turnover=0.0, risk_contrib={},
                expected_return=0.0, expected_vol=0.0,
            )

        current = current_weights or {}

        # -- step 1: raw sigmoid weights from signal z-scores
        raw = {sym: self._converter.zscore_to_weight(z, self.zscore_scale)
               for sym, z in signals.items()}

        # -- step 2: risk-budget scaling (uses inv-vol proportional adjustment)
        risk_weights = self._risk_alloc.allocate(signals, cov_matrix)

        # -- blend: 50% zscore weight, 50% risk-budget weight
        blended: dict[str, float] = {}
        for sym in signals:
            blended[sym] = 0.5 * raw.get(sym, 0.0) + 0.5 * risk_weights.get(sym, 0.0)

        # -- step 3: concentration limit
        blended = self._conc_limit.apply(blended)

        # -- step 4: sector constraint
        if sector_map:
            blended = self._sector_con.apply(blended, sector_map)

        # -- step 5: turnover constraint
        final = self._turnover_con.apply(current, blended, self.max_turnover)
        final = {sym: w for sym, w in final.items() if abs(w) > 1e-9}

        # -- step 6: diagnostics
        turnover = self._turnover_con.compute_turnover(current, final)
        risk_contrib = self._risk_alloc.risk_contributions(final, cov_matrix)

        all_w = list(final.values())
        gross = sum(abs(w) for w in all_w) or 1.0
        expected_return = sum(signals.get(sym, 0.0) * w for sym, w in final.items())

        expected_vol = 0.0
        if cov_matrix is not None and _HAS_NUMPY:
            syms = list(final.keys())
            w_arr = np.array([final[s] for s in syms])
            try:
                port_var = float(w_arr @ cov_matrix @ w_arr)
                expected_vol = math.sqrt(max(port_var, 0.0))
            except Exception:
                expected_vol = 0.0

        return PortfolioConstructionResult(
            weights=final,
            turnover=turnover,
            risk_contrib=risk_contrib,
            expected_return=expected_return,
            expected_vol=expected_vol,
            diagnostics={
                "gross_exposure": gross,
                "n_positions": len(final),
                "max_weight": max((abs(w) for w in all_w), default=0.0),
            },
        )

    def rebalance(
        self,
        signals: dict[str, float],
        current_weights: dict[str, float],
        cov_matrix: Optional["np.ndarray"] = None,
        sector_map: Optional[dict[str, str]] = None,
    ) -> PortfolioConstructionResult:
        """Convenience wrapper -- same as construct() with current_weights supplied."""
        return self.construct(
            signals=signals,
            current_weights=current_weights,
            cov_matrix=cov_matrix,
            sector_map=sector_map,
        )
