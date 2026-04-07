# execution/portfolio_construction/dynamic_sizing.py
# Dynamic position sizing integrating regime signals, Kelly criterion,
# vol targeting, and turnover constraints.
#
# Implements:
#   - DynamicSizer: fractional Kelly + vol targeting, regime overlays,
#     turnover-constrained rebalancing
#   - TargetPortfolio: dataclass summarising a computed allocation

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Fractional Kelly multiplier -- production default.
# Full Kelly is theoretically optimal but sensitive to estimation error.
# 0.25 (quarter-Kelly) is a common conservative choice.
KELLY_FRACTION: float = 0.25

# Regime-specific weight scalars.
REGIME_MOM_BOOST: float = 1.3      # BH active + trending: momentum boost
REGIME_MR_BOOST: float = 1.2       # BH inactive + mean-reverting: MR boost
REGIME_HIGH_VOL_SCALAR: float = 0.7  # High vol: scale down all positions
REGIME_EVENT_SCALAR: float = 0.5    # Event calendar active: halve exposure

# Floor on effective portfolio vol used in vol targeting (avoid div by zero).
MIN_PORTFOLIO_VOL: float = 1e-6


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RegimeBHState(Enum):
    """Bull/high-vol regime indicator."""
    ACTIVE = auto()
    INACTIVE = auto()
    UNKNOWN = auto()


class RegimeTrend(Enum):
    """Trend / mean-reversion indicator."""
    TRENDING = auto()
    MEAN_REVERTING = auto()
    NEUTRAL = auto()


class VolRegime(Enum):
    """Volatility regime."""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()


# ---------------------------------------------------------------------------
# RegimeState: inputs from the ensemble regime module
# ---------------------------------------------------------------------------


@dataclass
class RegimeState:
    """
    Snapshot of regime state used to adjust portfolio weights.

    Fields mirror the outputs of the regime_ensemble module.

    Attributes
    ----------
    bh_active : RegimeBHState
        Whether the bull/high-momentum regime is active.
    trend : RegimeTrend
        Whether the market is trending or mean-reverting.
    vol_regime : VolRegime
        Current volatility regime.
    event_calendar_active : bool
        True when a high-impact event (FOMC, CPI, expiry) is imminent.
    asset_regimes : dict
        Optional per-asset regime override.  Maps asset index (int) to a
        dict with keys 'is_momentum' (bool) and 'is_mean_reversion' (bool).
    """

    bh_active: RegimeBHState = RegimeBHState.UNKNOWN
    trend: RegimeTrend = RegimeTrend.NEUTRAL
    vol_regime: VolRegime = VolRegime.NORMAL
    event_calendar_active: bool = False
    asset_regimes: Dict[int, Dict[str, bool]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TargetPortfolio
# ---------------------------------------------------------------------------


@dataclass
class TargetPortfolio:
    """
    Result of a sizing computation.

    Attributes
    ----------
    weights : np.ndarray
        Target weight vector summing to 1.
    expected_sharpe : float
        Ex-ante Sharpe estimate.
    expected_vol : float
        Ex-ante annualised portfolio volatility.
    regime : RegimeState
        Regime state at the time of computation.
    timestamp : datetime
        Wall-clock time of computation.
    kelly_weights_unscaled : np.ndarray, optional
        Raw Kelly weights before vol targeting and regime scaling.
    notes : list[str]
        Audit trail of adjustments applied.
    """

    weights: np.ndarray
    expected_sharpe: float
    expected_vol: float
    regime: RegimeState
    timestamp: datetime
    kelly_weights_unscaled: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DynamicSizer
# ---------------------------------------------------------------------------


class DynamicSizer:
    """
    Computes dynamic position sizes integrating:
    - Fractional Kelly criterion
    - Portfolio volatility targeting
    - Regime-conditional weight scaling
    - Turnover constraints

    Usage example:
        sizer = DynamicSizer(kelly_fraction=0.25, target_vol=0.15)
        weights = sizer.size_kelly_vol_target(signals, vols, corr)
        final_weights = sizer.regime_adjusted_weights(weights, regime_state)
        final_weights = sizer.turnover_constrained(current_weights, final_weights)
    """

    def __init__(
        self,
        kelly_fraction: float = KELLY_FRACTION,
        target_vol: float = 0.15,
        min_weight: float = 0.0,
        max_weight: float = 0.30,
        long_only: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        kelly_fraction : float
            Fraction of full Kelly to apply.  Must be in (0, 1].
        target_vol : float
            Annualised portfolio volatility target.
        min_weight : float
            Floor per-asset weight.  Set negative for long/short.
        max_weight : float
            Cap per-asset weight.
        long_only : bool
            If True, negative weights are zeroed before vol targeting.
        """
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
        if target_vol <= 0:
            raise ValueError(f"target_vol must be positive, got {target_vol}")

        self.kelly_fraction = kelly_fraction
        self.target_vol = target_vol
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.long_only = long_only

    # ------------------------------------------------------------------
    # Core sizing: fractional Kelly + vol targeting
    # ------------------------------------------------------------------

    def size_kelly_vol_target(
        self,
        signals: np.ndarray,
        vols: np.ndarray,
        corr: np.ndarray,
        target_vol: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute fractional Kelly weights scaled to a portfolio vol target.

        Full Kelly for a multi-asset portfolio (first-order approximation):

            f_i = mu_i / sigma_i^2 - sum_{j != i} rho_{ij} * f_j * sigma_j / sigma_i

        In matrix form:  f = Sigma^{-1} * mu
        where Sigma is the covariance matrix (not just diagonal), and mu are
        the expected returns encoded in `signals`.

        Fractional Kelly:  f_frac = kelly_fraction * f

        Vol targeting:
            scale = target_vol / portfolio_vol(f_frac)
            w = f_frac * scale  (then renorm to sum to 1)

        Parameters
        ----------
        signals : np.ndarray, shape (n,)
            Alpha / expected return signals.  Units: annualised return fraction.
        vols : np.ndarray, shape (n,)
            Per-asset annualised volatilities.
        corr : np.ndarray, shape (n, n)
            Correlation matrix.
        target_vol : float, optional
            Override instance default target vol.
        kelly_fraction : float, optional
            Override instance default Kelly fraction.

        Returns
        -------
        np.ndarray, shape (n,)
            Weight vector.  May not sum exactly to 1 if clipping is applied;
            normalisation is performed after clipping.
        """
        signals = np.asarray(signals, dtype=float)
        vols = np.asarray(vols, dtype=float)
        corr = np.asarray(corr, dtype=float)
        n = len(signals)

        if vols.shape[0] != n:
            raise ValueError(f"vols shape {vols.shape} does not match signals length {n}")
        if corr.shape != (n, n):
            raise ValueError(f"corr shape {corr.shape} != ({n}, {n})")
        if np.any(vols <= 0):
            raise ValueError("All vols must be strictly positive.")

        tv = target_vol if target_vol is not None else self.target_vol
        kf = kelly_fraction if kelly_fraction is not None else self.kelly_fraction

        # Build covariance matrix from correlation and vols.
        cov = _corr_vols_to_cov(corr, vols)
        cov = _ensure_psd(cov)

        # Full Kelly weights: f = Sigma^{-1} * mu
        # Use pseudoinverse for numerical stability.
        try:
            cov_inv = np.linalg.inv(cov + np.eye(n) * 1e-12)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        kelly_full = cov_inv @ signals

        # Fractional Kelly.
        kelly_frac = kf * kelly_full

        # Long-only: zero out negatives.
        if self.long_only:
            kelly_frac = np.clip(kelly_frac, 0.0, None)

        # Clip to max weight before vol targeting.
        kelly_frac = np.clip(kelly_frac, self.min_weight, self.max_weight)

        # Compute current portfolio vol.
        port_var = float(kelly_frac @ cov @ kelly_frac)
        port_vol = np.sqrt(max(port_var, MIN_PORTFOLIO_VOL ** 2))

        # Vol-target scaling.
        scale = tv / port_vol
        weights = kelly_frac * scale

        # Re-clip and normalise.
        weights = np.clip(weights, self.min_weight, self.max_weight)
        wsum = weights.sum()
        if wsum > 1e-12:
            weights = weights / wsum
        else:
            weights = np.full(n, 1.0 / n)

        return weights

    # ------------------------------------------------------------------
    # Regime overlay
    # ------------------------------------------------------------------

    def regime_adjusted_weights(
        self,
        base_weights: np.ndarray,
        regime_state: RegimeState,
    ) -> np.ndarray:
        """
        Apply regime-conditional scaling to base portfolio weights.

        Rules (applied in order):

        1. BH active + trending:
           - Assets classified as momentum (asset_regimes[i]['is_momentum'])
             are boosted by REGIME_MOM_BOOST (1.3x).
           - Non-momentum weights are unchanged.

        2. BH inactive + mean-reverting:
           - Momentum assets are set to zero.
           - Assets classified as mean-reversion (asset_regimes[i]['is_mean_reversion'])
             are boosted by REGIME_MR_BOOST (1.2x).

        3. High vol regime:
           - All weights are scaled by REGIME_HIGH_VOL_SCALAR (0.7x).
           - Applied after step 1/2 boosts.

        4. Event calendar active:
           - All weights are scaled by REGIME_EVENT_SCALAR (0.5x).
           - Applied last.

        After all adjustments, weights are re-normalised to sum to 1.

        Parameters
        ----------
        base_weights : np.ndarray, shape (n,)
            Input portfolio weights (should sum to 1).
        regime_state : RegimeState

        Returns
        -------
        np.ndarray, shape (n,)
        """
        base_weights = np.asarray(base_weights, dtype=float)
        weights = base_weights.copy()
        n = len(weights)
        notes: List[str] = []

        # Step 1 / 2: BH state + trend interaction.
        bh_active = regime_state.bh_active == RegimeBHState.ACTIVE
        bh_inactive = regime_state.bh_active == RegimeBHState.INACTIVE
        trending = regime_state.trend == RegimeTrend.TRENDING
        mean_reverting = regime_state.trend == RegimeTrend.MEAN_REVERTING

        asset_regimes = regime_state.asset_regimes

        if bh_active and trending:
            # Boost momentum assets.
            for i in range(n):
                is_mom = asset_regimes.get(i, {}).get("is_momentum", False)
                if is_mom:
                    weights[i] *= REGIME_MOM_BOOST
            notes.append(f"BH active + trending: applied {REGIME_MOM_BOOST}x momentum boost")

        elif bh_inactive and mean_reverting:
            # Flatten momentum; boost mean-reversion.
            for i in range(n):
                asset_info = asset_regimes.get(i, {})
                is_mom = asset_info.get("is_momentum", False)
                is_mr = asset_info.get("is_mean_reversion", False)
                if is_mom:
                    weights[i] = 0.0
                elif is_mr:
                    weights[i] *= REGIME_MR_BOOST
            notes.append(
                f"BH inactive + mean-reverting: zeroed momentum, "
                f"applied {REGIME_MR_BOOST}x MR boost"
            )

        # Step 3: High vol regime.
        if regime_state.vol_regime == VolRegime.HIGH:
            weights *= REGIME_HIGH_VOL_SCALAR
            notes.append(f"High vol regime: scaled weights by {REGIME_HIGH_VOL_SCALAR}")

        # Step 4: Event calendar.
        if regime_state.event_calendar_active:
            weights *= REGIME_EVENT_SCALAR
            notes.append(f"Event calendar active: scaled weights by {REGIME_EVENT_SCALAR}")

        # Renormalise.
        if self.long_only:
            weights = np.clip(weights, 0.0, None)

        wsum = weights.sum()
        if wsum > 1e-12:
            weights = weights / wsum
        else:
            # All weights zeroed -- return equal-weight as safety fallback.
            weights = np.full(n, 1.0 / n)
            notes.append("All weights zeroed after regime adjustment -- fallback to equal weight")

        if notes:
            logger.debug("Regime adjustments: %s", "; ".join(notes))

        return weights

    # ------------------------------------------------------------------
    # Turnover constraint
    # ------------------------------------------------------------------

    def turnover_constrained(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        max_turnover: float = 0.10,
    ) -> np.ndarray:
        """
        Clip portfolio weight changes to respect a one-way turnover budget.

        One-way turnover is defined as:
            TO = 0.5 * sum(|target_i - current_i|)

        If TO <= max_turnover, the target weights are returned unchanged.

        If TO > max_turnover, changes are scaled back proportionally:
            delta_i_clipped = delta_i * (max_turnover / TO)

        This preserves the relative direction and magnitude of trades --
        i.e. if asset A was being increased by twice as much as asset B in
        percentage terms, it remains twice as much after clipping.

        After clipping, the weights are renormalised to sum to 1 to ensure
        they remain a valid portfolio allocation.

        Parameters
        ----------
        current_weights : np.ndarray, shape (n,)
            Current portfolio weights.  Should sum to 1.
        target_weights : np.ndarray, shape (n,)
            Desired portfolio weights.  Should sum to 1.
        max_turnover : float
            Maximum one-way turnover per rebalance.  E.g. 0.10 = 10%.

        Returns
        -------
        np.ndarray, shape (n,)
            Feasible weights respecting the turnover budget.
        """
        current_weights = np.asarray(current_weights, dtype=float)
        target_weights = np.asarray(target_weights, dtype=float)

        if current_weights.shape != target_weights.shape:
            raise ValueError(
                f"current_weights shape {current_weights.shape} != "
                f"target_weights shape {target_weights.shape}"
            )

        delta = target_weights - current_weights
        one_way_turnover = 0.5 * float(np.sum(np.abs(delta)))

        if one_way_turnover <= max_turnover + 1e-10:
            # Already within budget.
            return target_weights.copy()

        # Proportional scaling of deltas to hit the turnover budget exactly.
        scale = max_turnover / one_way_turnover
        delta_clipped = delta * scale
        clipped_weights = current_weights + delta_clipped

        # Ensure non-negative for long-only.
        if self.long_only:
            clipped_weights = np.clip(clipped_weights, 0.0, None)

        # Renormalise.
        wsum = clipped_weights.sum()
        if wsum > 1e-12:
            clipped_weights = clipped_weights / wsum
        else:
            clipped_weights = np.full(len(clipped_weights), 1.0 / len(clipped_weights))

        return clipped_weights

    # ------------------------------------------------------------------
    # Full pipeline convenience method
    # ------------------------------------------------------------------

    def compute_target_portfolio(
        self,
        signals: np.ndarray,
        vols: np.ndarray,
        corr: np.ndarray,
        regime_state: Optional[RegimeState] = None,
        current_weights: Optional[np.ndarray] = None,
        max_turnover: float = 0.10,
        asset_names: Optional[Sequence[str]] = None,
    ) -> TargetPortfolio:
        """
        End-to-end pipeline: Kelly sizing -> regime overlay -> turnover clipping.

        Parameters
        ----------
        signals : np.ndarray, shape (n,)
            Alpha signals (expected returns).
        vols : np.ndarray, shape (n,)
            Per-asset annualised vols.
        corr : np.ndarray, shape (n, n)
            Correlation matrix.
        regime_state : RegimeState, optional
            If None, a neutral regime state is used.
        current_weights : np.ndarray, shape (n,), optional
            Current holdings.  Required if max_turnover < 1.0.
        max_turnover : float
            One-way turnover budget.
        asset_names : sequence, optional

        Returns
        -------
        TargetPortfolio
        """
        signals = np.asarray(signals, dtype=float)
        vols = np.asarray(vols, dtype=float)
        corr = np.asarray(corr, dtype=float)
        n = len(signals)

        if regime_state is None:
            regime_state = RegimeState()

        notes: List[str] = []

        # Step 1: Kelly + vol targeting.
        kelly_weights = self.size_kelly_vol_target(signals, vols, corr)
        notes.append("Applied fractional Kelly + vol targeting")

        # Step 2: Regime overlay.
        regime_weights = self.regime_adjusted_weights(kelly_weights, regime_state)
        notes.append("Applied regime overlay")

        # Step 3: Turnover constraint.
        if current_weights is not None:
            cw = np.asarray(current_weights, dtype=float)
            if cw.shape[0] != n:
                warnings.warn(
                    f"current_weights length {cw.shape[0]} != n {n}; skipping turnover constraint.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                final_weights = regime_weights
            else:
                final_weights = self.turnover_constrained(cw, regime_weights, max_turnover)
                notes.append(f"Applied turnover constraint (max={max_turnover:.1%})")
        else:
            final_weights = regime_weights

        # Compute ex-ante statistics.
        cov = _corr_vols_to_cov(corr, vols)
        port_vol = float(np.sqrt(final_weights @ cov @ final_weights))
        port_ret = float(final_weights @ signals)
        sharpe = port_ret / port_vol if port_vol > MIN_PORTFOLIO_VOL else 0.0

        return TargetPortfolio(
            weights=final_weights,
            expected_sharpe=sharpe,
            expected_vol=port_vol,
            regime=regime_state,
            timestamp=datetime.utcnow(),
            kelly_weights_unscaled=kelly_weights,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _corr_vols_to_cov(corr: np.ndarray, vols: np.ndarray) -> np.ndarray:
    """Build covariance matrix from correlation matrix and volatility vector."""
    D = np.diag(vols)
    return D @ corr @ D


def _ensure_psd(cov: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """Project onto the PSD cone by clipping negative eigenvalues."""
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
