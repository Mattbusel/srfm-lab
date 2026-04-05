"""
execution/tca/market_impact_model.py
======================================
Pre-trade market impact estimation and post-trade model calibration.

Two models are implemented:

Square-root (Almgren et al.)
    impact = eta * sigma * sqrt(Q / V)

    where:
        eta   = market-impact coefficient (calibrated from historical fills)
        sigma = daily volatility of the instrument
        Q     = order size (USD notional)
        V     = estimated daily volume (USD)

Almgren-Chriss decomposition
    total_impact = permanent_impact + temporary_impact
    permanent  = gamma * sigma * sqrt(Q / V)
    temporary  = eta   * sigma * sqrt(Q / (V * T))  (T = execution duration)

Model calibration uses OLS regression on historical fills:
    actual_impact ~ alpha + beta * model_impact

Once calibrated the model's beta deviates from 1.0 and can be used to
scale predictions.  Calibration requires at least MIN_CALIBRATION_FILLS fills.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("execution.market_impact_model")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ETA:   float = 0.1     # square-root coefficient (unitless)
DEFAULT_GAMMA: float = 0.05    # Almgren-Chriss permanent impact coefficient
MIN_CALIBRATION_FILLS: int = 20


# ---------------------------------------------------------------------------
# Impact estimate value object
# ---------------------------------------------------------------------------

@dataclass
class ImpactEstimate:
    """Pre-trade market impact estimate."""
    symbol:           str
    order_notional:   float       # USD
    daily_volume:     float       # USD
    daily_volatility: float       # fraction, e.g. 0.02 = 2 %
    execution_min:    float       # execution window in minutes (0 for instantaneous)

    # Filled by compute()
    sqrt_model_bps:   float = 0.0
    permanent_bps:    float = 0.0
    temporary_bps:    float = 0.0
    total_ac_bps:     float = 0.0
    recommended_max_notional: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol":               self.symbol,
            "order_notional":       self.order_notional,
            "daily_volume":         self.daily_volume,
            "daily_volatility":     self.daily_volatility,
            "sqrt_model_bps":       self.sqrt_model_bps,
            "permanent_bps":        self.permanent_bps,
            "temporary_bps":        self.temporary_bps,
            "total_ac_bps":         self.total_ac_bps,
            "recommended_max_notional": self.recommended_max_notional,
        }


# ---------------------------------------------------------------------------
# MarketImpactModel
# ---------------------------------------------------------------------------

class MarketImpactModel:
    """
    Estimates pre-trade market impact and calibrates from historical fills.

    Parameters
    ----------
    eta : float
        Square-root model coefficient.  Will be updated after calibration.
    gamma : float
        Almgren-Chriss permanent impact coefficient.
    impact_threshold_bps : float
        Threshold used in ``recommend_order_size`` — returned notional
        keeps expected impact below this value.
    """

    def __init__(
        self,
        eta:                   float = DEFAULT_ETA,
        gamma:                 float = DEFAULT_GAMMA,
        impact_threshold_bps:  float = 10.0,
    ) -> None:
        self._eta                 = eta
        self._gamma               = gamma
        self._impact_threshold    = impact_threshold_bps
        self._calibration_data:   list[tuple[float, float]] = []  # (model_impact, actual_impact)
        self._calibrated_beta:    float = 1.0
        self._calibrated_alpha:   float = 0.0
        self._lock                = threading.RLock()

    # ------------------------------------------------------------------
    # Square-root model
    # ------------------------------------------------------------------

    def estimate_sqrt_impact(
        self,
        order_notional: float,
        daily_volume:   float,
        daily_vol:      float,
        eta:            Optional[float] = None,
    ) -> float:
        """
        Compute square-root market impact in basis points.

        Parameters
        ----------
        order_notional : float  USD notional of the order.
        daily_volume : float    Estimated daily turnover in USD.
        daily_vol : float       Daily price volatility (fraction).
        eta : float | None      Override model eta.

        Returns
        -------
        float  Expected market impact in bps.
        """
        e = eta if eta is not None else self._eta
        if daily_volume <= 0 or daily_vol <= 0:
            return 0.0
        participation = order_notional / daily_volume
        impact_frac   = e * daily_vol * math.sqrt(participation)
        return impact_frac * 10_000

    # ------------------------------------------------------------------
    # Almgren-Chriss model
    # ------------------------------------------------------------------

    def estimate_almgren_chriss(
        self,
        order_notional: float,
        daily_volume:   float,
        daily_vol:      float,
        execution_min:  float = 0.0,
        trading_day_min: float = 480.0,
    ) -> tuple[float, float]:
        """
        Almgren-Chriss permanent + temporary impact decomposition.

        Returns
        -------
        (permanent_bps, temporary_bps)
        """
        if daily_volume <= 0 or daily_vol <= 0:
            return 0.0, 0.0
        participation = order_notional / daily_volume
        permanent     = self._gamma * daily_vol * math.sqrt(participation) * 10_000

        if execution_min > 0:
            speed         = participation / (execution_min / trading_day_min)
            temporary     = self._eta * daily_vol * math.sqrt(speed) * 10_000
        else:
            temporary     = self.estimate_sqrt_impact(order_notional, daily_volume, daily_vol)

        return permanent, temporary

    # ------------------------------------------------------------------
    # Full estimate
    # ------------------------------------------------------------------

    def estimate(
        self,
        symbol:           str,
        order_notional:   float,
        daily_volume:     float,
        daily_volatility: float,
        execution_min:    float = 0.0,
    ) -> ImpactEstimate:
        """
        Compute a complete ImpactEstimate for a proposed order.

        Parameters
        ----------
        symbol : str
        order_notional : float    USD value of the order.
        daily_volume : float      Estimated USD daily volume.
        daily_volatility : float  Daily vol (fraction).
        execution_min : float     Planned execution window in minutes.

        Returns
        -------
        ImpactEstimate
        """
        est = ImpactEstimate(
            symbol           = symbol,
            order_notional   = order_notional,
            daily_volume     = daily_volume,
            daily_volatility = daily_volatility,
            execution_min    = execution_min,
        )

        with self._lock:
            raw_sqrt = self.estimate_sqrt_impact(
                order_notional, daily_volume, daily_volatility
            )
            # Apply calibrated scaling if available
            est.sqrt_model_bps = self._calibrated_alpha + self._calibrated_beta * raw_sqrt

            perm, temp         = self.estimate_almgren_chriss(
                order_notional, daily_volume, daily_volatility, execution_min
            )
            est.permanent_bps   = perm
            est.temporary_bps   = temp
            est.total_ac_bps    = perm + temp

            est.recommended_max_notional = self.recommend_order_size(
                daily_volume, daily_volatility
            )

        return est

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------

    def recommend_order_size(
        self,
        daily_volume:   float,
        daily_vol:      float,
        threshold_bps:  Optional[float] = None,
    ) -> float:
        """
        Return the maximum order notional (USD) that keeps expected
        square-root impact below *threshold_bps*.

        Derived by inverting the square-root model:
            impact = eta * sigma * sqrt(Q/V)  ->  Q = V * (impact / (eta*sigma))^2
        """
        threshold = threshold_bps if threshold_bps is not None else self._impact_threshold
        if daily_vol <= 0 or self._eta <= 0:
            return 0.0
        threshold_frac = threshold / 10_000.0
        participation  = (threshold_frac / (self._eta * daily_vol)) ** 2
        return participation * daily_volume

    # ------------------------------------------------------------------
    # OLS calibration
    # ------------------------------------------------------------------

    def add_calibration_point(
        self,
        model_impact_bps:  float,
        actual_impact_bps: float,
    ) -> None:
        """
        Record one (model, actual) data point for OLS calibration.

        Automatically re-calibrates once MIN_CALIBRATION_FILLS points
        have been accumulated.
        """
        with self._lock:
            self._calibration_data.append((model_impact_bps, actual_impact_bps))
            if len(self._calibration_data) >= MIN_CALIBRATION_FILLS:
                self._recalibrate()

    def _recalibrate(self) -> None:
        """OLS regression: actual ~ alpha + beta * model."""
        xs = [p[0] for p in self._calibration_data]
        ys = [p[1] for p in self._calibration_data]
        n  = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        ss_xy  = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        ss_xx  = sum((x - x_mean) ** 2 for x in xs)
        if ss_xx < 1e-12:
            return
        self._calibrated_beta  = ss_xy / ss_xx
        self._calibrated_alpha = y_mean - self._calibrated_beta * x_mean
        log.info(
            "MarketImpactModel recalibrated: alpha=%.4f beta=%.4f (n=%d)",
            self._calibrated_alpha, self._calibrated_beta, n,
        )

    def calibration_stats(self) -> dict:
        """Return current calibration parameters and sample count."""
        with self._lock:
            return {
                "n_points":  len(self._calibration_data),
                "alpha":     self._calibrated_alpha,
                "beta":      self._calibrated_beta,
                "eta":       self._eta,
                "gamma":     self._gamma,
                "calibrated": len(self._calibration_data) >= MIN_CALIBRATION_FILLS,
            }
